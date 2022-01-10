package org.penistrong.bayesclassifier;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.penistrong.bayesclassifier.inputformat.UnclassifiedDocCombineInputFormat;
import org.penistrong.bayesclassifier.mapper.ConditionalProbabilityMapper;
import org.penistrong.bayesclassifier.reducer.MaxCondProbReducer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.AbstractMap;
import java.util.Hashtable;
import java.util.Map;

/**
 * 朴素贝叶斯分类器驱动类，包括利用存储于HDFS中的类别统计结果计算先验概率P(C_i)和后验概率P(d|C_i)的静态方法
 */
public class NaiveBayesClassifier {
    //存储先验概率的哈希表，即<类别C_i， 先验概率P(C_i)>
    public static Hashtable<String, Float> prior = new Hashtable<>();
    //存储类型为C_i的文档中出现的总词数的HashTable
    public static Hashtable<String, Integer> termCounts = new Hashtable<>();
    //存储某类别对应的不同词的词典大小，应用在Add-one Smoothing策略中
    public static Hashtable<String, Integer> classVocabSize = new Hashtable<>();
    //存储词(Term)的条件概率的哈希表，即<词t, 条件概率P(t|C_i)>，给定类别C_i，该词t在该类别中出现的概率
    //这里采用复合哈希表的形式，<<类别C_i, 词t>, 条件概率P(t|C_i)>
    public static Hashtable<Map.Entry<String, String>, Float> posterior = new Hashtable<>();

    /**
     * 计算先验概率P(C_i)
     * @param conf: Hadoop Configuration
     * @param fileCountResult: 每行以 <Class \t fileCount> 形式存储的每个类别对应的文档总数统计结果的HDFS文件路径
     */
    public static void calculatePrior(Configuration conf, Path fileCountResult) throws IOException {
        FileSystem fs = FileSystem.get(conf);
        FSDataInputStream in = null;
        BufferedReader line_reader = null;
        String line;
        //临时使用的HashTable<String, Int>储存各类别对应的文档数
        Hashtable<String, Integer> classFileCount = new Hashtable<>();
        try {
            int sum = 0;
            in = fs.open(fileCountResult);
            line_reader = new BufferedReader(new InputStreamReader(in));
            while ((line = line_reader.readLine()) != null) {
                String[] kv = line.split("\t");
                //忽略统计结果文件中的异常行(不是 C_i \t Count 形式的行)
                if (kv.length != 2) continue;
                classFileCount.put(kv[0], Integer.parseInt(kv[1]));
                //累加文档总数
                sum += Integer.parseInt(kv[1]);
            }
            //按行读取统计结果完毕后，计算先验概率并保存至HashTable
            float finalSum = (float) sum;
            classFileCount.forEach((k, v) -> {
                prior.put(k, v / finalSum);
            });
        }finally {
            //关闭相关输入流及文件系统
            if (line_reader != null)
                line_reader.close();
            IOUtils.closeStream(in);
            fs.close();
        }
    }

    /**
     * 计算后验概率，利用词独立性假设: P(d|C_i)=P(t_1,t_2,...,t_{n_d}|C_i)=\prod_{1<=k<=n_d} P(t_k|C_i)
     * 为了防止出现在预测阶段存在没有在训练集里出现过的单词而使\hat{P}(t_k|C_i)=0，使用Add-one Smoothing策略
     * @param conf: Hadoop Configuration
     * @param wordCountResult: 每行以 <Class \t Word \t Count> 形式存储的各类别文档中不同词出现次数的统计结果的HDFS文件路径
     */
    public static void calculatePosterior(Configuration conf, Path wordCountResult) throws IOException {
        FileSystem fs = FileSystem.get(conf);
        FSDataInputStream in = null;
        BufferedReader line_reader = null;
        String line;

        try{
            in = fs.open(wordCountResult);
            line_reader = new BufferedReader(new InputStreamReader(in));

            //两趟扫描
            //第一趟: 统计类型为C_i的文档中出现的term的总次数 以及 不同term的个数(词典大小|V|)
            while ((line = line_reader.readLine()) != null) {
                String[] classTermCount = line.split("\t");
                //忽略异常行
                if (classTermCount.length != 3) continue;
                String cls = classTermCount[0];
                String count = classTermCount[2];
                //哈希表中没有该类别对应的键，初始化
                if (!termCounts.containsKey(cls))
                    termCounts.put(cls, 0);
                //更新总数, compute()函数只在给定的键存在时才会执行BiFunction
                termCounts.compute(cls, (k, v) -> {
                   return v + Integer.parseInt(count);
                });
                //更新该类别的词典大小
                if (!classVocabSize.containsKey(cls))
                    classVocabSize.put(cls, 0);
                classVocabSize.compute(cls, (k, v) -> {
                    return v + 1;
                });
            }
            line_reader.close();
            IOUtils.closeStream(in);

            //第二趟: 计算P(t_k|C_i)并存储在posterior中
            //注意条件概率的计算要使用Add-one Smoothing策略
            //P(t_k|C_i)=(T_{ct} + 1) / (\sum_{t' \in V}T_{ct'} + |V|)
            in = fs.open(wordCountResult);
            line_reader = new BufferedReader(new InputStreamReader(in));
            String cls, term, count;
            while( (line = line_reader.readLine()) != null) {
                String[] classTermCount = line.split("\t");
                if (classTermCount.length != 3) continue;
                cls = classTermCount[0];
                term = classTermCount[1];
                count = classTermCount[2];
                posterior.put(new AbstractMap.SimpleEntry<>(cls, term),
                        (Float.parseFloat(count) + 1) / (termCounts.get(cls) + classVocabSize.get(cls)));
            }
        }finally {
            if (line_reader != null) line_reader.close();
            IOUtils.closeStream(in);
            fs.close();
        }
    }

    public static void main(String[] args) throws Exception{
        Configuration conf = new Configuration();
        String[] givenArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        //至少给出4个参数，前两个为上一阶段的类别文档总数和类别文档分词出现次数的统计结果
        //而后给出要预测的未分类文档所在的目录，可以给出多个
        //最后一个为预测结果的输出目录
        if (givenArgs.length < 4) {
            System.err.println("Usage: NaiveBayesClassifier " +
                    "<in 1: /path/to/classFileCount> " +
                    "<in 2: /path/to/classWordCount> " +
                    "<in 3: /path/to/unclassified_docs> [<in 3>...]" +
                    "<out: /path/to/prediction_result>");
            System.exit(2);
        }
        //调用静态方法计算先验概率与后验概率
        calculatePrior(conf, new Path(givenArgs[0]));
        calculatePosterior(conf, new Path(givenArgs[1]));

        Job job = Job.getInstance(conf, "Naive Bayes Classifier");
        job.setJarByClass(NaiveBayesClassifier.class);

        job.setInputFormatClass(UnclassifiedDocCombineInputFormat.class);
        job.setMapperClass(ConditionalProbabilityMapper.class);
        job.setReducerClass(MaxCondProbReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        //添加输入目录
        for (int i = 2; i < givenArgs.length - 1; i++)
            UnclassifiedDocCombineInputFormat.addInputPath(job, new Path(givenArgs[i]));
        //设置递归处理所有输入路径下不同级目录中的文件
        UnclassifiedDocCombineInputFormat.setInputDirRecursive(job, true);
        //设定输出路径
        FileOutputFormat.setOutputPath(job, new Path(givenArgs[givenArgs.length - 1]));

        //执行Job
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
