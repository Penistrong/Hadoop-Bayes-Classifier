package org.penistrong.bayesclassifier;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.log4j.Logger;
import org.penistrong.bayesclassifier.inputformat.UnclassifiedDocCombineInputFormat;
import org.penistrong.bayesclassifier.mapper.ConditionalProbabilityMapper;
import org.penistrong.bayesclassifier.reducer.MaxCondProbReducer;

import java.io.*;
import java.util.*;

/**
 * 朴素贝叶斯分类器驱动类，包括利用存储于HDFS中的类别统计结果计算先验概率P(C_i)和后验概率P(d|C_i)的静态方法
 */
public class NaiveBayesClassifier {
    public static Logger logger = Logger.getLogger(NaiveBayesClassifier.class);

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
                if (classTermCount.length != 3){
                    logger.info("[**DEBUG**] Encounter an abnormal line...Skip this line : ["+line+"]");
                    continue;
                }
                String cls = classTermCount[0];
                String count = classTermCount[2];
                //哈希表中没有该类别对应的键，初始化
                if (!termCounts.containsKey(cls))
                    termCounts.put(cls, 0);
                //更新总数, compute()函数只在给定的键存在时才会执行BiFunction
                termCounts.compute(cls, (k, v) -> v + Integer.parseInt(count));
                //更新该类别的词典大小
                if (!classVocabSize.containsKey(cls))
                    classVocabSize.put(cls, 0);
                classVocabSize.compute(cls, (k, v) -> v + 1);
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

    //将哈希表序列化为二进制文件并存储到HDFS上以供位于别的节点上的不同JVM内的Mapper.class调用
    public static void storeResultAnalysisOnHDFS(Configuration conf) throws IOException {
        FileSystem fs = FileSystem.get(conf);
        conf.set("prior", "/tmp/n-b-c-hashtables/prior");
        FSDataOutputStream out = fs.create(new Path("/tmp/n-b-c-hashtables/prior"), true);
        ObjectOutputStream oos = new ObjectOutputStream(out);
        oos.writeObject(prior);
        oos.close();
        conf.set("termCounts", "/tmp/n-b-c-hashtables/termCounts");
        out = fs.create(new Path("/tmp/n-b-c-hashtables/termCounts"), true);
        oos = new ObjectOutputStream(out);
        oos.writeObject(termCounts);
        oos.close();
        conf.set("classVocabSize", "/tmp/n-b-c-hashtables/classVocabSize");
        out = fs.create(new Path("/tmp/n-b-c-hashtables/classVocabSize"), true);
        oos = new ObjectOutputStream(out);
        oos.writeObject(classVocabSize);
        oos.close();
        conf.set("posterior", "/tmp/n-b-c-hashtables/posterior");
        out = fs.create(new Path("/tmp/n-b-c-hashtables/posterior"), true);
        oos = new ObjectOutputStream(out);
        oos.writeObject(posterior);
        oos.close();
        IOUtils.closeStream(out);
        fs.close();
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

        //注意，由于是分布式集群，其他节点如果想要访问到存储相关信息的HashTable，不能跨JVM访问
        //有一种跨JVM传递变量的方法，那就是在本驱动类中，将这些静态变量写入到JobConf的配置字段中
        //在Mapper和Reducer中使用setup()方法，初始化这些字段即可得到这些变量
        //但是，Configuration类提供的只是设置long,String等简单类型的配置字段，无法传递HashTable
        //所以，将这些HashTable序列化为文件后存储到HDFS上，在Mapper的setup()阶段读取这些文件并反序列化即可
        storeResultAnalysisOnHDFS(conf);

        logger.info("[**DEBUG**] Executed static method to calculate prior and posterior!");
        logger.info("[**DEBUG**] Total Class Num :" + prior.size());
        int show_len = 10;
        logger.info("[**DEBUG**] posterior Size is " + posterior.size() + ", show random " + show_len + " entries :");
        for (Map.Entry<Map.Entry<String, String>, Float> entry : posterior.entrySet()) {
            if (--show_len < 0)
                break;
            logger.info(entry.getKey().getKey() + ": " + entry.getKey().getValue() + "\t" + entry.getValue());
        }

        // 由于在datanode上运行的Mapper需要读取HDFS上的文件，使用的是同一个conf下得到的文件系统
        // 当任务提交到集群上面以后，多个datanode在getFileSystem过程中，由于Configuration一样，会得到同一个FileSystem
        // 如果有一个datanode在使用完关闭连接，其它的datanode在访问就会出现上述异常
        // 禁用FileSystem内部的static cache即可
        conf.setBoolean("fs.hdfs.impl.disable.cache", true);

        Job job = Job.getInstance(conf, "Naive Bayes Classifier");
        job.setJarByClass(NaiveBayesClassifier.class);

        job.setInputFormatClass(UnclassifiedDocCombineInputFormat.class);
        //设置每个CombineFileSplit切片的最大大小为4MB
        UnclassifiedDocCombineInputFormat.setMaxInputSplitSize(job, 4194304);

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
        if (!job.waitForCompletion(true))
            System.exit(1);

        //执行预测结果评估
        Evaluator.evaluate(conf, new Path(givenArgs[2]), FileOutputFormat.getOutputPath(job));
    }
}
