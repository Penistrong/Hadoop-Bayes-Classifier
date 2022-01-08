package org.penistrong.bayesclassifier;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.util.GenericOptionsParser;
import org.penistrong.bayesclassifier.inputformat.ClassFileSumCombineInputFormat;
import org.penistrong.bayesclassifier.inputformat.ClassWordCountCombineTextInputFormat;
import org.penistrong.bayesclassifier.inputformat.ClassWordCountRecordReaderWrapper;
import org.penistrong.bayesclassifier.outputformat.ClassWordCountOutputFormat;

import java.io.IOException;
import java.util.StringTokenizer;

/**
 * 训练后验概率
 */
public class TrainPosteriorProbability {

    public static class TokenizerCountMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

        private Text pair = new Text();
        private final static IntWritable one = new IntWritable(1);

        /*
        @Override
        protected void setup(Mapper<LongWritable, Text, Text, IntWritable>.Context context) throws IOException, InterruptedException {
            //InputSplit split = context.getInputSplit();
            //获取当前输入的split的路径
            //Path path = ((FileSplit) split).getPath();
            //将其父目录名作为类别名保存到私有变量wordClass中
            //wordClass.set(path.getParent().getName());
            wordClass.set(context.getConfiguration().get("mapreduce.map.classwordcount.currentclass", "wtf!!!"));
        }*/

        /**
         * 每读取1个split，按行读内容，分词后输出<类别+单词, 出现次数>的键值对
         * 这里即<<Class, Word>, 1>注意前面的<Class, Word>使用一个Text存储，用分隔符":"进行分割
         * @param key: 使用的CombineTextInputFormat输出的每行内容的偏移量
         * @param value: 文档的一行内容
         * @param context: 上下文
         * @throws IOException
         * @throws InterruptedException
         */
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            //利用同一JVM下的RecordReader.class中的静态方法得到当前(CombineFileSplit)切片中的实际小文件的路径
            Path filePath = ClassWordCountRecordReaderWrapper.getCurrentSmallFilePath();
            //获取其父目录名
            String wordClass = filePath.getParent().getName();

            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                //使用setup时保存的当前分片所属类别名,去组装"类别,单词",key是偏移量用不到
                pair.set(wordClass + ":" + itr.nextToken());
                context.write(pair, one);
            }
        }
    }

    public static class PairSumCombiner extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable count = new IntWritable();
        /**
         * 本地聚合的Combiner,将<<Class, Word>, 1>进行聚合
         * @param key:
         * @param values
         * @param context
         * @throws IOException
         * @throws InterruptedException
         */
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable value : values)
                sum += value.get();
            count.set(sum);
            context.write(key, count);
        }
    }

    public static class PairSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable count = new IntWritable();

        /**
         * 与Combiner不同的是，要把键拆分为以"\t"分割写入文件
         * @param key:
         * @param values
         * @param context
         * @throws IOException
         * @throws InterruptedException
         */
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable value : values)
                sum += value.get();
            count.set(sum);
            String[] keys = key.toString().split(":");
            key.set(keys[0] + "\t" + keys[1]);
            context.write(key, count);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] givenArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (givenArgs.length < 2) {
            System.err.println("Usage: class word count <in: /path/to/dataset>[<in>...] <out: /path/to/resultFile>");
            System.exit(2);
        }

        Job job = Job.getInstance(conf, "Class Word Count");
        job.setJarByClass(TrainPosteriorProbability.class);
        //job.setInputFormatClass(ClassWordCountCombineTextInputFormat.class);
        //测试:使用CombineTextInputFormat, 在map时处理切分机制
        job.setInputFormatClass(ClassWordCountCombineTextInputFormat.class);
        //设置每个切片的最大大小为4MB，防止所有训练集的所有小文件全部放到1个CombineSplit(这样导致只有1个Mapper运作)
        ClassWordCountCombineTextInputFormat.setMaxInputSplitSize(job, 4194304);

        //设置Mapper,Combiner,Reducer
        job.setMapperClass(TokenizerCountMapper.class);
        job.setCombinerClass(PairSumCombiner.class);
        job.setReducerClass(PairSumReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        //job.setOutputFormatClass(ClassWordCountOutputFormat.class);

        //将给定的多个数据集源路径添加到文件输入路径中
        for (int i = 0; i < givenArgs.length - 1;i++)
            ClassFileSumCombineInputFormat.addInputPath(job, new Path(givenArgs[i]));
        //!设定递归读取树形目录结构下的所有文件
        ClassFileSumCombineInputFormat.setInputDirRecursive(job, true);
        //设定输出路径
        ClassWordCountOutputFormat.setOutputPath(job, new Path(givenArgs[givenArgs.length - 1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
