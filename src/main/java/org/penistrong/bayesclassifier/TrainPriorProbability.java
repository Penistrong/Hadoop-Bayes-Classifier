package org.penistrong.bayesclassifier;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.penistrong.bayesclassifier.inputformat.ClassFileSumCombineInputFormat;
import org.penistrong.bayesclassifier.inputformat.ClassWordCountCombineTextInputFormat;

import java.io.IOException;

/**
 * 训练先验概率，只统计不同类别的文档总数，而不实际处理文档内容
 * 目录结构如下
 * --NBCorpus
 *   --Country
 *     --<Country Name as Class Name>
 *   --Industry
 *     --<Industry Index as Class Name>
 */
public class TrainPriorProbability {

    public static class ClassFileSumMapper extends Mapper<Text, IntWritable, Text, IntWritable> {
        /**
         * 每读取一个文件，不需要计算其中词频，输入<ClassName, 1>并直接输出
         * @param key: 文档所属类别ClassName
         * @param value: 文档本身所占的1文件数
         * @param context: Mapper上下文
         * @throws IOException: IO异常
         * @throws InterruptedException: 中断异常
         */
        public void map(Text key, IntWritable value, Context context)
                throws IOException, InterruptedException {
            context.write(key, value);
        }
    }

    public static class ClassFileSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();
        /**
         * Map生成键值对<k,v>后框架会合并它们并利用Reducer处理<k, values>
         * 作为Combiner时,values为list<IntWritable(1)>,于执行Map任务的节点本地进行局部聚合
         * 作为Reducer时，values为list<IntWritable(local count)>,发送给执行Reduce任务的节点进行最终聚合
         * @param key: 类别
         * @param values: 类别k的局部文档个数
         * @param context: Reducer上下文
         */
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for(IntWritable value : values)
                sum += value.get();
            result.set(sum);
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] givenArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (givenArgs.length < 2) {
            System.err.println("Usage: class files num count <in: /path/to/dataset>[<in>...] <out: /path/to/resultFile>");
            System.exit(2);
        }
        Job job = Job.getInstance(conf, "class files num count");
        job.setJarByClass(TrainPriorProbability.class);

        //设置自定义InputFormat
        job.setInputFormatClass(ClassFileSumCombineInputFormat.class);
        //设置每个切片的最大大小为4MB，防止所有训练集的所有小文件全部放到1个CombineSplit(这样导致只有1个Mapper运作)
        ClassWordCountCombineTextInputFormat.setMaxInputSplitSize(job, 4194304);
        //设置Mapper,Combiner,Reducer
        job.setMapperClass(ClassFileSumMapper.class);
        job.setCombinerClass(ClassFileSumReducer.class);
        job.setReducerClass(ClassFileSumReducer.class);

        //输出键值对的类型定义，输出文件每一行为<类别ClassName, 文档个数TotalCount>
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        //将给定的多个数据集源路径添加到文件输入路径中
        for (int i = 0; i < givenArgs.length - 1;i++)
            ClassFileSumCombineInputFormat.addInputPath(job, new Path(givenArgs[i]));
        //!设定递归读取树形目录结构下的所有文件
        ClassFileSumCombineInputFormat.setInputDirRecursive(job, true);
        //设定输出路径
        FileOutputFormat.setOutputPath(job, new Path(givenArgs[givenArgs.length - 1]));

        //int exitCode = ToolRunner.run(conf, new TrainPosteriorProbability(), args);
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
