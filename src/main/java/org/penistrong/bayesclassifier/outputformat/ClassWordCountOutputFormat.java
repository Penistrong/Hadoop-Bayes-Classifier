package org.penistrong.bayesclassifier.outputformat;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;

public class ClassWordCountOutputFormat extends FileOutputFormat<Text, IntWritable> {
    @Override
    public RecordWriter<Text, IntWritable> getRecordWriter(TaskAttemptContext context) throws IOException, InterruptedException {
        //将上下文和当前输出文件夹传递给自定义的RecordWriter构造函数
        //在configuration中，输出路径的键为"mapreduce.output.fileoutputformat.outputdir"
        return new ClassWordCountRecordWriter(context, getOutputPath(context));
    }
}
