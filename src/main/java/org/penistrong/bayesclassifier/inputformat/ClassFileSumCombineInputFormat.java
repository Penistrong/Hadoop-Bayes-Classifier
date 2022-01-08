package org.penistrong.bayesclassifier.inputformat;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.CombineFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.CombineFileRecordReader;
import org.apache.hadoop.mapreduce.lib.input.CombineFileSplit;

import java.io.IOException;

public class ClassFileSumCombineInputFormat extends CombineFileInputFormat<Text, IntWritable> {

    //重写isSplitable函数，指明每个文件不可分割，即1个Mapper每次处理1个文件整体
    @Override
    protected boolean isSplitable(JobContext context, Path filename) {
        return false;
    }

    @Override
    public RecordReader<Text, IntWritable> createRecordReader(InputSplit split, TaskAttemptContext context)
            throws IOException {
        return new CombineFileRecordReader<>(
                (CombineFileSplit) split,
                context,
                ClassFileSumRecordReader.class
        );
    }
}
