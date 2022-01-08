package org.penistrong.bayesclassifier.inputformat;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.*;

import java.io.IOException;

public class ClassWordCountCombineTextInputFormat extends CombineFileInputFormat<LongWritable, Text> {

    /*
    @Override
    protected boolean isSplitable(JobContext context, Path filename) {
        return false;
    }*/
    //继承CombineFileInputFormat，类似CombineTextInputFormat，但是使用了一个自定义的RecordReader包装类
    @Override
    public RecordReader<LongWritable, Text> createRecordReader(InputSplit split, TaskAttemptContext context)
            throws IOException {
        return new CombineFileRecordReader<>(
                (CombineFileSplit)split,
                context,
                ClassWordCountRecordReaderWrapper.class
                //ClassWordCountRecordReader.class
                //ClassWordCountCombineTextInputFormat.TextRecordReaderWrapper.class
        );
    }

    /*
    private static class TextRecordReaderWrapper extends CombineFileRecordReaderWrapper<LongWritable, Text> {
        public TextRecordReaderWrapper(CombineFileSplit split, TaskAttemptContext context, Integer idx)
                throws IOException, InterruptedException {
            super(new TextInputFormat(), split, context, idx);
            //获取当前idx对应的每个子split的路径
            Path path = split.getPath(idx);
            Configuration conf = context.getConfiguration();
            //利用配置字段进行传值
            conf.set("mapreduce.map.classwordcount.currentclass", path.getParent().getName());
        }
    }
    */
}
