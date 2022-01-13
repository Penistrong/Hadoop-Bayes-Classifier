package org.penistrong.bayesclassifier.inputformat;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.*;

import java.io.IOException;

public class ClassWordCountCombineTextInputFormat extends CombineFileInputFormat<LongWritable, Text> {

    //继承CombineFileInputFormat，类似CombineTextInputFormat，但是使用了一个自定义的RecordReader包装类
    @Override
    public RecordReader<LongWritable, Text> createRecordReader(InputSplit split, TaskAttemptContext context)
            throws IOException {
        return new CombineFileRecordReader<>(
                (CombineFileSplit)split,
                context,
                ClassWordCountRecordReaderWrapper.class
        );
    }
}
