package org.penistrong.bayesclassifier.inputformat;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.CombineFileRecordReaderWrapper;
import org.apache.hadoop.mapreduce.lib.input.CombineFileSplit;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;

import java.io.IOException;

public class ClassWordCountRecordReaderWrapper extends CombineFileRecordReaderWrapper<LongWritable, Text> {
    //采用这个包装类的目的是在于，如果直接使用CombineTextInputFormat，在mapper里使用context.getInputSplit()
    //获取的是合并的CombineFileSplit，这样的话没法得到每个小文件的实际路径，这里使用包装类
    //定义一个静态私有成员变量保存当前小文件的路径，由于mapper和recordReader在同一个jvm下工作
    //直接使用CLassWordCountRecordReaderWrapper.getCurrentSmallFilePath()方法获取当前小文件的路径
    private static Path currentSmallFilePath;

    public ClassWordCountRecordReaderWrapper(CombineFileSplit split, TaskAttemptContext context, Integer idx)
        throws IOException, InterruptedException {
        super(new TextInputFormat(), split, context, idx); //与CombineTextInputFormat使用的包装类相似
        FileSplit smallFile = new FileSplit(split.getPath(idx),
                                            split.getOffset(idx),
                                            split.getLength(idx),
                                            split.getLocations());
        currentSmallFilePath = smallFile.getPath();
    }

    public static Path getCurrentSmallFilePath() {
        return currentSmallFilePath;
    }
}
