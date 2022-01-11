package org.penistrong.bayesclassifier.inputformat;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.CombineFileSplit;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.log4j.Logger;
import org.penistrong.bayesclassifier.NaiveBayesClassifier;

import java.io.IOException;

public class UnclassifiedDocRecordReader extends RecordReader<Text, Text> {
    //对于每个小文件，一次读取整个文件内容作为value
    private Configuration conf;
    private FileSystem fs;
    private CombineFileSplit combineFileSplit;
    private FileSplit smallSplit;
    private int curIdx;
    private Text key = new Text();      //docId, 直接使用文件名
    private Text content = new Text();  //Content，小文件的全部内容
    private boolean processed = false;

    public UnclassifiedDocRecordReader(CombineFileSplit split, TaskAttemptContext context, Integer idx) {
        this.combineFileSplit = split;
        this.curIdx = idx;
    }

    @Override
    public void initialize(InputSplit split, TaskAttemptContext context) throws IOException, InterruptedException {
        this.conf = context.getConfiguration();
        this.fs = FileSystem.get(conf);
        this.combineFileSplit = (CombineFileSplit) split;
        this.smallSplit = new FileSplit(
                combineFileSplit.getPath(curIdx),
                combineFileSplit.getOffset(curIdx),
                combineFileSplit.getLength(curIdx),
                combineFileSplit.getLocations()
        );
    }

    @Override
    public boolean nextKeyValue() throws IOException, InterruptedException {
        if (curIdx >= 0 && curIdx < this.combineFileSplit.getNumPaths()) {
            if (!processed) {
                Path path = this.smallSplit.getPath();
                //获得该小文件的文件名，并填入key(Text)中
                key.set(path.getName());
                //新建字节缓冲数组，长度为该小文件的长度
                int len = (int) smallSplit.getLength();
                byte[] buffer = new byte[len];
                //利用IOUtils.readFully一次读入所有内容
                FSDataInputStream in = fs.open(path);
                IOUtils.readFully(in, buffer, 0, len);
                //buffer数组填入value(Text)中
                content.set(buffer);
                //关闭流
                IOUtils.closeStream(in);
                processed = true;
                return true;
            }
            return false;
        }
        return false;
    }

    @Override
    public Text getCurrentKey() throws IOException, InterruptedException {
        return key;
    }

    @Override
    public Text getCurrentValue() throws IOException, InterruptedException {
        return content;
    }

    @Override
    public float getProgress() throws IOException, InterruptedException {
        return processed? 1.0f : 0;
    }

    @Override
    public void close() throws IOException {
        fs.close();
    }
}
