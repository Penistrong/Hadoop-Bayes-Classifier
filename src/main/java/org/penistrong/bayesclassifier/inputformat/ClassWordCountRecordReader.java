package org.penistrong.bayesclassifier.inputformat;

import com.google.common.base.Charsets;
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
import org.apache.hadoop.mapreduce.lib.input.SplitLineReader;

import java.io.IOException;

public class ClassWordCountRecordReader extends RecordReader<Text, Text> {

    private Configuration conf;
    private CombineFileSplit combineFileSplit;
    private FileSplit split;
    private int curIdx;
    private int totalSum;
    private Text key = new Text();
    private Text value = new Text();
    //processed用来判断当前的小split是否已读取完成
    private boolean processed = false;
    //缓冲区
    private byte[] contents;
    //输入流
    private FSDataInputStream fileIn;
    //按行读取的LineReader
    private SplitLineReader in;
    //分割行的分隔符的字节形式
    private byte[] recordDelimiterBytes;

    public ClassWordCountRecordReader(CombineFileSplit combineFileSplit, TaskAttemptContext context, Integer idx) {
        this.conf = context.getConfiguration();
        this.combineFileSplit = combineFileSplit;
        this.curIdx = idx;
        this.totalSum = combineFileSplit.getPaths().length;
    }

    @Override
    public void initialize(InputSplit inputSplit, TaskAttemptContext context) throws IOException, InterruptedException {
        this.combineFileSplit = (CombineFileSplit) inputSplit;
        this.split = new FileSplit(
                combineFileSplit.getPath(curIdx),
                combineFileSplit.getOffset(curIdx),
                combineFileSplit.getLength(curIdx),
                combineFileSplit.getLocations()
        );
        this.contents = new byte[ (int) split.getLength()];
        //获得文件系统
        final Path file = split.getPath();
        final FileSystem fs = file.getFileSystem(conf);
        this.fileIn = fs.open(file);
        //获得分隔符的字节
        String delimiter = context.getConfiguration().get(
                "textinputformat.record.delimiter");
        byte[] recordDelimiterBytes = null;
        if (null != delimiter)
            recordDelimiterBytes = delimiter.getBytes(Charsets.UTF_8);

        this.in = new SplitLineReader(fileIn, recordDelimiterBytes);
    }

    @Override
    public boolean nextKeyValue() throws IOException, InterruptedException {
        if (curIdx >= 0 && curIdx < this.totalSum) {
            if (!processed) {
                in.readLine(key);
                processed = true;
            }
            return true;
        }
        return false;
    }

    @Override
    public Text getCurrentKey() throws IOException, InterruptedException {
        return key;
    }

    @Override
    public Text getCurrentValue() throws IOException, InterruptedException {
        return value;
    }

    @Override
    public float getProgress() throws IOException, InterruptedException {
        return 0;
    }

    @Override
    public void close() throws IOException {
        IOUtils.closeStream(this.fileIn);
    }
}
