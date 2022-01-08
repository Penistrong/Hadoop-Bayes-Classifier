package org.penistrong.bayesclassifier.outputformat;

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.TaskAttemptContext;

import java.io.IOException;

public class ClassWordCountRecordWriter extends RecordWriter<Text, IntWritable> {

    private FSDataOutputStream out = null;

    public ClassWordCountRecordWriter(TaskAttemptContext context, Path outputDir) {
        try {
            FileSystem fs = FileSystem.get(context.getConfiguration());
            Path out_file = new Path(outputDir.toString() + "/class-word-count");
            out = fs.create(out_file);
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    @Override
    public void write(Text key, IntWritable val) throws IOException, InterruptedException {
        //将key:"类名,单词"按分割符","进行分割
        //利用制表符，按照"类名\t单词\t出现次数"的方式写入文件
        String[] pair = key.toString().split(",");
        String result = pair[0] + "\t" + pair[1] + "\t" + val.get() + "\r\n";
        out.write(result.getBytes());
    }

    @Override
    public void close(TaskAttemptContext context) throws IOException, InterruptedException {
        IOUtils.closeStream(out);
    }
}
