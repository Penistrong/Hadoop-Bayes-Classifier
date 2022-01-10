package org.penistrong.bayesclassifier.inputformat;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.CombineFileSplit;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import java.io.IOException;


public class ClassFileSumRecordReader extends RecordReader<Text, IntWritable> {

    private CombineFileSplit combineFileSplit;
    private FileSplit split;    //每个小文件
    private int currentIndex;   //小文件在合并的CombineFileSplit中的索引
    //private Configuration conf;
    private Text key = new Text();
    private final static IntWritable one = new IntWritable(1);
    private boolean processed = false;

    public ClassFileSumRecordReader(CombineFileSplit combineFileSplit, TaskAttemptContext context, Integer index) {
        this.combineFileSplit = combineFileSplit;
        //当前要处理的小文件在CombineFileSplit中的索引
        this.currentIndex = index;
    }

    @Override
    public void initialize(InputSplit inputSplit, TaskAttemptContext taskAttemptContext) throws IOException, InterruptedException {
        //conf = taskAttemptContext.getConfiguration();
        //此时的inputSplit存放的是文件分割数组，包含每个文件的全路径及其长度
        this.combineFileSplit = (CombineFileSplit) inputSplit;
        this.split = new FileSplit(
                combineFileSplit.getPath(currentIndex),
                combineFileSplit.getOffset(currentIndex),
                combineFileSplit.getLength(currentIndex),
                combineFileSplit.getLocations()
        );
    }

    @Override
    public boolean nextKeyValue() throws IOException, InterruptedException {
        if (currentIndex >= 0 && currentIndex < this.combineFileSplit.getNumPaths()) {
            if (!processed) {
                //获取每个小文件所在的路径
                Path path = this.split.getPath();
                //FileSystem fs = path.getFileSystem(conf);
                //获得其上级目录的名称，对应该文档所属的分类ClassName
                key.set(path.getParent().getName());
                processed = true;
                return true;
            }
            //注意，读完一个文件就终止，且不读取文件内容，故nextKeyValue()返回False，转而处理下一个小split
            return false;
        }
        return false;
    }

    @Override
    public Text getCurrentKey() throws IOException, InterruptedException {
        //键值对的键为该文件的类别ClassName
        return key;
    }

    @Override
    public IntWritable getCurrentValue() throws IOException, InterruptedException {
        //键值对的值固定为1
        return one;
    }

    @Override
    public float getProgress() throws IOException, InterruptedException {
        //官方文档:getProgress():How much of the input has the RecordReader consumed
        //由于每个CombineFileSplit里包含许多小文件，因此要计算具体进度
        /*
        if (currentIndex >= 0 && currentIndex < this.paths.length)
            return (float) currentIndex / this.paths.length;
        */
        return processed? 1.0f : 0;
    }

    @Override
    public void close() throws IOException {

    }
}
