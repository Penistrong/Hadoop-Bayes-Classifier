package org.penistrong.bayesclassifier.reducer;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.*;

/**
 * 输入的是<docId, List<Class, CondProb>>，输出<docId, maxCondProbClassName>
 */
public class MaxCondProbReducer extends Reducer<Text, Text, Text, Text> {

    private Text maxCondProbCls = new Text();

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        //拆分以字符串"Class:CondProb"形式保存的value，并找到最大值
        Map<String, Double> clsCondProbs = new HashMap<>();
        for (Text val : values) {
            String[] splits = val.toString().split(":");
            clsCondProbs.put(splits[0], Double.parseDouble(splits[1]));
        }
        //按条件概率(value)的自然排序comparator进行排序，直接从Optional容器中取出Entry并拿出对应的键(对应的最大条件概率类别maxCls)
        maxCondProbCls.set(clsCondProbs.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey());
        //输出<docId, maxCls>
        //logger.info("[Reducer]Generated reduce result <k-v>:=<"+key+"-"+maxCondProbCls+">");
        context.write(key, maxCondProbCls);
    }
}
