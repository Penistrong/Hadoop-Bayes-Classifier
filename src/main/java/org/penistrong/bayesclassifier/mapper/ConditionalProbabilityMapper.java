package org.penistrong.bayesclassifier.mapper;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.penistrong.bayesclassifier.NaiveBayesClassifier;

import java.io.IOException;
import java.util.AbstractMap;
import java.util.Map;
import java.util.StringTokenizer;

/**
 * 预测文档所属类别的条件概率Mapper,输入的k-v为<docId, content>，输出的k-v为<docId, <Class, Probability>>
 */
public class ConditionalProbabilityMapper extends Mapper<Text, Text, Text, Text> {

    private Text clsCondProb = new Text();

    public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
        String content = value.toString();
        //从prior,termCounts,classVocabSize这3个哈希表任选其一的键集合中都可以拿到所有的类别名
        for (String cls : NaiveBayesClassifier.prior.keySet()) {
            Double condProb = calConditionalProbabilityForClass(content, cls);
            clsCondProb.set(cls + ":" + condProb);
            context.write(key, clsCondProb);
        }
    }

    /**
     * 根据给定的文档内容和类别名，计算该文档属于该类别的条件概率，注意是用log处理后的结果
     * @param content: 整个文档(小文件)的内容，使用String装载
     * @param cls: 类别名
     * @return 条件概率condProb
     */
    private static Double calConditionalProbabilityForClass(String content, String cls) {
        //首先是log P(C_i)
        Double condProb = Math.log(NaiveBayesClassifier.prior.get(cls));

        //计算该文档所有的词的条件概率 log P(t_k|C_i) ，利用StringTokenizer分词处理
        StringTokenizer itr = new StringTokenizer(content);
        String term;
        Map.Entry<String, String> pair = null;  //用以查询哈希表posterior的键
        //从驱动类的静态成员变量中按照键<cls, term>取对应的条件概率，如果不存在该键(训练集中该类别中不含该词)即为固定概率
        Double rawTermCondProb = Math.log(
                1.0f / (NaiveBayesClassifier.termCounts.get(cls) + NaiveBayesClassifier.classVocabSize.get(cls)));
        while (itr.hasMoreTokens()) {
            term = itr.nextToken();
            pair = new AbstractMap.SimpleEntry<>(cls, term);
            if (NaiveBayesClassifier.posterior.containsKey(pair)) {
                condProb += Math.log(NaiveBayesClassifier.posterior.get(pair));
            } else {
                //训练集中该类别未出现该单词
                condProb += rawTermCondProb;
            }
        }

        return condProb;
    }
}
