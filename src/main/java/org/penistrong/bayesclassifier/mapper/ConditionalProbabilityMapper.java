package org.penistrong.bayesclassifier.mapper;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.AbstractMap;
import java.util.Hashtable;
import java.util.Map;
import java.util.StringTokenizer;

/**
 * 预测文档所属类别的条件概率Mapper,输入的k-v为<docId, content>，输出的k-v为<docId, <Class, Probability>>
 */
public class ConditionalProbabilityMapper extends Mapper<Text, Text, Text, Text> {

    public static Logger logger = Logger.getLogger(ConditionalProbabilityMapper.class);

    //存储先验概率的哈希表，即<类别C_i， 先验概率P(C_i)>
    public static Hashtable<String, Float> prior = new Hashtable<>();
    //存储类型为C_i的文档中出现的总词数的HashTable
    public static Hashtable<String, Integer> termCounts = new Hashtable<>();
    //存储某类别对应的不同词的词典大小，应用在Add-one Smoothing策略中
    public static Hashtable<String, Integer> classVocabSize = new Hashtable<>();
    //存储词(Term)的条件概率的哈希表，即<词t, 条件概率P(t|C_i)>，给定类别C_i，该词t在该类别中出现的概率
    //这里采用复合哈希表的形式，<<类别C_i, 词t>, 条件概率P(t|C_i)>
    public static Hashtable<Map.Entry<String, String>, Float> posterior = new Hashtable<>();

    private Text clsCondProb = new Text();

    //得到此前驱动类(main所在类)处理统计结果后得到的相关哈希表，从HDFS中获取其序列化文件并使用反序列化得到静态对象
    @Override
    protected void setup(Mapper<Text, Text, Text, Text>.Context context) throws IOException, InterruptedException {
        super.setup(context);
        try {
            decodeResultAnalysisOnHDFS(context.getConfiguration());
        } catch (ClassNotFoundException e) {
            logger.info("[Mapper]Could not deserialize object into HashTable");
            e.printStackTrace();
        }
    }

    public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
        String content = value.toString();
        //从prior,termCounts,classVocabSize这3个哈希表任选其一的键集合中都可以拿到所有的类别名
        for (String cls : prior.keySet()) {
            Double condProb = calConditionalProbabilityForClass(content, cls);
            clsCondProb.set(cls + ":" + condProb);
            context.write(key, clsCondProb);
        }
    }

    //解码相应结果
    public static void decodeResultAnalysisOnHDFS(Configuration conf) throws IOException, ClassNotFoundException {
        FileSystem fs = FileSystem.get(conf);
        FSDataInputStream in = fs.open(new Path(conf.get("prior", "/tmp/n-b-c-hashtables/prior")));
        ObjectInputStream ois = new ObjectInputStream(in);
        prior = (Hashtable<String, Float>) ois.readObject();
        ois.close();
        in = fs.open(new Path(conf.get("termCounts", "/tmp/n-b-c-hashtables/termCounts")));
        ois = new ObjectInputStream(in);
        termCounts = (Hashtable<String, Integer>) ois.readObject();
        ois.close();
        in = fs.open(new Path(conf.get("classVocabSize", "/tmp/n-b-c-hashtables/classVocabSize")));
        ois = new ObjectInputStream(in);
        classVocabSize = (Hashtable<String, Integer>) ois.readObject();
        ois.close();
        in = fs.open(new Path(conf.get("posterior", "/tmp/n-b-c-hashtables/posterior")));
        ois = new ObjectInputStream(in);
        posterior = (Hashtable<Map.Entry<String, String>, Float>) ois.readObject();
        ois.close();
        IOUtils.closeStream(in);
        fs.close();
    }

    /**
     * 根据给定的文档内容和类别名，计算该文档属于该类别的条件概率，注意是用log处理后的结果
     * @param content: 整个文档(小文件)的内容，使用String装载
     * @param cls: 类别名
     * @return 条件概率condProb
     */
    private static Double calConditionalProbabilityForClass(String content, String cls) {
        //首先是log P(C_i)
        Double condProb = Math.log(prior.get(cls));

        //计算该文档所有的词的条件概率 log P(t_k|C_i) 之和，利用StringTokenizer分词处理
        StringTokenizer itr = new StringTokenizer(content);
        String term;
        Map.Entry<String, String> pair = null;  //用以查询哈希表posterior的键
        //从驱动类的静态成员变量中按照键<cls, term>取对应的条件概率，如果不存在该键(训练集中该类别中不含该词)即为固定概率
        Double rawTermCondProb = Math.log(
                1.0f / (termCounts.get(cls) + classVocabSize.get(cls)));
        while (itr.hasMoreTokens()) {
            term = itr.nextToken();
            pair = new AbstractMap.SimpleEntry<>(cls, term);
            if (posterior.containsKey(pair)) {
                condProb += Math.log(posterior.get(pair));
            } else {
                //训练集中该类别未出现该单词
                condProb += rawTermCondProb;
            }
        }

        return condProb;
    }
}
