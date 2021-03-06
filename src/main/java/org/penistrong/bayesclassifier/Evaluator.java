package org.penistrong.bayesclassifier;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.util.hash.Hash;
import org.apache.log4j.Logger;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.stream.Collectors;

public class Evaluator {

    public static Logger logger = Logger.getLogger(Evaluator.class);

    //保存预测的<文档，预测分类>键值对的哈希表prediction
    private static Hashtable<String, String> prediction = new Hashtable<>();
    //保存实际的<文档，真实分类>键值对的哈希表groundTruth
    private static Hashtable<String, String> groundTruth = new Hashtable<>();
    //保存所有类别的HashSet
    private static HashSet<String> trueCls = new LinkedHashSet<>();

    private static Hashtable<String, Integer> TP = new Hashtable<>();
    private static Hashtable<String, Integer> FP = new Hashtable<>();
    private static Hashtable<String, Integer> FN = new Hashtable<>();
    private static Hashtable<String, Integer> TN = new Hashtable<>();

    public static void initialize(Configuration conf, Path testDocPath, Path outputPath) throws IOException {
        //初始化各哈希表
        FileSystem fs = FileSystem.get(conf);
        Path predictionPath = new Path(outputPath, "part-r-00000");
        FSDataInputStream in = null;
        try {
            in = fs.open(predictionPath);
            BufferedReader line_reader = new BufferedReader(new InputStreamReader(in));
            String line;
            while((line = line_reader.readLine()) != null){
                String[] splits = line.split("\t");
                prediction.put(splits[0], splits[1]);
            }
        } finally {
            IOUtils.closeStream(in);
        }
        //遍历测试文件所在目录，采用递归遍历得到所有文件(而不是文件夹)，并据此获得其父目录名作为类别
        RemoteIterator<LocatedFileStatus> files = fs.listFiles(testDocPath, true);
        while (files.hasNext()) {
            LocatedFileStatus file = files.next();
            groundTruth.put(file.getPath().getName(), file.getPath().getParent().getName());
        }
        //获得测试文件的所有类型,初始化各储存分类结果的哈希表,且利用自动去重的HashSet统计所有的类别
        groundTruth.forEach( (docId, cls) -> { TP.put(cls, 0); trueCls.add(cls); });
        FP.putAll(TP);
        FN.putAll(TP);
        TN.putAll(TP);
    }

    //Evaluation静态方法，统计TP,FP,FN,TN，并计算精度P,召回率R,调和平均数F1-score
    public static void evaluate(Configuration conf, Path testDocPath, Path outputPath) throws IOException {
        logger.info("[EVALUATION] Begin evaluation, calculating TP,FP,FN,TN,Precision,Recall and F1-score...");
        initialize(conf, testDocPath, outputPath);
        //首先计算各个类别的四个参数TP,FP,FN,TN
        //采取两趟扫描，第一趟遍历预测分类结果哈希表，根据预测分类是否映射到真实分类更新TP与FP
        for (Map.Entry<String, String> kv : prediction.entrySet()) {
            String docId = kv.getKey();
            String predict_cls = kv.getValue();
            if (groundTruth.get(docId).equals(predict_cls))
                //预测分类与真实分类都为A,TP++
                TP.compute(predict_cls, (cls, curTP) -> curTP + 1);
            else
                //预测分类为A,但是真实分类不是A,FP++
                FP.compute(predict_cls, (cls, curFP) -> curFP + 1);
        }
        //第二趟遍历分类哈希表，统计所有真实类型为A，但是分类器结果不是A的文档个数，据此更新FN
        for (String A : trueCls) {
            for (Map.Entry<String, String> kv : groundTruth.entrySet()) {
                //首先过滤出真实类型为A的文档名kv.getKey()，查看其在prediction中的预测分类是否为A(仅处理预测分类不为A的)
                if (kv.getValue().equals(A) && !prediction.get(kv.getKey()).equals(A))
                    //真实类型为A，但分类器结果不是A
                    FN.compute(A, (cls, curFN) -> curFN + 1);
                else if (!prediction.get(kv.getKey()).equals(A))
                    //真实类型不是A，且分类器结果也不是A
                    TN.compute(A, (cls, curTN) -> curTN + 1);
            }
        }
        logger.info("[EVALUATION] Evaluated "+groundTruth.size()+" files of "+trueCls.size()+" classes");
        //利用宏平均Macro-Averaging计算，分别计算每个类各自的P,R,F1，再求平均
        Hashtable<String, Float> P_perCls = new Hashtable<>();
        Hashtable<String, Float> R_perCls = new Hashtable<>();
        Hashtable<String, Float> F1_perCls = new Hashtable<>();
        //如果遇到TP=FP=0或者TP=FN=0的情况，注意要防止0/0情况出现
        TP.forEach((cls, curTP) -> {
            float p = (curTP == 0 && FP.get(cls) == 0) ? 0: ((float)curTP) / (curTP + FP.get(cls));
            float r = (curTP == 0 && FN.get(cls) == 0) ? 0: ((float)curTP) / (curTP + FN.get(cls));
            P_perCls.put(cls, p);
            R_perCls.put(cls, r);
            F1_perCls.put(cls, (p == 0 && r == 0)? 0: 2*p*r/(p+r));
        });
        double P = P_perCls.values().stream().collect(Collectors.summarizingDouble(v -> v)).getAverage();
        double R = R_perCls.values().stream().collect(Collectors.summarizingDouble(v -> v)).getAverage();
        double F1 = F1_perCls.values().stream().collect(Collectors.summarizingDouble(v -> v)).getAverage();

        logger.info("[EVALUATION] After {Macro-Averaging}, Precision="+P+" Recall="+R+" F1-score="+F1);

        //利用微平均Micro-Averaging计算，将各个类的TP,FP,FN,TN合并后再计算具体的评估参数
        long tp = TP.values().stream().collect(Collectors.summarizingInt(v -> v)).getSum();
        long fp = FP.values().stream().collect(Collectors.summarizingInt(v -> v)).getSum();
        long fn = FN.values().stream().collect(Collectors.summarizingInt(v -> v)).getSum();
        long tn = TN.values().stream().collect(Collectors.summarizingInt(v -> v)).getSum();
        P = ((double)tp) / (tp + fp);
        R = ((double)tp) / (tp + fn);
        F1 = 2 * P * R / (P + R);

        logger.info("[EVALUATION] After {Micro-Averaging}, total TP="+tp+" FP="+fp+" FN="+fn+" TN="+tn);
        logger.info("[EVALUATION] Precision="+P+" Recall="+R+" F1-score="+F1);
    }
}
