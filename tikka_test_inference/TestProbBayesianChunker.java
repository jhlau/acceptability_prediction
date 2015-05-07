/*
Author: Jey Han Lau
Date: May 14

Model = Bayesian Chunker 

*/

import tikka.bhmm.models.*;
import tikka.bhmm.model.base.*;
import tikka.bhmm.apps.*;
import tikka.structures.*;
import tikka.opennlp.io.*;
import org.apache.commons.cli.*;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Collections;
import java.util.Comparator;
import java.lang.Math;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.StatUtils;


public class TestProbBayesianChunker extends TestProbAbstract {
    private String chunkerModelFile = "";
    private HMMBase bhmm2 = null;

    protected void addModel2Options(Options options) {
        options.addOption("xchunker", "bayesian-chunker", true, "path of bayesian chunker model");
    }

    protected void setupModel2(Options options, String[] args) {
        try {
            CommandLineParser optparse = new PosixParser();
            CommandLine cline = optparse.parse(options, args);
            if (cline.hasOption("xchunker")) {
                chunkerModelFile = cline.getOptionValue("xchunker");
            }
            String[] args2 = new String[2];
            args2[0] = "-l";
            args2[1] = chunkerModelFile;

            CommandLineOptions modelOptions = new CommandLineOptions(optparse.parse(options, args2));
            SerializableModel serializableModel = new SerializableModel();

            bhmm2 = serializableModel.loadModel(modelOptions, modelOptions.getModelInputPath());
            bhmm2.resetTrainDataDir(); // reset the previous train directory (avoid IOException)
            bhmm2.initializeFromLoadedModel2(modelOptions);

            if (debug) {
                System.out.println("model name = " + bhmm2.getModelName());
                System.out.println("alpha = " + bhmm2.getAlpha());
                System.out.println("beta = " + bhmm2.getBeta());
                System.out.println("gamma = " + bhmm2.getGamma());
                System.out.println("delta = " + bhmm2.getDelta());
                System.out.println("phash = " + bhmm2.getPhash());
                System.out.println("stateC = " + bhmm2.getStateC());
                System.out.println("stateF = " + bhmm2.getStateF());
                System.out.println("stateS = " + bhmm2.getStateS());
                System.out.println("topicK = " + bhmm2.getTopicK());
                System.out.println("chunkA = " + bhmm2.getChunkA());
                System.out.println("wordW = " + bhmm2.getWordW());
                System.out.println("wordN = " + bhmm2.getWordN());
                System.out.println("iterations = " + bhmm2.getIterations());
                System.out.println("random seed = " + bhmm2.getRandomSeed());
                System.out.println("useTrigram = " + bhmm2.getUseTrigram());

                /*
                System.out.println("\nState to StateID:");
                for (Map.Entry entry : bhmm2.getTrainWordIdx().entrySet()) {
                    System.out.println(entry.getKey() + " : " + entry.getValue());
                }*/

                System.out.println("\nTop-100 chunks:");
                int xx = 0;
                int yy = 0;
                int zz = 0;
                List<Map.Entry<String, Integer>> list =
                    new ArrayList<Map.Entry<String, Integer>>(bhmm2.getChunkFreq().entrySet());
                Collections.sort(list, new ValueThenKeyComparator<String, Integer>());
                for (Map.Entry item : list) {
                    String[] parts = ((String) item.getKey()).split("\\+");
                    if (xx < 100) {
                        System.out.println(item.getKey() + " : " + item.getValue());
                    }
                    if (parts.length > 5)  {
                        //System.out.println("\t" + item.getKey() + " : " + item.getValue());
                        yy++;
                    }
                    xx++;
                    zz += (Integer) item.getValue();
                }
                System.out.println();
                System.out.println("Number of chunk types greater than length 5 = " + yy);
                System.out.println("Number of chunk types = " + bhmm2.getChunkFreq().size());
                System.out.println("Number of chunk tokens = " + zz);
            }
            
        } catch (Exception e) {
            System.err.println("Error loading bayesian chunker model");
            System.exit(0);
        }
    }


    protected double computeModel2Probs(int[] stateVector, int[] stateCounts, int wordN,
        ArrayList<Double> m2UnigramLogProbResult, ArrayList<Double> m2WordLogProbResult,
        ArrayList<Double> m2WordMeanLogProbResult, ArrayList<Double> m2StateTranLogProbResult,
        HashMap<String, Double> m2Results)
        throws IOException {
        // compute the probability
        ArrayList<ArrayList<Double>> stateTranProbs = new ArrayList<ArrayList<Double>>();
        ArrayList<ArrayList<Double>> wordStateProbs = new ArrayList<ArrayList<Double>>();
        ArrayList<int[]> dummy = new ArrayList<int[]>();
        HashMap<String, Integer> trainWordIdx = bhmm2.getTrainWordIdx();

        // convert the stateVectors to word ids of model 2
        ArrayList<Integer> testSentence = new ArrayList<Integer>();
        for (int state : stateVector) {
            String key = "S" + state;
            testSentence.add(trainWordIdx.get(key));
        }

        if (debug) {
            System.out.println("-------------------------------------------------------");
            System.out.println("Original stateVector = " + Arrays.toString(stateVector));
            System.out.println("stateVector in ID    = " + convertToString(testSentence));
        }

        ArrayList<Double> sentenceProbs = new ArrayList<Double>();
        bhmm2.computeTestSentenceProb(testSentence, sentenceProbs, stateTranProbs,
            wordStateProbs, dummy);
        double model2Prob = (new DescriptiveStatistics(convertToArray(sentenceProbs))).getMean();

        for (int i=0; i<stateVector.length;i++) {
            int s = stateVector[i];
            DescriptiveStatistics wordStats = new
                DescriptiveStatistics(convertToArray(wordStateProbs.get(i)));
            double wordLogProb = wordStats.getMean();
            double uProb = ((double) stateCounts[s])/wordN;
            double wordMeanLogProb = wordLogProb / Math.log10(uProb) * -1.0;
            m2WordLogProbResult.add(wordLogProb);
            m2UnigramLogProbResult.add(Math.log10(uProb));
            m2WordMeanLogProbResult.add(wordMeanLogProb);
            m2StateTranLogProbResult.add(0.0);
        }

        DescriptiveStatistics stats =
            new DescriptiveStatistics(convertToArray(m2WordMeanLogProbResult));
        m2Results.put("nlogprob_1q", stats.getPercentile(25.0));
        m2Results.put("nlogprob_2q", stats.getPercentile(50.0));
        m2Results.put("nlogprob_mean", stats.getMean());
        m2Results.put("nlogprob_m1q",getMeanQ(m2WordMeanLogProbResult, m2Results.get("nlogprob_1q")));
        m2Results.put("nlogprob_m2q",getMeanQ(m2WordMeanLogProbResult, m2Results.get("nlogprob_2q")));

        // dummy m2Results for statetran
        m2Results.put("statetran_1q", 0.0);
        m2Results.put("statetran_2q", 0.0);
        m2Results.put("statetran_mean", 0.0);
        m2Results.put("statetran_m1q", 0.0);
        m2Results.put("statetran_m2q", 0.0);

        return model2Prob;
    }
    
    protected String convertToString(ArrayList<Integer> x) {
        String z = "";
        for (int y : x) {
            z += (y + " ");
        }
        return z;
    }

    public static void main(String[] args) {
        new TestProbBayesianChunker().go(args);
    }

    private class ValueThenKeyComparator<K extends Comparable<? super K>,
                                        V extends Comparable<? super V>>
        implements Comparator<Map.Entry<K, V>> {

        public int compare(Map.Entry<K, V> a, Map.Entry<K, V> b) {
            int cmp1 = b.getValue().compareTo(a.getValue());
            if (cmp1 != 0) {
                return cmp1;
            } else {
                return b.getKey().compareTo(a.getKey());
            }
        }

    }
}
