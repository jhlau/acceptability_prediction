/*
Author: Jey Han Lau
Date: May 14

Abstract class that takes in HMM and a second model file: first induce the word classes using HMM,
next feed in the induced word classes to model 2 and compute sentence probabilities.

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
import java.io.File;
import java.io.PrintWriter;
import java.io.FileReader;
import java.io.BufferedReader;
import java.lang.Math;
import java.util.Collections;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.StatUtils;


public abstract class TestProbAbstract {
    protected boolean debug = false;

    protected abstract void setupModel2(Options options, String[] args);

    protected abstract double computeModel2Probs(int[] stateVector, int[] stateCounts, int wordN,
        ArrayList<Double> m2UnigramLogProbResult, ArrayList<Double> m2WordLogProbResult,
        ArrayList<Double> m2WordMeanLogProbResult, ArrayList<Double> m2StateTranLogProbResult,
        HashMap<String, Double> m2Results) throws IOException;

    protected abstract void addModel2Options(Options options);

    protected void go(String[] args) {
        CommandLineParser optparse = new PosixParser();
        CommandLine cline = null;
        Options options = setOptions();
        addModel2Options(options);
        String testFile = "";
        int maxSentLen = 0;
        boolean individualProbs = false;

        try {
            cline = optparse.parse(options, args);
            if (cline.hasOption('h')) {
                HelpFormatter formatter = new HelpFormatter();
                formatter.printHelp("java TestProb", options);
                System.exit(0);
            }
            if (cline.hasOption("xdebug")) {
                debug = true;
            }
            if (cline.hasOption("z")) {
                testFile = cline.getOptionValue("z");
            }
            if (cline.hasOption("xms")) {
                maxSentLen = Integer.parseInt(cline.getOptionValue("xms"));
            }
            if (cline.hasOption("xnoindprob")) {
                individualProbs = false;
            }
        } catch (ParseException parseException) {
            System.err.println("Error parsing command line arguments");
            System.exit(0);
        }

        // load the model
        HMMBase bhmm = null;
        try {
            CommandLineOptions modelOptions = new CommandLineOptions(cline);
            SerializableModel serializableModel = new SerializableModel();

            bhmm = serializableModel.loadModel(modelOptions,
                modelOptions.getModelInputPath());
            bhmm.resetTrainDataDir(); // reset the previous train directory (avoid IOException)
            bhmm.initializeFromLoadedModel2(modelOptions);
        } catch (IOException e) {
            System.err.println("Error loading hmm model");
            System.exit(0);
        }
        // setup model 2
        setupModel2(options, args);

        // get unigram counts
        HashMap<Integer, Integer> wordCounts = new HashMap<Integer, Integer>();
        int wordN = bhmm.getWordN();
        int[] wordVector = bhmm.getWordVector();
        for (int i = 0; i < wordN; i++) {
            int wordId = wordVector[i];
            int count = wordCounts.containsKey(wordId) ? wordCounts.get(wordId) : 0;
            wordCounts.put(wordId, count + 1);
        }

        // print the csv header
        System.out.print("id,ppl,sent_length,logprob,unigram_logprob," +
            "mean_logprob,norm_logprob_div,norm_logprob_sub,slor,");

        if (individualProbs) {
            System.out.println(
            "wlogprob-bot-1,wlogprob-bot-2,wlogprob-bot-3,wlogprob-bot-4,wlogprob-bot-5," +
            "wlogprob_mean,wlogprob_m1q,wlogprob_m2q");
        } else {
            System.out.println();
        }

        // read in test sentences and compute various probability measures
        try{
            DataReader dataReader = new Conll2kReader(new File(testFile));
            HashMap<String, Integer> trainWordIdx = bhmm.getTrainWordIdx();
            String[][] sentence;
            int line_id = 0;

            while ((sentence = dataReader.nextSequence()) != null) {
                ArrayList<Integer> testSentence = new ArrayList<Integer>();
                String text = "";
                String textInWordId = "";
                int numWords = 0;
                for (String[] line : sentence) {
                    String word = "";
                    int wordId = 0;
                    if (!trainWordIdx.containsKey(line[0])) {
                        word = "UNK";
                    } else {
                        word = line[0];
                    }
                    wordId = trainWordIdx.get(word);
                    try {
                        testSentence.add(wordId);
                    } catch (Exception e) {
                        System.err.println("Error getting word = " + word +
                            " from training data; Error = " + e.getMessage());
                        System.exit(0);
                    }
                    text += word + " ";
                    textInWordId += wordId + " ";
                    numWords += 1;
                }

                if (debug) {
                    System.out.println("===============================================");
                    System.out.println("Sentence Text = " + text);
                }

                // check sentence length; generate dummy "inf" values if sentence length exceeds
                // maxSentLen
                if ((maxSentLen == 0) || (numWords <= maxSentLen)) {

                    // compute the probability
                    ArrayList<ArrayList<Double>> stateTranProbs = new ArrayList<ArrayList<Double>>();
                    ArrayList<ArrayList<Double>> wordStateProbs = new ArrayList<ArrayList<Double>>();
                    ArrayList<Double> sentenceProbs = new ArrayList<Double>();
                    ArrayList<int[]> stateVectors = new ArrayList<int[]>();

                    bhmm.computeTestSentenceProb(testSentence, sentenceProbs, stateTranProbs,
                        wordStateProbs, stateVectors);

                    // call to model 2
                    ArrayList<Double> m2WordLogProbResult = new ArrayList<Double>();
                    ArrayList<Double> m2WordMeanLogProbResult = new ArrayList<Double>();
                    ArrayList<Double> m2StateTranLogProbResult = new ArrayList<Double>();
                    ArrayList<Double> m2UnigramLogProbResult = new ArrayList<Double>();
                    HashMap<String, Double> m2Results = new HashMap<String, Double>();
                    double model2Prob = computeModel2Probs(stateVectors.get(0), bhmm.getStateCounts(),
                        wordN, m2UnigramLogProbResult, m2WordLogProbResult, m2WordMeanLogProbResult,
                        m2StateTranLogProbResult, m2Results);

                    // compute various sentence probs
                    double seqLogProb = model2Prob;
                    double unigramLogProb = calcStateUnigramLogProb(stateVectors.get(0),
                        bhmm.getStateCounts(), wordN);
                    double seqWLogProb = seqLogProb / numWords;
                    double seqFMeanLogProb = (seqLogProb / unigramLogProb) * -1.0;
                    double seqFMeanLogProb2 = seqLogProb - unigramLogProb;
                    double seqSlor = seqFMeanLogProb2 / numWords;
                    double seqPplex = Math.pow(10, (-1.0 * seqWLogProb));

                    String outputList = "";
                    //get the bottom five probabilities
                    Collections.sort(m2UnigramLogProbResult);
                    Collections.sort(m2WordLogProbResult);
                    Collections.sort(m2WordMeanLogProbResult);
                    Collections.sort(m2StateTranLogProbResult);
                    List<ArrayList<Double>> allResults = new ArrayList<ArrayList<Double>>();
                    //allResults.add(m2UnigramLogProbResult);
                    //allResults.add(m2WordLogProbResult);
                    allResults.add(m2WordMeanLogProbResult);
                    //allResults.add(m2StateTranLogProbResult);
                    // bot-5 results
                    for (int v=0; v<allResults.size();v++) {
                        int u = 0;
                        for (double val : allResults.get(v)) {
                            outputList = outputList + "," + val;
                            u++;
                            if (u >= 5) {
                                break;
                            }
                        }
                    }

                    // print the probabilities
                    System.out.print(line_id + ",," + numWords + "," + seqLogProb +
                        "," + unigramLogProb + "," + seqWLogProb + "," + seqFMeanLogProb+ "," +
                         seqFMeanLogProb2 + "," + seqSlor);
                    if (individualProbs) {
                        System.out.println(outputList + "," +
                        m2Results.get("nlogprob_mean") + "," + m2Results.get("nlogprob_m1q") +
                        "," + m2Results.get("nlogprob_m2q"));
                    } else {
                        System.out.println();
                    }
                } else {
                    System.out.print(line_id + ",," + numWords);
                    for (int i=0; i<37; i++) {
                        System.out.print(",inf");
                    }
                    System.out.println();
                }

                line_id += 1;
            }
        } catch (IOException e) {}
        System.out.flush();

    }

    /* function to collect options */
    protected static Options setOptions() {

        Options options = new Options();
        options.addOption("h", "help", false, "print help");
        options.addOption("l", "hmm-model", true, "path of hmm model");
        options.addOption("z", "test-sentences", true, "text file that contains the test " +
            "sentences (one sentence per line format)");
        options.addOption("xdebug", "debug-mode", false, "debug mode");
        options.addOption("xms", "max-sent-length", true, "maximum sentence length " +
            "(0 = no restriction; default = 0)");
        options.addOption("xnoindprob", "no-individual-probs", false, "disable outputting " +
            "individual probabilities");

        return options;
    }

    protected double[] convertToArray(ArrayList<Double> x) {
        double[] y = new double[x.size()];
        for (int i=0; i < x.size(); i++) {
            y[i] = x.get(i);
        }   
        return y;
    }
    

    protected String convToString(int[] sent) {
        String sentInStr = "";
        boolean first = true;
        for (int word : sent) {
            if (first) {
                sentInStr = "S" + word;
                first = false;
            } else {
                sentInStr += " S" + word;
            }
        } 
        return sentInStr;
    }

    protected double calcStateUnigramLogProb(int[] stateVector, int[] stateCounts, int wordN) {
        double unigramProb = 0;
        for (int state : stateVector) {
            double prob = ((double) stateCounts[state]) / wordN;
            unigramProb += Math.log10(prob);
        }
        return unigramProb;
    }

    protected double calcUnigramLogProb(ArrayList<Integer> testSentence,
        HashMap<Integer, Integer> wordCounts, int wordN) {
        double unigramProb = 0;
        for (int i=0; i<testSentence.size();i++) {
            double prob = (double)(wordCounts.get(testSentence.get(i)))/wordN;
            unigramProb += Math.log10(prob);
        }
        return unigramProb;
    }

    protected double getMeanQ(ArrayList<Double> results, double maxValue) {
        ArrayList<Double> passedResults = new ArrayList<Double> ();
        for (Double v : results) {
            if ( v <= maxValue) {
                passedResults.add(v);
            }
        }
        return StatUtils.mean(convertToArray(passedResults));
    }

}
