/*
Author: Jey Han Lau
Date: Nov 13

Reads the binary HMM model file produced by tikka, and computes (various) probability measures for 
the input test sentences/sequences (outputs a csv file)

Argument 1: Binary model file produced by tikka
Argument 2: Test Sentences (one line each sentence)
*/

import tikka.bhmm.model.base.*;
import tikka.bhmm.models.*;
import tikka.bhmm.apps.*;
import tikka.structures.*;
import tikka.opennlp.io.*;
import tikka.utils.ec.util.MersenneTwisterFast;
import org.apache.commons.cli.*;
import java.io.IOException;
import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;
import java.util.Comparator;
import java.util.Iterator;
import java.util.Collections;
import java.lang.Math;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.StatUtils;

public class TestProbSimple {
    static boolean debug = false;
    static boolean debug2 = false;
    static boolean debug3 = false;
 
    public static void main(String[] args) {
        new TestProbSimple().go(args);
    }

    private void go(String[] args) {
        // debug flag

        CommandLineParser optparse = new PosixParser();
        Options options = setOptions();
        PrintWriter wcOut = null;

        try {
            CommandLine cline = optparse.parse(options, args);
            if (cline.hasOption('h')) {
                HelpFormatter formatter = new HelpFormatter();
                formatter.printHelp("java TestProbSimple", options);
                System.exit(0);
            }
            if (cline.hasOption("xdebug")) {
                debug2 = true;
            }
            if (cline.hasOption("xwc")) {
                wcOut = new PrintWriter(cline.getOptionValue("xwc"));
            }

            CommandLineOptions modelOptions = new CommandLineOptions(cline);
            SerializableModel serializableModel = new SerializableModel();

            HMMBase bhmm = serializableModel.loadModel(modelOptions,
                modelOptions.getModelInputPath());
            bhmm.resetTrainDataDir(); // reset the previous train directory (avoid IOException)
            bhmm.initializeFromLoadedModel2(modelOptions);


            /* Various debug prints to check the model */
            if (debug2) {
                System.out.println("model name = " + bhmm.getModelName());
                System.out.println("alpha = " + bhmm.getAlpha());
                System.out.println("beta = " + bhmm.getBeta());
                System.out.println("gamma = " + bhmm.getGamma());
                System.out.println("delta = " + bhmm.getDelta());
                System.out.println("phash = " + bhmm.getPhash());
                System.out.println("stateC = " + bhmm.getStateC());
                System.out.println("stateF = " + bhmm.getStateF());
                System.out.println("stateS = " + bhmm.getStateS());
                System.out.println("topicK = " + bhmm.getTopicK());
                System.out.println("chunkA = " + bhmm.getChunkA());
                System.out.println("wordW = " + bhmm.getWordW());
                System.out.println("wordN = " + bhmm.getWordN());
                System.out.println("iterations = " + bhmm.getIterations());
                System.out.println("random seed = " + bhmm.getRandomSeed());
                System.out.println("useTrigram = " + bhmm.getUseTrigram());

                System.out.println("\nWord to WordID:");
                for (Map.Entry entry : bhmm.getTrainWordIdx().entrySet()) {
                    //System.out.println(entry.getKey() + " : " + entry.getValue());
                }

                if (bhmm.getTrainIdxToPos() != null) {
                    System.out.println("\nPOS Tag Ids:");
                    Iterator it = bhmm.getTrainIdxToPos().entrySet().iterator();
                    while (it.hasNext()) {
                        Map.Entry pairs = (Map.Entry)it.next();
                        System.out.println(pairs.getKey() + "=" + pairs.getValue());
                    }
                }

                if (bhmm.getNumWordsPerState() != null) {
                    System.out.println("\nNumber of Words Per State:");
                    for (int i=0; i<bhmm.getNumWordsPerState().length; i++) {
                        System.out.println("State " + i + " = " + bhmm.getNumWordsPerState()[i]);
                    }
                }

                if (bhmm.getStatesForWord() != null) {
                    System.out.println("\nPossible States For Each Word:");
                    Iterator it = bhmm.getStatesForWord().entrySet().iterator();
                    while (it.hasNext()) {
                        Map.Entry pairs = (Map.Entry)it.next();
                        System.out.println("Word " + pairs.getKey() + " = " + pairs.getValue());
                    }
                }

                System.out.println("\nTop Words Per State:");
                bhmm.normalize();

                StringDoublePair[][] x = bhmm.getTopWordsPerState();
                for (int i = 0; i < x.length; i++) {
                    System.out.println("State = " + i);
                    for (int j = 0; j < (x[i]).length; j++) {
                        System.out.println("\t" + x[i][j].stringValue + " = " +
                            x[i][j].doubleValue);
                    }
                }

    
                if (bhmm.getModelName().equals("m4")) {
                    System.out.println("\nTop Words Per Topic:");
                    x = bhmm.getTopWordsPerTopic();
                    for (int i = 0; i < x.length; i++) {
                        System.out.println("Topic = " + i);
                        for (int j = 0; j < (x[i]).length; j++) {
                            System.out.println("\t" + x[i][j].stringValue + " = " +
                                x[i][j].doubleValue);
                        }
                    }
                }

                if (bhmm.getModelName().equals("m7")) {
                    System.out.println("\nChunk Transition:");
                    int[] ct = bhmm.getChunkTransition();
                    int chunkA = bhmm.getChunkA();
                    int stateS = bhmm.getStateS();
                    for (int i=0; i<chunkA; i++) {
                        for (int j=0; j<stateS; j++) {
                            for (int k=0; k<chunkA; k++) {
                                if (ct[i*stateS*chunkA + j*chunkA + k] > 0) {
                                    System.out.println(i+"->"+j+"->"+k+ " = " +
                                        ct[i*stateS*chunkA + j*chunkA + k]);
                                }
                            }
                        }
                    }

                    System.out.println("\nState Transition:");
                    int[] st = bhmm.getStateTransition();
                    //for(int h=0; h<chunkA; h++) {
                    for (int i=0; i<chunkA; i++) {
                        for (int j=0; j<stateS; j++) {
                            for (int k=0; k<stateS; k++) {
                                if (st[i*stateS*stateS+j*stateS+k] > 0) {
                                    System.out.println(i+"->"+j+"->"+k+ " = " +
                                        st[i*stateS*stateS+j*stateS+k]);
                                }
                            }
                        }
                    }
                    //}
                }

                if (bhmm.getModelName().equals("m8")) {
                    System.out.println("\nTop-100 chunks:");
                    int xx = 0;
                    int yy = 0;
                    List<Map.Entry<String, Integer>> list =
                        new ArrayList<Map.Entry<String, Integer>>(bhmm.getChunkFreq().entrySet());
                    Collections.sort(list, new ValueThenKeyComparator<String, Integer>());
                    for (Map.Entry item : list) {
                        String[] parts = ((String) item.getKey()).split("\\+");
                        if (xx < 100) {
                            System.out.println(item.getKey() + " : " + item.getValue());
                        }
                        if (parts.length > 5)  {
                            System.out.println("\t" + item.getKey() + " : " + item.getValue());
                            yy++;
                        }
                        xx++;
                    }
                    System.out.println();
                    System.out.println("yy = " + yy);
                    System.out.println("chunkFreq size = " + bhmm.getChunkFreq().size());
                }
            }

            if (debug) {
                System.out.println("stateByWord = " + Arrays.toString(bhmm.getStateByWord()));
                System.out.println("firstOrderTransitions = " +
                    Arrays.toString(bhmm.getFirstOrderTransitions()));
                System.out.println("secondOrderTransitions = " +
                    Arrays.toString(bhmm.getSecondOrderTransitions()));
                System.out.println("stateCounts = " + Arrays.toString(bhmm.getStateCounts()));


                System.out.println("\nDocument Vector = " +
                    Arrays.toString(bhmm.getDocumentVector()));
                System.out.println("Word Vector = " + Arrays.toString(bhmm.getWordVector()));
                System.out.println("Sentence Vector = "+Arrays.toString(bhmm.getSentenceVector()));
                System.out.println("State Vector = " + Arrays.toString(bhmm.getStateVector()));
            }

            // check the counts
            if (!bhmm.getModelName().equals("m4") && !bhmm.getModelName().equals("m7") &&
                !bhmm.getModelName().equals("m8")) {
                if ((bhmm.getWordN() != sumArray(bhmm.getStateByWord())) ||
                    (bhmm.getWordN() != sumArray(bhmm.getStateCounts())) ||
                    (bhmm.getWordN() != sumArray(bhmm.getFirstOrderTransitions())) ||
                    (bhmm.getWordN() != sumArray(bhmm.getSecondOrderTransitions()))) {
                    System.out.println("Error: mismatched word counts");
                    System.out.println("wordN = " + bhmm.getWordN());
                    System.out.println("sum(stateByWord) = " + sumArray(bhmm.getStateByWord()));
                    System.out.println("sum(stateCounts) = " + sumArray(bhmm.getStateCounts()));
                    System.out.println("sum(firstOrderTransitions) = " +
                        sumArray(bhmm.getFirstOrderTransitions()));
                    System.out.println("sum(secondOrderTransitions) = " +
                        sumArray(bhmm.getSecondOrderTransitions()));
                    System.exit(1);
                }
            }
            if (bhmm.getModelName().equals("m4")) {
                int diff = bhmm.getWordN() - sumArray(bhmm.getStateByWord());
                if ((sumArray(bhmm.getTopicByWord()) != diff) ||
                    (sumArray(bhmm.getTopicCounts()) != diff)) {
                    System.out.println("Error: mismatched word counts");
                    System.out.println("wordN = " + bhmm.getWordN());
                    System.out.println("sum(topicByWord) = " + sumArray(bhmm.getTopicByWord()));
                    System.out.println("sum(topicCounts) = " + sumArray(bhmm.getTopicCounts()));
                    System.out.println("sum(stateByWord) = " + sumArray(bhmm.getStateByWord()));
                    System.exit(1);
                }
            }
            if (bhmm.getModelName().equals("m7") && 
                ((bhmm.getWordN() != sumArray(bhmm.getChunkTransition())) ||
                (bhmm.getWordN() != sumArray(bhmm.getStateTransition())) ||
                (bhmm.getWordN() != sumArray(bhmm.getStateByWord())) ||
                (bhmm.getWordN() != sumArray(bhmm.getStateCounts())))) {
                System.out.println("Error: mismatched word counts");
                System.out.println("wordN = " + bhmm.getWordN());
                System.out.println("sum(stateByWord) = " + sumArray(bhmm.getStateByWord()));
                System.out.println("sum(stateCounts) = " + sumArray(bhmm.getStateCounts()));
                System.out.println("sum(chunkTransition) = " + sumArray(bhmm.getChunkTransition()));
                System.out.println("sum(stateTransition) = " + sumArray(bhmm.getStateTransition()));
            }

            HashMap<Integer, Double> unigramProb = new HashMap<Integer, Double>();
            double[][] stProb = null;
            double[][][] stProb2 = null; //second order transition
            double[][] emissionProb = null;
            int[] chunkTransitionSumA = null;
            int[] stateTransitionSumS = null;
            
            // build the hmm if using MAP estimate logprob
            if (!bhmm.getModelName().equals("m7") && !bhmm.getModelName().equals("m8")) {
                stProb = calcStateTransitionProb(bhmm.getFirstOrderTransitions(), bhmm.getStateS(),
                    bhmm.getGamma());
                if (bhmm.getUseTrigram()) {
                    stProb2 = calcSecondOrderStateTransitionProb(bhmm.getSecondOrderTransitions(),
                        bhmm.getStateS(), bhmm.getGamma());
                }
                emissionProb = calcWordEmissionProb(bhmm.getModelName(), bhmm.getStateCounts(),
                    bhmm.getStateByWord(), bhmm.getStateS(), bhmm.getStateC(), bhmm.getWordW(),
                    bhmm.getBeta(), bhmm.getDelta(), unigramProb, bhmm.getStatesForWord(),
                    bhmm.getNumWordsPerState());
            }
 

            // get the number of tokens for each unique word
            if (debug) {
                System.out.println("\nGenerating word counts:");
            }
            HashMap<Integer, Integer> wordCounts = new HashMap<Integer, Integer>();
            int[] wordVector = bhmm.getWordVector();
            for (int i = 0; i < bhmm.getWordN(); i++) {
                int wordId = wordVector[i];
                int count = wordCounts.containsKey(wordId) ? wordCounts.get(wordId) : 0;
                wordCounts.put(wordId, count + 1);
                if (debug) {
                    System.out.println("Word Vector " + i + " = " + wordId);
                    System.out.println("\twordCounts[" + wordId + "] = " +
                        wordCounts.get(wordId));
                }
            }
    
            // get the state -> word proabilities for each word
            if (debug) {
                System.out.println("\nUnigram Probabilities:");
                Iterator it = unigramProb.entrySet().iterator();
                while (it.hasNext()) {
                    Map.Entry pairs = (Map.Entry)it.next();
                    System.out.println(pairs.getKey() + "=" + pairs.getValue());
                }
            }
            

            // read in test sentences and compute various probability measures
            File testFile = new File(cline.getOptionValue("z"));
            DataReader dataReader = new Conll2kReader(testFile);
            HashMap<String, Integer> trainWordIdx = bhmm.getTrainWordIdx();
            String[][] sentence;
            int line_id = 0;

            // print the csv header
            System.out.println("id,ppl,sent_length,logprob,unigram_logprob," +
                "mean_logprob,norm_logprob_div,norm_logprob_sub,slor," +
                "wlogprob-bot-1,wlogprob-bot-2,wlogprob-bot-3,wlogprob-bot-4,wlogprob-bot-5," +
                "wlogprob_mean,wlogprob_m1q,wlogprob_m2q");
    
            try{
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
                    //unigramLogProb = unigramLogProb * -1.0;
                    //unigramLogProb2 = unigramLogProb2 * -1.0;

                    // compute the probability
                    HashMap<String, Double> results = new HashMap<String, Double>();
                    ArrayList<Double> wordLogProbResult = new ArrayList<Double>();
                    ArrayList<Double> wordMeanLogProbResult = new ArrayList<Double>();
                    ArrayList<Double> unigramLogProbResult = new ArrayList<Double>();
                    ArrayList<Double> stateTranLogProbResult = new ArrayList<Double> ();

                    ArrayList<ArrayList<Double>> stateTranProbs = new ArrayList<ArrayList<Double>>();
                    ArrayList<ArrayList<Double>> wordStateProbs = new ArrayList<ArrayList<Double>>();
                    ArrayList<Double> sentenceProbs = new ArrayList<Double>();
                    ArrayList<int[]> stateVectors = new ArrayList<int[]>();
                    if (!bhmm.getModelName().equals("m7") && !bhmm.getModelName().equals("m8")) {
                        calcMAPLogProb(testSentence, bhmm.getModelName(), stProb, stProb2, 
                            emissionProb,
                            bhmm.getFirstOrderTransitions(), bhmm.getSecondOrderTransitions(),
                            bhmm.getStateByWord(), bhmm.getStateCounts(), bhmm.getTopicByWord(),
                            bhmm.getTopicCounts(), bhmm.getChunkTransition(), bhmm.
                            getStateTransition(),
                            chunkTransitionSumA, stateTransitionSumS,
                            bhmm.getStateS(), bhmm.getStateC(), bhmm.getStateF(),
                            bhmm.getTopicK(), bhmm.getChunkA(), bhmm.getWordW(),
                            bhmm.getGamma(), bhmm.getDelta(), bhmm.getBeta(), bhmm.getAlpha(),
                            bhmm.getStatesForWord(),
                            bhmm.getNumWordsPerState(), bhmm.getUseTrigram(),
                            wordCounts, bhmm.getWordN(), results, wordLogProbResult,
                            wordMeanLogProbResult, unigramLogProbResult, stateTranLogProbResult,
                            stateVectors);
                    } else {
                        bhmm.computeTestSentenceProb(testSentence, sentenceProbs, stateTranProbs,
                            wordStateProbs, stateVectors);
                        collectSentenceStatistics(testSentence, wordCounts, bhmm.getWordN(),
                            sentenceProbs, stateTranProbs, wordStateProbs, results,
                            wordLogProbResult, wordMeanLogProbResult, unigramLogProbResult,
                            stateTranLogProbResult);

                    }

                    if (wcOut != null) {
                        int[] st = stateVectors.get(stateVectors.size()-1);
                        for (int i=0; i<st.length;i++) {
                            wcOut.write("S" + st[i]);
                            if (i != (st.length-1)) {
                                wcOut.write(" ");
                            } else {
                                wcOut.write("\n");
                            }
                        }
                        wcOut.flush();
                    }


                    //scoring functions
                    double seqLogProb = results.get("logprob_mean");
                    double unigramLogProb = StatUtils.sum(convertToArray(unigramLogProbResult));

                    double seqWLogProb = seqLogProb / numWords;
                    double seqFMeanLogProb = (seqLogProb / unigramLogProb) * -1.0;
                    double seqFMeanLogProb2 = seqLogProb - unigramLogProb;
                    double seqSlor = seqFMeanLogProb2 / numWords;
                    double seqPplex = Math.pow(10, (-1.0 * seqWLogProb));

                    String outputList = "";
                    //get the bottom five probabilities
                    Collections.sort(wordLogProbResult);
                    Collections.sort(wordMeanLogProbResult);
                    Collections.sort(unigramLogProbResult);
                    Collections.sort(stateTranLogProbResult);
                    List<ArrayList<Double>> allResults = new ArrayList<ArrayList<Double>>();
                    //allResults.add(unigramLogProbResult);
                    //allResults.add(wordLogProbResult);
                    allResults.add(wordMeanLogProbResult);
                    //allResults.add(stateTranLogProbResult);
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

                    if (debug2) {
                        System.out.println("\nSentence Text = " + text);
                        System.out.println("Sentence Text in Word ID = " + textInWordId);
                        System.out.println("Final State Vector = " +
                            Arrays.toString(stateVectors.get(stateVectors.size()-1)));
                        System.out.println("Unigram Logprob = " + unigramLogProb);
                        System.out.println("Sentence Log Prob = " + seqLogProb);
                        System.out.println("Sentence Logprob / Num Words = " + seqWLogProb);
                        System.out.println("Sentence Logprob / Freq Unigram Logprob = " +
                            seqFMeanLogProb);
                        System.out.println("Sentence Logprob - Freq Unigram Logprob = " +
                            seqFMeanLogProb2);
                        System.out.println("Sentence statetran_1q = " + results.get("statetran_1q"));
                        System.out.println("Sentence statetran_2q = " + results.get("statetran_2q"));
                        System.out.println("Sentence statetran_mean = " +
                            results.get("statetran_mean"));
                        System.out.println("Sentence statetran_m1q = " +
                            results.get("statetran_m1q"));
                        System.out.println("Sentence statetran_m2q = " +
                            results.get("statetran_m2q"));
                        System.out.println("Sentence Perplexity = " + seqPplex);
                    }
                    //output the probabilities (csv format)
                    System.out.println(line_id + ",," + numWords + "," + seqLogProb +
                        "," + unigramLogProb + "," + seqWLogProb + "," + seqFMeanLogProb+ "," +
			seqFMeanLogProb2 + "," + seqSlor + outputList + "," +
                        results.get("nlogprob_mean") + "," + results.get("nlogprob_m1q") +
                        "," + results.get("nlogprob_m2q"));


                    line_id += 1;
                }
            } catch (IOException e) {
            }
            System.out.flush();

        } catch (ParseException exp) {
            System.err.println("Exception in parsing command line options: " + exp.getMessage());
        } catch (IOException exp) {
            System.err.println("IOException: " + exp.getMessage());
            System.exit(0);
        }

        if (wcOut != null) {
            wcOut.close();
        }

    }

    /* function to collect options */
    public static Options setOptions() {

        Options options = new Options();
        options.addOption("h", "help", false, "print help");
        options.addOption("l", "model-input-path", true, "full path of model to be loaded");
        options.addOption("z", "test-sentences", true, "text file that contains the test " +
            "sentences (one sentence per line format)");
        options.addOption("xdebug", "debug-mode", false, "debug mode");
        options.addOption("xwc", "wordclass-output", true, "output the word classes of test " +
            "sentences");

        return options;
    }

    // get sum of array
    protected int sumArray(int[] a) {
        int result = 0;
        for (int i=0; i<a.length; i++) {
            result += a[i];
        }
        return result;
    }

    private static void calcMAPLogProb(ArrayList<Integer> testSentence, String modelName,
        double[][]stProb, double[][][] stProb2, double[][] emissionProb,
        int[] firstOrderTransitions, int[] secondOrderTransitions, int[] stateByWord,
        int[] stateCounts, int[] topicByWord, int[] topicCounts, int[] chunkTransition,
        int[] stateTransition, int[] chunkTransitionSumA, int[] stateTransitionSumS,
        int stateS, int stateC, int stateF, int topicK, int chunkA,
        int wordW, double gamma, double delta, double beta, double alpha,
        HashMap<Integer,ArrayList<Integer>> statesForWord, int[] numWordsPerState,
        boolean useTrigram, HashMap<Integer,Integer> wordCounts, int wordN,
        HashMap<String, Double> results, ArrayList<Double> wordLogProbResult,
        ArrayList<Double> wordMeanLogProbResult, ArrayList<Double> unigramLogProbResult,
        ArrayList<Double> stateTranLogProbResult, ArrayList<int[]> stateVectors) {

        int wordid, topicid, stateid, chunkid;
        int prev = stateS-1, current = stateS-1, next = stateS-1, nnext = stateS-1;
        int currentChunk = chunkA-1, nextChunk = chunkA-1;
        double max = 0, totalprob = 0;
        double r = 0;
        int wordstateoff, stateoff, secondstateoff, wordtopicoff;
        int[] stateVector = new int[testSentence.size()];
        int[] chunkVector = new int[testSentence.size()];
        int[] topicVector = new int[testSentence.size()];
        int[] documentByTopic = null;
        double[] topicProbs = null;
        double[] chunkProbs = null;
        if (topicK != -1) {
            documentByTopic  = new int[topicK];
            topicProbs = new double[topicK];
        }
        double[] stateProbs = new double[stateS];
        if (chunkA != -1) {
            chunkProbs = new double[chunkA];
        }
        
        int[] contentStateBySentence = new int[stateC];
        int sentenceCounts = 0;
        int i,j;
        // use seed 1 for replicability
        MersenneTwisterFast mtfRand = new MersenneTwisterFast(1);
        ArrayList<Double> probList = new ArrayList<Double>();
        ArrayList<ArrayList<Double>> probMinList = new ArrayList<ArrayList<Double>>();
        for (int u=0; u<5; u++) {
            probMinList.add(new ArrayList<Double>());
        }
        ArrayList<ArrayList<Double>> wordStateProbList = new ArrayList<ArrayList<Double>>();
        ArrayList<ArrayList<Double>> stateTranProbList = new ArrayList<ArrayList<Double>>();
        for (int u=0; u<testSentence.size(); u++) {
            wordStateProbList.add(new ArrayList<Double>());
            stateTranProbList.add(new ArrayList<Double>());
        }

        // initialise state vector with random states
        for (i = 0; i < testSentence.size(); i++) {
            wordid = testSentence.get(i);
            if ((statesForWord != null) && (statesForWord.containsKey(wordid))) {
                stateVector[i] = statesForWord.get(wordid).get(
                mtfRand.nextInt(statesForWord.get(wordid).size()));
            } else { 
                if (modelName.equals("m2")) {
                    stateVector[i] = mtfRand.nextInt(stateS-1);
                } else if (modelName.equals("m7")) {
                    stateVector[i] = mtfRand.nextInt(stateS-1);
                    chunkVector[i] = mtfRand.nextInt(chunkA-1); 
                } else {
                    // roll a random content state
                    if (mtfRand.nextDouble() > 0.5) {
                        stateVector[i] = mtfRand.nextInt(stateC);
                    } else { // roll a random function state
                        stateVector[i] = stateC + mtfRand.nextInt(stateF);
                    }

                    // roll a random topic state if it's LDAHMM
                    if (modelName.equals("m4")) {
                        topicVector[i] = mtfRand.nextInt(topicK);
                    }
                    
                }
            }
            if (stateVector[i] < stateC)  {
                contentStateBySentence[stateVector[i]]++; 
                sentenceCounts++;
                if (modelName.equals("m4")) {
                    documentByTopic[topicVector[i]]++;
                }
            }
        }

        if (debug3) {
            System.out.println("\nSentence = " + testSentence);
            System.out.println("Initial States = " + Arrays.toString(stateVector));
        }

        /*
        System.out.print("stateVector After Initialisation = ");
        for (int tmp : stateVector)
            System.out.print(tmp + ",");
        System.out.print("\ncontentStateBySentence = ");
        for (int tmp : contentStateBySentence)
            System.out.print(tmp + ",");*/

        // do the sampling
        double sgamma = stateS*gamma;
        double aalpha = chunkA*alpha;
        double wdelta = wordW*delta;
        for (int iter = 0; iter < 5000; ++iter) {
            current = stateS-1;
            prev = stateS-1;
            currentChunk = chunkA-1;
            nextChunk = chunkA-1;
            for (i = 0; i < testSentence.size(); i++) {
                if (i==0) {
                    current = stateS-1;
                    prev = stateS-1;
                    currentChunk = chunkA-1;
                }
                if (useTrigram && i==1) {
                    prev = stateS-1;
                }

                try {
                    next = stateVector[i + 1];
                    nextChunk = chunkVector[i + 1];
                } catch (ArrayIndexOutOfBoundsException e) {
                    next = stateS-1;
                    nextChunk = chunkA-1;
                }
                try {
                    nnext = stateVector[i + 2];
                } catch (ArrayIndexOutOfBoundsException e) {
                    nnext = stateS-1;
                }

                wordid = testSentence.get(i);
                wordstateoff = wordid * stateS;
                wordtopicoff = wordid * topicK;
                stateoff = current * stateS;
                secondstateoff = prev*stateS*stateS + current*stateS;

                if (stateVector[i] < stateC)  {
                    contentStateBySentence[stateVector[i]]--;
                    sentenceCounts--;
                    if (modelName.equals("m4")) {
                        documentByTopic[topicVector[i]]--;
                    }
                }

                try {
                    // roll a topic if it's LDAHMM
                    if (modelName.equals("m4")) {
                        totalprob = 0;
                        for(j=0; j<topicK;j++) {
                            topicProbs[j] = documentByTopic[j] + alpha;
                            if (stateVector[i] == 0) {
                                topicProbs[j] *= (topicByWord[wordtopicoff + j] + beta)
                                    / (topicCounts[j] + (wordW*beta));
                            }
                            totalprob += topicProbs[j];
                        }

                        r = mtfRand.nextDouble() * totalprob;
                        max = topicProbs[0];

                        topicid = 0;
                        while (r > max) {
                            topicid++;
                            max += topicProbs[topicid];
                        }
                        topicVector[i] = topicid;

                    }
                    // roll a chunk if it's HMMTwoTier
                    if (modelName.equals("m7")) {
                        totalprob = 0;
                        for (j=0; j<(chunkA-1); j++) {
                            int chunkoff = currentChunk*stateS*chunkA + current*chunkA;
                            int nextchunkoff = j*stateS*chunkA + stateVector[i]*chunkA;
                            stateoff = j*stateS*stateS + current*stateS;
                            chunkProbs[j] = (chunkTransition[chunkoff+j] + alpha) *
                                (chunkTransition[nextchunkoff+nextChunk] + alpha) /
                                (chunkTransitionSumA[j*stateS + stateVector[i]] + aalpha) *
                                (stateTransition[stateoff + stateVector[i]] + gamma) /
                                (stateTransitionSumS[j*stateS + current] + sgamma);
                            totalprob += chunkProbs[j];
                        }   
                        r = mtfRand.nextDouble() * totalprob;
                        chunkid = 0;
                        max = chunkProbs[chunkid];
                        while (r > max) {
                            chunkid++;
                            max += chunkProbs[chunkid];
                        }   
                        chunkVector[i] = chunkid;
                    }

                    totalprob = 0;
                    for (j = 0;j<(stateS-1); j++) {
                        if (modelName.equals("m7")) {
                            chunkid = chunkVector[i];
                            stateoff = chunkid*stateS*stateS + current*stateS;
                            int nextstateoff = nextChunk*stateS*stateS + j*stateS;
                            int nextchunkoff = chunkid*stateS*chunkA + j*chunkA;
                            stateProbs[j] = (chunkTransition[nextchunkoff+nextChunk]+alpha) /
                                (chunkTransitionSumA[chunkid*stateS + j] + aalpha) *
                                (stateTransition[stateoff + j] + gamma) *
                                (stateTransition[nextstateoff + next] + gamma) /
                                (stateTransitionSumS[nextChunk*stateS + j] + sgamma) *
                                (stateByWord[wordstateoff + j] + delta) /
                                (stateCounts[j] + wdelta);
                        } else {
                            if ((statesForWord != null && statesForWord.containsKey(wordid) &&
                                statesForWord.get(wordid).contains(j)) ||
                                (statesForWord !=null && !statesForWord.containsKey(wordid)) ||
                                (statesForWord == null)) {

                                double wdeltaState = wordW * delta;
                                if (statesForWord != null) {
                                    wdeltaState = numWordsPerState[j] * delta;
                                }

                                // x is the word emission probability
                                double x = 0.0;
                                x = ((stateByWord[wordstateoff + j] + delta) /
                                            (stateCounts[j] + wdeltaState));


                                // use different formulae if it's other models
                                /*
                                System.out.println("\nmodelName = " + modelName);
                                System.out.println("j = " + j);
                                System.out.println(stateByWord[wordstateoff+j] + "," + 
                                    stateCounts[j]);
                                */
                                if ((modelName.equals("m3")) && (j<stateC))
                                {
                                    x = ((stateByWord[wordstateoff + j] + beta) /
                                        (stateCounts[j] + wordW*beta));
                                } else if ((modelName.equals("m4")) && (j==0)) {
                                    x = (topicByWord[wordtopicoff + topicVector[i]] + beta) /
                                        (topicCounts[topicVector[i]] + wordW*beta);
                                } else if (((modelName.equals("m5")) || (modelName.equals("m6"))) &&
                                    (j<stateC)) {
                                    x = ((stateByWord[wordstateoff + j] + beta) /
                                        (stateCounts[j] + wordW*beta)) *
                                        (contentStateBySentence[j] + alpha) /
                                        (sentenceCounts + stateC*alpha);
                                }
                                //System.out.println(contentStateBySentence[j] + "," +
                                //    sentenceCounts);
                                //System.out.println("x = " + x);

                                // trigram model
                                if (useTrigram) {
                                    // see words as 'abxcd', where x is the current word
                                    int S2 = stateS*stateS;
                                    double abx =
                                        (secondOrderTransitions[secondstateoff + j] + gamma);
                                    double bxc =
                                        (secondOrderTransitions[(current*S2+j*stateS+next)]+gamma) /
                                        (firstOrderTransitions[current*stateS + j] + sgamma);
                                    double xcd =
                                        (secondOrderTransitions[(j*S2+next*stateS+nnext)]+gamma) /
                                        (firstOrderTransitions[j*stateS + next] + sgamma);
                                    stateProbs[j] = x*abx*bxc*xcd;
                                // bigram model
                                } else {
                                    stateProbs[j] = x *
                                        (firstOrderTransitions[stateoff + j] + gamma) /
                                        (stateCounts[j] + (stateS*gamma)) *
                                        (firstOrderTransitions[j * stateS + next] + gamma);
                                }
                            } else {
                                stateProbs[j] = 0.0;
                            }
                        }
                        totalprob += stateProbs[j];

                    }
                } catch (ArrayIndexOutOfBoundsException e) {
                }
                double x = mtfRand.nextDouble();
                r = x * totalprob;
                stateid = 0;
                max = stateProbs[stateid];
                while (r > max) {
                    stateid++;
                    max += stateProbs[stateid];
                }
                stateVector[i] = stateid;
                prev = current;
                current = stateid;
                currentChunk = chunkVector[i];

                if (stateVector[i] < stateC)  {
                    contentStateBySentence[stateVector[i]]++;
                    sentenceCounts++;
                    if (modelName.equals("m4")) {
                        documentByTopic[topicVector[i]]++;
                    }
                }
                /*
                System.out.println("\nOut: stateVector[i] = " + stateVector[i]);
                System.out.print("\n\n\nstateVector = ");
                for (int tmp : stateVector)
                    System.out.print(tmp + ",");
                System.out.print("\ncontentStateBySentence = ");
                for (int tmp : contentStateBySentence)
                    System.out.print(tmp + ","); */
            }

            if ((iter >= 4500) && (iter%10==0)) {
                // sample the log prob after burn in
                // compute the log prob given the states
                int scurrentState = stateS-1;
                int sprevState = stateS-1;
                int scurrentChunk = chunkA-1;
                double wordStateProb;
                double logProb = 0;

                for (i=0; i < testSentence.size(); i++) {
                    int swordid = testSentence.get(i);
                    int sstateid = stateVector[i];
                    int schunkid = chunkVector[i];
                    double sx = emissionProb[sstateid][swordid];


        
                    //System.out.println("\n" + x + "," + sstateid);
                    if (((modelName.equals("m5")) ||(modelName.equals("m6")))&&(sstateid<stateC)) {
                        sx *= (contentStateBySentence[sstateid] + alpha) /
                            (sentenceCounts + (stateC*alpha));
                        //System.out.println("\t"+sx+","+contentStateBySentence[sstateid] + "," +
                        //    sentenceCounts);
    
                    } else if ((modelName.equals("m4")) && (sstateid == 0)) {
                        sx = (documentByTopic[topicVector[i]]+alpha)/(sentenceCounts + topicK*alpha)
                            * (topicByWord[swordid*topicK + topicVector[i]] + beta) /
                            (topicCounts[topicVector[i]] + wordW*beta);
                    }

                    if (modelName.equals("m7")) {
                        double stp =
                            (chunkTransition[scurrentChunk*stateS*chunkA+scurrentState*chunkA+
                            schunkid] + alpha) /
                            (chunkTransitionSumA[scurrentChunk*stateS+scurrentState] + aalpha) *
                            (stateTransition[schunkid*stateS*stateS+scurrentState*stateS+sstateid] +
                            gamma) /
                            (stateTransitionSumS[schunkid*stateS+scurrentState] + sgamma);
                        wordStateProb = stp*sx;
                        stateTranProbList.get(i).add(Math.log10(stp));
                    } else {
                        if (useTrigram) {
                            wordStateProb = (stProb2[sprevState][scurrentState][sstateid]) * sx;
                            stateTranProbList.get(i).add(
                                Math.log10(stProb2[sprevState][scurrentState][sstateid]));
                            
                        } else {
                            wordStateProb = (stProb[scurrentState][sstateid]) * sx;
                            stateTranProbList.get(i).add(
                                Math.log10(stProb[scurrentState][sstateid]));
                        }
                    }

                    logProb += Math.log10(wordStateProb);
                    wordStateProbList.get(i).add(Math.log10(wordStateProb));
                    sprevState = scurrentState;
                    scurrentState = sstateid;
                    scurrentChunk = schunkid;
                }
                probList.add(logProb); // use log prob because we want to do harmonic mean
            }
        }

        stateVectors.add(stateVector);

        collectSentenceStatistics(testSentence, wordCounts, wordN, probList,
            stateTranProbList, wordStateProbList, results, wordLogProbResult,
            wordMeanLogProbResult, unigramLogProbResult,
            stateTranLogProbResult);

        return;
    }

    private static void collectSentenceStatistics(ArrayList<Integer> testSentence,
        HashMap<Integer, Integer> wordCounts, int wordN, ArrayList<Double> sentenceProbs,
        ArrayList<ArrayList<Double>> stateTranProbs, ArrayList<ArrayList<Double>> wordStateProbs,
        HashMap<String, Double> results,
        ArrayList<Double> wordLogProbResult, ArrayList<Double> wordMeanLogProbResult,
        ArrayList<Double> unigramLogProbResult, ArrayList<Double> stateTranLogProbResult) {

        /*
        System.out.println("testSentence = " + testSentence);
        System.out.println("sentenceProbs = " + Arrays.toString(convertToArray(sentenceProbs)));*/

        // get the individual mean logprob for each word
        for (int i=0; i<testSentence.size();i++) {
            DescriptiveStatistics wordStats = new
                DescriptiveStatistics(convertToArray(wordStateProbs.get(i)));
            double wordLogProb = wordStats.getMean();
            double uProb = (double)(wordCounts.get(testSentence.get(i)))/wordN;
            double wordMeanLogProb = wordLogProb / Math.log10(uProb) * -1.0;
            wordLogProbResult.add(wordLogProb);
            unigramLogProbResult.add(Math.log10(uProb));
            wordMeanLogProbResult.add(wordMeanLogProb);

            stateTranLogProbResult.add(StatUtils.mean(convertToArray(stateTranProbs.get(i))));

            /*
            System.out.println("\nGoing through the testSentence word by word, i =" + i +
                "; word = " + testSentence.get(i));
            System.out.println("wordStateProbs[i] = " +
                Arrays.toString(convertToArray(wordStateProbs.get(i))));
            System.out.println("stateTranProbs[i] = " +
                Arrays.toString(convertToArray(stateTranProbs.get(i))));
            System.out.println("wordLogProb for i = " +
                wordLogProbResult.get(wordLogProbResult.size()-1));
            System.out.println("uProb for i = " + uProb + ";" + wordCounts.get(testSentence.get(i)) +
                ";" + wordN);
            System.out.println("wordMeanLogProb for i = " +
                wordMeanLogProbResult.get(wordMeanLogProbResult.size()-1));
            System.out.println("stateTranLogProb for i = " +
                stateTranLogProbResult.get(stateTranLogProbResult.size()-1));*/

        }

        // sentence (harmonic) mean log prob
        DescriptiveStatistics stats = new DescriptiveStatistics(convertToArray(sentenceProbs));
        results.put("logprob_mean", stats.getMean());
        results.put("sample_size", (double)sentenceProbs.size());

        // individual word (harmonic) mean log prob
        stats = new DescriptiveStatistics(convertToArray(wordMeanLogProbResult));
        results.put("nlogprob_1q", stats.getPercentile(25.0));
        results.put("nlogprob_2q", stats.getPercentile(50.0));
        results.put("nlogprob_mean", stats.getMean());
        results.put("nlogprob_m1q", getMeanQ(wordMeanLogProbResult, results.get("nlogprob_1q")));
        results.put("nlogprob_m2q", getMeanQ(wordMeanLogProbResult, results.get("nlogprob_2q")));

        // individual word (harmonic) mean state tran prob
        stats = new DescriptiveStatistics(convertToArray(stateTranLogProbResult));
        results.put("statetran_1q", stats.getPercentile(25.0));
        results.put("statetran_2q", stats.getPercentile(50.0));
        results.put("statetran_mean", stats.getMean());
        results.put("statetran_m1q", getMeanQ(stateTranLogProbResult, results.get("statetran_1q")));
        results.put("statetran_m2q", getMeanQ(stateTranLogProbResult, results.get("statetran_2q")));

        /*
        System.out.println("\nnlogprob_1q = " + results.get("nlogprob_1q"));
        System.out.println("nlogprob_2q = " + results.get("nlogprob_2q"));
        System.out.println("nlogprob_mean = " + results.get("nlogprob_mean"));
        System.out.println("nlogprob_m1q = " + results.get("nlogprob_m1q"));
        System.out.println("nlogprob_m2q = " + results.get("nlogprob_m2q"));
        System.out.println("\nstatetran_1q = " + results.get("statetran_1q"));
        System.out.println("statetran_2q = " + results.get("statetran_2q"));
        System.out.println("statetran_mean = " + results.get("statetran_mean"));
        System.out.println("statetran_m1q = " + results.get("statetran_m1q"));
        System.out.println("statetran_m2q = " + results.get("statetran_m2q"));*/
    }

    private static double getMeanQ(ArrayList<Double> results, double maxValue) {
        ArrayList<Double> passedResults = new ArrayList<Double> ();
        for (Double v : results) {
            if ( v <= maxValue) {
                passedResults.add(v);
            }
        }
        return StatUtils.mean(convertToArray(passedResults));
    }

    private static double calc_entropy(double[] x) {
        double result = 0.0;
        for (double a : x) {
            result += -1.0 * a * (Math.log(a)/Math.log(2));
        }
        return result;
    }

    private static double[] convertToArray(ArrayList<Double> x) {
        return convertToArray(x, false);
    }

    private static double[] convertToArray(ArrayList<Double> x, boolean convToPos) {
        double[] y = new double[x.size()];
        for (int i=0; i < x.size(); i++) {
            if (convToPos && (x.get(i) <0)) {
                y[i] = x.get(i) * -1.0;
            } else {
                y[i] = x.get(i);
            }
        }   
        return y;
    }

    /* compute the second order state transition probabilities */
    private static double[][][] calcSecondOrderStateTransitionProb(int[] sot, int stateS,
        double gamma) {

        int[][] stateBigramCounts = new int[stateS][stateS];
        double[][][] stProb2 = new double[stateS][stateS][stateS];
        int i,j,k;

        /* get the correct state bigram counts from second order transition array */
        for (i=0; i<stateS; i++) {
            for (j=0; j<stateS; j++) {
                int total = 0;
                for (k=0; k<stateS; k++) {
                    total += sot[i*stateS*stateS + j*stateS + k];
                }
                stateBigramCounts[i][j] = total;
            }
        }

        if (debug) {
            System.out.println("\nSecond Order State Transition Probabilities:");
        }

        for (i=0; i<stateS; i++) {
            for (j=0; j<stateS; j++) {
                for (k=0; k<stateS; k++) {
                    stProb2[i][j][k] = (double)((sot[i*stateS*stateS + j*stateS + k] + gamma) /
                        (stateBigramCounts[i][j] + stateS*gamma));
                    if (debug) {
                        System.out.println(i + "->" + j + "->" + k + " = " + stProb2[i][j][k]);
                    }
                }
            }
        }

        return stProb2;
    }

    /* compute the state transition probabilities */
    private static double[][] calcStateTransitionProb(int[] fot, int stateS, double gamma) {
        int[] sc = new int[stateS];
        double[][] stProb = new double[stateS][stateS];
        int i,j;

        /* get the *modified* state counts (the model's original stateCounts is off by 1 for the 
           starting and ending token for counting state transition probabilities) */
        for (i = 0; i < stateS; i++) {
            int x = 0;
            for (j = 0; j < stateS; j++) {
                x += fot[i*stateS + j];
            }
            sc[i] = x;
        }
        
        if (debug) {
            System.out.println("\nFirst Order State Transition Probabilities:");
        }

        /* state transition probabilities */
        for (i = 0; i < stateS; i++) {
            for (j = 0; j < stateS; j++) {
                int count = fot[i*stateS + j];
                stProb[i][j] = (double)((count + gamma) / (sc[i] + stateS * gamma));
                if (debug) {
                    System.out.println(i + "->" + j + " = " + stProb[i][j]);
                }
            }
        }

        return stProb;

    }

    /* compute the word emission probabilities (i.e. state -> word) */
    private static double[][] calcWordEmissionProb(String modelName, int[] stateCounts,
        int[] stateByWord, int stateS, int stateC, int wordW, double beta, double delta,
        HashMap<Integer,Double> unigramProb,
        HashMap<Integer,ArrayList<Integer>> statesForWord, int[] numWordsPerState) {
        double[][] emissionProb = new double[stateS][wordW];
        int i,j;

        if (debug) {
            System.out.println("\nWord Emission Probabilities:");
        }


        for (i = 0; i < stateS; i++) {
            double[] wordProbs = new double[wordW];
            double statenorm = wordW*delta;
            double prior = delta;
            for (j = 0; j < wordW; j++) {
                int count = stateByWord[j*stateS + i];
                if (statesForWord == null) {
                    if ((!modelName.equals("m2")) && (i<stateC)) {
                        prior = beta;
                        statenorm = wordW*beta;
                    }
                    emissionProb[i][j] = (double)((count + prior) / (stateCounts[i] + statenorm));
                    
                    //System.out.println("\t" + i + "," + count + "," + prior + "," +
                    //    stateCounts[i] + "," + statenorm);
                } else {
                    int wordWState = numWordsPerState[i];
                    if ((statesForWord.containsKey(j) && statesForWord.get(j).contains(i)) || 
                        (!statesForWord.containsKey(j))) {
                        emissionProb[i][j] = (double)((count + delta) /
                            (stateCounts[i] + wordWState * delta));
                    } else {
                        emissionProb[i][j] = 0.0;
                    }

                }
                if (debug) {
                    System.out.println(i + "->" + j + " = " + emissionProb[i][j]);
                }

                if (unigramProb.containsKey(j)) {
                    unigramProb.put(j, unigramProb.get(j) + emissionProb[i][j]);
                } else {
                    unigramProb.put(j, emissionProb[i][j]);
                }
            }
        }

        return emissionProb;
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


