package tikka.bhmm.models;

import java.io.BufferedWriter;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;

import tikka.bhmm.apps.CommandLineOptions;
import tikka.structures.*;
import tikka.utils.annealer.Annealer;
import tikka.bhmm.model.base.*;
import tikka.utils.ec.util.MersenneTwisterFast;

/**
 * Parallelised hmm-lda
 *
 * @author JH Lau
 */
public class ParLDAHMM extends HMM {

    /**
     * Array of probabilities by topic
     */
    protected double[] topicProbs;
    /**
     * Table of top {@link #outputPerClass} words per topic. Used in
     * normalization and printing.
     */
    protected StringDoublePair[][] topWordsPerTopic;

    public ParLDAHMM(CommandLineOptions options) {
        super(options);
        topicK = options.getTopics();

        phash = -1.0;
        stateC = 1;
        stateS = stateF + stateC + 1;
        chunkA = -1;
        S2 = stateS*stateS;
        S1 = stateS;
    }

    // get the top words per topic
    @Override
    public StringDoublePair[][] getTopWordsPerTopic() {
        return topWordsPerTopic;
    }


    /**
     * Initializes arrays for counting occurrences. These need to be initialized
     * regardless of whether the model being trained from raw data or whether
     * it is loaded from a saved model.
     */
    @Override
    protected void initializeCountArrays() {
        super.initializeCountArrays();

        topicCounts = new int[topicK];
        topicProbs = new double[topicK];
        for (int i = 0; i < topicK; ++i) {
            topicCounts[i] = 0;
            topicProbs[i] = 0.;
        }

        topicVector = new int[wordN];
        try {
            for (int i = 0;; ++i) {
                topicVector[i] = 0;
            }
        } catch (ArrayIndexOutOfBoundsException e) {
        }

        topicByWord = new int[topicK * wordW];
        try {
            for (int i = 0;; ++i) {
                topicByWord[i] = 0;
            }
        } catch (ArrayIndexOutOfBoundsException e) {
        }

        documentByTopic = new int[documentD * topicK];
        try {
            for (int i = 0;; ++i) {
                documentByTopic[i] = 0;
            }
        } catch (ArrayIndexOutOfBoundsException e) {
        }
    }

    /**
     * Overriding method to load model
     */
    @Override
    public void initializeFromLoadedModel2(CommandLineOptions options) throws IOException {
        super.initializeFromLoadedModel2(options);
        topicProbs = new double[topicK];
    }


    /**
     * Normalize the sample counts.
     */
    @Override
    public void normalize() {
        normalizeTopics();
        normalizeStates();
    }

    /**
     * Normalize the sample counts for words given topic.
     */
    protected void normalizeTopics() {
        // initialise the topic probs 
        topicProbs = new double[topicK];

        topWordsPerTopic = new StringDoublePair[topicK][];
        for (int i = 0; i < topicK; ++i) {
            topWordsPerTopic[i] = new StringDoublePair[outputPerClass];
        }


        double sum = 0.;
        for (int i = 0; i < topicK; ++i) {
            sum += topicProbs[i] = topicCounts[i] + wbeta;
            ArrayList<DoubleStringPair> topWords =
                  new ArrayList<DoubleStringPair>();
            /**
             * Start at one to leave out EOSi
             */
            for (int j = 0; j < wordW; ++j) {
                topWords.add(new DoubleStringPair(
                      topicByWord[j * topicK + i] + beta, trainIdxToWord.get(
                      j)));
            }
            Collections.sort(topWords);
            for (int j = 0; j < outputPerClass; ++j) {
                if (j < topWords.size()) {
                    topWordsPerTopic[i][j] = new StringDoublePair(
                        topWords.get(j).stringValue, topWords.get(j).doubleValue
                        / topicProbs[i]);
                } else {
                    topWordsPerTopic[i][j] =
                        new StringDoublePair("Null", 0.0);
                }

            }
        }

        for (int i = 0; i < topicK; ++i) {
            topicProbs[i] /= sum;
        }
    }

    /**
     * Normalize the sample counts for words given state. Unlike the base class,
     * it marginalizes word probabilities over the topics for the topic state,
     * i.e. state 0.
     */
    @Override
    protected void normalizeStates() {
        topWordsPerState = new StringDoublePair[stateS][];
        for (int i = 0; i < stateS; ++i) {
            topWordsPerState[i] = new StringDoublePair[outputPerClass];
        }

        double sum = 0.;
        double[] marginalwordprobs = new double[wordW];
        try {
            for (int i = 0;; ++i) {
                marginalwordprobs[i] = 0;
            }
        } catch (ArrayIndexOutOfBoundsException e) {
        }

        for (int j = 0; j < wordW; ++j) {
            int wordoff = j * topicK;
            for (int i = 0; i < topicK; ++i) {
                marginalwordprobs[j] += topicProbs[i]
                      * (topicByWord[wordoff + i] + beta)
                      / (topicCounts[i] + wbeta);
            }
        }

        // collect the top words for the content state first
        sum += stateProbs[0] = stateCounts[0] + wdelta;
        ArrayList<DoubleStringPair> topWords =
                new ArrayList<DoubleStringPair>();
        for (int j = 0; j < wordW; ++j) {
            topWords.add(new DoubleStringPair(
                    marginalwordprobs[j], trainIdxToWord.get(
                    j)));
        }
        Collections.sort(topWords);
        for (int j = 0; j < outputPerClass; ++j) {
            if (j < topWords.size()) {
                topWordsPerState[0][j] =
                    new StringDoublePair(
                    topWords.get(j).stringValue,
                    topWords.get(j).doubleValue);
            } else {
                topWordsPerState[0][j] =
                    new StringDoublePair("Null", 0.0);
            }
        }

        for (int i = 1; i < stateS; ++i) {
            sum += stateProbs[i] = stateCounts[i] + wdelta;
            topWords.clear();
            /**
             * Start at one to leave out EOSi
             */
            for (int j = 0; j < wordW; ++j) {
                topWords.add(new DoubleStringPair(
                      stateByWord[j * stateS + i] + delta, trainIdxToWord.get(
                      j)));
            }
            Collections.sort(topWords);
            for (int j = 0; j < outputPerClass; ++j) {
                if (j < topWords.size()) {
                    topWordsPerState[i][j] =
                        new StringDoublePair(
                        topWords.get(j).stringValue,
                        topWords.get(j).doubleValue / stateProbs[i]);
                } else {
                    topWordsPerState[i][j] =
                        new StringDoublePair("Null", 0.0);
                }
            }
        }

        for (int i = 0; i < stateS; ++i) {
            stateProbs[i] /= sum;
        }
    }

    /**
     * Print the normalized sample counts to out. Print only the top {@link
     * #outputPerTopic} per given state and topic.
     *
     * @param out Output buffer to write to.
     * @throws IOException
     */
    @Override
    public void printTabulatedProbabilities(BufferedWriter out) throws
          IOException {
        printStates(out);
        printNewlines(out, 4);
        printTopics(out);
        out.close();
    }

    /**
     * Print the normalized sample counts for each topic to out. Print only the top {@link
     * #outputPerTopic} per given topic.
     * 
     * @param out
     * @throws IOException
     */
    protected void printTopics(BufferedWriter out) throws IOException {
        int startt = 0, M = 4, endt = M;
        out.write("***** Word Probabilities by Topic *****\n\n");
        while (startt < topicK) {
            for (int i = startt; i < endt; ++i) {
                String header = "Topic_" + i;
                header = String.format("%25s\t%6.5f\t", header, topicProbs[i]);
                out.write(header);
            }

            out.newLine();
            out.newLine();

            for (int i = 0; i < outputPerClass; ++i) {
                for (int c = startt; c < endt; ++c) {
                    String line = String.format("%25s\t%6.5f\t",
                          topWordsPerTopic[c][i].stringValue,
                          topWordsPerTopic[c][i].doubleValue);
                    out.write(line);
                }
                out.newLine();
            }
            out.newLine();
            out.newLine();

            startt = endt;
            endt = java.lang.Math.min(topicK, startt + M);
        }
    }

    @Override
    public void initializeParametersRandom() {
        int wordid, docid, topicid, stateid;
        int prev = (stateS-1), current = (stateS-1);
        int wordtopicoff, wordstateoff, docoff, stateoff, secondstateoff;

        /**
         * Initialize by assigning random topic indices to words
         */
        for (int i = 0; i < wordN; ++i) {
            wordid = wordVector[i];
            wordstateoff = stateS * wordid;

            docid = documentVector[i];
            wordtopicoff = topicK * wordid;
            docoff = topicK * docid;

            double tmpRoll = mtfRand.nextDouble();
            if (tmpRoll > 0.5) {
                stateid = 1;
            } else {
                stateid = 0;
            }

            // selects a random topic
            topicid = mtfRand.nextInt(topicK);
            topicVector[i] = topicid;

            if (stateid == 1) {
                // selects a random non-content state
                stateid = mtfRand.nextInt(stateS-2)+1;
            }
            stateVector[i] = stateid;

            if (stateid == 0) {
                topicByWord[wordtopicoff + topicid]++;
                documentByTopic[docoff + topicid]++;
                topicCounts[topicid]++;
            } else {
                stateByWord[wordstateoff + stateid]++;
            }

            stateCounts[stateid]++;
            firstOrderTransitions[(current*S1) + stateid]++;
            secondOrderTransitions[(prev*S2) + (current*S1) + stateid]++;
            first[i] = current;
            second[i] = prev;
            prev = current;
            current = stateid;
        }
        /*
        System.out.println("After initialisation:");
        System.out.println("WordVector = " + Arrays.toString(wordVector));
        System.out.println("SentenceVector = " + Arrays.toString(sentenceVector));
        System.out.println("documentVector = " + Arrays.toString(documentVector));
        System.out.println("StateVector = " + Arrays.toString(stateVector));
        System.out.println("StateCounts = " + Arrays.toString(stateCounts));
        System.out.println("StateByWord = " + Arrays.toString(stateByWord));
        System.out.println("TopicByWord = " + Arrays.toString(topicByWord));
        System.out.println("TopicCounts = " + Arrays.toString(topicCounts));
        System.out.println("DocumentByTopic = " + Arrays.toString(documentByTopic));
        System.out.println("first = " + Arrays.toString(first));
        System.out.println("second = " + Arrays.toString(second));
        System.out.println("firstOrderTransitions = " + Arrays.toString(firstOrderTransitions));
        System.out.println("secondOrderTransitions = "+Arrays.toString(secondOrderTransitions));
        System.out.println("----------------------------------------------------------");*/
    }

    @Override
    protected void trainInnerIter(int itermax, Annealer annealer) {

    	int[] stateByWordDiff = new int[stateS*wordW];
    	int[] stateCountsDiff = new int[stateS];
    	int[] firstOrderTransitionsDiff = new int[stateS*stateS]; 
    	int[] secondOrderTransitionsDiff = new int[stateS*stateS*stateS]; 
    	int[] topicByWordDiff = new int[topicK*wordW];
    	int[] topicCountsDiff = new int[topicK];
	    SerializableModel serializableModel;

        // parallelisation parameter/initialisations
        ExecutorService executor = Executors.newFixedThreadPool(numThread);
        int[] windex = super.getWorkerStartIndex(numThread, sentenceVector);

        long start = System.currentTimeMillis();
        for (int iter = 0; iter < itermax; ++iter) {
            System.err.println("\n\niteration " + iter + " (Elapsed time = +" +
                (System.currentTimeMillis()-start)/1000 + "s)");

            // collect the results from each worker
            List<Future<ArrayList<int[]>>> list = new ArrayList<Future<ArrayList<int[]>>>();
            for (int np = 0; np < numThread; np++) {
                int startId = windex[np];
                int endId = wordN;
                if (np != (numThread-1)) {
                    endId = windex[np+1];
                }
                
                Callable<ArrayList<int[]>> worker = new Worker(startId, endId, stateByWord, 
                    stateCounts, firstOrderTransitions, secondOrderTransitions,
                    topicByWord, topicCounts, workerMtfRand[np]);
                Future<ArrayList<int[]>> submit = executor.submit(worker);
                list.add(submit);
            }

            Arrays.fill(stateByWordDiff, 0);
            Arrays.fill(stateCountsDiff, 0);
            Arrays.fill(firstOrderTransitionsDiff, 0);
            Arrays.fill(secondOrderTransitionsDiff, 0);
            Arrays.fill(topicByWordDiff, 0);
            Arrays.fill(topicCountsDiff, 0);

            /*
            System.out.println("Original stateCounts = " + Arrays.toString(stateCounts));
            System.out.println("Original stateCountsDiff = " + Arrays.toString(stateCountsDiff));
            System.out.println("Original topicCounts = " + Arrays.toString(topicCounts));
            System.out.println("Original topicCountsDiff = " + Arrays.toString(topicCountsDiff));*/
            for (Future<ArrayList<int[]>> future : list) {
                try {
                    stateByWordDiff = super.sumList(stateByWordDiff,
                        super.sumList(future.get().get(0), stateByWord, "minus"), "plus");
                    stateCountsDiff = super.sumList(stateCountsDiff,
                        super.sumList(future.get().get(1), stateCounts, "minus"), "plus");
                    firstOrderTransitionsDiff = super.sumList(firstOrderTransitionsDiff,
                        super.sumList(future.get().get(2), firstOrderTransitions, "minus"), "plus");
                    secondOrderTransitionsDiff = super.sumList(secondOrderTransitionsDiff,
                        super.sumList(future.get().get(3), secondOrderTransitions, "minus"), "plus");
                    topicByWordDiff = super.sumList(topicByWordDiff,
                        super.sumList(future.get().get(4), topicByWord, "minus"), "plus");
                    topicCountsDiff = super.sumList(topicCountsDiff,
                        super.sumList(future.get().get(5), topicCounts, "minus"), "plus");
                    /*
                    System.out.println("\nworker stateCounts Result = " +
                        Arrays.toString(future.get().get(1)));
                    System.out.println("updated stateCountsDiff = " +
                        Arrays.toString(stateCountsDiff));
                    System.out.println("worker topicCounts Result = " +
                        Arrays.toString(future.get().get(5)));
                    System.out.println("updated topicCountsDiff = " +
                        Arrays.toString(topicCountsDiff));*/
                } catch (Exception e) {
                    System.err.println("Failed to collect results. Message = " +
                        e.getMessage());
                }
            }

            // sum up all the differences and update the matrixes
            stateByWord = super.sumList(stateByWord, stateByWordDiff, "plus");
            stateCounts = super.sumList(stateCounts, stateCountsDiff, "plus");
            firstOrderTransitions = super.sumList(firstOrderTransitions,
                firstOrderTransitionsDiff, "plus");
            secondOrderTransitions = super.sumList(secondOrderTransitions,
                secondOrderTransitionsDiff, "plus");
            topicByWord = super.sumList(topicByWord, topicByWordDiff, "plus");
            topicCounts = super.sumList(topicCounts, topicCountsDiff, "plus");

            /*
            System.out.println("\nFinal stateCounts = " + Arrays.toString(stateCounts));
            System.out.println("Final topicCounts = " + Arrays.toString(topicCounts));
            System.out.println("wordN = " + wordN);
            System.out.println("sum(stateByWord) = " + sumArray(stateByWord));
            System.out.println("sum(stateCounts) = " + sumArray(stateCounts));
            System.out.println("sum(firstOrderTransitions) = " + sumArray(firstOrderTransitions));
            System.out.println("sum(secondOrderTransitions) = " + sumArray(secondOrderTransitions));
            System.out.println("sum(topicByWord) = " + sumArray(topicByWord));
            System.out.println("sum(topicCounts) = " + sumArray(topicCounts));*/

            if (iter%1000 == 0) {
                try {
                    System.err.println("\nSaving model to :"
                        + modelOutputPath + "." + iter);
                    serializableModel = new SerializableModel(this);
                    serializableModel.saveModel(modelOutputPath + "." + iter);
                }
                catch (Exception e) {}
            }

        }
        executor.shutdown();

    }

    private class Worker implements Callable<ArrayList<int[]>> {

        private int start, end;
        private int[] wStateByWord, wStateCounts, wFirstOrderTransitions, wSecondOrderTransitions,
            wTopicByWord, wTopicCounts;
        private double[] wStateProbs;
        private double[] wTopicProbs;
        private MersenneTwisterFast wMtfRand;
        
        Worker(int start, int end, int[] wStateByWord, int[] wStateCounts,
            int[] wFirstOrderTransitions, int[] wSecondOrderTransitions,
            int[] wTopicByWord, int[] wTopicCounts,
            MersenneTwisterFast wMtfRand) {
            this.start = start;
            this.end = end;
            this.wStateByWord = wStateByWord.clone();
            this.wStateCounts = wStateCounts.clone();
            this.wFirstOrderTransitions = wFirstOrderTransitions.clone();
            this.wSecondOrderTransitions = wSecondOrderTransitions.clone();
            this.wTopicByWord = wTopicByWord.clone();
            this.wTopicCounts = wTopicCounts.clone();
            this.wStateProbs = new double[stateS];
            this.wTopicProbs = new double[topicK];
            this.wMtfRand = wMtfRand;
        }

        public ArrayList<int[]> call() {

            int wordid, docid, topicid, stateid;
            int prev = (stateS-1), current = (stateS-1), next = (stateS-1), nnext = (stateS-1);
            double max = 0, totalprob = 0;
            double r = 0;
            int wordtopicoff, wordstateoff, docoff;
            int pprevsentid = -1; 
            int prevsentid = -1; 
            int nextsentid = -1; 
            int nnextsentid = -1; 

            for (int i = start; i < end; i++) {
                wordid = wordVector[i];
                stateid = stateVector[i];
                wordstateoff = wordid * stateS;

                docid = documentVector[i];
                topicid = topicVector[i];
                wordtopicoff = wordid * topicK;
                docoff = docid * topicK;

                if (stateid == 0) {
                    wTopicByWord[wordtopicoff + topicid]--;
                    documentByTopic[docoff + topicid]--;
                    wTopicCounts[topicid]--;
                } else {
                    wStateByWord[wordstateoff + stateid]--;
                }
                wStateCounts[stateid]--;
                wFirstOrderTransitions[first[i] * S1 + stateid]--;
                wSecondOrderTransitions[(second[i]*S2) + (first[i]*S1) + stateid]--;

                totalprob = 0;
                try {
                    for (int j = 0;; j++) {
                        wTopicProbs[j] = documentByTopic[docoff + j] + alpha;
                        if (stateid == 0) {
                            wTopicProbs[j] *= (wTopicByWord[wordtopicoff + j] + beta)
                                  / (wTopicCounts[j] + wbeta);
                        }
                        totalprob += wTopicProbs[j];
                    }
                } catch (java.lang.ArrayIndexOutOfBoundsException e) {
                }
                r = wMtfRand.nextDouble() * totalprob;
                max = wTopicProbs[0];
                topicid = 0;
                while (r > max) {
                    topicid++;
                    max += wTopicProbs[topicid];
                }
                topicVector[i] = topicid;

                try {
                    next = stateVector[i + 1];
                    nextsentid = sentenceVector[i + 1];
                } catch (ArrayIndexOutOfBoundsException e) {
                    next = stateS-1;
                    nextsentid = -1;
                }
                try {
                    nnext = stateVector[i + 2];
                    nnextsentid = sentenceVector[i + 2];
                } catch (ArrayIndexOutOfBoundsException e) {
                    nnext = stateS-1;
                    nnextsentid = -1;
                }
                if (sentenceVector[i] != prevsentid) {
                    current = stateS-1;
                    prev = stateS-1;
                }  else if (sentenceVector[i] != pprevsentid) {
                    prev = stateS-1;
                }
                if (sentenceVector[i] != nextsentid) {
                    next = stateS-1;
                    nnext = stateS-1;
                }  else if (sentenceVector[i] != nnextsentid) {
                    nnext = stateS-1;
                }

                totalprob = 0;
                for (int j = 0; j < (stateS-1); j++) {
                    // see words as 'abxcd', where x is the current word
                    double x = 0.0;
                    if (j==0) {
                        x = (wTopicByWord[wordtopicoff + topicid] + beta) /
                            (wTopicCounts[topicid] + wbeta);
                    } else {
                        x = (wStateByWord[wordstateoff + j] + delta) / (wStateCounts[j] + wdelta);
                    }
                    double abx =
                            (wSecondOrderTransitions[(prev*S2+current*stateS+j)]+gamma);
                    double bxc =
                            (wSecondOrderTransitions[(current*S2+j*stateS+next)]+gamma) /
                            (wFirstOrderTransitions[current*stateS + j] + sgamma);
                    double xcd =
                            (wSecondOrderTransitions[(j*S2+next*stateS+nnext)]+gamma) /
                            (wFirstOrderTransitions[j*stateS + next] + sgamma);
                    wStateProbs[j] = x*abx*bxc*xcd;
                    totalprob += wStateProbs[j];
                }
                r = wMtfRand.nextDouble() * totalprob;
                stateid = 0;
                max = wStateProbs[stateid];
                while (r > max) {
                    stateid++;
                    max += wStateProbs[stateid];
                }
                stateVector[i] = stateid;

                if (stateid == 0) {
                    wTopicByWord[wordtopicoff + topicid]++;
                    documentByTopic[docoff + topicid]++;
                    wTopicCounts[topicid]++;
                } else {
                    wStateByWord[wordstateoff + stateid]++;
                }

                wStateCounts[stateid]++;
                wFirstOrderTransitions[current*stateS + stateid]++;
                wSecondOrderTransitions[prev*S2 + current*stateS+ stateid]++;
                first[i] = current;
                second[i] = prev;
                prev = current;
                current = stateid;
                pprevsentid = prevsentid;
                prevsentid = sentenceVector[i];
            }
            ArrayList<int[]> results = new ArrayList<int[]>();
            results.add(wStateByWord);
            results.add(wStateCounts);
            results.add(wFirstOrderTransitions);
            results.add(wSecondOrderTransitions);
            results.add(wTopicByWord);
            results.add(wTopicCounts);
            System.err.println("\tWorker (" + start + "," + end + ") completed.");
            //System.out.println("results size = " + results.size());
            return results;
        }
        /*
        System.out.println("After sampling:");
        System.out.println("WordVector = " + Arrays.toString(wordVector));
        System.out.println("SentenceVector = " + Arrays.toString(sentenceVector));
        System.out.println("documentVector = " + Arrays.toString(documentVector));
        System.out.println("StateVector = " + Arrays.toString(stateVector));
        System.out.println("StateCounts = " + Arrays.toString(wStateCounts));
        System.out.println("StateByWord = " + Arrays.toString(wStateByWord));
        System.out.println("TopicByWord = " + Arrays.toString(wTopicByWord));
        System.out.println("TopicCounts = " + Arrays.toString(wTopicCounts));
        System.out.println("DocumentByTopic = " + Arrays.toString(documentByTopic));
        System.out.println("first = " + Arrays.toString(first));
        System.out.println("second = " + Arrays.toString(second));
        System.out.println("firstOrderTransitions = " + Arrays.toString(wFirstOrderTransitions));
        System.out.println("secondOrderTransitions = "+Arrays.toString(wSecondOrderTransitions));
        System.out.println("----------------------------------------------------------");*/
    }

    /**
     * Creates a string stating the parameters used in the model. The
     * string is used for pretty printing purposes and clarity in other
     * output routines.
     */
    @Override
    public void setModelParameterStringBuilder() {
        super.setModelParameterStringBuilder();
        String line = null;
        line = String.format("topicK:%d", topicK) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("documentD:%d", documentD) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("alpha:%f", alpha) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("beta:%f", beta) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("wbeta:%f", wbeta) + newline;
        modelParameterStringBuilder.append(line);
    }

    /*
    @Override
    public void initializeFromLoadedModel(CommandLineOptions options) throws
          IOException {
        super.initializeFromLoadedModel(options);

        int current = 0;
        int wordid = 0, stateid = 0, docid, topicid;
        int stateoff, wordstateoff, wordtopicoff, docoff;

        for (int i = 0; i < wordN; ++i) {
            wordid = wordVector[i];
            docid = documentVector[i];
            stateid = stateVector[i];
            topicid = topicVector[i];

            stateoff = current * stateS;
            wordstateoff = wordid * stateS;
            wordtopicoff = wordid * topicK;
            docoff = docid * topicK;

            if (stateid == 0) {
                topicByWord[wordtopicoff + topicid]++;
                documentByTopic[docoff + topicid]++;
                topicCounts[topicid]++;
            } else {
                stateByWord[wordstateoff + stateid]++;
            }

            stateByWord[wordstateoff + stateid]++;
            stateCounts[stateid]++;
            firstOrderTransitions[stateoff + stateid]++;
            first[i] = current;
            current = stateid;
        }
    }*/
}
