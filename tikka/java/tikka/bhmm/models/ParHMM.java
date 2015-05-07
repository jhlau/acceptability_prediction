package tikka.bhmm.models;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Collections;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;

import tikka.bhmm.apps.CommandLineOptions;
import tikka.bhmm.model.base.*;
import tikka.utils.ec.util.MersenneTwisterFast;
import tikka.structures.*;
import tikka.utils.annealer.Annealer;


/**
 * Parallelised BHMM.
 *
 * @author jhlau
 */
public class ParHMM extends HMMBase {

    public ParHMM(CommandLineOptions options) {
        super(options);
        alpha = -1.0;
        beta = -1.0;
        phash = -1.0;
        stateC = 0;
        stateS = stateF + 1;
        topicK = -1;
        chunkA = -1;
        S2 = stateS*stateS;
        S1 = stateS;
    }


    /**
     * Randomly initialize learning parameters
     */
    @Override
    public void initializeParametersRandom() {
        int wordid, stateid;
        int prev = (stateS-1), current = (stateS-1);
        double max = 0, totalprob = 0;
        double r = 0;
        int wordstateoff, stateoff, secondstateoff;

        /**
         * Initialize by assigning random topic indices to words
         */
        for (int i = 0; i < wordN; ++i) {
            wordid = wordVector[i];
            wordstateoff = stateS * wordid;

            totalprob = 0;
            stateoff = current * stateS;
            secondstateoff = (prev*S2) + (current*stateS);
            try {
                for (int j = 0; j<(stateS-1); j++) {
                    /*
                    totalprob += stateProbs[j] =
                        (stateByWord[wordstateoff + j] + delta)
                        / (stateCounts[j] + wdelta)
                        * (firstOrderTransitions[stateoff + j] + gamma);
                    */
                    if ((statesForWord != null && statesForWord.containsKey(wordid) &&
                        statesForWord.get(wordid).contains(j)) ||
                        (statesForWord !=null && !statesForWord.containsKey(wordid)) ||
                        (statesForWord == null)) {
                        totalprob += stateProbs[j] = 1.0;
                    } else {
                        totalprob += stateProbs[j] = 0.0;
                    }
                }
            } catch (java.lang.ArrayIndexOutOfBoundsException e) {
            }

            r = mtfRand.nextDouble() * totalprob;
            stateid = 0;
            max = stateProbs[stateid];
            while (r > max) {
                stateid++;
                max += stateProbs[stateid];
            }
            stateVector[i] = stateid;
            firstOrderTransitions[stateoff + stateid]++;
            secondOrderTransitions[secondstateoff + stateid]++;
            stateByWord[wordstateoff + stateid]++;
            stateCounts[stateid]++;
            first[i] = current;
            second[i] = prev;
            prev = current;
            current = stateid;

        }
        /*
        System.out.println("After initialisation:");
        System.out.println("WordVector = " + Arrays.toString(wordVector));
        System.out.println("SentenceVector = " + Arrays.toString(sentenceVector));
        System.out.println("StateVector = " + Arrays.toString(stateVector));
        System.out.println("StateCounts = " + Arrays.toString(stateCounts));
        System.out.println("StateByWord = " + Arrays.toString(stateByWord));
        System.out.println("first = " + Arrays.toString(first));
        System.out.println("second = " + Arrays.toString(second));
        System.out.println("firstOrderTransitions = " + Arrays.toString(firstOrderTransitions));
        System.out.println("secondOrderTransitions = "+Arrays.toString(secondOrderTransitions));
        System.out.println("----------------------------------------------------------");
        */
    }

    /**
     * Training routine for the inner iterations
     */
    @Override
    protected void trainInnerIter(int itermax, Annealer annealer) {

        // parallelisation parameter/initialisations
        ExecutorService executor = Executors.newFixedThreadPool(numThread);
        int[] windex = super.getWorkerStartIndex(numThread, sentenceVector);

        //System.out.println("sentenceVector = " + Arrays.toString(sentenceVector));
        //System.out.println("worker indices = " + Arrays.toString(windex));

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
                    stateCounts, firstOrderTransitions, secondOrderTransitions, workerMtfRand[np]);
                Future<ArrayList<int[]>> submit = executor.submit(worker);
                list.add(submit);
            }

            int[] stateByWordDiff = new int[stateS*wordW];
            int[] stateCountsDiff = new int[stateS];
            int[] firstOrderTransitionsDiff = new int[stateS*stateS]; 
            int[] secondOrderTransitionsDiff = new int[stateS*stateS*stateS]; 
            //System.out.println("Original stateCounts = " + Arrays.toString(stateCounts));
            //System.out.println("Original stateCountsDiff = " + Arrays.toString(stateCountsDiff));
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
                    //System.out.println("\nworker stateCounts Result = " + Arrays.toString(future.get().get(1)));
                    //System.out.println("updated stateCountsDiff = " + Arrays.toString(stateCountsDiff));
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

            /*
            System.out.println("\nFinal stateCounts = " + Arrays.toString(stateCounts));
            System.out.println("wordN = " + wordN);
            System.out.println("sum(stateByWord) = " + sumArray(stateByWord));
            System.out.println("sum(stateCounts) = " + sumArray(stateCounts));
            System.out.println("sum(firstOrderTransitions) = " + sumArray(firstOrderTransitions));
            System.out.println("sum(secondOrderTransitions) = " + sumArray(secondOrderTransitions));
            */

            if (iter%1000 == 0) {
                try {
                    System.err.println("\nSaving model to :"
                        + modelOutputPath + "." + iter);
                    SerializableModel serializableModel = new SerializableModel(this);
                    serializableModel.saveModel(modelOutputPath + "." + iter);
                }
                catch (Exception e) {}
            }

        }
        executor.shutdown();
    }

    private class Worker implements Callable<ArrayList<int[]>> {

        private int start, end;
        private int[] wStateByWord, wStateCounts, wFirstOrderTransitions, wSecondOrderTransitions;
        private double[] wStateProbs;
        private MersenneTwisterFast wMtfRand;
        
        Worker(int start, int end, int[] wStateByWord, int[] wStateCounts,
            int[] wFirstOrderTransitions, int[] wSecondOrderTransitions,
            MersenneTwisterFast wMtfRand) {
            this.start = start;
            this.end = end;
            this.wStateByWord = wStateByWord.clone();
            this.wStateCounts = wStateCounts.clone();
            this.wFirstOrderTransitions = wFirstOrderTransitions.clone();
            this.wSecondOrderTransitions = wSecondOrderTransitions.clone();
            this.wStateProbs = new double[stateS];
            this.wMtfRand = wMtfRand;
        }

        public ArrayList<int[]> call() {

            //System.out.println("Worker " + start + "," + end + " starting work");

            int wordid, stateid;
            int prev = (stateS-1), current = (stateS-1), next = (stateS-1), nnext = (stateS-1);
            double max = 0, totalprob = 0;
            double r = 0;
            int wordstateoff;

            /*
                System.out.println("stateByWord = " + Arrays.toString(wStateByWord));
                System.out.println("firstOrderTransitions = " +
                    Arrays.toString(wFirstOrderTransitions));
                System.out.println("secondOrderTransitions = " +
                    Arrays.toString(wSecondOrderTransitions));
                System.out.println("stateCounts = " + Arrays.toString(wStateCounts));
                System.out.println("wordVector = " + Arrays.toString(wordVector));
                System.out.println("stateVector = " + Arrays.toString(stateVector));
            */


            int pprevsentid = -1;
            int prevsentid = -1;
            int nextsentid = -1;
            int nnextsentid = -1;
            for (int i = start; i < end; i++) {
                //System.out.println("Worker " + start + "," + end + " word " + i);
                /*
                if (i % 100000 == 0) {
                    System.err.print(((float)i/1000000) + "M, ");
                }*/
                wordid = wordVector[i];
                stateid = stateVector[i];
                wordstateoff = wordid * stateS;

                wStateByWord[wordstateoff + stateid]--;
                wStateCounts[stateid]--;
                wFirstOrderTransitions[first[i] * stateS + stateid]--;
                wSecondOrderTransitions[(second[i]*S2) + (first[i]*stateS) + stateid]--;

                /*
                System.out.println("\n\nnew counts after decrements:");
                System.out.println("StateCounts = " + Arrays.toString(wStateCounts));
                System.out.println("StateByWord = " + Arrays.toString(wStateByWord));
                System.out.println("first = " + Arrays.toString(first));
                System.out.println("second = " + Arrays.toString(second));
                System.out.println("firstOrderTransitions = " +
                    Arrays.toString(wFirstOrderTransitions));
                System.out.println("secondOrderTransitions = "+
                    Arrays.toString(wSecondOrderTransitions));*/
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

                totalprob = 0;
                try {
                    for (int j = 0;j<(stateS-1); j++) {
                        if ((statesForWord != null && statesForWord.containsKey(wordid) &&
                            statesForWord.get(wordid).contains(j)) ||
                            (statesForWord !=null && !statesForWord.containsKey(wordid)) ||
                            (statesForWord == null)) {

                            double wdeltaState = wdelta;
                            if (statesForWord != null) {
                                wdeltaState = numWordsPerState[j] * delta;
                            }
                            double x = 0.0;
                            x = (wStateByWord[wordstateoff + j] + delta) /
                                (wStateCounts[j] + wdeltaState);

                            // use trigram
                            if (useTrigram) {
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
                                // see words as 'abxcd', where x is the current word
                                double abx =
                                        (wSecondOrderTransitions[(prev*S2+current*stateS+j)]+gamma);
                                double bxc =
                                        (wSecondOrderTransitions[(current*S2+j*stateS+next)]+gamma) /
                                        (wFirstOrderTransitions[current*stateS + j] + sgamma);
                                double xcd =
                                        (wSecondOrderTransitions[(j*S2+next*stateS+nnext)]+gamma) /
                                        (wFirstOrderTransitions[j*stateS + next] + sgamma);

                                wStateProbs[j] = x*abx*bxc*xcd;
                            } else {
                                if (sentenceVector[i] != prevsentid) {
                                    current = stateS-1;
                                }
                                if (sentenceVector[i] != nextsentid) {
                                    next = stateS-1;
                                }

                                wStateProbs[j] =
                                     x *
                                    (wFirstOrderTransitions[(current*stateS + j)] + gamma) /
                                    (wStateCounts[j] + sgamma) *
                                    (wFirstOrderTransitions[j * stateS + next] + gamma);
                            }
                        } else {
                            wStateProbs[j] = 0.0;
                        }
                        totalprob += wStateProbs[j];
                    }
                } catch (ArrayIndexOutOfBoundsException e) {
                }

                r = wMtfRand.nextDouble() * totalprob;
                stateid = 0;
                max = wStateProbs[stateid];
                while (r > max) {
                    stateid++;
                    max += wStateProbs[stateid];
                }
                stateVector[i] = stateid;

                wStateByWord[wordstateoff + stateid]++;
                wStateCounts[stateid]++;
                wFirstOrderTransitions[current*stateS + stateid]++;
                wSecondOrderTransitions[prev*S2 + current*stateS+ stateid]++;
                first[i] = current;
                second[i] = prev;
                prev = current;
                current = stateid;
                pprevsentid = prevsentid;
                prevsentid = sentenceVector[i];

                /*
                System.out.println("\n\nnew counts after update:");
                System.out.println("StateCounts = " + Arrays.toString(wStateCounts));
                System.out.println("StateByWord = " + Arrays.toString(wStateByWord));
                System.out.println("first = " + Arrays.toString(first));
                System.out.println("second = " + Arrays.toString(second));
                System.out.println("firstOrderTransitions = " +
                    Arrays.toString(wFirstOrderTransitions));
                System.out.println("secondOrderTransitions = "+
                    Arrays.toString(wSecondOrderTransitions));*/
            }

            /*
            System.out.println("\n\nEnd of iteration:");
            System.out.println("WordVector = " + Arrays.toString(wordVector));
            System.out.println("SentenceVector = " + Arrays.toString(sentenceVector));
            System.out.println("StateVector = " + Arrays.toString(stateVector));
            System.out.println("StateCounts = " + Arrays.toString(wStateCounts));
            System.out.println("StateByWord = " + Arrays.toString(wStateByWord));
            System.out.println("first = " + Arrays.toString(first));
            System.out.println("second = " + Arrays.toString(second));
            System.out.println("firstOrderTransitions = " + Arrays.toString(wFirstOrderTransitions));
            System.out.println("secondOrderTransitions = "+Arrays.toString(wSecondOrderTransitions));
            System.out.println("----------------------------------------------------------");*/
            ArrayList<int[]> results = new ArrayList<int[]>();
            results.add(wStateByWord);
            results.add(wStateCounts);
            results.add(wFirstOrderTransitions);
            results.add(wSecondOrderTransitions);
            System.err.println("\tWorker (" + start + "," + end + ") completed.");
            //System.out.println("results size = " + results.size());
            return results;
        }
    }

    /**
     * Normalize the sample counts for words given state.
     */
    @Override
    protected void normalizeStates() {
        topWordsPerState = new StringDoublePair[stateS][];
        for (int i = 0; i < stateS; ++i) {
            topWordsPerState[i] = new StringDoublePair[outputPerClass];
        }

        double sum = 0.;
        for (int i = 0; i < stateS; ++i) {
            sum += stateProbs[i] = stateCounts[i] + wdelta;
            ArrayList<DoubleStringPair> topWords =
                  new ArrayList<DoubleStringPair>();
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
     * Creates a string stating the parameters used in the model. The
     * string is used for pretty printing purposes and clarity in other
     * output routines.
     */
    @Override
    public void setModelParameterStringBuilder() {
        modelParameterStringBuilder = new StringBuilder();
        String line = null;
        line = String.format("stateS:%d", stateS) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("wordW:%d", wordW) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("wordN:%d", wordN) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("gamma:%f", gamma) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("delta:%f", delta) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("initialTemperature:%f", initialTemperature) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("temperatureDecrement:%f", temperatureDecrement) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("targetTemperature:%f", targetTemperature) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("iterations:%d", iterations) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("tagSet:%s", tagMap.getTagSetName()) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("reduction-level:%s", tagMap.getReductionLevel().toString()) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("randomSeed:%d", randomSeed) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("rootDir:%s", trainDataDir) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("testRootDir:%s", testDataDir) + newline;
        modelParameterStringBuilder.append(line);
    }

}
