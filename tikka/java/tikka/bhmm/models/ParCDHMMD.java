package tikka.bhmm.models;

import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;
import java.io.IOException;

import tikka.bhmm.model.base.*;
import tikka.utils.ec.util.MersenneTwisterFast;
import tikka.bhmm.apps.CommandLineOptions;
import tikka.utils.annealer.Annealer;

/**
 * Parallelised CDHMM-d model
 *
 * @author JH Lau
 */
public class ParCDHMMD extends HMMBase {

    public ParCDHMMD(CommandLineOptions options) {
        super(options);
        phash = -1.0;
        topicK = -1;
        chunkA = -1;
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

            int wordid, stateid, docid;
            int prev = (stateS-1), current = (stateS-1), next = (stateS-1), nnext = (stateS-1);
            double max = 0, totalprob = 0;
            double r = 0;
            int wordstateoff, docoff;
            int pprevsentid = -1; 

            int prevsentid = -1; 
            int nextsentid = -1;
            int nnextsentid = -1;

            for (int i = start; i < end; ++i) {
                wordid = wordVector[i];
                docid = documentVector[i];
                stateid = stateVector[i];
                wordstateoff = wordid * stateS;
                docoff = docid * stateC;

                if (stateid < stateC) {
                    contentStateByDocument[docoff + stateid]--;
                    documentCounts[docid]--;
                }
                wStateByWord[wordstateoff + stateid]--;
                wStateCounts[stateid]--;
                wFirstOrderTransitions[first[i] * stateS + stateid]--;
                wSecondOrderTransitions[(second[i]*S2) + (first[i]*stateS) + stateid]--;

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
                for (int j=0; j < (stateS-1); j++) {
                    // see words as 'abxcd', where x is the current word
                    double x = 0.0;
                    if (j<stateC) {
                        x = (wStateByWord[wordstateoff + j] + beta) /
                            (wStateCounts[j] + wbeta) *
                            ((contentStateByDocument[docoff + j] + alpha) /
                            (documentCounts[docid] + calpha));
                    } else {
                        x = (wStateByWord[wordstateoff + j] + delta) /
                            (wStateCounts[j] + wdelta);
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
                max = wStateProbs[0];
                stateid = 0;
                while (r > max) {
                    stateid++;
                    max += wStateProbs[stateid];
                }
                stateVector[i] = stateid;

                if (stateid < stateC) {
                    contentStateByDocument[docoff + stateid]++;
                    documentCounts[docid]++;
                }

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
            }

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
     * Randomly initialize learning parameters
     */
    @Override
    public void initializeParametersRandom() {

        int wordid, docid, stateid;
        int prev = (stateS-1), current = (stateS-1);
        double max = 0, totalprob = 0;
        double r = 0;
        int wordstateoff, docoff, stateoff, secondstateoff;

        /**
         * Initialize by assigning random topic indices to words
         */
        for (int i = 0; i < wordN; ++i) {
            wordid = wordVector[i];
            wordstateoff = wordid * stateS;

            docid = documentVector[i];
            stateoff = current * stateS;
            secondstateoff = (prev*S2) + (current*stateS);
            docoff = docid * stateC;

            totalprob = 0;

            if (mtfRand.nextDouble() > 0.5) {
                for (int j = 0; j < stateC; j++) {
                    totalprob += stateProbs[j] = 1.0;
                }
                stateid = 0;
            } else {
                for (int j = stateC; j < (stateS-1); j++) {
                    totalprob += stateProbs[j] = 1.0;
                }
                r = mtfRand.nextDouble() * totalprob;
                stateid = stateC;
            }

            r = mtfRand.nextDouble() * totalprob;
            max = stateProbs[0];
            while (r > max) {
                stateid++;
                max += stateProbs[stateid];
            }
            stateVector[i] = stateid;

            if (stateid < stateC) {
                contentStateByDocument[docoff + stateid]++;
                documentCounts[docid]++;
            }

            firstOrderTransitions[stateoff + stateid]++;
            secondOrderTransitions[secondstateoff + stateid]++;
            stateByWord[wordstateoff + stateid]++;
            stateCounts[stateid]++;
            first[i] = current;
            second[i] = prev;
            prev = current;
            current = stateid;
        }
    }

    /*
    @Override
    public void initializeFromLoadedModel(CommandLineOptions options) throws
          IOException {
        super.initializeFromLoadedModel(options);

        int current = 0;
        int wordid = 0, stateid = 0, docid;
        int stateoff, wordstateoff, docoff;

        for (int i = 0; i < wordN; ++i) {
            wordid = wordVector[i];
            docid = documentVector[i];
            stateid = stateVector[i];

            stateoff = current * stateS;
            wordstateoff = wordid * stateS;
            docoff = docid * stateC;

            if (stateid < stateC) {
                contentStateByDocument[docoff + stateid]++;
                documentCounts[docid]++;
            }
            stateByWord[wordstateoff + stateid]++;
            stateCounts[stateid]++;
            firstOrderTransitions[stateoff + stateid]++;
            first[i] = current;
            current = stateid;
        }
    }*/
}
