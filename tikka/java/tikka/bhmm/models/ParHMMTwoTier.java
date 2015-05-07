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
 * Parallelised Two Tier HMM Model. 
 * First tier latent variables = 'state'; second tier latent variables = 'chunk'
 *
 * @author Jey Han Lau
 */
public class ParHMMTwoTier extends HMMBase {

    // array to store chunk probabilities
    protected double[] chunkProbs;

    public ParHMMTwoTier(CommandLineOptions options) {
        super(options);

        beta = -1.0;
        phash = -1.0;
        stateC = 0;
        stateS = stateF + 1;
        topicK = -1;
    }

    /**
     * Initializes arrays for counting occurrences. Overriding the parent method as HMMTwoTier has
     * quite different structures.
     */
    @Override
    protected void initializeCountArrays() {
        stateProbs = new double[stateS];
        chunkProbs = new double[chunkA];
        chunkVector = new int[wordN];
        stateByWord = new int[stateS * wordW];
        stateCounts = new int[stateS];
        chunkTransition = new int[chunkA * stateS * chunkA];
        chunkTransitionSumA = new int[chunkA * stateS];
        stateTransition = new int[chunkA * stateS * stateS];
        stateTransitionSumS = new int[chunkA * stateS];
    }

    /**
     * Overriding method to load model
     */
    @Override
    public void initializeFromLoadedModel2(CommandLineOptions options) throws IOException {
        super.initializeFromLoadedModel2(options);
        chunkProbs = new double[chunkA];
    }


    /**
     * Randomly initialize learning parameters
     */
    @Override
    public void initializeParametersRandom() {
        int wordid, stateid, chunkid;
        int prevsentid = -1;
        int currentChunk = (chunkA-1);
        int prevState = (stateS-1), currentState = (stateS-1);
        int wordstateoff, stateoff, chunkoff;

        /**
         * Initialize by assigning random indices to words
         */
        for (int i = 0; i < wordN; ++i) {
            wordid = wordVector[i];
            if (sentenceVector[i] != prevsentid) {
                currentChunk = (chunkA-1);
                prevState = (stateS-1);
                currentState = (stateS-1);
            }

            //randomise a chunk
            chunkid = mtfRand.nextInt(chunkA-1);
            // randomise a state
            stateid = mtfRand.nextInt(stateS-1);

            wordstateoff = stateS * wordid;
            chunkoff = currentChunk*stateS*chunkA + currentState*chunkA;
            stateoff = chunkid*stateS*stateS + currentState*stateS;

            stateVector[i] = stateid;
            chunkVector[i] = chunkid;

            chunkTransition[chunkoff + chunkid]++;
            chunkTransitionSumA[currentChunk*stateS + currentState]++;
            stateTransition[stateoff + stateid]++;
            stateTransitionSumS[chunkid*stateS + currentState]++;
            stateByWord[wordstateoff + stateid]++;
            stateCounts[stateid]++;

            /*
            System.out.println("\nword = " + i);
            System.out.println("randomised chunkid = " + chunkid);
            System.out.println("randomised stateid = " + stateid);
            System.out.println("currentChunk = " + currentChunk);
            System.out.println("currentState = " + currentState);
            System.out.println("Updating chunkTransition index = " + (chunkoff+chunkid));
            System.out.println("Updating chunkTransitionSumA index = " +
                (currentChunk*stateS+currentState));
            System.out.println("Updating stateTransition index = " + (stateoff+stateid));
            System.out.println("Updating stateTransitionSumS index = " +
                (chunkid*stateS+currentState));
            */

            prevState = currentState;
            currentState = stateid;
            currentChunk = chunkid;
            prevsentid = sentenceVector[i];

        }
        /*
        System.out.println("After initialisation:");
        System.out.println("WordVector = " + Arrays.toString(wordVector));
        System.out.println("SentenceVector = " + Arrays.toString(sentenceVector));
        System.out.println("StateVector = " + Arrays.toString(stateVector));
        System.out.println("ChunkVector = " + Arrays.toString(chunkVector));
        System.out.println("StateCounts = " + Arrays.toString(stateCounts));
        System.out.println("StateByWord = " + Arrays.toString(stateByWord));
        System.out.println("chunkTransition = " + Arrays.toString(chunkTransition));
        System.out.println("chunkTransitionSumA = " + Arrays.toString(chunkTransitionSumA));
        System.out.println("stateTransition = " + Arrays.toString(stateTransition));
        System.out.println("stateTransitionSumS = " + Arrays.toString(stateTransitionSumS));
        System.out.println("==========================================================");*/
    }

   // compute log probability for a test sentence
    @Override
    public void computeTestSentenceProb(ArrayList<Integer> testSentence,
        ArrayList<Double> sentenceProbs, ArrayList<ArrayList<Double>> stateTranProbs,
        ArrayList<ArrayList<Double>> wordStateProbs, ArrayList<int[]> stateVectors) {

        // revive some of the variables
        sgamma = stateS*gamma;
        wdelta = wordW*delta;

        // initialise the vectors
        wordVector = new int[testSentence.size()];
        stateVector = new int[testSentence.size()];
        chunkVector = new int[testSentence.size()];
        sentenceVector = new int[testSentence.size()];
        for (int i=0; i<testSentence.size(); i++) {
            wordVector[i] = testSentence.get(i);
            stateVector[i] = mtfRand.nextInt(stateS-1);
            chunkVector[i] = mtfRand.nextInt(chunkA-1);
            sentenceVector[i] = 1;
            stateTranProbs.add(new ArrayList<Double>());
            wordStateProbs.add(new ArrayList<Double>());
        }

        // fix stateTransitionSumSumS and chunkTransitionSumA
        chunkTransitionSumA = new int [chunkA*stateS];
        stateTransitionSumS = new int [chunkA*stateS];
        for (int i=0; i<chunkA; i++) {
            for (int j=0; j<stateS; j++) {
                int sumA = 0;
                for (int k=0; k<chunkA; k++) {
                    sumA += chunkTransition[i*stateS*chunkA + j*chunkA + k];
                }
                chunkTransitionSumA[i*stateS + j] = sumA;
            }
        }
        for (int i=0; i<chunkA; i++) {
            for (int j=0; j<stateS; j++) {
                int sumS = 0;
                for (int k=0; k<stateS; k++) {
                    sumS += stateTransition[i*stateS*stateS+j*stateS+k];
                }
                stateTransitionSumS[i*stateS+j] = sumS;
            }
        }

        Worker worker = new Worker(0, testSentence.size(), stateByWord, 
            stateCounts, chunkTransition, chunkTransitionSumA,
            stateTransition, stateTransitionSumS, workerMtfRand[0], true, false);
        boolean computeProb = false;
        for (int iter = 0; iter < 5000; ++iter) {
            computeProb = false;
            if ((iter >= 4500) && (iter%10==0)) {
                computeProb = true;
            }
            worker.setComputeProb(computeProb);
            try {
                ArrayList<Object> results = worker.call();
                if (computeProb) {
                    sentenceProbs.add((Double) results.get(6)); //sentenceProbs
                    for (int j=0; j<testSentence.size(); j++) {
                        stateTranProbs.get(j).add(
                            ((ArrayList<Double>) results.get(7)).get(j));
                        wordStateProbs.get(j).add(
                            ((ArrayList<Double>) results.get(8)).get(j));
                    }
                    stateVectors.add(stateVector.clone());
                }
            } catch (Exception e) {
                System.err.println("Error calling worker. Message = " + e.getMessage());
            }

        }

        return;
    }


    /**
     * Training routine for the inner iterations
     */
    @Override
    protected void trainInnerIter(int itermax, Annealer annealer) {
        int[] chunkTransitionDiff = new int[chunkA*stateS*chunkA];
        int[] chunkTransitionSumADiff = new int[chunkA*stateS];
        int[] stateTransitionDiff = new int[chunkA*stateS*stateS];
        int[] stateTransitionSumSDiff = new int[chunkA*stateS];
        int[] stateByWordDiff = new int[stateS*wordW];
        int[] stateCountsDiff = new int[stateS];

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
            List<Future<ArrayList<Object>>> list = new ArrayList<Future<ArrayList<Object>>>();
            for (int np = 0; np < numThread; np++) {
                int startId = windex[np];
                int endId = wordN;
                if (np != (numThread-1)) {
                    endId = windex[np+1];
                }
                
                Callable<ArrayList<Object>> worker = new Worker(startId, endId, stateByWord, 
                    stateCounts, chunkTransition, chunkTransitionSumA,
                    stateTransition, stateTransitionSumS, workerMtfRand[np], false, false);
                Future<ArrayList<Object>> submit = executor.submit(worker);
                list.add(submit);
            }

            Arrays.fill(chunkTransitionDiff, 0);
            Arrays.fill(chunkTransitionSumADiff, 0);
            Arrays.fill(stateTransitionDiff, 0);
            Arrays.fill(stateTransitionSumSDiff, 0);
            Arrays.fill(stateByWordDiff, 0);
            Arrays.fill(stateCountsDiff, 0);
            
            /*
            System.out.println("Original chunkTransitionSumA = " +
                Arrays.toString(chunkTransitionSumA));
            System.out.println("Original chunkTransitionSumADiff = " +
               Arrays.toString(chunkTransitionSumADiff));
            */
            for (Future<ArrayList<Object>> future : list) {
                try {
                    stateByWordDiff = super.sumList(stateByWordDiff,
                        super.sumList(((int[]) future.get().get(0)), stateByWord, "minus"), "plus");
                    stateCountsDiff = super.sumList(stateCountsDiff,
                        super.sumList(((int[]) future.get().get(1)), stateCounts, "minus"), "plus");
                    chunkTransitionDiff = super.sumList(chunkTransitionDiff,
                        super.sumList(((int[]) future.get().get(2)), chunkTransition, "minus"),
                        "plus");
                    chunkTransitionSumADiff = super.sumList(chunkTransitionSumADiff,
                        super.sumList(((int[]) future.get().get(3)), chunkTransitionSumA, "minus"),
                        "plus");
                    stateTransitionDiff = super.sumList(stateTransitionDiff,
                        super.sumList(((int[]) future.get().get(4)), stateTransition, "minus"),
                        "plus");
                    stateTransitionSumSDiff = super.sumList(stateTransitionSumSDiff,
                        super.sumList(((int[]) future.get().get(5)), stateTransitionSumS, "minus"),
                        "plus");

                    /*
                    System.out.println("\nworker chunkTransitionSumA = " +
                        Arrays.toString(future.get().get(3)));
                    System.out.println("updated chunkTransitionSumADiff = " +
                        Arrays.toString(chunkTransitionSumADiff));*/
                } catch (Exception e) {
                    System.err.println("Failed to collect results. Message = " +
                        e.getMessage());
                }
            }

            // sum up all the differences and update the matrixes
            stateByWord = super.sumList(stateByWord, stateByWordDiff, "plus");
            stateCounts = super.sumList(stateCounts, stateCountsDiff, "plus");
            chunkTransition = super.sumList(chunkTransition, chunkTransitionDiff, "plus");
            chunkTransitionSumA = super.sumList(chunkTransitionSumA, chunkTransitionSumADiff, "plus");
            stateTransition = super.sumList(stateTransition, stateTransitionDiff, "plus");
            stateTransitionSumS = super.sumList(stateTransitionSumS, stateTransitionSumSDiff, "plus");

            /*
            System.out.println("\nFinal chunkTransition = " + Arrays.toString(chunkTransition));
            System.out.println("Final chunkTransitionSumA = " + Arrays.toString(chunkTransitionSumA));
            System.out.println("Final stateTransition = " + Arrays.toString(stateTransition));
            System.out.println("Final stateTransitionSumA = " + Arrays.toString(stateTransitionSumS));
            System.out.println("wordN = " + wordN);
            System.out.println("sum(stateByWord) = " + sumArray(stateByWord));
            System.out.println("sum(stateCounts) = " + sumArray(stateCounts));
            System.out.println("sum(chunkTransition) = " + sumArray(chunkTransition));
            System.out.println("sum(chunkTransitionSumA) = " + sumArray(chunkTransitionSumA));
            System.out.println("sum(stateTransition) = " + sumArray(stateTransition));
            System.out.println("sum(stateTransitionSumS) = " + sumArray(stateTransitionSumS));*/

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

    private class Worker implements Callable<ArrayList<Object>> {

        private int start, end;
        private int[] wStateByWord, wStateCounts, wChunkTransition, wChunkTransitionSumA;
        private int[] wStateTransition, wStateTransitionSumS;
        private double[] wChunkProbs, wStateProbs;
        private MersenneTwisterFast wMtfRand;
        private boolean testMode;
        private boolean computeProb;
        
        Worker(int start, int end, int[] wStateByWord, int[] wStateCounts,
            int[] wChunkTransition, int[] wChunkTransitionSumA,
            int[] wStateTransition, int[] wStateTransitionSumS,
            MersenneTwisterFast wMtfRand, boolean testMode, boolean computeProb) {
            this.start = start;
            this.end = end;

            if (!testMode) {
                this.wStateByWord = wStateByWord.clone();
                this.wStateCounts = wStateCounts.clone();
                this.wChunkTransition = wChunkTransition.clone();
                this.wChunkTransitionSumA = wChunkTransitionSumA.clone();
                this.wStateTransition = wStateTransition.clone();
                this.wStateTransitionSumS = wStateTransitionSumS.clone();
            } else {
                this.wStateByWord = wStateByWord;
                this.wStateCounts = wStateCounts;
                this.wChunkTransition = wChunkTransition;
                this.wChunkTransitionSumA = wChunkTransitionSumA;
                this.wStateTransition = wStateTransition;
                this.wStateTransitionSumS = wStateTransitionSumS;
            }
            this.wStateProbs = new double[stateS];
            this.wChunkProbs = new double[chunkA];
            this.wMtfRand = wMtfRand;
            this.testMode = testMode;
            this.computeProb = computeProb;
        }

        public void setComputeProb(boolean flag) {
            this.computeProb = flag;
        }

        public ArrayList<Object> call() {

            //System.out.println("Worker " + start + "," + end + " starting work");

            int wordid, stateid, chunkid;
            int prevState = (stateS-1), currentState = (stateS-1),
                nextState = (stateS-1), nnextState = (stateS-1);
            int prevStateOld = (stateS-1), currentStateOld = (stateS-1);
            int currentChunk = (chunkA-1), nextChunk = (chunkA-1);
            int currentChunkOld = (chunkA-1);
            double max = 0, totalprob = 0;
            double r = 0;
            int wordstateoff;
            double aalpha = alpha * chunkA;

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
                chunkid = chunkVector[i];
                wordstateoff = wordid * stateS;

                try {
                    nextState = stateVector[i + 1];
                    nextsentid = sentenceVector[i + 1];
                    nextChunk = chunkVector[i + 1];
                } catch (ArrayIndexOutOfBoundsException e) {
                    nextState = stateS-1;
                    nextsentid = -1;
                    nextChunk = chunkA-1;
                }
                try {
                    nnextState = stateVector[i + 2];
                    nnextsentid = sentenceVector[i + 2];
                } catch (ArrayIndexOutOfBoundsException e) {
                    nnextState = stateS-1;
                    nnextsentid = -1;
                }
                if (sentenceVector[i] != prevsentid) {
                    currentState = stateS-1;
                    prevState = stateS-1;
                    currentChunk = chunkA-1;
                    currentStateOld = stateS-1;
                    prevStateOld = stateS-1;
                    currentChunkOld = chunkA-1;
                }  else if (sentenceVector[i] != pprevsentid) {
                    prevState = stateS-1;
                    prevStateOld = stateS-1;
                }
                if (sentenceVector[i] != nextsentid) {
                    nextState = stateS-1;
                    nnextState = stateS-1;
                    nextChunk = chunkA-1;
                }  else if (sentenceVector[i] != nnextsentid) {
                    nnextState = stateS-1;
                }

                if (!testMode) {
                    wStateByWord[wordstateoff + stateid]--;
                    wStateCounts[stateid]--;
                    wChunkTransition[currentChunkOld*stateS*chunkA + currentStateOld*chunkA +
                        chunkid]--;
                    wChunkTransitionSumA[currentChunkOld*stateS + currentStateOld]--;
                    wStateTransition[chunkid*stateS*stateS + currentStateOld*stateS + stateid]--;
                    wStateTransitionSumS[chunkid*stateS + currentStateOld]--;
                }

                /*
                System.out.println("=====================================================");
                System.out.println("i = " + i);
                System.out.println("wordid = " + wordid);
                System.out.println("sentid = " + sentenceVector[i]);
                System.out.println("prevsentid = " + prevsentid);
                System.out.println("chunkid = " + chunkid);
                System.out.println("stateid = " + stateid);
                System.out.println("currentChunk = " + currentChunk);
                System.out.println("currentChunkOld = " + currentChunkOld);
                System.out.println("currentState = " + currentState);
                System.out.println("currentStateOld = " + currentStateOld);
                System.out.println("Decrementing wChunkTransition index = " +
                    (currentChunkOld*stateS*chunkA + currentStateOld*chunkA + chunkid));
                System.out.println("Decrementing wChunkTransitionSumA index = " +
                    (currentChunkOld*stateS + currentStateOld));
                System.out.println("Decrementing wStateTransition index = " +
                    (chunkid*stateS*stateS + currentStateOld*stateS + stateid));
                System.out.println("Decrementing wStateTransitionSumS index = " +
                    (chunkid*stateS + currentStateOld));*/

                // update *Old
                prevStateOld = currentStateOld;
                currentStateOld = stateid;
                currentChunkOld = chunkid;

                // roll a chunk state first
                totalprob = 0;
                for (int j=0; j<(chunkA-1); j++) {
                    int chunkoff = currentChunk*stateS*chunkA + currentState*chunkA;
                    int nextchunkoff = j*stateS*chunkA + stateid*chunkA;
                    int stateoff = j*stateS*stateS + currentState*stateS;
                    wChunkProbs[j] = (wChunkTransition[chunkoff+j] + alpha) *
                        (wChunkTransition[nextchunkoff+nextChunk] + alpha) /
                        (wChunkTransitionSumA[j*stateS + stateid] + aalpha) *
                        (wStateTransition[stateoff + stateid] + gamma) /
                        (wStateTransitionSumS[j*stateS + currentState] + sgamma);
                    totalprob += wChunkProbs[j];
                }
                r = wMtfRand.nextDouble() * totalprob;
                chunkid = 0;
                max = wChunkProbs[chunkid];
                while (r > max) {
                    chunkid++;
                    max += wChunkProbs[chunkid];
                }
                chunkVector[i] = chunkid;

                // now roll for the state
                totalprob = 0;
                for (int j = 0;j<(stateS-1); j++) {
                    int stateoff = chunkid*stateS*stateS + currentState*stateS;
                    int nextstateoff = nextChunk*stateS*stateS + j*stateS;
                    int nextchunkoff = chunkid*stateS*chunkA + j*chunkA;
                    wStateProbs[j] = (wChunkTransition[nextchunkoff+nextChunk]+alpha) /
                        (wChunkTransitionSumA[chunkid*stateS + j] + aalpha) *
                        (wStateTransition[stateoff + j] + gamma) *
                        (wStateTransition[nextstateoff + nextState] + gamma) /
                        (wStateTransitionSumS[nextChunk*stateS + j] + sgamma) *
                        (wStateByWord[wordstateoff + j] + delta) /
                        (wStateCounts[j] + wdelta);
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

                if (!testMode) {
                    // update the arrays
                    wStateByWord[wordstateoff + stateid]++;
                    wStateCounts[stateid]++;
                    wChunkTransition[currentChunk*stateS*chunkA + currentState*chunkA + chunkid]++;
                    wChunkTransitionSumA[currentChunk*stateS + currentState]++;
                    wStateTransition[chunkid*stateS*stateS + currentState*stateS + stateid]++;
                    wStateTransitionSumS[chunkid*stateS + currentState]++;
                }

                /*
                System.out.println("\nNew chunkid = " + chunkid);
                System.out.println("New stateid = " + stateid);
                System.out.println("Incrementing wChunkTransition index = " +
                    (currentChunk*stateS*chunkA + currentState*chunkA + chunkid));
                System.out.println("Incrementing wChunkTransitionSumA index = " +
                    (currentChunk*stateS + currentState));
                System.out.println("Incrementing wStateTransition index = " +
                    (chunkid*stateS*stateS + currentState*stateS + stateid));
                System.out.println("Incrementing wStateTransitionSumS index = " +
                    (chunkid*stateS + currentState));
                */

                prevState = currentState;
                currentState = stateid;
                currentChunk = chunkid;
                pprevsentid = prevsentid;
                prevsentid = sentenceVector[i];
            }

            /*
            System.out.println("End of iteration:");
            System.out.println("WordVector = " + Arrays.toString(wordVector));
            System.out.println("SentenceVector = " + Arrays.toString(sentenceVector));
            System.out.println("StateVector = " + Arrays.toString(stateVector));
            System.out.println("StateCounts = " + Arrays.toString(wStateCounts));
            System.out.println("StateByWord = " + Arrays.toString(wStateByWord));
            System.out.println("first = " + Arrays.toString(first));
            System.out.println("second = " + Arrays.toString(second));
            System.out.println("firstOrderTransitions = " + Arrays.toString(wFirstOrderTransitions));
            System.out.println("secondOrderTransitions = "+Arrays.toString(wSecondOrderTransitions));
            System.out.println("----------------------------------------------------------");
            */
            ArrayList<Object> results = new ArrayList<Object>();
            results.add(wStateByWord);
            results.add(wStateCounts);
            results.add(wChunkTransition);
            results.add(wChunkTransitionSumA);
            results.add(wStateTransition);
            results.add(wStateTransitionSumS);

            if (computeProb) {
                ArrayList<Object> sentenceProbResult = new ArrayList<Object>();
                computeSentenceProb(sentenceProbResult);
                results.add((Double) sentenceProbResult.get(0)); // sent log prob
                results.add((ArrayList<Double>) sentenceProbResult.get(1)); //statetran probs
                results.add((ArrayList<Double>) sentenceProbResult.get(2)); //wordstate probs
            }

            if (!testMode) {
                System.err.println("\tWorker (" + start + "," + end + ") completed.");
            }
            return results;
        }
    }

    // compute the log probability, state transition and state transition * word emission 
    // probabilities of the test sentence
    private void computeSentenceProb(ArrayList<Object> sentenceProbResult) {
        double logProb = 0;
        ArrayList<Double> stateTranProbs = new ArrayList<Double>();
        ArrayList<Double> wordStateProbs = new ArrayList<Double>();

        double aalpha = chunkA*alpha;
        int currentState = stateS-1;
        int currentChunk = chunkA-1;

        for (int i=0; i<wordVector.length; i++) {
            int wordid = wordVector[i];
            int stateid = stateVector[i];
            int chunkid = chunkVector[i];

            double sx = (stateByWord[stateS*wordid+stateid] + delta) /
                (stateCounts[stateid] + wdelta);
            double stp =
                (chunkTransition[currentChunk*stateS*chunkA+currentState*chunkA+
                chunkid] + alpha) /
                (chunkTransitionSumA[currentChunk*stateS+currentState] + aalpha) *
                (stateTransition[chunkid*stateS*stateS+currentState*stateS+stateid] +
                gamma) /
                (stateTransitionSumS[chunkid*stateS+currentState] + sgamma);
            double wordStateProb = stp*sx;

            logProb += Math.log10(wordStateProb);
            stateTranProbs.add(Math.log10(stp));
            wordStateProbs.add(Math.log10(wordStateProb));

            currentState = stateid;
            currentChunk = chunkid;
        }

        sentenceProbResult.add(logProb);
        sentenceProbResult.add(stateTranProbs);
        sentenceProbResult.add(wordStateProbs);
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



}
