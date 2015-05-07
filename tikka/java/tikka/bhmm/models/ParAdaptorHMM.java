package tikka.bhmm.models;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Collections;
import java.util.Arrays;
import java.util.Map;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;
import java.util.Comparator;
import java.lang.Math;

import tikka.bhmm.apps.CommandLineOptions;
import tikka.bhmm.model.base.*;
import tikka.utils.ec.util.MersenneTwisterFast;
import tikka.structures.*;
import tikka.utils.annealer.Annealer;


/**
 * AdapterHMM
 *
 * @author JH Lau
 */
public class ParAdaptorHMM extends HMM {

    // wordN vector that contains segment boundaries (0 = no segment; 1 = segment)
    int[] segVector;
    // an arraylist of linkedlist (one linkedlist for each thread/worker)
    ArrayList<LinkedList<String>> chunkChain;
    ArrayList<LinkedList<Integer>> chunkChainSentId;
    String dummyState;

    public ParAdaptorHMM(CommandLineOptions options) {
        super(options);
        beta = -1.0;
        stateC = 0;
        stateS = stateC + stateF + 1;
        topicK = -1;
        chunkA = -1;
        dummyState = Integer.toString(stateS-1);
    }

    /**
     * Initializes arrays for counting occurrences. Overriding the parent method as it has
     * different structures.
     */
    @Override
    protected void initializeCountArrays() {
        segVector = new int[wordN];
        chunkChain = new ArrayList<LinkedList<String>>();
        chunkChainSentId = new ArrayList<LinkedList<Integer>>();
        stateProbs = new double[stateS];
        stateByWord = new int[stateS * wordW];
        stateCounts = new int[stateS];
        chunkFreq = new HashMap<String, Integer>();
        chunk1T = new HashMap<String, Integer>();
        chunk2T = new HashMap<String, Integer>();
        chunkTypes = 0;
        chunkTokens = 0;
    }

    @Override
    public void initializeFromLoadedModel2(CommandLineOptions options) throws
          IOException {
            super.initializeFromLoadedModel2(options);

            // correct dummyState
            dummyState = Integer.toString(stateS-1);

            // correct chunk1T
            chunk1T.put(flatten(dummyState, dummyState, ","), sentenceS);
        }

    private int incHashMap(HashMap<String, Integer> hmap, String key, int types) {
        Integer value = 0;
        try {
            value = hmap.get(key);
        } catch (Exception e) {}
        if (value == null) {
            value = 0;
        }


        value++;
        hmap.put(key, value);

        if (value == 1) {
            types++;
        }
        
        return types;
    }

    private int decHashMap(HashMap<String, Integer> hmap, String key, int types) {
        Integer value = 0;
        try {
            value = hmap.get(key);
        } catch (Exception e) {}
        if (value == null) {
            value = 0;
        }

        if (value == 0) {
            System.err.println("Decrementing non-existant value, something is wrong. Key = " +
                key);
            System.exit(1);
        }
        
        value--;
        hmap.put(key, value);

        if (value == 0) {
            types--;
        }

        return types;
    }

    private String flatten(String s1, String s2, String s3, String symbol) {
        return s1 + symbol + s2 + symbol + s3;
    }

    private String flatten(String s1, String s2, String symbol) {
        return s1 + symbol + s2;
    }


    private String flatten(int[] stateids, String symbol) {
        String result = "";
        for (int stateid : stateids) {
            if (result.length() == 0) {
                result = Integer.toString(stateid);
            } else {
                result = result + symbol + Integer.toString(stateid);
            }
        }
        return result;
    }

    private String flatten(String[] chunks, String symbol) {
        String result = "";
        for (String c : chunks) {
            if (result.length() == 0) {
                result = c;
            } else {
                result = result + symbol + c;
            }
        }
        return result;
    }

    private Object[] getChunk(int[] segVec, int[] stateVec, int[] sentenceVec, int sentid,
        int idx, int nx) {
        int nxStart=idx-1, nxEnd=-1;
        int nxSentId = -1;
        int n = 0;

        try {
            for (int i=idx;;i++) {
                if ((segVec[i]==1)) {
                    n++;
                    if (n == nx) {
                        nxStart = i;
                    } else if ( n == (nx+1) ) {
                        nxEnd = i;
                        nxSentId = sentenceVec[i];
                        break;
                    }
                }
            }
        } catch (Exception e) {}

        if ((nxEnd == -1) || ((nxSentId != sentid) && (sentid != -1))) {
            return (new Object[] {nxSentId, dummyState});
        } else {
            String c = flatten(Arrays.copyOfRange(stateVec, nxStart+1, nxEnd+1),"+");
            return (new Object[] {nxSentId, c});
        }
    }

    private void debugPrint() {
        System.out.println("WordVector     = " + Arrays.toString(wordVector));
        System.out.println("SentenceVector = " + Arrays.toString(sentenceVector));
        System.out.println("SegVector      = " + Arrays.toString(segVector));
        System.out.println("StateVector    = " + Arrays.toString(stateVector));
        System.out.println("StateCounts = " + Arrays.toString(stateCounts));
        System.out.println("StateByWord = " + Arrays.toString(stateByWord));
        System.out.println("chunkTypes = " + chunkTypes);
        System.out.println("chunkTokens = " + chunkTokens);
        System.out.println("\n\nchunkFreq =");
        for (Map.Entry entry: chunkFreq.entrySet()) {
            System.out.println("\t" + entry.getKey() + " = " + entry.getValue());
        }
        System.out.println("\nchunk1T =");
        for (Map.Entry entry: chunk1T.entrySet()) {
            System.out.println("\t" + entry.getKey() + " = " + entry.getValue());
        }
        System.out.println("\nchunk2T =");
        for (Map.Entry entry: chunk2T.entrySet()) {
            System.out.println("\t" + entry.getKey() + " = " + entry.getValue());
        }

        System.out.println("----------------------------------------------------------");
    }

    private void updateChunkMetadata() {
        // get some chunk information
        String prevChunk = dummyState;
        int prevchunksentid = -1;
        String currChunk = dummyState;
        int currchunksentid = -1;
        Object[] chunkData = getChunk(segVector, stateVector, sentenceVector, -1, 0, 0);
        int chunksentid = (Integer) chunkData[0];
        String chunk = (String) chunkData[1];
        chunkTypes = 0;
        chunkTokens = 0;
        chunkFreq = new HashMap<String, Integer>();
        chunk1T = new HashMap<String, Integer>();
        chunk2T = new HashMap<String, Integer>();
        
        // update chunkFreq, chunk1T, chunk2T
        for (int i = 0; i < wordN; i++) {
            if (segVector[i] == 1) {
                chunkTypes = incHashMap(chunkFreq, chunk, chunkTypes);
                chunkTokens++;
                incHashMap(chunk1T, flatten(currChunk, chunk, ","), 0);
                incHashMap(chunk2T, flatten(prevChunk, currChunk, chunk, ","), 0);

                if (i != (wordN-1)) {
                    prevChunk = currChunk;
                    prevchunksentid = currchunksentid;
                    currChunk = chunk;
                    currchunksentid = chunksentid;
                    
                    chunkData = getChunk(segVector, stateVector, sentenceVector, -1, i+1, 0);
                    chunksentid = (Integer) chunkData[0];
                    chunk = (String) chunkData[1];

                    // update previous chunk information
                    if (chunksentid != currchunksentid) {
                        currChunk = dummyState;
                        prevChunk = dummyState;
                    } else if (chunksentid != prevchunksentid) {
                        prevChunk = dummyState;
                    }
                }
            }
        }
    }


    // compute log probability for a test sentence
    @Override
    public void computeTestSentenceProb(ArrayList<Integer> testSentence,
        ArrayList<Double> sentenceProbs, ArrayList<ArrayList<Double>> stateTranProbs,
        ArrayList<ArrayList<Double>> wordStateProbs, ArrayList<int[]> stateVectors) {
        wordVector = new int[testSentence.size()];
        stateVector = new int[testSentence.size()];
        segVector = new int[testSentence.size()];
        sentenceVector = new int[testSentence.size()];

        for (int i=0; i<testSentence.size(); i++) {
            wordVector[i] = testSentence.get(i);
            stateVector[i] = mtfRand.nextInt(stateS-1);
            sentenceVector[i] = 1;
            if ((i == (testSentence.size()-1)) || (mtfRand.nextDouble() > 0.5)) {
                segVector[i] = 1; 
            } else {
                segVector[i] = 0;
            }
            stateTranProbs.add(new ArrayList<Double>());
            wordStateProbs.add(new ArrayList<Double>());
        }

        //debugPrint();

        ExecutorService executor = Executors.newFixedThreadPool(numThread);
        // 5000 iterations for the test
        for (int iter = 0; iter < 5000; ++iter) {
            boolean computeProb = false;
            if ((iter >= 4500) && (iter%10==0)) {
                computeProb = true;
            }

            // collect the results from each worker
            List<Future<ArrayList<Object>>> list = new ArrayList<Future<ArrayList<Object>>>();
            Callable<ArrayList<Object>> worker = new Worker(0, 0, testSentence.size(), stateByWord, 
                stateCounts, chunkFreq, chunk1T, chunk2T, chunkTypes, chunkTokens,
                workerMtfRand[0], true, computeProb);
            Future<ArrayList<Object>> submit = executor.submit(worker);
            list.add(submit);

            for (Future<ArrayList<Object>> future : list) {
                try {
                    future.get().get(0); //stateByWord
                    future.get().get(1); //stateCounts
                    if (computeProb) {
                        sentenceProbs.add((Double) future.get().get(2)); //sentenceProbs
                        for (int j=0; j<testSentence.size(); j++) {
                            stateTranProbs.get(j).add(
                                ((ArrayList<Double>) future.get().get(3)).get(j));
                            wordStateProbs.get(j).add(
                                ((ArrayList<Double>) future.get().get(4)).get(j));
                        }
                        stateVectors.add(stateVector);
                    }
                } catch (Exception e) {
                    System.err.println("Failed to collect results. Message = " +
                        e.getMessage());
                    System.exit(1);
                }
            }

        }
        executor.shutdown();

        return;
    }


    /**
     * Randomly initializeFull learning parameters
     */
    @Override
    public void initializeParametersRandom() {
        int wordid, stateid, wordstateoff, nextsentid;

        //initialise by giving random segment boundary to words
        for (int i = 0; i < wordN; i++) {
            if (i!=(wordN-1)) {
                nextsentid = sentenceVector[i+1];
            } else {
                nextsentid = -1;
            }

            if (sentenceVector[i] != nextsentid) {
                segVector[i] = 1;
            } else if (mtfRand.nextDouble() > 0.5) {
                segVector[i] = 1; 
            } else {
                segVector[i] = 0;
            }
        }
        // initialise with random states
        for (int i = 0; i < wordN; i++) {
            wordid = wordVector[i];
            wordstateoff = stateS * wordid;
            stateid = mtfRand.nextInt(stateS-1);
            stateVector[i] = stateid;
            stateCounts[stateid]++;
            stateByWord[wordstateoff + stateid]++;
        }

        updateChunkMetadata();

        System.err.println("Completed random initialisation.");
        System.err.println("Number of chunk types = " + chunkTypes);

        /*
        System.out.println("After initialisation:");
        debugPrint();*/
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
            List<Future<ArrayList<Object>>> list = new ArrayList<Future<ArrayList<Object>>>();
            for (int np = 0; np < numThread; np++) {
                int startId = windex[np];
                int endId = wordN;
                if (np != (numThread-1)) {
                    endId = windex[np+1];
                }
                
                Callable<ArrayList<Object>> worker = new Worker(np, startId, endId, stateByWord, 
                    stateCounts, chunkFreq, chunk1T, chunk2T, chunkTypes, chunkTokens,
                    workerMtfRand[np], false, false);
                Future<ArrayList<Object>> submit = executor.submit(worker);
                list.add(submit);
            }

            int[] stateByWordDiff = new int[stateS*wordW];
            int[] stateCountsDiff = new int[stateS];
            for (Future<ArrayList<Object>> future : list) {
                try {
                    long timestart = System.nanoTime();

                    stateByWordDiff = super.sumList(stateByWordDiff,
                        super.sumList(((int[]) future.get().get(0)), stateByWord, "minus"), "plus");
                    stateCountsDiff = super.sumList(stateCountsDiff,
                        super.sumList(((int[]) future.get().get(1)), stateCounts, "minus"), "plus");

                } catch (Exception e) {
                    System.err.println("Failed to collect results. Message = " +
                        e.getMessage());
                }
            }

            // sum up all the differences and update the matrixes
            long timestart = System.nanoTime();

            stateByWord = super.sumList(stateByWord, stateByWordDiff, "plus");
            stateCounts = super.sumList(stateCounts, stateCountsDiff, "plus");
            updateChunkMetadata();

            
            /*
            System.out.println("Time spent updating dictionary = " +
                ((float) (System.nanoTime()-timestart))/1000000000 + "s");*/
            System.err.println("Number of chunk types = " + chunkTypes);
            System.err.println("chunk1T size = " + chunk1T.size());
            System.err.println("chunk2T size = " + chunk2T.size());

            /*
            System.out.println("\nIteration " + iter + " completed:");
            debugPrint(wordVector, segVector, sentenceVector,
                stateVector, stateCounts, stateByWord, chunkTypes, chunkTokens,
                chunkChain, chunkChainSentId, chunkFreq, chunk1T, chunk2T);*/
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

    private class Worker implements Callable<ArrayList<Object>> {

        private int workerid, start, end;
        private int[] wStateByWord, wStateCounts;
        private HashMap<String, Integer> wChunkFreq;
        private HashMap<String, Integer> wChunk1T;
        private HashMap<String, Integer> wChunk2T;
        private int wChunkTypes, wChunkTokens;
        private double h0, h1;
        private double[] wStateProbs;
        private MersenneTwisterFast wMtfRand;
        private boolean testMode;
        private boolean computeProb;

        Worker(int workerid, int start, int end, int[] wStateByWord, int[] wStateCounts,
            HashMap<String, Integer> wChunkFreq, HashMap<String, Integer> wChunk1T,
            HashMap<String, Integer> wChunk2T, int wChunkTypes, int wChunkTokens,
            MersenneTwisterFast wMtfRand, boolean testMode, boolean computeProb) {
            this.workerid = workerid;
            this.start = start;
            this.end = end;
            
            if (!testMode) {
                this.wStateByWord = wStateByWord.clone();
                this.wStateCounts = wStateCounts.clone();
                this.wChunkFreq = new HashMap<String, Integer>(wChunkFreq);
                this.wChunk1T = new HashMap<String, Integer>(wChunk1T);
                this.wChunk2T = new HashMap<String, Integer>(wChunk2T);
            } else {
                this.wStateByWord = wStateByWord;
                this.wStateCounts = wStateCounts;
                this.wChunkFreq = wChunkFreq;
                this.wChunk1T = wChunk1T;
                this.wChunk2T = wChunk2T;
            }

            this.wChunkTypes = wChunkTypes;
            this.wChunkTokens = wChunkTokens;
            this.wStateProbs = new double[stateS];
            this.wMtfRand = wMtfRand;
            this.testMode = testMode;
            this.computeProb = computeProb;
        }

        private int getVal(HashMap<String, Integer> hmap, String key) {
            int result = 0;
            try {
                result = hmap.get(key);
            } catch (Exception e) {}

            return result;
        }

        private int getValFast(HashMap<String, Integer> hmap, String[] chunks, int lookAt,
            int parentVal) {
            if (parentVal == 0) {
                return 0;
            }

            for (int i=lookAt; i<chunks.length; i++) {
                if (chunks[i].equals(dummyState)) {
                    return 0;
                }
            }

            return getVal(hmap, flatten(chunks, ","));
        }

        private double calcChunkProb(String ch, HashMap<String, Integer> wMap, int N, int[] counts,
            int denomAdd) {
            String[] posArray = ch.split("\\+");
            double p0 = 1.0;

            /*
            System.out.println("\nComputing Chunk prob:");
            System.out.println("chunk = " + ch);
            System.out.println("chunkFreq = " + wMap);
            System.out.println("chunkTokens = " + N);
            System.out.println("stateCounts = " + Arrays.toString(stateCounts) + "\n");*/

            for (String pos : posArray) {
                int posInt = Integer.parseInt(pos);
                double p = ((double) counts[posInt])/wordN;
                p0 = p0 * p;

            }
            p0 = p0 * phash * Math.pow((1-phash), posArray.length-1);

            double chunkProb = ((double) (getVal(wMap, ch) + (alpha*p0))) / (N + alpha + denomAdd);
            
            return chunkProb;
        }

        private double calcChunkProb(String ch, HashMap<String, Integer> wMap, int N, int[] counts) {
            return calcChunkProb(ch, wMap, N, counts, 0);
        }

        private int getChunkEnd(int[] segVec, int now) {
            int chunkEnd = 0;
            for (int i=now; i<segVec.length; i++) {
                if (segVec[i] == 1) {
                    chunkEnd = i;
                    break;
                }
            }

            return chunkEnd;
        }

        // split->merge( a|llx|rr|b -> a|llxrr|b )
        private void S2Mchunk1T(HashMap<String, Integer> c1T, String llx, String rr,
            String llxrr, String a, String b) {
            // remove link a,llx
            decHashMap(c1T, flatten(a, llx, ","), 0);
            // remove link llx,rr
            decHashMap(c1T, flatten(llx, rr, ","), 0);
            // add link a, llxrr
            incHashMap(c1T, flatten(a, llxrr, ","), 0);
            // remove link rr,b; add link llxrr,b
            if (!b.equals(dummyState)) {
                decHashMap(c1T, flatten(rr, b, ","), 0);
                incHashMap(c1T, flatten(llxrr, b, ","), 0);
            }

        }

        // split->merge ( aa|a|llx|rr|b|bb -> aa|a|llxrr|b|bb )
        private void S2Mchunk2T(HashMap<String, Integer> c2T, String llx, String rr,
            String llxrr, String aa, String a, String b, String bb) {
            // remove link aa,a,llx
            decHashMap(c2T, flatten(aa, a, llx, ","), 0);
            // remove link a,llx,rr
            decHashMap(c2T, flatten(a, llx, rr, ","), 0);
            // add link aa,a,llxrr
            incHashMap(c2T, flatten(aa, a, llxrr, ","), 0);
            if (!b.equals(dummyState)) {
                // remove link llx,rr,b
                decHashMap(c2T, flatten(llx, rr, b, ","), 0);
                // add link a,llxrr,b
                incHashMap(c2T, flatten(a, llxrr, b, ","), 0);
                if (!bb.equals(dummyState)) {
                    // remove link rr,b,bb
                    decHashMap(c2T, flatten(rr, b, bb, ","), 0);
                    // add link llxrr,b,bb
                    incHashMap(c2T, flatten(llxrr, b, bb, ","), 0);
                }
            }
        }

        // merge->split ( a|llxrr|b -> a|llx|rr|b )
        private void M2Schunk1T(HashMap<String, Integer> c1T, String llxrr, String llx,
            String rr, String a, String b) {
            // remove link a,llxrr
            decHashMap(c1T, flatten(a, llxrr, ","), 0);
            // add link a,llx
            incHashMap(c1T, flatten(a, llx, ","), 0);
            // add link llx,rr
            incHashMap(c1T, flatten(llx, rr, ","), 0);
            // remove link llxrr,b; add link rr,b
            if (!b.equals(dummyState)) {
                decHashMap(c1T, flatten(llxrr, b, ","), 0);
                incHashMap(c1T, flatten(rr, b, ","), 0);
            }

        }

        // merge->split ( aa|a|llxrr|b|bb -> aa|a|llx|rr|b|bb )
        private void M2Schunk2T(HashMap<String, Integer> c2T, String llxrr, String llx,
            String rr, String aa, String a, String b, String bb) {
            // remove link aa,a,llxrr
            decHashMap(c2T, flatten(aa, a, llxrr, ","), 0);
            // add link aa,a,llx
            incHashMap(c2T, flatten(aa, a, llx, ","), 0);
            // add link a,llx,rr
            incHashMap(c2T, flatten(a, llx, rr, ","), 0);
            if (!b.equals(dummyState)) {
                // remove link a,llxrr,b; add link llx,rr,b
                decHashMap(c2T, flatten(a, llxrr, b, ","), 0);
                incHashMap(c2T, flatten(llx, rr, b, ","), 0);
                if (!bb.equals(dummyState)) {
                    // remove link llxrr,b,bb; add link rr,b,bb
                    decHashMap(c2T, flatten(llxrr, b, bb, ","), 0);
                    incHashMap(c2T, flatten(rr, b, bb, ","), 0);
                }
            }
        }

        /*

        private void debugPrint() {
            log.println("WordVector     = " + Arrays.toString(wordVector));
            log.println("SentenceVector = " + Arrays.toString(sentenceVector));
            log.println("SegVector      = " + Arrays.toString(segVector));
            log.println("StateVector    = " + Arrays.toString(stateVector));
            log.println("StateCounts = " + Arrays.toString(wStateCounts));
            log.println("StateByWord = " + Arrays.toString(wStateByWord));
            log.println("chunkTypes = " + wChunkTypes);
            log.println("chunkTokens = " + wChunkTokens);
            
            log.println("\n\nchunkFreq =");
            for (String key : chunkFreq.keys()) {
                log.println("\t" + key + " = " + chunkFreq.get(key));
            }
            log.println("\nchunk1T =");
            for (String key : chunk1T.keys()) {
                log.println("\t" + key + " = " + chunk1T.get(key));
            }
            log.println("\nchunk2T =");
            for (String key : chunk2T.keys()) {
                log.println("\t" + key + " = " + chunk2T.get(key));
            }

            log.println("----------------------------------------------------------");
            log.flush();
        }*/

        // compute the log probability and various statistics of the test sentence
        private void computeSentenceProb(ArrayList<Object> sentenceProbResult) {
            double mgamma = (double) wChunkTypes * gamma;
            double logProb = 0;
            ArrayList<Double> stateTranProbs = new ArrayList<Double>();
            ArrayList<Double> wordStateProbs = new ArrayList<Double>();

            /*
            System.out.println("=============================================");
            System.out.println("Computing Sentence Probability:");
            debugPrint();*/


            // get chunk information
            String prevChunk = dummyState, currChunk = dummyState;

            Object[] chunkData = getChunk(segVector, stateVector, sentenceVector, -1, 0, 0);
            String c = (String) chunkData[1];
            Object[] nextChunkData = getChunk(segVector, stateVector, sentenceVector, -1, 0, 1);
            String nextChunk = (String) nextChunkData[1];
            for (int i=0; i<wordVector.length; i++) {
                int segid, wordid, stateid, wordstateoff;
                double seg=1.0, statetran=1.0, wordemission=1.0;

                segid = segVector[i];
                wordid = wordVector[i];
                stateid = stateVector[i];
                wordstateoff = stateS * wordid;

                /*
                System.out.println("\n\ni = " + i);
                System.out.println("wordid = " + wordid);
                System.out.println("segid = " + segid);
                System.out.println("stateid = " + stateid);
                System.out.println("prevChunk = " + prevChunk);
                System.out.println("currChunk = " + currChunk);
                System.out.println("c = " + c);
                System.out.println("nextChunk = " + nextChunk);*/

                // probability of a segment or non-segment
                if (i == (wordVector.length-1)) {
                    seg = 1.0;
                } else if (segid == 0) {
                    seg = calcChunkProb(c, wChunkFreq, wChunkTokens, wStateCounts);
                } else {
                    seg = calcChunkProb(c, wChunkFreq, wChunkTokens, wStateCounts) *
                        calcChunkProb(nextChunk, wChunkFreq, wChunkTokens, wStateCounts, 1);
                }

                // probability of the state transition
                String abx_key = flatten(prevChunk, currChunk, c, ",");
                String ab_key = flatten(prevChunk, currChunk, ",");
                statetran = ((double) getVal(wChunk2T, abx_key) + gamma) /
                    (getVal(wChunk1T, ab_key) + mgamma);

                // word emission probability
                wordemission = (wStateByWord[wordstateoff + stateid] + delta) /
                    (wStateCounts[stateid] + wdelta);

                stateTranProbs.add(Math.log10(seg) + Math.log10(statetran));
                wordStateProbs.add(Math.log10(seg)+Math.log10(statetran)+Math.log10(wordemission));
                logProb += Math.log10(seg)+Math.log10(statetran)+Math.log10(wordemission);

                /*
                System.out.println("\nseg = " + seg);
                System.out.println("statetran = " + statetran);
                System.out.println("wordemission = " + wordemission);
                System.out.println("logProb =" + logProb);
                System.out.println("stateTranProbs[-1] =" +
                    stateTranProbs.get(stateTranProbs.size()-1));
                System.out.println("wordStateProbs[-1] =" +
                    wordStateProbs.get(wordStateProbs.size()-1));*/
                

                // update chunk information
                if ((segid==1) && (i!=(wordVector.length-1))) {
                    prevChunk = currChunk;
                    currChunk = c;
                    chunkData = getChunk(segVector, stateVector, sentenceVector,-1,i+1,0);
                    c = (String) chunkData[1];
                    nextChunkData = getChunk(segVector, stateVector, sentenceVector, -1, i+1, 1);
                    nextChunk = (String) nextChunkData[1];
                }
            }
            sentenceProbResult.add(logProb);
            sentenceProbResult.add(stateTranProbs);
            sentenceProbResult.add(wordStateProbs);
        }

        public ArrayList<Object> call() {

            int wordid, stateid, segid;
            String prevChunk=dummyState, currChunk=dummyState;
            String c, newchunk, nextChunk, nnextChunk;
            double max = 0, totalprob = 0;
            double r = 0;
            int wordstateoff;
            int prevchunksentid=-1, currchunksentid=-1;
            int chunksentid, nextchunksentid, nnextchunksentid;
            int segStart=start, segEnd=0;

            long timea=0;
            long timeb=0;
            long timesetup=0;
            long timedec=0;
            long timerolldie=0;
            long timeupdate=0;
            long timesetup2=0;
            long timedec2=0;
            long timerolldie2=0;
            long timeupdate2=0;
            long timeforward2=0;
            long timesub=0;
            long timeflatten=0;
            long timefetch=0;
            long timemult=0;

            // get chunk information
            Object[] chunkData = getChunk(segVector, stateVector, sentenceVector, -1, start, 0);
            chunksentid = (Integer) chunkData[0];
            c = (String) chunkData[1];
            segEnd = getChunkEnd(segVector, start);
            Object[] nextChunkData = getChunk(segVector, stateVector, sentenceVector,
                chunksentid, start, 1);
            nextchunksentid = (Integer) nextChunkData[0];
            nextChunk = (String) nextChunkData[1];
            Object[] nnextChunkData = getChunk(segVector, stateVector, sentenceVector,
                chunksentid, start, 2);
            nnextchunksentid = (Integer) nnextChunkData[0];
            nnextChunk = (String) nnextChunkData[1];
            
            for (int i = start; i < end; i++) {

                long timestart = System.nanoTime();

                segid = segVector[i];

                timesetup += (System.nanoTime() - timestart);
                long timeastart = System.nanoTime();

                /*
                System.out.println("====================================================");
                System.out.println("i = " + i);
                System.out.println("end = " + end);
                System.out.println("chunksentid = " +chunksentid );
                System.out.println("chunk = " + c);
                System.out.println("\ncurrchunksentid = " + currchunksentid );
                System.out.println("currChunk = " + currChunk);
                System.out.println("\nprevchunksentid = " + prevchunksentid );
                System.out.println("prevChunk = " + prevChunk);
                System.out.println("\nnextchunksentid = " + nextchunksentid );
                System.out.println("nextChunk = " + nextChunk);
                System.out.println("\nnnextchunksentid = " + nnextchunksentid );
                System.out.println("nnextChunk = " + nnextChunk);
                System.out.println("\nsegid = " + segid);
                System.out.println("\n\n\nNow choosing whether to create a segment boundary");
                debugPrint();*/

                /**
                 * Select whether to create a segment boundary for the ith word
                 **/
                // the case where there has been a boundary before llx | rr | ...
                if (segid == 1) {

                    /*
                    System.out.println("CASE: there is a boundary previuosly (llx|rr)");
                    System.out.println("nextChunk = " + nextChunk);*/

                    // decrement the counts
                    if (!testMode) {
                        wChunkTypes = decHashMap(wChunkFreq, c, wChunkTypes);
                        wChunkTokens--;
                        if (!nextChunk.equals(dummyState)) {
                            wChunkTypes = decHashMap(wChunkFreq, nextChunk, wChunkTypes);
                            wChunkTokens--;
                        }
                    }

                    timedec += (System.nanoTime() - timeastart);
                    timeastart = System.nanoTime();

                    /*
                    System.out.println("\nDecrementing counts");
                    System.out.println("wChunkFreq = " + wChunkFreq);
                    System.out.println("wChunkTypes = " + wChunkTypes);
                    System.out.println("wChunkTokens = " + wChunkTokens);*/


                    newchunk = flatten(c, nextChunk, "+");
                    // always create a segment if it's a sentence boundary
                    if (chunksentid != nextchunksentid) {
                        h0 = 0.0;
                        h1 = 1.0;
                    } else {
                        // see it as one unit llxrr
                        h0 = calcChunkProb(newchunk, wChunkFreq, wChunkTokens, wStateCounts);
                        // see it as two units llx, rr
                        h1 = calcChunkProb(c, wChunkFreq, wChunkTokens, wStateCounts) *
                            calcChunkProb(nextChunk, wChunkFreq, wChunkTokens, wStateCounts, 1);
                    }

                    totalprob = h0+h1;
                    r = wMtfRand.nextDouble() * totalprob;

                    timerolldie += (System.nanoTime() - timeastart);
                    timeastart = System.nanoTime();


                    /*
                    System.out.println("\nnewchunk = " + newchunk);
                    System.out.println("h0 = " + h0);
                    System.out.println("h1 = " + h1);
                    System.out.println("totalprob = " + totalprob);
                    System.out.println("r = " + r);*/
                    // change to merge the segments llxrr)
                    if (r < h0) {
                        segVector[i] = 0;

                        Object[] nnnextChunkData = getChunk(segVector, stateVector, sentenceVector,
                            chunksentid, i, 2);
                        int nnnextchunksentid = (Integer) nnnextChunkData[0];
                        String nnnextChunk = (String) nnnextChunkData[1];
    
                        if (!testMode) {
                            wChunkTypes = incHashMap(wChunkFreq, newchunk, wChunkTypes);
                            wChunkTokens++;
                            S2Mchunk1T(wChunk1T, c, nextChunk, newchunk, currChunk, nnextChunk);
                            S2Mchunk2T(wChunk2T, c, nextChunk, newchunk, prevChunk, currChunk,
                                nnextChunk, nnnextChunk);
                        }

                        // update next chunks
                        c = newchunk;
                        nextchunksentid = nnextchunksentid;
                        nextChunk = nnextChunk;
                        nnextchunksentid = nnnextchunksentid;
                        nnextChunk = nnnextChunk;
                        segEnd = getChunkEnd(segVector, i);
                    } else {
                        if(!testMode) {
                            wChunkTypes = incHashMap(wChunkFreq, c, wChunkTypes);
                            wChunkTokens++;
                            if (!nextChunk.equals(dummyState)) {
                                wChunkTypes = incHashMap(wChunkFreq, nextChunk, wChunkTypes);
                                wChunkTokens++;
                            }
                        }
                    }

                    timeupdate += (System.nanoTime() - timeastart);


                } else {
                // the case where there hasn't been a boundary before  llxrr | ....

                    //decrement the counts
                    if (!testMode) {
                        wChunkTypes = decHashMap(wChunkFreq, c, wChunkTypes);
                        wChunkTokens--;    
                    }

                    timedec += (System.nanoTime() - timeastart);
                    timeastart = System.nanoTime();

                    /*
                    System.out.println("CASE: there isn't a boundary previuosly (llxrr)");
                    System.out.println("\nDecrementing counts");
                    System.out.println("wChunkFreq = " + wChunkFreq);
                    System.out.println("wChunkTypes = " + wChunkTypes);
                    System.out.println("wChunkTokens = " + wChunkTokens);*/
    

                    //see it as one unit llxrr
                    h0 = calcChunkProb(c, wChunkFreq, wChunkTokens, wStateCounts);
                    //see it as two units llx, rr
                    String lchunk="", rchunk="";
                    lchunk = flatten(Arrays.copyOfRange(stateVector, segStart, i+1), "+");
                    rchunk = flatten(Arrays.copyOfRange(stateVector, i+1, segEnd+1), "+");

                    h1 = calcChunkProb(lchunk, wChunkFreq, wChunkTokens, wStateCounts) *
                        calcChunkProb(rchunk, wChunkFreq, wChunkTokens, wStateCounts, 1);

                    totalprob = h0+h1;
                    r = wMtfRand.nextDouble() * totalprob;

                    timerolldie += (System.nanoTime() - timeastart);
                    timeastart = System.nanoTime();

                    /*
                    System.out.println("\nlchunk = " + lchunk);
                    System.out.println("rchunk = " + rchunk);
                    System.out.println("h0 = " + h0);
                    System.out.println("h1 = " + h1);
                    System.out.println("totalprob = " + totalprob);
                    System.out.println("r = " + r);*/


                    // change to create a boundary: llx | rr
                    if (r > h0) {
                        segVector[i] = 1;

                        if (!testMode) {
                            wChunkTypes = incHashMap(wChunkFreq, lchunk, wChunkTypes);
                            wChunkTypes = incHashMap(wChunkFreq, rchunk, wChunkTypes);
                            wChunkTokens++;
                            wChunkTokens++;

                            M2Schunk1T(wChunk1T, c, lchunk, rchunk, currChunk, nextChunk);
                            M2Schunk2T(wChunk2T, c, lchunk, rchunk, prevChunk, currChunk,
                                nextChunk, nnextChunk);
                        }

                        // update next chunks
                        c = lchunk;
                        nnextchunksentid = nextchunksentid;
                        nnextChunk = nextChunk;            
                        nextchunksentid = chunksentid;
                        nextChunk = rchunk;
                        segEnd = getChunkEnd(segVector, i);
                    } else {
                        if (!testMode) {
                            wChunkTypes = incHashMap(wChunkFreq, c, wChunkTypes);
                            wChunkTokens++;    
                        }
                    }

                    timeupdate += (System.nanoTime() - timeastart);
                }
            
                
                timea += (System.nanoTime()-timestart);
                timestart = System.nanoTime();

                /*
                System.out.println("\nChunk metadata all updated:");
                ArrayList<LinkedList<String>> tmp = new ArrayList<LinkedList<String>>();
                tmp.add(wChunkChain);
                ArrayList<LinkedList<Integer>> tmp2 = new ArrayList<LinkedList<Integer>>();
                tmp2.add(wChunkChainSentId);
                debugPrint(wordVector, segVector, sentenceVector,
                    stateVector, wStateCounts, wStateByWord, wChunkTypes, wChunkTokens,
                    tmp, tmp2, wChunkFreq, wChunk1T, wChunk2T);
                System.out.println("\n\n\nNow choosing the state of the word");*/


                /**
                 * Select a state for the ith word
                 **/
                wordid = wordVector[i];
                stateid = stateVector[i];
                wordstateoff = wordid * stateS;

                timesetup2 += (System.nanoTime() - timestart);
                long timebstart = System.nanoTime();

                /*
                System.out.println("wordid = " + wordid);
                System.out.println("stateid = " + stateid);
                System.out.println("wordstateoff = " + wordstateoff);
                System.out.println("prevChunk = " + prevChunk);
                System.out.println("currChunk = " + currChunk);
                System.out.println("nextChunk = " + nextChunk);
                System.out.println("nnextChunk = " + nnextChunk);*/

                // decrement the counts
                if (!testMode) {
                    wStateByWord[wordstateoff + stateid]--;
                    wStateCounts[stateid]--;
                    wChunkTypes = decHashMap(wChunkFreq, c, wChunkTypes);
                    wChunkTokens--;
                    decHashMap(wChunk1T, flatten(currChunk, c,","), 0);
                    decHashMap(wChunk2T, flatten(prevChunk,currChunk,c,","), 0);
                    if (!nextChunk.equals(dummyState)) {
                        decHashMap(wChunk1T, flatten(c, nextChunk,","), 0);
                        decHashMap(wChunk2T,
                            flatten(currChunk,c,nextChunk,","), 0);
                        if (!nnextChunk.equals(dummyState)) {
                            decHashMap(wChunk2T,
                                flatten(c,nextChunk,nnextChunk,","), 0);
                        }
                    }
                }

                timedec2 += (System.nanoTime() - timebstart);
                timebstart = System.nanoTime();

                /*
                System.out.println("\nDecrementing counts:");
                System.out.println("wStateByWord = " + Arrays.toString(wStateByWord));
                System.out.println("wStateCounts = " + Arrays.toString(wStateCounts));
                System.out.println("wChunkChain = " + wChunkChain);
                System.out.println("wChunkFreq = " + wChunkFreq);
                System.out.println("wChunk1T = " + wChunk1T);
                System.out.println("wChunk2T = " + wChunk2T);
                System.out.println("wChunkTypes = " + wChunkTypes);
                System.out.println("wChunkTokens = " + wChunkTokens);*/
                

                // now select the state
                totalprob = 0;
                double mgamma = (double) wChunkTypes * gamma;
                for (int j=0; j<(stateS-1); j++) {

                    long timestart2 = System.nanoTime();

                    //see words as abxcd
                    stateVector[i] = j;
                    String jchunk = flatten(Arrays.copyOfRange(stateVector, segStart, segEnd+1),"+");

                    timesub += (System.nanoTime() - timestart2);
                    timestart2 = System.nanoTime();

                    // get the counts
                    int bx = getValFast(wChunk1T, new String[]{currChunk, jchunk}, 2, -1);
                    int xc = getValFast(wChunk1T, new String[]{jchunk, nextChunk}, 1, -1);
                    int abx = getValFast(wChunk2T, new String[]{prevChunk, currChunk, jchunk}, 3,
                        -1);
                    int bxc = getValFast(wChunk2T, new String[]{currChunk, jchunk, nextChunk}, 2,
                        bx);
                    int xcd = getValFast(wChunk2T, new String[]{jchunk, nextChunk, nnextChunk}, 1,
                        xc);

                    timefetch += (System.nanoTime() - timestart2);
                    timestart2 = System.nanoTime();

                    double abx_ab = ((double ) (abx + gamma));
                    double bxc_bx = ((double ) (bxc + gamma)) / (bx + mgamma);
                    double xcd_xc = ((double ) (xcd + gamma)) / (xc + mgamma);
                    double x = (wStateByWord[wordstateoff + j] + delta) /
                                (wStateCounts[j] + wdelta);

                    wStateProbs[j] = x * abx_ab * bxc_bx * xcd_xc;
                    totalprob += wStateProbs[j];

                    timemult += (System.nanoTime() - timestart2);
                }
                
                r = wMtfRand.nextDouble() * totalprob;
                stateid = 0;
                max = wStateProbs[stateid];
                while (r > max) {
                    stateid++;
                    max += wStateProbs[stateid];
                }
                stateVector[i] = stateid;

                timerolldie2 += (System.nanoTime() - timebstart);
                timebstart = System.nanoTime();

                /*
                System.out.println("\n\ntotalprob = " + totalprob);
                System.out.println("r = " + r);
                System.out.println("selected state = " + stateid);*/

                // increment the counts
                if (!testMode) {
                    wStateByWord[wordstateoff + stateid]++;
                    wStateCounts[stateid]++;
                    c = flatten(Arrays.copyOfRange(stateVector, segStart, segEnd+1),"+");
                    wChunkTypes = incHashMap(wChunkFreq, c, wChunkTypes);
                    wChunkTokens++;
                    incHashMap(wChunk1T, flatten(currChunk, c,","), 0);
                    incHashMap(wChunk2T, flatten(prevChunk,currChunk,c,","), 0);
                    if (!nextChunk.equals(dummyState)) {
                        incHashMap(wChunk1T, flatten(c, nextChunk,","), 0);
                        incHashMap(wChunk2T,
                            flatten(currChunk,c,nextChunk,","), 0);
                        if (!nnextChunk.equals(dummyState)) {
                            incHashMap(wChunk2T,
                                flatten(c,nextChunk,nnextChunk,","), 0);
                        }
                    }
                }

                timeupdate2 += (System.nanoTime()- timebstart);
                timebstart = System.nanoTime();

                // update chunk id and position in the chunk
                if ((segVector[i] == 1) && (i!=(end-1))) {
                    prevChunk = currChunk;
                    currChunk = c;
                    prevchunksentid = currchunksentid;
                    currchunksentid = chunksentid;

                    //update chunk information
                    chunkData = getChunk(segVector, stateVector, sentenceVector, -1, i+1, 0);
                    chunksentid = (Integer) chunkData[0];
                    c = (String) chunkData[1];
                    nextChunkData = getChunk(segVector, stateVector, sentenceVector,
                        chunksentid, i+1, 1);
                    nextchunksentid = (Integer) nextChunkData[0];
                    nextChunk = (String) nextChunkData[1];
                    nnextChunkData = getChunk(segVector, stateVector, sentenceVector,
                        chunksentid, i+1, 2);
                    nnextchunksentid = (Integer) nnextChunkData[0];
                    nnextChunk = (String) nnextChunkData[1];

                    // update previous chunk information
                    if (chunksentid != currchunksentid) {
                        currChunk = dummyState;
                        prevChunk = dummyState;
                    } else if (chunksentid != prevchunksentid) {
                        prevChunk = dummyState;
                    }

                    segStart = i+1;
                    segEnd = getChunkEnd(segVector, i+1);
                }

                /*
                System.out.println("\nIncrementing counts and updating metadata:");
                System.out.println("new segVector[i] = " + segVector[i]);
                System.out.println("new stateVector[i] = " + stateVector[i]);
                ArrayList<LinkedList<String>> tmp = new ArrayList<LinkedList<String>>();
                tmp.add(wChunkChain);
                ArrayList<LinkedList<Integer>> tmp2 = new ArrayList<LinkedList<Integer>>();
                tmp2.add(wChunkChainSentId); */
                //debugPrint(wordVector, segVector, sentenceVector,
                //    stateVector, wStateCounts, wStateByWord, wChunkTypes, wChunkTokens,
                //    wChunkFreq, wChunk1T, wChunk2T);
                //debugPrint();

                timeforward2 += (System.nanoTime() - timebstart);
                timeb += (System.nanoTime() - timestart);

            }
            
            /*
            System.out.println("Worker " + workerid + " done.");
            System.out.println("timea = " + (((float)timea)/1000000000) + "s; timeb = " +
                (((float) timeb)/1000000000) + "s; "+
                "\n  timesetup = " + (((float)timesetup)/1000000000) + "s; " +
                "timedec = " + (((float)timedec)/1000000000) + "s; " +
                "\n    timerolldie = " + (((float)timerolldie)/1000000000) + "s; " +
                "timeupdate = " + (((float)timeupdate)/1000000000) + "s; " +
                "\n  timesetup2 = " + (((float)timesetup2)/1000000000) + "s; " +
                "timedec2 = " + (((float)timedec2)/1000000000) + "s; " +
                "\n    timerolldie2 = " + (((float)timerolldie2)/1000000000) + "s; " +
                "timeupdate2 = " + (((float)timeupdate2)/1000000000) + "s; " +
                "timeforward2 = " + (((float)timeforward2)/1000000000) + "s; " +
                "\n      timesub = " + (((float)timesub)/1000000000) + "s; " +
                "timeflatten = " + (((float)timeflatten)/1000000000) + "s; " +
                "timefetch = " + (((float)timefetch)/1000000000) + "s; " +
                "timemult = " + (((float)timemult)/1000000000) + "s");*/

            ArrayList<Object> results = new ArrayList<Object>();
            results.add(wStateByWord);
            results.add(wStateCounts);

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
            //System.out.println("results size = " + results.size());
            return results;
        }
    }

}
