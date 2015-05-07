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
 * Bayesian Chunker model
 *
 * @author JH Lau
 */
public class ParBayesianChunker extends HMM {

    int[] wordCounts;
    String dummyState;

    public ParBayesianChunker(CommandLineOptions options) {
        super(options);
        delta = -1.0;
        gamma = -1.0;
        stateC = 0;
        stateS = 0;
        topicK = -1;
        chunkA = -1;
        dummyState = Integer.toString(0);
    }

    /**
     * Initializes arrays for counting occurrences. Overriding the parent method as it has
     * quite different structures.
     */
    @Override
    protected void initializeCountArrays() {
        segVector = new int[wordN];
        stateProbs = null;
        stateByWord = null;
        stateCounts = null;
        stateVector = null;
        chunkFreq = new HashMap<String, Integer>();
        chunk1T = new HashMap<String, Integer>();
        chunk2T = new HashMap<String, Integer>();
        chunkTypes = 0;
        chunkTokens = 0;
    
        initializeWordCounts();
    }

    @Override
    public void initializeFromLoadedModel2(CommandLineOptions options) throws
          IOException {
        super.initializeFromLoadedModel2(options);

        // correct dummyState
        dummyState = Integer.toString(0);

        initializeWordCounts();
    }

    @Override
    public void normalize() {}

    @Override
    public StringDoublePair[][] getTopWordsPerState() {
        return (new StringDoublePair[0][0]);
    }

    private void initializeWordCounts() {
        wordCounts = new int[wordW];
        
        for (int i=0; i<wordN; i++) {
            wordCounts[wordVector[i]]++;
        }

        return;
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

    private Object[] getChunk(int[] segVec, int[] wordVec, int[] sentenceVec, int sentid,
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
            String c = flatten(Arrays.copyOfRange(wordVec, nxStart+1, nxEnd+1),"+");
            return (new Object[] {nxSentId, c});
        }
    }

    private void debugPrint() {
        System.out.println("WordVector     = " + Arrays.toString(wordVector));
        System.out.println("SentenceVector = " + Arrays.toString(sentenceVector));
        System.out.println("SegVector      = " + Arrays.toString(segVector));
        System.out.println("chunkTypes = " + chunkTypes);
        System.out.println("chunkTokens = " + chunkTokens);
        System.out.println("wordCounts = " + Arrays.toString(wordCounts));
        System.out.println("\n\nchunkFreq =");
        for (Map.Entry entry: chunkFreq.entrySet()) {
            System.out.println("\t" + entry.getKey() + " = " + entry.getValue());
        }
        System.out.println("\nchunk1T =");
        for (Map.Entry entry: chunk1T.entrySet()) {
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
        Object[] chunkData = getChunk(segVector, wordVector, sentenceVector, -1, 0, 0);
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
                //incHashMap(chunk2T, flatten(prevChunk, currChunk, chunk, ","), 0);

                if (i != (wordN-1)) {
                    prevChunk = currChunk;
                    prevchunksentid = currchunksentid;
                    currChunk = chunk;
                    currchunksentid = chunksentid;
                    
                    chunkData = getChunk(segVector, wordVector, sentenceVector, -1, i+1, 0);
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

        // add dummyStates to chunkFreq; this count won't matter in the training gibbs sampler as
        // this term will exist in both h0 and h1 but it'll be needed for computing
        // test probability
        chunkFreq.put(dummyState, sentenceS);
    }


    // compute log probability for a test sentence
    @Override
    public void computeTestSentenceProb(ArrayList<Integer> testSentence,
        ArrayList<Double> sentenceProbs, ArrayList<ArrayList<Double>> stateTranProbs,
        ArrayList<ArrayList<Double>> wordStateProbs, ArrayList<int[]> stateVectors) {
        wordVector = new int[testSentence.size()];
        segVector = new int[testSentence.size()];
        sentenceVector = new int[testSentence.size()];

        for (int i=0; i<testSentence.size(); i++) {
            wordVector[i] = testSentence.get(i);
            sentenceVector[i] = 1;
            if ((i == (testSentence.size()-1)) || (mtfRand.nextDouble() > 0.5)) {
                segVector[i] = 1; 
            } else {
                segVector[i] = 0;
            }
            stateTranProbs.add(new ArrayList<Double>());
            wordStateProbs.add(new ArrayList<Double>());
        }

        Worker worker = new Worker(0, testSentence.size(), 
            chunkFreq, chunk1T, chunk2T, chunkTypes, chunkTokens,
            workerMtfRand[0], true, false);
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
                    sentenceProbs.add((Double) results.get(0)); //sentenceProbs
                    //System.out.println("\tSegVector = " + Arrays.toString(segVector));
                    //System.out.println("\t\tSentenceProb = " + results.get(0));
                    for (int j=0; j<testSentence.size(); j++) {
                        wordStateProbs.get(j).add(
                            ((ArrayList<Double>) results.get(1)).get(j));
                    }
            }
            } catch (Exception e) {
                System.err.println("Error calling worker. Message = " + e.getMessage());
            }

        }

        //System.out.println("segVector = " + Arrays.toString(segVector));

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
            } else if (mtfRand.nextDouble() < 0.5) {
                segVector[i] = 1; 
            } else {
                segVector[i] = 0;
            }
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
                
                Callable<ArrayList<Object>> worker = new Worker(startId, endId, 
                    chunkFreq, chunk1T, chunk2T, chunkTypes, chunkTokens,
                    workerMtfRand[np], false, false);
                Future<ArrayList<Object>> submit = executor.submit(worker);
                list.add(submit);
            }

            for (Future<ArrayList<Object>> future : list) {
                try {
                    //long timestart = System.nanoTime();
                    future.get().get(0);
                } catch (Exception e) {
                    System.err.println("Failed to collect results. Message = " +
                        e.getMessage());
                }
            }

            // sum up all the differences and update the matrixes
            //long timestart = System.nanoTime();

            updateChunkMetadata();

            
            /*
            System.out.println("Time spent updating dictionary = " +
                ((float) (System.nanoTime()-timestart))/1000000000 + "s");*/
            System.err.println("Number of chunk types = " + chunkTypes);
            System.err.println("chunk1T size = " + chunk1T.size());
            System.err.println("chunk2T size = " + chunk2T.size());

            /*
            System.out.println("wordN = " + wordN);
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

        private int start, end;
        private HashMap<String, Integer> wChunkFreq;
        private HashMap<String, Integer> wChunk1T;
        private HashMap<String, Integer> wChunk2T;
        private int wChunkTypes, wChunkTokens;
        private double h0, h1;
        private MersenneTwisterFast wMtfRand;
        private boolean testMode;
        private boolean computeProb;

        Worker(int start, int end,
            HashMap<String, Integer> wChunkFreq, HashMap<String, Integer> wChunk1T,
            HashMap<String, Integer> wChunk2T, int wChunkTypes, int wChunkTokens,
            MersenneTwisterFast wMtfRand, boolean testMode, boolean computeProb) {
            this.start = start;
            this.end = end;
            
            if (!testMode) {
                this.wChunkFreq = new HashMap<String, Integer>(wChunkFreq);
                this.wChunk1T = new HashMap<String, Integer>(wChunk1T);
                this.wChunk2T = new HashMap<String, Integer>(wChunk2T);
            } else {
                this.wChunkFreq = wChunkFreq;
                this.wChunk1T = wChunk1T;
                this.wChunk2T = wChunk2T;
            }

            this.wChunkTypes = wChunkTypes;
            this.wChunkTokens = wChunkTokens;
            this.wMtfRand = wMtfRand;
            this.testMode = testMode;
            this.computeProb = computeProb;
        }

        public void setComputeProb(boolean flag) {
            this.computeProb = flag;
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
            System.out.println("chunkTokens = " + N);*/

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
            // add link a, llxrr
            incHashMap(c1T, flatten(a, llxrr, ","), 0);
            // remove link rr,b; add link llxrr,b
            if (!b.equals(dummyState)) {
                incHashMap(c1T, flatten(llxrr, b, ","), 0);
            }

        }

        // merge->split ( a|llxrr|b -> a|llx|rr|b )
        private void M2Schunk1T(HashMap<String, Integer> c1T, String llxrr, String llx,
            String rr, String a, String b) {
            // add link a,llx
            incHashMap(c1T, flatten(a, llx, ","), 0);
            // add link llx,rr
            incHashMap(c1T, flatten(llx, rr, ","), 0);
            // remove link llxrr,b; add link rr,b
            if (!b.equals(dummyState)) {
                incHashMap(c1T, flatten(rr, b, ","), 0);
            }

        }

        private double calcTransProb(String a, String b) {
            if (b.equals(dummyState)) {
                return 1.0;
            }
            int a_b_count = getVal(wChunk1T, flatten(a, b, ","));
            double a_b = ((double) a_b_count +
                beta*calcChunkProb(b, wChunkFreq, wChunkTokens, wordCounts)) /
                (getVal(wChunkFreq, a) + beta*wChunkTypes);

            return a_b;
        }

        private void workerDebugPrint() {
            System.out.println("WordVector     = " + Arrays.toString(wordVector));
            System.out.println("SentenceVector = " + Arrays.toString(sentenceVector));
            System.out.println("SegVector      = " + Arrays.toString(segVector));
            System.out.println("chunkTypes = " + wChunkTypes);
            System.out.println("chunkTokens = " + wChunkTokens);
            System.out.println("wordCounts = " + Arrays.toString(wordCounts));
            System.out.println("\n\nchunkFreq =");
            for (Map.Entry entry: wChunkFreq.entrySet()) {
                System.out.println("\t" + entry.getKey() + " = " + entry.getValue());
            }
            System.out.println("\nchunk1T =");
            for (Map.Entry entry: wChunk1T.entrySet()) {
                System.out.println("\t" + entry.getKey() + " = " + entry.getValue());
            }

            System.out.println("----------------------------------------------------------");
        }

        // compute the log probability and various statistics of the test sentence
        private void computeSentenceProb(ArrayList<Object> sentenceProbResult) {
            double logProb = 0;
            ArrayList<Double> wordStateProbs = new ArrayList<Double>();

            /*
            System.out.println("=============================================");
            System.out.println("Computing Sentence Probability:");
            debugPrint();*/


            // get chunk information
            String prevChunk = dummyState, currChunk = dummyState;

            Object[] chunkData = getChunk(segVector, wordVector, sentenceVector, -1, 0, 0);
            String c = (String) chunkData[1];
            Object[] nextChunkData = getChunk(segVector, wordVector, sentenceVector, -1, 0, 1);
            String nextChunk = (String) nextChunkData[1];
            for (int i=0; i<wordVector.length; i++) {
                int segid, wordid;
                double seg=1.0;

                segid = segVector[i];
                wordid = wordVector[i];

                /*
                System.out.println("\n\ni = " + i);
                System.out.println("wordid = " + wordid);
                System.out.println("segid = " + segid);
                System.out.println("prevChunk = " + prevChunk);
                System.out.println("currChunk = " + currChunk);
                System.out.println("c = " + c);
                System.out.println("nextChunk = " + nextChunk);*/


                /*
                if (i == (wordVector.length-1)) {
                    seg = 1.0;
                } else if (segid == 0) {
                    seg = calcChunkProb(c, wChunkFreq, wChunkTokens, wordCounts);
                } else {
                    seg = calcChunkProb(c, wChunkFreq, wChunkTokens, wordCounts) *
                        calcChunkProb(nextChunk, wChunkFreq, wChunkTokens, wordCounts, 1);
                }*/

                /*
                wordStateProbs.add(Math.log10(seg));
                logProb += Math.log10(seg);*/

                if (segid == 1) {
                    logProb += Math.log10(calcTransProb(currChunk, c));
                }
                wordStateProbs.add(0.0);
                



                // probability of the state transition
                //String abx_key = flatten(prevChunk, currChunk, c, ",");
                //String ab_key = flatten(prevChunk, currChunk, ",");
                //statetran = ((double) getVal(wChunk2T, abx_key) + gamma) /
                //    (getVal(wChunk1T, ab_key) + mgamma);

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
                    chunkData = getChunk(segVector, wordVector, sentenceVector,-1,i+1,0);
                    c = (String) chunkData[1];
                    nextChunkData = getChunk(segVector, wordVector, sentenceVector, -1, i+1, 1);
                    nextChunk = (String) nextChunkData[1];
                }
            }
            sentenceProbResult.add(logProb);
            sentenceProbResult.add(wordStateProbs);
        }

        public ArrayList<Object> call() {

            int wordid, segid;
            String prevChunk=dummyState, currChunk=dummyState;
            String c, newchunk, nextChunk, nnextChunk;
            double max = 0, totalprob = 0;
            double r = 0;
            int prevchunksentid=-1, currchunksentid=-1;
            int chunksentid, nextchunksentid, nnextchunksentid;
            int segStart=start, segEnd=0;

            // get chunk information
            Object[] chunkData = getChunk(segVector, wordVector, sentenceVector, -1, start, 0);
            chunksentid = (Integer) chunkData[0];
            c = (String) chunkData[1];
            segEnd = getChunkEnd(segVector, start);
            Object[] nextChunkData = getChunk(segVector, wordVector, sentenceVector,
                chunksentid, start, 1);
            nextchunksentid = (Integer) nextChunkData[0];
            nextChunk = (String) nextChunkData[1];
            Object[] nnextChunkData = getChunk(segVector, wordVector, sentenceVector,
                chunksentid, start, 2);
            nnextchunksentid = (Integer) nnextChunkData[0];
            nnextChunk = (String) nnextChunkData[1];
            
            for (int i = start; i < end; i++) {
                segid = segVector[i];

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
                workerDebugPrint();*/

                /**
                 * Select whether to create a segment boundary for the ith word
                 **/
                // the case where there has been a boundary before a | llx | rr | b ...
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
                            // remove link llx,rr
                            decHashMap(wChunk1T, flatten(c, nextChunk, ","), 0);
                        }
                        // remove link a,llx
                        decHashMap(wChunk1T, flatten(currChunk, c, ","), 0);
                        if (!nnextChunk.equals(dummyState)) {
                            // remove link rr,b
                            decHashMap(wChunk1T, flatten(nextChunk, nnextChunk, ","), 0);
                        }
                    }

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
                        // see it as one unit a|llxrr|b
                        double a_llxrr = calcTransProb(currChunk, newchunk);
                        double llxrr_b = calcTransProb(newchunk, nnextChunk);
                        h0 = a_llxrr * llxrr_b;
                        // see it as two units a|llx|rr|b
                        double a_llx = calcTransProb(currChunk, c);
                        double llx_rr = calcTransProb(c, nextChunk);
                        double rr_b = calcTransProb(nextChunk, nnextChunk);
                        h1 = a_llx * llx_rr * rr_b;

                        /*
                        System.out.println("\na_llxrr = " + a_llxrr);
                        System.out.println("llxrr_b = " + llxrr_b);
                        System.out.println("a_llx = " + a_llx);
                        System.out.println("llx_rr = " + llx_rr);
                        System.out.println("rr_b = " + rr_b);*/
                    }


                    totalprob = h0+h1;
                    r = wMtfRand.nextDouble() * totalprob;

                    /*
                    System.out.println("\nnewchunk = " + newchunk);
                    System.out.println("h0 = " + h0);
                    System.out.println("h1 = " + h1);
                    System.out.println("totalprob = " + totalprob);
                    System.out.println("r = " + r);*/

                    // change to merge the segments llxrr)
                    if (r < h0) {
                        segVector[i] = 0;

                        Object[] nnnextChunkData = getChunk(segVector, wordVector, sentenceVector,
                            chunksentid, i, 2);
                        int nnnextchunksentid = (Integer) nnnextChunkData[0];
                        String nnnextChunk = (String) nnnextChunkData[1];
    
                        if (!testMode) {
                            wChunkTypes = incHashMap(wChunkFreq, newchunk, wChunkTypes);
                            wChunkTokens++;
                            S2Mchunk1T(wChunk1T, c, nextChunk, newchunk, currChunk, nnextChunk);
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
                                // add link llx,rr
                                incHashMap(wChunk1T, flatten(c, nextChunk, ","), 0);
                            }
                            // add link a,llx
                            incHashMap(wChunk1T, flatten(currChunk, c, ","), 0);
                            if (!nnextChunk.equals(dummyState)) {
                                // add link rr,b
                                incHashMap(wChunk1T, flatten(nextChunk, nnextChunk, ","), 0);
                            }
                        }
                    }
                } else {
                // the case where there hasn't been a boundary before  a|llxrr|b ....

                    //decrement the counts
                    if (!testMode) {
                        wChunkTypes = decHashMap(wChunkFreq, c, wChunkTypes);
                        wChunkTokens--;    
                        if (!nextChunk.equals(dummyState)) {
                            // remove link llx,rr
                            decHashMap(wChunk1T, flatten(c, nextChunk, ","), 0);
                        }
                        // remove link a,llx
                        decHashMap(wChunk1T, flatten(currChunk, c, ","), 0);
                    }

                    /*
                    System.out.println("CASE: there isn't a boundary previuosly (llxrr)");
                    System.out.println("\nDecrementing counts");
                    System.out.println("wChunkFreq = " + wChunkFreq);
                    System.out.println("wChunkTypes = " + wChunkTypes);
                    System.out.println("wChunkTokens = " + wChunkTokens);*/
    
                    // see it as one unit a|llxrr|b
                    double a_llxrr = calcTransProb(currChunk, c);
                    double llxrr_b = calcTransProb(c, nextChunk);
                    h0 = a_llxrr * llxrr_b;
                    // see it as two units a|llx|rr|b
                    String lchunk = flatten(Arrays.copyOfRange(wordVector, segStart, i+1), "+");
                    String rchunk = flatten(Arrays.copyOfRange(wordVector, i+1, segEnd+1), "+");
                    double a_llx = calcTransProb(currChunk, lchunk);
                    double llx_rr = calcTransProb(lchunk, rchunk);
                    double rr_b = calcTransProb(rchunk, nextChunk);
                    h1 = a_llx * llx_rr * rr_b;

                    totalprob = h0+h1;
                    r = wMtfRand.nextDouble() * totalprob;

                    /*
                    System.out.println("\nlchunk = " + lchunk);
                    System.out.println("rchunk = " + rchunk);
                    System.out.println("a_llxrr = " + a_llxrr);
                    System.out.println("llxrr_b = " + llxrr_b);
                    System.out.println("a_llx = " + a_llx);
                    System.out.println("llx_rr = " + llx_rr);
                    System.out.println("rr_b = " + rr_b);
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
                            //M2Schunk2T(wChunk2T, c, lchunk, rchunk, prevChunk, currChunk,
                            //    nextChunk, nnextChunk);
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
                            if (!nextChunk.equals(dummyState)) {
                                // add link llx,rr
                                incHashMap(wChunk1T, flatten(c, nextChunk, ","), 0);
                            }
                            // add link a,llx
                            incHashMap(wChunk1T, flatten(currChunk, c, ","), 0);
                        }
                    }
                }
            
                /*
                System.out.println("\nChunk metadata all updated:");
                ArrayList<LinkedList<String>> tmp = new ArrayList<LinkedList<String>>();
                tmp.add(wChunkChain);
                ArrayList<LinkedList<Integer>> tmp2 = new ArrayList<LinkedList<Integer>>();
                tmp2.add(wChunkChainSentId);
                System.out.println("\n\n\nNow choosing the state of the word");*/


                // update chunk id and position in the chunk
                if ((segVector[i] == 1) && (i!=(end-1))) {
                    prevChunk = currChunk;
                    currChunk = c;
                    prevchunksentid = currchunksentid;
                    currchunksentid = chunksentid;

                    //update chunk information
                    chunkData = getChunk(segVector, wordVector, sentenceVector, -1, i+1, 0);
                    chunksentid = (Integer) chunkData[0];
                    c = (String) chunkData[1];
                    nextChunkData = getChunk(segVector, wordVector, sentenceVector,
                        chunksentid, i+1, 1);
                    nextchunksentid = (Integer) nextChunkData[0];
                    nextChunk = (String) nextChunkData[1];
                    nnextChunkData = getChunk(segVector, wordVector, sentenceVector,
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
                workerDebugPrint();*/
            }
            

            ArrayList<Object> results = new ArrayList<Object>();

            if (computeProb) {
                ArrayList<Object> sentenceProbResult = new ArrayList<Object>();
                computeSentenceProb(sentenceProbResult);
                results.add((Double) sentenceProbResult.get(0)); // sent log prob
                // sent individual seg prob
                results.add((ArrayList<Double>) sentenceProbResult.get(1));
            } else {
                results.add(0.0);
            }
            
            if (!testMode) {
                System.err.println("\tWorker (" + start + "," + end + ") completed.");
            }
            //System.out.println("results size = " + results.size());
            return results;
        }
    }

}
