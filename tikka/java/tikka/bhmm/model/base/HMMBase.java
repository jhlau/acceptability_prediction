///////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2010 Taesun Moon, The University of Texas at Austin
//
//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation; either
//  version 3 of the License, or (at your option) any later version.
//
//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
//
//  You should have received a copy of the GNU Lesser General Public
//  License along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
///////////////////////////////////////////////////////////////////////////////
package tikka.bhmm.model.base;

import tikka.bhmm.apps.CommandLineOptions;
import tikka.opennlp.io.*;
import tikka.structures.*;
import tikka.utils.ec.util.MersenneTwisterFast;
import tikka.utils.annealer.*;
import tikka.utils.normalizer.*;
import tikka.utils.postags.*;
import tikka.exceptions.IgnoreTagException;

import java.io.BufferedWriter;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.StatUtils;

/**
 * The "barely hidden markov model" or "bicameral hidden markov model"
 *
 * @author tsmoon
 */
public abstract class HMMBase extends HMMFields {

    public HMMBase(CommandLineOptions options) {
        try {
            initializeFromOptions(options);
        } catch (IOException e) {
        }
    }

    // resets the path to the train data directory (only used when loading the model to avoid IO 
    // Exception)
    public void resetTrainDataDir() {
        trainDataDir = null;
    }

    // gets the model name
    public String getModelName() {
        return modelName;
    }

    // get the first order transition array
    public int[] getFirstOrderTransitions() {
        return firstOrderTransitions;
    }

    // get the second order transition array
    public int[] getSecondOrderTransitions() {
        return secondOrderTransitions;
    }

    // get the state to word array
    public int[] getStateByWord() {
        return stateByWord;
    }

    // get the state counts
    public int[] getStateCounts() {
        return stateCounts;
    }

    // get the topic to word array
    public int[] getTopicByWord() {
        return topicByWord;
    }

    // get the topic counts
    public int[] getTopicCounts() {
        return topicCounts;
    }

    // get the content state by sentence counts
    public int[] getContentStateBySentence() {
        return contentStateBySentence;
    }

    // get the sentence counts
    public int [] getSentenceCounts() {
        return sentenceCounts;
    }

    // get the number of unique words
    public int getWordW() {
        return wordW;
    }
    
    // get the number of total words/tokens
    public int getWordN() {
        return wordN;
    }

    // get the alpha prior
    public double getAlpha() {
        return alpha;
    }

    // get the beta prior
    public double getBeta() {
        return beta;
    }

    // get the gamma prior
    public double getGamma() {
        return gamma;
    }

    // get the delta prior
    public double getDelta() {
        return delta;
    }

    public int getTopicK() {
        return topicK;
    }

    // get the number of content states
    public int getStateC() {
        return stateC;
    }

    // get the number of function states
    public int getStateF() {
        return stateF;
    }

    // get the total number of states (stateC + stateF)
    public int getStateS() {
        return stateS;
    }

    // get the hash map of word to word id
    public HashMap<String, Integer> getTrainWordIdx() {
        return trainWordIdx;
    }

    // get the top words per state
    public StringDoublePair[][] getTopWordsPerState() {
        return topWordsPerState;
    }

    // get the top words per state (to be overriden by HMMLDA)
    public StringDoublePair[][] getTopWordsPerTopic() {
        return null;
    }

    // get the word vector
    public int[] getWordVector() {
        return wordVector;
    }

    // get the sentence vector
    public int[] getSentenceVector() {
        return sentenceVector;
    }

    // get the document vector
    public int[] getDocumentVector() {
        return documentVector;
    }

    // get the state vector
    public int[] getStateVector() {
        return stateVector;
    }

    // get the topic vector
    public int[] getTopicVector() {
        return topicVector;
    }

    // get the number of iterations
    public int getIterations() {
        return iterations;
    }

    // get the seed
    public int getRandomSeed() {
        return randomSeed;
    }

    // get the pos/state constraints for each word
    public HashMap<Integer, ArrayList<Integer>> getStatesForWord() {
        return statesForWord;
    }

    // get the number of words for each state
    public int[] getNumWordsPerState() {
        return numWordsPerState;
    }

    // get the pos tag names
    public HashMap<Integer, String> getTrainIdxToPos() {
        return trainIdxToPos;
    }

    // get the boolean flag of useTrigram
    public boolean getUseTrigram() {
        return useTrigram;
    }

    // get the number of threads used during training
    public int getNumThread() {
        return numThread;
    }

    // get the chunk transition of model HMMTwoTier
    public int[] getChunkTransition() {
        return chunkTransition;
    }

    // get the state transition of model HMMTwoTier
    public int[] getStateTransition() {
        return stateTransition;
    }

    // get the number of chunks (used only by model HMMTwoTier)
    public int getChunkA() {
        return chunkA;
    }

    // get the p-hash parameter (used only by model Bayesian Chunker)
    public double getPhash() {
        return phash;
    }

    // get chunkFreq (m8)
    public HashMap<String, Integer> getChunkFreq() {
        return chunkFreq;
    }

    // get chunk1T(m8)
    public HashMap<String, Integer> getChunk1T() {
        return chunk1T;
    }

    // get chunk2T(m8)
    public HashMap<String, Integer> getChunk2T() {
        return chunk2T;
    }
      

    /**
     * Initialize basic parameters from the command line. Depending on need
     * many parameters will be overwritten in subsequent initialization stages.
     *
     * @param options Command line options
     * @throws IOException
     */
    protected void initializeFromOptions(CommandLineOptions options) throws
          IOException {
        /**
         * Setting input data
         */
        modelOutputPath = options.getModelOutputPath();
        dataFormat = options.getDataFormat();
        trainDataDir = options.getTrainDataDir();
        if (trainDataDir != null) {
            trainDirReader = new DirReader(trainDataDir, dataFormat);
        } else {
            trainDataDir = "";
        }

        testDataDir = options.getTestDataDir();
        if (testDataDir != null) {
            testDirReader = new DirReader(testDataDir, dataFormat);
            testWordIdx = new HashMap<String, Integer>();
            testIdxToWord = new HashMap<Integer, String>();
            testWordIdx.put(EOSw, EOSi);
            testIdxToWord.put(EOSi, EOSw);
        } else {
            testDataDir = "";
        }

        /**
         * Whether to use trigram
         */
        useTrigram = options.useTrigram();

        /**
         * Number of threads/processors for parallelisation
         */
        numThread = options.getNumThread();

        /**
         * Setting the POS-tag dictionary
         */
        posDictPath = options.getPosDictPath();

        /**
         * Setting lexicons
         */
        trainWordIdx = new HashMap<String, Integer>();
        trainIdxToWord = new HashMap<Integer, String>();
        trainWordIdx.put(EOSw, EOSi);
        trainIdxToWord.put(EOSi, EOSw);

        /**
         * Setting dimensions
         */
        stateC = options.getContentStates();
        stateF = options.getFunctionStates();
        stateS = stateF + stateC + 1; // +1 state for dummy sentence marker
        outputPerClass = options.getOutputPerClass();
        S3 = stateS * stateS * stateS;
        S2 = stateS * stateS;
        S1 = stateS;
        chunkA = options.getChunks() + 1;

        /**
         * Setting iterations and temperatures
         */
        iterations = options.getNumIterations();
        initialTemperature = options.getInitialTemperature();
        temperature = initialTemperature;
        temperatureReciprocal = 1 / temperature;
        temperatureDecrement = options.getTemperatureDecrement();
        targetTemperature = options.getTargetTemperature();
        innerIterations = iterations;
        outerIterations =
              (int) Math.round((initialTemperature - targetTemperature)
              / temperatureDecrement) + 1;
        samples = options.getSamples();
        lag = options.getLag();
        testSetBurninIterations = options.getTestSetBurninIterations();

        /**
         * Setting hyperparameters
         */
        alpha = options.getAlpha();
        beta = options.getBeta();
        delta = options.getDelta();
        gamma = options.getGamma();
        phash = options.getPhash();

        /**
         * Initializing random number generator, etc.
         */
        randomSeed = options.getRandomSeed();
        initRandGen(randomSeed);

        tagMap = TagMapGenerator.generate(options.getTagSet(), options.getReductionLevel(), stateS);
        switch (options.getTagSet()) {
            case PTB:
            case BROWN:
                wordNormalizer = new WordNormalizer(tagMap);
                break;
            case NONE:
                wordNormalizer = new WordNormalizerToLowerNoTag(tagMap);
                break;
            default:
                wordNormalizer = new WordNormalizerToLower(tagMap);
        }

        modelName = options.getExperimentModel();
    }

    
    // get the start index for each worker (for parallelisation)
    protected int[] getWorkerStartIndex(int numProc, int[] vector) {
        int[] startIndices = new int[numProc];
        double numWorkPerWorker = (double) (vector[vector.length-1]+1)/numProc;
        int j = 0;
        startIndices[j] = 0;
        for (int i=0; i<vector.length; i++) {
            if (vector[i] >= (numWorkPerWorker*(j+1))) {
                j++;
                startIndices[j] = i;
            }
        }
        return startIndices;
    }

    // summing two arrays
    protected int[] sumList(int[] a, int[] b, String operation) {
        if ((!(operation.equals("minus")) && !(operation.equals("plus"))) ||
            (a.length != b.length)) {
            System.err.println("Invalid argument 'operation' or length of arrays mismatched");
            System.exit(1);
        }

        int[] c = new int[a.length];
        if (operation.equals("plus")) {
            for (int i=0; i<a.length; i++) {
                c[i] = a[i] + b[i];
            }   
        } else if (operation.equals("minus")) {
            for (int i=0; i<a.length; i++) {
                c[i] = a[i] - b[i];
            }   
        }   
        return c;
    }

    // get sum of array
    protected int sumArray(int[] a) {
        int result = 0;
        for (int i=0; i<a.length; i++) {
            result += a[i];
        }
        return result;
    }

    /**
     * Initialize data structures needed for inference from training data.
     */
    public void initializeFromTrainingData() {
        int numTokens = 0;
        try {
            numTokens = countNumberTokens(new DirReader(trainDataDir, dataFormat));
        } catch (IOException e) { } //ignore as it would have been caught before
        initializeTokenArrays(trainDirReader, trainWordIdx, trainIdxToWord, numTokens);
        initializeCountArrays();
        initializePosDict();

        // check that the number of sentences/documents is less than number of threads
        if (modelName.equals("m4") || modelName.equals("m6")) {
            if (numThread > documentD) {
                numThread = documentD;
            }
        } else {
            if (numThread > sentenceS) {
                numThread = sentenceS;
            }
        }

        System.err.println("\nModel Parameters ('-1' denotes unused parameter):");
        System.err.println("modelName = " + modelName);
        System.err.println("alpha = " + alpha);
        System.err.println("beta = " + beta);
        System.err.println("gammma = " + gamma);
        System.err.println("delta = " + delta);
        System.err.println("phash = " + phash);
        System.err.println("stateC = " + stateC);
        System.err.println("stateF = " + stateF);
        System.err.println("stateS = " + stateS);
        System.err.println("topicK = " + topicK);
        System.err.println("chunkA = " + chunkA);
        System.err.println("numThread = " + numThread);
        System.err.println("iterations = " + innerIterations);
        System.err.println("random seed = " + randomSeed);
        System.err.println("Number of sentences = " + sentenceS);
        System.err.println("Number of documents = " + documentD);
        System.err.println("wordW = " + wordW);
        System.err.println("wordN = " + wordN);
    }

    /**
     * Randomly initializeFull parameters for training
     */
    public abstract void initializeParametersRandom();

    /**
     * Initialize arrays that will be used to track the state, topic, split
     * position and switch of each token. The DocumentByTopic array is also
     * rewritten in sampling for test sets.
     *
     * @param dirReader Object to walk through files and directories
     * @param wordIdx   Dictionary from word to index
     * @param idxToWord Dictionary from index to word
     */
    protected void initializeTokenArrays(DirReader dirReader,
          HashMap<String, Integer> wordIdx, HashMap<Integer, String> idxToWord, int numTokens) {
        documentD = sentenceS = 0;
        wordVector = new int[numTokens];
        sentenceVector = new int[numTokens];
        if (modelName.equals("m4") || modelName.equals("m6")) {
            documentVector = new int[numTokens];
            //goldTagVector = new int[numTokens];
        }

        int i = 0;
        boolean incDocAtFinish = true;
        while ((dataReader = dirReader.nextDocumentReader()) != null) {
            try {
                String[][] sentence;
                while ((sentence = dataReader.nextSequence()) != null) {
                    if ((sentence.length == 1) && (sentence[0][0].equals("<PAGEBOUNDARY>"))) {
                        incDocAtFinish = false;
                        documentD++;
                    } else {
                        for (String[] line : sentence) {
                            try {
                                wordNormalizer.normalize(line);
                                String word = wordNormalizer.getWord();
                                String tag = wordNormalizer.getTag();
                                if (!word.isEmpty() && tag != null) {
                                    if (!wordIdx.containsKey(word)) {
                                        wordIdx.put(word, wordIdx.size());
                                        idxToWord.put(idxToWord.size(), word);
                                    }
                                    wordVector[i] = wordIdx.get(word);
                                    sentenceVector[i] = sentenceS;
                                    if (modelName.equals("m4") || modelName.equals("m6")) {
                                        documentVector[i] = documentD;
                                        //goldTagVector[i] = tagMap.get(tag);
                                    }
                                    i++;
                                }
                            } catch (IgnoreTagException e) {
                            }
                        }
                        sentenceS++;
                    }
                }
            } catch (IOException e) {
            } finally {
                try {
                    dataReader.close();
                }
                catch (IOException e) {
                    System.err.println("Error closing file = " + dataReader.getFilename() + ";" +
                        "\nMessage = " + e.getMessage());
                }
            }
            if (incDocAtFinish) {
                documentD++;
            }
        }
        System.err.println();

        wordN = numTokens;
        wordW = wordIdx.size();
        wbeta = beta * wordW;
        wdelta = delta * wordW;
        calpha = alpha * stateC;
        sgamma = gamma * stateS;

        if (!modelName.equals("m7")) {
            first = new int[wordN];
            second = new int[wordN];
        }
        stateVector = new int[wordN];

        /*
        if (!modelName.equals("m2")) {
            System.out.println("modelName = "
                + modelName);
            //third = new int[wordN];
        }*/
    }

    /**
     * Initializes arrays for counting occurrences. These need to be initialized
     * regardless of whether the model being trained from raw data or whether
     * it is loaded from a saved model.
     */
    protected void initializeCountArrays() {

        stateCounts = new int[stateS];
        stateProbs = new double[stateS];
        for (int i = 0; i < stateS; ++i) {
            stateCounts[i] = 0;
            stateProbs[i] = 0;
        }

        stateByWord = new int[stateS * wordW];
        for (int i = 0; i < stateS * wordW; ++i) {
            stateByWord[i] = 0;
        }

        contentStateBySentence = new int[stateC * sentenceS];
        for (int i = 0; i < stateC * sentenceS; ++i) {
            contentStateBySentence[i] = 0;
        }

        sentenceCounts = new int[sentenceS];
        for (int i = 0; i < sentenceS; ++i) {
            sentenceCounts[i] = 0;
        }

        documentCounts = new int[documentD];
        for (int i = 0; i < documentD; ++i) {
            documentCounts[i] = 0;
        }

        contentStateByDocument = new int[stateC * documentD];
        for (int i = 0; i < stateC * documentD; ++i) {
            contentStateByDocument[i] = 0;
        }

        functionStateByDocument = new int[stateS * documentD];
        for (int i = 0; i < stateS * documentD; ++i) {
            functionStateByDocument[i] = 0;
        }

        /*
        if (!modelName.equals("m2")) {
            thirdOrderTransitions = new int[stateS * stateS * stateS * stateS];

            for (int i = 0; i < stateS * stateS * stateS * stateS; ++i) {
                thirdOrderTransitions[i] = 0;
            }
        }*/

        secondOrderTransitions = new int[stateS * stateS * stateS];
        firstOrderTransitions = new int[stateS * stateS];

        for (int i = 0; i < stateS * stateS * stateS; ++i) {
            secondOrderTransitions[i] = 0;
        }

        for (int i = 0; i < stateS * stateS; ++i) {
            firstOrderTransitions[i] = 0;
        }

        sampleProbs = new double[samples];
        for (int i = 0; i < samples; ++i) {
            sampleProbs[i] = 0;
        }
    }

    /**
     * Initialize the pos-tag dictionary
     */
    protected void initializePosDict() {
        int i;
        if (posDictPath != null) {
            // initialize to give every state wordW
            numWordsPerState = new int[stateS];
            for (i=0; i<stateS; i++) {
                numWordsPerState[i] = wordW;
            }

            // pos ID to pos name
            trainIdxToPos = new HashMap<Integer, String>();

            // read the pos tag dictionary and initialize the state constraints for each word
            statesForWord = new HashMap<Integer, ArrayList<Integer>>();
            try {
                BufferedReader reader = new BufferedReader(new FileReader(posDictPath));
                String line = null;
                boolean tagDef = true;
                String[] parts;
                while ((line = reader.readLine()) != null) {
                    if (tagDef) {
                        parts = line.split(":");
                        if (parts.length == 2) {
                            int val = Integer.parseInt(parts[0]);
                            trainIdxToPos.put(val, parts[1]);
                        } else {
                            tagDef = false;
                        }
                    }

                    if (!tagDef) {
                        parts = line.split("\\s");
                        if (trainWordIdx.containsKey(parts[0])) {
                            int wordId = trainWordIdx.get(parts[0]);
                            ArrayList<Integer> postags = new ArrayList<Integer>();
                            for (i=1; i<(parts.length); i++) {
                                int val = Integer.parseInt(parts[i]);
                                postags.add(val);
                            }
                            statesForWord.put(wordId, postags);

                            // decrement state counts for these words
                            for (i=0; i<stateS; i++) {
                                if (!postags.contains(i)) {
                                    numWordsPerState[i]--;
                                }
                            }
                        }
                        
                    }

                }
            }
            catch (Exception e) {
                System.err.println("Error parsing the POS dictionary; message = " + e.getMessage());
                System.exit(0);
            }

        
        } else {
            statesForWord = null;
            numWordsPerState = null;
            trainIdxToPos = null;
        }
    }

    /* count the number of tokens */
    protected int countNumberTokens(DirReader dirReader) {

        int numTokens = 0;
        while ((dataReader = dirReader.nextDocumentReader()) != null) {
            try {
                String[][] sentence;
                while ((sentence = dataReader.nextSequence()) != null) {
                    if (!((sentence.length == 1) && (sentence[0][0].equals("<PAGEBOUNDARY>")))) {
                        for (String[] line : sentence) {
                            try {
                                wordNormalizer.normalize(line);
                                String word = wordNormalizer.getWord();
                                String tag = wordNormalizer.getTag();
                                if (!word.isEmpty() && tag != null) {
                                    numTokens++;
                                }
                            } catch (IgnoreTagException e) {
                            }
                        }
                    }
                }
            } catch (IOException e) {
            } finally {
                try {
                    dataReader.close();
                }
                catch (IOException e) {
                    System.err.println("Error closing file = " + dataReader.getFilename() + ";" +
                        "\nMessage = " + e.getMessage());
                }
            }
        
        }

        return numTokens;
    }




    /**
     * Learn parameters
     */
    public void train() {
        train(true);
    }

    public void train(boolean randomInitialization) {
        if (randomInitialization) {
            initializeParametersRandom();
        }

        Annealer annealer = new SimulatedAnnealer();
        /**
         * Training iterations
         */
        for (int outiter = 0; outiter < outerIterations;
              ++outiter) {
            System.err.print("\nouter iteration " + outiter + ":");
            System.err.println("annealing temperature " + temperature);
            annealer.stabilizeTemperature();
            annealer.setTemperatureReciprocal(temperatureReciprocal);
            trainInnerIter(innerIterations, annealer);
//            trainInnerIter(innerIterations, "inner iteration");
            temperature -= temperatureDecrement;
            temperatureReciprocal = 1 / temperature;
        }
        /**
         * Increment it so sampling resumes at same temperature if it is loaded
         * from a model
         */
        temperature += temperatureDecrement;
    }

    /**
     * Training routine for the inner iterations
     *
     * @param itermax Maximum number of iterations to perform
     * @param annealer Callback to annealing process
     * @see HDPHMMLDA#sampleFromTrain()
     */
    protected abstract void trainInnerIter(int itermax, Annealer annealer);

    /**
     * Maximum posterior decoding of tag sequence
     */
    public void decode() {
        Annealer annealer = new MaximumPosteriorDecoder();
        trainInnerIter(1, annealer);
    }

    public void evaluate() {
        evaluator = new Evaluator(tagMap, DistanceMeasureEnum.Measure.JACCARD);
        evaluator.evaluateTags(stateVector, goldTagVector);
        System.err.print(
              String.format("%f\t%f\t%f\t%f", evaluator.getFullOneToOneAccuracy(),
              evaluator.getFullManyToOneAccuracy(),
              evaluator.getReducedOneToOneAccuracy(),
              evaluator.getReducedManyToOneAccuracy()));
        System.err.print(
              String.format("\t%f\t%f\t%f\t%f",
              evaluator.getFullPairwisePrecision(),
              evaluator.getFullPairwiseRecall(),
              evaluator.getFullPairwiseFScore(),
              evaluator.getFullVariationOfInformation()));
        System.err.println(
              String.format("\t%f\t%f\t%f\t%f",
              evaluator.getReducedPairwisePrecision(),
              evaluator.getReducedPairwiseRecall(),
              evaluator.getReducedPairwiseFScore(),
              evaluator.getReducedVariationOfInformation()));
    }

    public void printEvaluationScore(BufferedWriter out) throws IOException {
        out.write(modelParameterStringBuilder.toString());
        printNewlines(out, 2);
        out.write(
              String.format("%f\t%f\t%f\t%f", evaluator.getFullOneToOneAccuracy(),
              evaluator.getFullManyToOneAccuracy(),
              evaluator.getReducedOneToOneAccuracy(),
              evaluator.getReducedManyToOneAccuracy()));
        out.write(
              String.format("\t%f\t%f\t%f\t%f",
              evaluator.getFullPairwisePrecision(),
              evaluator.getFullPairwiseRecall(),
              evaluator.getFullPairwiseFScore(),
              evaluator.getFullVariationOfInformation()));
        out.write(
              String.format("\t%f\t%f\t%f\t%f",
              evaluator.getReducedPairwisePrecision(),
              evaluator.getReducedPairwiseRecall(),
              evaluator.getReducedPairwiseFScore(),
              evaluator.getReducedVariationOfInformation()));
    }

    /**
     * Normalize the sample counts.
     */
    public void normalize() {
        normalizeStates();
    }

    /**
     * Normalize the sample counts for words given state.
     */
    protected void normalizeStates() {
        topWordsPerState = new StringDoublePair[stateS][];
        for (int i = 0; i < stateS; ++i) {
            topWordsPerState[i] = new StringDoublePair[outputPerClass];
        }


        double sum = 0.;
        int i = 0;
        /**
         * Normalize content states
         */
        for (; i < stateC; ++i) {
            sum += stateProbs[i] = stateCounts[i] + wbeta;
            ArrayList<DoubleStringPair> topWords =
                  new ArrayList<DoubleStringPair>();
            /**
             * Start at one to leave out EOSi
             */
            for (int j = 0; j < wordW; ++j) {
                topWords.add(new DoubleStringPair(
                      stateByWord[j * stateS + i] + beta, trainIdxToWord.get(
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

        /**
         * Normalize function states
         */
        for (; i < stateS; ++i) {
            sum += stateProbs[i] = stateCounts[i] + wdelta;
            ArrayList<DoubleStringPair> topWords =
                  new ArrayList<DoubleStringPair>();
            /**
             * Start at one to leave out EOSi
             */
            for (int j = EOSi + 1; j < wordW; ++j) {
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

        for (i = 0; i < stateS; ++i) {
            stateProbs[i] /= sum;
        }
    }

    /**
     * Print text that has been segmented/tagged in a training sample to output.
     *
     * @param outDir Root of path to generate output to
     * @throws IOException
     */
    public void printAnnotatedTrainText(String outDir) throws IOException {
        printAnnotatedText(outDir, trainDataDir, trainDirReader, trainIdxToWord);
    }

    /**
     * Print annotated text.
     *
     * @param outDir Root of path to generate output to
     * @param dataDir   Origin of data
     * @param dirReader DirReader for data
     * @param idxToWord Dictionary from index to word
     * @throws IOException
     * @see HDPHMMLDA#printAnnotatedTestText(java.lang.String)
     * @see HDPHMMLDA#printAnnotatedTrainText(java.lang.String)
     */
    public void printAnnotatedText(String outDir, String dataDir,
          DirReader dirReader, HashMap<Integer, String> idxToWord)
          throws IOException {
        DirWriter dirWriter = new DirWriter(outDir, dataDir, dirReader);
        String root = dirWriter.getRoot();

        BufferedWriter bufferedWriter;

        int docid = 0, cursent = 0, prevsent = 0;
        String word;
        bufferedWriter = dirWriter.nextOutputBuffer();

        for (int i = 0; i < wordN; ++i) {
            cursent = sentenceVector[i];
            if (docid != documentVector[i]) {
                bufferedWriter.close();
                bufferedWriter = dirWriter.nextOutputBuffer();
                docid = documentVector[i];
            }

            int wordid = wordVector[i];

            if (cursent != prevsent) {
                bufferedWriter.newLine();
            }

            word = idxToWord.get(wordid);
            bufferedWriter.write(word);
            bufferedWriter.write("\t");
            int stateid = stateVector[i];
            int goldid = goldTagVector[i];
            String tag = String.format("N:%d", stateid);
            bufferedWriter.write(tag);
            bufferedWriter.write("\t");
            tag = String.format("F:%s", tagMap.getOneToOneTagString(stateid));
            bufferedWriter.write(tag);
            bufferedWriter.write("\t");
            tag = String.format("R:%s", tagMap.getManyToOneTagString(stateid));
            bufferedWriter.write(tag);
            bufferedWriter.write("\t");
            tag = String.format("GF:%s", tagMap.getGoldTagString(goldid));
            bufferedWriter.write(tag);
            bufferedWriter.write("\t");
            tag = String.format("GR:%s", tagMap.getGoldReducedTagString(goldid));
            bufferedWriter.write(tag);
            bufferedWriter.newLine();

            prevsent = cursent;
        }
        bufferedWriter.close();

        bufferedWriter = new BufferedWriter(new OutputStreamWriter(
              new FileOutputStream(root + File.separator + "PARAMETERS")));

        bufferedWriter.write(modelParameterStringBuilder.toString());
        bufferedWriter.close();
    }

    /**
     * Print the normalized sample counts to out. Print only the top {@link
     * #outputPerTopic} per given state and topic.
     *
     * @param out Output buffer to write to.
     * @throws IOException
     */
    public void printTabulatedProbabilities(BufferedWriter out) throws
          IOException {
        printStates(out);
    }

    /**
     * Prints empty newlines in output. For pretty printing purposes.
     *
     * @param out   Destination of output
     * @param n     Number of new lines to create in output
     * @throws IOException
     */
    protected void printNewlines(BufferedWriter out, int n) throws IOException {
        for (int i = 0; i < n; ++i) {
            out.newLine();
        }
    }

    /**
     * Print the normalized sample counts for each state to out. Print only the top {@link
     * #outputPerTopic} per given state.
     *
     * @param out
     * @throws IOException
     */
    protected void printStates(BufferedWriter out) throws IOException {
        int startt = 0, M = 4, endt = Math.min(M + startt, stateProbs.length);
        out.write("***** Word Probabilities by State *****\n\n");
        while (startt < stateS) {
            for (int i = startt; i < endt; ++i) {
                String header = "S_" + i;
                header = String.format("%25s\t%6.5f\t",
                      String.format("%s:%s:%s", header,
                      tagMap.getOneToOneTagString(i),
                      tagMap.getManyToOneTagString(i)),
                      stateProbs[i]);
                out.write(header);
            }

            out.newLine();
            out.newLine();

            for (int i = 0; i < outputPerClass; ++i) {
                for (int c = startt; c < endt; ++c) {
                    String line = String.format("%25s\t%6.5f\t",
                          topWordsPerState[c][i].stringValue,
                          topWordsPerState[c][i].doubleValue);
                    out.write(line);
                }
                out.newLine();
            }
            out.newLine();
            out.newLine();

            startt = endt;
            endt = java.lang.Math.min(stateS, startt + M);
        }
    }

    /**
     * Creates a string stating the parameters used in the model. The
     * string is used for pretty printing purposes and clarity in other
     * output routines.
     */
    public void setModelParameterStringBuilder() {
        modelParameterStringBuilder = new StringBuilder();
        String line = null;
        line = String.format("alpha:%f", alpha) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("beta:%f", beta) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("gamma:%f", gamma) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("delta:%f", delta) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("stateC:%d", stateC) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("stateF:%d", stateF) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("stateS:%d", stateS) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("wordW:%d", wordW) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("wordN:%d", wordN) + newline;
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

    public void setNumIterations(int x) {
        iterations = x;
    }

    /**
     * 
     * @param options
     * @throws IOException
     */
    public void initializeFromLoadedModel(CommandLineOptions options)
          throws IOException {
        if (randomSeed == -1) {
            mtfRand = new MersenneTwisterFast();
        } else {
            mtfRand = new MersenneTwisterFast(randomSeed);
        }

        if (trainDataDir != null) {
            trainDirReader = new DirReader(trainDataDir, dataFormat);
        } else {
            trainDataDir = "";
        }

        initializeFromLoadedModel2(options); 

        initializeCountArrays();
    }

    public void initializeFromLoadedModel2(CommandLineOptions options) throws
          IOException {
        /** 
         * Revive some constants that will be used often
         */
        wbeta = beta * wordW;
        wdelta = delta * wordW;
        calpha = alpha * stateC;
        sgamma = gamma * stateS;


        stateS = stateF + stateC + 1;
        S3 = stateS * stateS * stateS;
        S2 = stateS * stateS;
        S1 = stateS;

        /** 
         * Revive the annealing regime
         */
        temperature = targetTemperature;

        first = new int[wordN];
        second = new int[wordN];
        stateProbs = new double[stateS];

        for (String word : trainWordIdx.keySet()) {
            trainIdxToWord.put(trainWordIdx.get(word), word);
        }  

        // correct stateS
        stateS = stateF + stateC + 1;

        if (stateVector != null) {
            // fill in first and second vector
            int wordid, stateid;
            int prev = (stateS-1), current = (stateS-1);
            int prevsentid = -1, pprevsentid = -1;
            int stateoff, secondstateoff, wordstateoff;
            for (int i = 0; i < wordN; ++i) {
                stateid = stateVector[i];

                if (sentenceVector[i] != prevsentid) {
                    current = stateS-1;
                    prev = stateS-1;
                }  else if (sentenceVector[i] != pprevsentid) {
                    prev = stateS-1;
                }   

                first[i] = current;
                second[i] = prev;
                prev = current;
                current = stateid;

                pprevsentid = prevsentid;
                prevsentid = sentenceVector[i];

            }   
        }
        initRandGen(randomSeed);
    }

    private void initRandGen(int sd) {
        if (sd == -1) {
            mtfRand = new MersenneTwisterFast();
        } else {
            mtfRand = new MersenneTwisterFast(sd);
        }

        // num generator for workers
        workerMtfRand = new MersenneTwisterFast[numThread];
        for (int w=0; w<numThread; w++) {
            if (sd == -1) {
                workerMtfRand[w] = new MersenneTwisterFast();
            } else {
                // use a different seed for each worker
                workerMtfRand[w] = new MersenneTwisterFast(sd+w);
            }
        }

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

    // to be overridden by child classes
    public void computeTestSentenceProb(ArrayList<Integer> testSentence,
        ArrayList<Double> sentenceProbs, ArrayList<ArrayList<Double>> stateTranProbs,
        ArrayList<ArrayList<Double>> wordStateProbs, ArrayList<int[]> stateVectors) {}

}
