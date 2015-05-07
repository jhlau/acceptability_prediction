///////////////////////////////////////////////////////////////////////////////
// To change this template, choose Tools | Templates
// and open the template in the editor.
///////////////////////////////////////////////////////////////////////////////
package tikka.bhmm.model.base;

import java.util.HashMap;
import java.util.TreeMap;
import java.util.List;
import java.util.ArrayList;

import tikka.opennlp.io.*;

import tikka.structures.*;

import tikka.utils.ec.util.MersenneTwisterFast;
import tikka.utils.normalizer.*;
import tikka.utils.postags.*;

/**
 *
 * @author tsmoon
 */
public abstract class HMMFields {

    /**
     * OS neutral newline character
     */
    protected final static String newline = System.getProperty("line.separator");
    /**
     * Random number generator. Preferred over Java native Rand.
     */
    static protected MersenneTwisterFast mtfRand;
    /**
     * Random number generator for workers
     */
    static protected MersenneTwisterFast[] workerMtfRand;
    /**
     * Seed for random number generator. Default is 0.
     */
    protected int randomSeed;
    /**
     * Hyperparameter for content-state-by-sentence prior.
     */
    protected double alpha;
    /**
     * Normalization term for content-state-by-sentence multinomial
     */
    protected double calpha;
    /**
     * Hyperparameter for word-by-content-state prior
     */
    protected double beta;
    /**
     * Normalization term for word-by-content-state multinomial
     */
    protected double wbeta;
    /**
     * Hyperparameter for word-by-function-state prior
     */
    protected double delta;
    /**
     * Normalization term for word-by-function-state multinomial
     */
    protected double wdelta;
    /**
     * Hyperparameter for state transition prior
     */
    protected double gamma;
    /**
     * Normalization term for state transition multinomial
     */
    protected double sgamma;
    /**
     * Hyperparameter for transition out of content states
     */
    protected double psi = 10;
    protected double spsi;
    /**
     * Number of sentences
     */
    protected int sentenceS;
    /**
     * Number of documents
     */
    protected int documentD;
    /**
     * End of sentence marker index
     */
    protected final int EOSi = 0;
    /**
     * End of sentence marker
     */
    protected final String EOSw = "DUMMYSENTENCEMARKER";
    /**
     * Number of word types
     */
    protected int wordW;
    /**
     * Number of word tokens
     */
    protected int wordN;
    /**
     * Number of types to printTabulatedProbabilities per class (topic and/or state)
     */
    protected int outputPerClass;
    /**
     * Array of word indexes. Of length {@link #wordN}.
     */
    protected int[] wordVector;
    /**
     * Array of sentence indexes. Of length {@link #wordN}.
     */
    protected int[] sentenceVector;
    /**
     * Array of document indexes
     */
    protected int[] documentVector;
    /**
     * Number of content states. It also includes the initial start state
     * as an offset.
     */
    protected int stateC;
    /**
     * Number of function states
     */
    protected int stateF;
    /**
     * Sum of states that combines the start state (0), the content states and
     * the function states
     */
    protected int stateS;
    /**
     * Offset for 3rd, 2nd, and 1st order state count array
     */
    protected int S3, S2, S1;
    /**
     * 3rd order state counts
     */
    protected int[] thirdOrderTransitions;
    /**
     * 2nd order state counts
     */
    protected int[] secondOrderTransitions;
    /**
     * 1st order state counts
     */
    protected int[] firstOrderTransitions;
    /**
     * Array of counts per state
     */
    protected int[] stateCounts;
    /**
     * Array of counts per sentence
     */
    protected int[] sentenceCounts;
    /**
     * Array of counts per document
     */
    protected int[] documentCounts;
    /**
     * Array of states one word before in previous iteration
     */
    protected int[] first;
    /**
     * Array of states two words before in previous iteration
     */
    protected int[] second;
    /**
     * Array of states over tokens
     */
    protected int[] stateVector;
    /**
     * Array of full gold tags
     */
    protected int[] goldTagVector;
//    /**
//     * Array of reduced gold tags
//     */
//    protected int[] goldReducedTagVector;
//    /**
//     * Hashtable from gold tag to index for evaluation
//     */
//    HashMap<String, Integer> goldTagToIdx;
//    /**
//     * Hashtable from index to gold tag
//     */
//    HashMap<Integer, String> idxToGoldTag;
//    /**
//     * Array of counts for words given content states.
//     */
//    protected int[] contentStateByWord;
//    /**
//     * Array of counts for words given function states.
//     */
//    protected int[] functionStateByWord;
    /**
     * Array of counts for words given all states
     */
    protected int[] stateByWord;
    /**
     * Array of counts for content states given sentence
     */
    protected int[] contentStateBySentence;
    /**
     * LDAHMM parameters
     */
    protected int[] topicByWord;
    protected int[] topicCounts;
    protected int[] documentByTopic;
    /**
     *
     */
    protected int[] contentStateByDocument;
    /**
     * Array of function states over documents
     */
    protected int[] functionStateByDocument;
    /**
     * array that contains all chunk states for every word (model HMMTwoTier)
     */
    protected int[] chunkVector;
    /**
     * sum of all chunks for chunk transition (model HMMTwoTier)
     */
    protected int[] chunkTransitionSumA;
    /**
     * sum of all states for state transition (model HMMTwoTier)
     */
    protected int[] stateTransitionSumS;
    /**
     * Array of counts for chunk transition (model HMMTwoTier)
     */
    protected int[] chunkTransition;
    /**
     * Array of counts for state transition (model HMMTwoTier)
     */
    protected int[] stateTransition;
    /**
     * Probability of each state
     */
    /**
     * Number of chunks
     */
    protected int chunkA;
    protected double[] stateProbs;

    /** 
     * wordN vector that contains segment boundaries (0 = no segment; 1 = segment)
       (model bayesian chunker)
     */
    protected int[] segVector;

    /**
     * Table of top {@link #outputPerClass} words per state. Used in
     * normalization and printing.
     */
    protected StringDoublePair[][] topWordsPerState;
    /**
     * Hashtable from word to index for training data.
     */
    protected HashMap<String, Integer> trainWordIdx;
    /**
     * Hashtable from index to word for training data.
     */
    protected HashMap<Integer, String> trainIdxToWord;
    /**
     * Hashtable from word to index for training data.
     */
    protected HashMap<String, Integer> testWordIdx;
    /**
     * Hashtable from index to word for training data.
     */
    protected HashMap<Integer, String> testIdxToWord;
    /**
     * Path of training data.
     */
    protected String trainDataDir;
    /**
     * Path of test data.
     */
    protected String testDataDir;
    /**
     * Reader for each document
     */
    protected DataReader dataReader;
    /**
     * Reader for walking through training directories
     */
    protected DirReader trainDirReader;
    /**
     * Reader for walking through test directories
     */
    protected DirReader testDirReader;
    /**
     * Temperature at which to start annealing process
     */
    protected double initialTemperature;
    /**
     * Decrement at which to reduce the temperature in annealing process
     */
    protected double temperatureDecrement;
    /**
     * Stop changing temperature after the following temp has been reached.
     */
    protected double targetTemperature;
    /**
     * The actual exponent in the process. Is the reciprocal of the temperature.
     */
    protected double temperatureReciprocal;
    /**
     * Current temperature for annealing.
     */
    protected double temperature;
    /**
     * Number of iterations
     */
    protected int iterations;
    /**
     * Number of iterations per temperature increment. This is only used
     * when simulated annealing is implemented. It is identical to
     * {@link #iteration}.
     */
    protected int innerIterations;
    /**
     * Number of outer iterations per annealing scheme. This is only used
     * when simulated annealing is implemented. It is set to
     * <pre>
     * (targetTemperature-initialTemperature)/temperatureDecrement+1
     * </pre>
     */
    protected int outerIterations;
    /**
     * Number of iterations for test set burnin
     */
    protected int testSetBurninIterations;
    /**
     * Format of the input data
     */
    protected DataFormatEnum.DataFormat dataFormat;
    /**
     * Specifies how to normalize words
     */
    protected WordNormalizer wordNormalizer;
    /**
     * Type of model that is being run.
     */
    protected String modelName;
    /**
     * Path to save the model
     */
    protected String modelOutputPath;
    /**
     * Number of samples to take
     */
    protected int samples;
    /**
     * Number of iterations between samples
     */
    protected int lag;
    /**
     * Probability table of tokens per sample.
     */
    protected double[] sampleProbs;
    /**
     * String for maintaining all model parameters. Only for printing purposes.
     */
    protected StringBuilder modelParameterStringBuilder;
    /**
     * Object that handles the both model and gold tag sets. It also finds
     * the best mapping from one to the other.
     */
    protected TagMap tagMap;
    /**
     * Class for dealing with evaluation
     */
    protected Evaluator evaluator;
    /**
     * Number of topic types.
     */
    protected int topicK;
    /**
     * Array of topic indexes. Of length {@link #wordN}.
     */
    protected int[] topicVector;
    /**
     * Path of POS-tag dictionary.
     */
    protected String posDictPath;
    /**
     * POS tag / state constraints for words.
     */
    protected HashMap<Integer, ArrayList<Integer>> statesForWord;
    /**
     * Number of words for each state
     */
    protected int[] numWordsPerState;
    /**
     * POS tag and its IDS
     */
    protected HashMap<Integer, String> trainIdxToPos;
    /**
     * Use Trigram
     */
    protected boolean useTrigram;
    /**
     * Number of threads/processors for parallelisation
     */
    protected int numThread;
    /**
     * p-hash parameter used by Bayesian Chunker
     */
    protected double phash;
    /**
     * HashMap of chunk (multi-POS unit) frequencies; used by Bayesian Chunker
     */
    protected HashMap<String, Integer> chunkFreq;
    /**
     * HashMap of chunk 2nd order transition; used by Bayesian Chunker
     */
    protected HashMap<String, Integer> chunk2T;
    /**
     * HashMap of chunk 1st order transition; used by Bayesian Chunker
     */
    protected HashMap<String, Integer> chunk1T;
    /**
     * Number of unique chunk types; used by Bayesian Chunker
     */
    protected int chunkTypes;
    /**
     * Number of total chunk tokens; used by Bayesian Chunker
     */
    protected int chunkTokens;
    /**
     * Copy a sequence of numbers from ta to array ia.
     *
     * @param <T>   Any number type
     * @param ia    Target array of integers to be copied to
     * @param ta    Source List<T> of numbers to be copied from
     */
    protected static <T extends Number> void copyToArray(int[] ia, List<T> ta) {
        for (int i = 0; i < ta.size(); ++i) {
            ia[i] = ta.get(i).intValue();
        }
    }
}
