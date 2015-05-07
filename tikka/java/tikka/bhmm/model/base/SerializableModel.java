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

import tikka.opennlp.io.DataFormatEnum;

import tikka.bhmm.apps.CommandLineOptions;

import tikka.utils.postags.TagMap;

import tikka.structures.*;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import java.util.HashMap;
import java.util.TreeMap;
import java.util.ArrayList;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;


/**
 * Object where model parameters are saved. Includes both constant parameters
 * and inferred parameters.
 * 
 * @author tsmoon
 */
public class SerializableModel implements Serializable {

    static private final long serialVersionUID = 42L;
    /**
     * Model buffer for loading from file. Fields are read to loadBuffer than
     * copied to the enclosing class (this).
     */
    protected SerializableModel loadBuffer = null;
    protected DataFormatEnum.DataFormat dataFormat;
    protected int randomSeed;
    protected int iterations;
    protected int wordW;
    protected int wordN;
    protected double alpha;
    protected double beta;
    protected double gamma;
    protected double delta;
    protected HashMap<String, Integer> wordIdx;
    protected int[] wordVector;
    protected double initialTemperature;
    protected double temperatureDecrement;
    protected double targetTemperature;
    protected TagMap tagMap;
    protected int stateC;
    protected int stateF;
    protected int topicK;
    protected int[] sentenceVector;
    protected int[] goldTagVector;
    protected int[] stateVector;
    protected String trainDataDir;
    protected String modelName;
    protected int outputPerClass;
    protected int documentD;
    protected int sentenceS;
    protected int[] documentVector;
    protected int[] topicVector;
    protected int[] chunkVector;
    protected int[] segVector;
    /**
     * The following are needed for the supervised experiments
     */
    protected int[] stateByWord;
    protected int[] stateCounts;
    protected int[] topicCounts;
    protected int[] topicByWord;
    protected int[] documentByTopic;
    protected int[] firstOrderTransitions;
    protected int[] secondOrderTransitions;
    protected int[] chunkTransitionSumA;
    protected int[] stateTransitionSumS;
    protected HashMap<Integer, ArrayList<Integer>> statesForWord;
    protected int[] numWordsPerState;
    protected HashMap<Integer, String>  trainIdxToPos;
    protected boolean useTrigram;
    protected int numThread;
    protected int[] chunkTransition;
    protected int[] stateTransition;
    protected int chunkA;
    protected double phash;
    protected HashMap<String, Integer> chunkFreq;
    protected HashMap<String, Integer> chunk1T;
    protected HashMap<String, Integer> chunk2T;
    protected int chunkTypes;
    protected int chunkTokens;

    /**
     * Constructor to use when model is being saved.
     * 
     * @param bhmm Model to be saved
     */
    public SerializableModel(HMMBase m) {
        alpha = m.alpha;
        beta = m.beta;
        dataFormat = m.dataFormat;
        delta = m.delta;
        documentD = m.documentD;
        documentVector = m.documentVector;
        firstOrderTransitions = m.firstOrderTransitions;
        secondOrderTransitions = m.secondOrderTransitions;
        chunkTransitionSumA = m.chunkTransitionSumA;
        stateTransitionSumS = m.stateTransitionSumS;
        gamma = m.gamma;
        goldTagVector = m.goldTagVector;
        initialTemperature = m.initialTemperature;
        iterations = m.iterations;
        modelName = m.modelName;
        outputPerClass = m.outputPerClass;
        randomSeed = m.randomSeed;
        trainDataDir = m.trainDataDir;
        sentenceS = m.sentenceS;
        sentenceVector = m.sentenceVector;
        stateByWord = m.stateByWord;
        stateCounts = m.stateCounts;
        topicByWord = m.topicByWord;
        documentByTopic = m.documentByTopic;
        topicCounts = m.topicCounts;
        stateVector = m.stateVector;
        statesForWord = m.statesForWord;
        numWordsPerState = m.numWordsPerState;
        trainIdxToPos = m.trainIdxToPos;
        useTrigram = m.useTrigram;
        numThread = m.numThread;
        stateC = m.stateC;
        stateF = m.stateF;
        tagMap = m.tagMap;
        targetTemperature = m.targetTemperature;
        temperatureDecrement = m.temperatureDecrement;
        topicK = m.topicK;
        topicVector = m.topicVector;
        chunkVector = m.chunkVector;
        segVector = m.segVector;
        wordIdx = m.trainWordIdx;
        wordN = m.wordN;
        wordVector = m.wordVector;
        wordW = m.wordW;
        chunkTransition = m.chunkTransition;
        stateTransition = m.stateTransition;
        chunkA = m.chunkA;
        phash = m.phash;
        chunkFreq = m.chunkFreq;
        chunk1T = m.chunk1T;
        chunk2T = m.chunk2T;
        chunkTypes = m.chunkTypes;
        chunkTokens = m.chunkTokens;
    }

    /**
     * Constructor to use when model is being loaded
     */
    public SerializableModel() {
    }

    /**
     * Load a previously trained model.
     *
     * @param filename  Full path of model location.
     * @return  The model that has been loaded.
     * @throws IOException
     * @throws FileNotFoundException
     */
    public HMMBase loadModel(CommandLineOptions options, String filename)
          throws IOException,
          FileNotFoundException {
        ObjectInputStream modelIn =
              new ObjectInputStream(new GZIPInputStream(new FileInputStream(
              filename)));
        try {
            loadBuffer = (SerializableModel) modelIn.readObject();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        copy(loadBuffer);
        loadBuffer = null;
        modelIn.close();

        HMMBase bhmm = ModelGenerator.generator(modelName, options);

        return copy(bhmm);
    }

    /**
     * Save the trained model.
     *
     * @param filename  Full path of model location.
     * @throws IOException
     */
    public void saveModel(String filename) throws IOException {
        ObjectOutputStream modelOut =
              new ObjectOutputStream(new GZIPOutputStream(
              new FileOutputStream(filename)));
        modelOut.writeObject(this);
        modelOut.close();
    }

    protected void copy(SerializableModel sm) {
        alpha = sm.alpha;
        beta = sm.beta;
        dataFormat = sm.dataFormat;
        delta = sm.delta;
        documentD = sm.documentD;
        documentVector = sm.documentVector;
        firstOrderTransitions = sm.firstOrderTransitions;
        secondOrderTransitions = sm.secondOrderTransitions;
        chunkTransitionSumA = sm.chunkTransitionSumA;
        stateTransitionSumS = sm.stateTransitionSumS;
        gamma = sm.gamma;   
        goldTagVector = sm.goldTagVector;
        initialTemperature = sm.initialTemperature;
        iterations = sm.iterations;
        modelName = sm.modelName;
        outputPerClass = sm.outputPerClass;
        randomSeed = sm.randomSeed;
        trainDataDir = sm.trainDataDir;
        sentenceS = sm.sentenceS;
        sentenceVector = sm.sentenceVector;
        stateByWord = sm.stateByWord;
        stateCounts = sm.stateCounts;
        topicByWord = sm.topicByWord;
        documentByTopic = sm.documentByTopic;
        topicCounts = sm.topicCounts;
        stateVector = sm.stateVector;
        statesForWord = sm.statesForWord;
        numWordsPerState = sm.numWordsPerState;
        trainIdxToPos = sm.trainIdxToPos;
        useTrigram = sm.useTrigram;
        numThread = sm.numThread;
        stateC = sm.stateC;
        stateF = sm.stateF;
        tagMap = sm.tagMap;
        targetTemperature = sm.targetTemperature;
        temperatureDecrement = sm.temperatureDecrement;
        topicK = sm.topicK;
        topicVector = sm.topicVector;
        chunkVector = sm.chunkVector;
        segVector = sm.segVector;
        wordIdx = sm.wordIdx;
        wordN = sm.wordN;
        wordVector = sm.wordVector;
        wordW = sm.wordW;
        chunkTransition = sm.chunkTransition;
        stateTransition = sm.stateTransition;
        chunkA = sm.chunkA;
        phash = sm.phash;
        chunkFreq = sm.chunkFreq;
        chunk1T = sm.chunk1T;
        chunk2T = sm.chunk2T;
        chunkTypes = sm.chunkTypes;
        chunkTokens = sm.chunkTokens;
    }

    protected HMMBase copy(HMMBase hmm) {
        hmm.alpha = alpha;
        hmm.beta = beta;
        hmm.dataFormat = dataFormat;
        hmm.delta = delta;
        hmm.documentD = documentD;
        hmm.documentVector = documentVector;
        hmm.firstOrderTransitions = firstOrderTransitions;
        hmm.secondOrderTransitions = secondOrderTransitions;
        hmm.chunkTransitionSumA = chunkTransitionSumA;
        hmm.stateTransitionSumS = stateTransitionSumS;
        hmm.gamma = gamma;
        hmm.goldTagVector = goldTagVector;
        hmm.initialTemperature = initialTemperature;
        hmm.iterations = iterations;
        hmm.modelName = modelName;
        hmm.outputPerClass = outputPerClass;
        hmm.randomSeed = randomSeed;
        hmm.trainDataDir = trainDataDir;
        hmm.sentenceS = sentenceS;
        hmm.sentenceVector = sentenceVector;
        hmm.stateByWord = stateByWord;
        hmm.stateCounts = stateCounts;
        hmm.topicByWord = topicByWord;
        hmm.documentByTopic = documentByTopic;
        hmm.topicCounts = topicCounts;
        hmm.stateVector = stateVector;
        hmm.statesForWord = statesForWord;
        hmm.numWordsPerState = numWordsPerState;
        hmm.trainIdxToPos = trainIdxToPos;
        hmm.useTrigram = useTrigram;
        hmm.numThread = numThread;
        hmm.stateC = stateC;
        hmm.stateF = stateF;
        hmm.tagMap = tagMap;
        hmm.targetTemperature = targetTemperature;
        hmm.temperatureDecrement = temperatureDecrement;
        hmm.topicK = topicK;
        hmm.topicVector = topicVector;
        hmm.chunkVector = chunkVector;
        hmm.segVector = segVector;
        hmm.trainWordIdx = wordIdx;
        hmm.wordN = wordN;
        hmm.wordVector = wordVector;
        hmm.wordW = wordW;
        hmm.chunkTransition = chunkTransition;
        hmm.stateTransition = stateTransition;
        hmm.chunkA = chunkA;
        hmm.phash = phash;
        hmm.chunkFreq = chunkFreq;
        hmm.chunk1T = chunk1T;
        hmm.chunk2T = chunk2T;
        hmm.chunkTypes = chunkTypes;
        hmm.chunkTokens = chunkTokens;

        return hmm;
    }
}
