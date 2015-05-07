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
package tikka.hhl.models.base;

import tikka.hhl.models.base.HDPHMMLDA;
import tikka.hhl.apps.CommandLineOptions;

import tikka.hhl.models.m1.HDPHMMLDAm1;
import tikka.hhl.models.m2.HDPHMMLDAm2;

import tikka.opennlp.io.DataFormatEnum;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import java.util.HashMap;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import tikka.hhl.models.m3.HDPHMM;
import tikka.hhl.models.m4.HDPLDA;

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
    /**
     * Format of the input data
     */
    protected DataFormatEnum.DataFormat dataFormat;
    /**
     * Seed for random number generator.
     */
    protected int randomSeed;
    /**
     * Number of topic types.
     */
    protected int topicK;
    /**
     * Number of iterations
     */
    protected int iterations;
    /**
     * Number of documents
     */
    protected int documentD;
    /**
     * Number of word types
     */
    protected int wordW;
    /**
     * Number of word tokens
     */
    protected int wordN;
    /**
     * Hyperparameter for topic-by-document prior.
     */
    protected double alpha;
    /**
     * Hyperparameter for word/stem-by-topic prior
     */
    protected double beta;
    /**
     * Hashtable from word to index.
     */
    protected HashMap<String, Integer> wordIdx;
    /**
     * Array of document indexes. Of length {@link #wordN}.
     */
    protected int[] documentVector;
    /**
     * Array of topic indexes. Of length {@link #wordN}.
     */
    protected int[] topicVector;
    /**
     * Array of word indexes. Of length {@link #wordN}.
     */
    protected int[] wordVector;
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
     * Hyperparameter for state emissions
     */
    protected double gamma;
    /**
     * Number of states including topic states and sentence boundary state
     */
    protected int stateS;
    /**
     * Token indexes for sentence boundaries; sentences begin here
     */
    protected int[] sentenceVector;
    /**
     * Array of states over tokens
     */
    protected int[] stateVector;
    /**
     * Number of topic states. This is a less than {@link #stateS} and entails
     * that the topic states are a subset of the full states. A count of one
     * must be added to whatever number is passed from {@link HybridHMMLDAOptions}
     * since state 0 is always the sentence boundary.
     */
    protected int topicSubStates;
    /**
     * Hyperparameter for state by affix by stem DP
     */
    protected double muStem;
    /**
     * Hyperparameter for state by affix by stem HDP base distribution
     */
    protected double muStemBase;
    /**
     * Hyperparameter for state by affix DP
     */
    protected double muAffix;
    /**
     * Hyperparameter for state by affix HDP base distribution
     */
    protected double muAffixBase;
    /**
     * Hyperparameter for topic by affix by stem DP
     */
    protected double betaStem;
    /**
     * Hyperparameter for topic by affix by stem HDP base distribution
     */
    protected double betaStemBase;
    /**
     * Hyperparameter for state transition prior. This overrides
     * {@link HMMLDA#gamma} to reflect notation in the paper. See
     * {@link HMMLDA#thirdOrderTransitions} and {@link HMMLDA#secondOrderTransitions}.
     */
    protected double psi;
    /**
     * Hyperparameter for "switch" prior. See {@link  #switchVector}
     * and {@link #fourthOrderSwitches}.
     */
    protected double xi;
    /**
     * Prior probability of a morpheme boundary for affixes. Equivalent to
     * <pre>P(#)</pre> in the model.
     */
    protected double affixBoundaryProb;
    /**
     * Prior probability of a morpheme boundary for stems. Equivalent to
     * <pre>P(#)</pre> in the model.
     */
    protected double stemBoundaryProb;
    /**
     * Prior probability of a morpheme non-boundary for affixes. Equivalent to
     * <pre>1-P(#)</pre> in the model.
     */
    protected double notAffixBoundaryProb;
    /**
     * Prior probability of a morpheme non-boundary for stems. Equivalent to
     * <pre>1-P(#)</pre> in the model.
     */
    protected double notStemBoundaryProb;
//    /**
//     * Number of types to print per class (topic and/or state)
//     */
//    protected int outputPerClass;
    /**
     * Array of where each token was segmented. For use in reconstruction
     * in the serializableModel.
     */
    protected int[] splitVector;
    /**
     * Array of switch indexes.
     */
    protected int[] switchVector;
    /**
     * Path of training data.
     */
    protected String trainDataDir;
    /**
     * Type of model that is being run.
     */
    protected String modelName;

    /**
     * Constructor to use when model is being saved.
     * 
     * @param hhl Model to be saved
     */
    public SerializableModel(HDPHMMLDA m) {
        affixBoundaryProb = m.affixBoundaryProb;
        alpha = m.alpha;
        beta = m.beta;
        betaStem = m.betaStem;
        betaStemBase = m.betaStemBase;
        dataFormat = m.dataFormat;
        documentD = m.documentD;
        documentVector = m.documentVector;
        gamma = m.gamma;
        initialTemperature = m.initialTemperature;
        iterations = m.iterations;
        modelName = m.modelName;
        muAffix = m.muAffix;
        muAffixBase = m.muAffixBase;
        muStem = m.muStem;
        muStemBase = m.muStemBase;
        notAffixBoundaryProb = m.notAffixBoundaryProb;
        notStemBoundaryProb = m.notStemBoundaryProb;
//        outputPerClass = m.outputPerClass;
        psi = m.psi;
        randomSeed = m.randomSeed;
        trainDataDir = m.trainDataDir;
        sentenceVector = m.sentenceVector;
        splitVector = m.splitVector;
        stateVector = m.stateVector;
        stateS = m.stateS;
        stemBoundaryProb = m.stemBoundaryProb;
        switchVector = m.switchVector;
        targetTemperature = m.targetTemperature;
        temperatureDecrement = m.temperatureDecrement;
        topicK = m.topicK;
        topicSubStates = m.topicSubStates;
        topicVector = m.topicVector;
        wordIdx = m.trainWordIdx;
        wordN = m.wordN;
        wordVector = m.wordVector;
        wordW = m.wordW;
        xi = m.xi;
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
    public HDPHMMLDA loadModel(CommandLineOptions options, String filename)
          throws IOException,
          FileNotFoundException {
        ObjectInputStream modelIn =
              new ObjectInputStream(new GZIPInputStream(new FileInputStream(
              filename)));
//        ObjectInputStream modelIn =
//                new ObjectInputStream(new FileInputStream(filename));
        try {
            loadBuffer = (SerializableModel) modelIn.readObject();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        copy(loadBuffer);
        loadBuffer = null;
        modelIn.close();

        HDPHMMLDA hhl = null;
        if (modelName.equals("m1")) {
            hhl = new HDPHMMLDAm1(options);
        } else if (modelName.equals("m2")) {
            hhl = new HDPHMMLDAm2(options);
        } else if (modelName.equals("m3")) {
            hhl = new HDPHMM(options);
        } else if (modelName.equals("m4")) {
            hhl = new HDPLDA(options);
        }

        hhl.affixBoundaryProb = affixBoundaryProb;
        hhl.alpha = alpha;
        hhl.beta = beta;
        hhl.betaStem = betaStem;
        hhl.betaStemBase = betaStemBase;
        hhl.dataFormat = dataFormat;
        hhl.documentD = documentD;
        hhl.documentVector = documentVector;
        hhl.gamma = gamma;
        hhl.initialTemperature = initialTemperature;
        hhl.iterations = iterations;
        hhl.modelName = modelName;
        hhl.muAffix = muAffix;
        hhl.muAffixBase = muAffixBase;
        hhl.muStem = muStem;
        hhl.muStemBase = muStemBase;
        hhl.notAffixBoundaryProb = notAffixBoundaryProb;
        hhl.notStemBoundaryProb = notStemBoundaryProb;
//        hhl.outputPerClass = outputPerClass;
        hhl.psi = psi;
        hhl.randomSeed = randomSeed;
        hhl.trainDataDir = trainDataDir;
        hhl.sentenceVector = sentenceVector;
        hhl.splitVector = splitVector;
        hhl.stateVector = stateVector;
        hhl.stateS = stateS;
        hhl.stemBoundaryProb = stemBoundaryProb;
        hhl.switchVector = switchVector;
        hhl.targetTemperature = targetTemperature;
        hhl.temperatureDecrement = temperatureDecrement;
        hhl.topicK = topicK;
        hhl.topicSubStates = topicSubStates;
        hhl.topicVector = topicVector;
        hhl.trainWordIdx = wordIdx;
        hhl.wordN = wordN;
        hhl.wordVector = wordVector;
        hhl.wordW = wordW;
        hhl.xi = xi;

        return hhl;
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
        affixBoundaryProb = sm.affixBoundaryProb;
        alpha = sm.alpha;
        beta = sm.beta;
        betaStem = sm.betaStem;
        betaStemBase = sm.betaStemBase;
        dataFormat = sm.dataFormat;
        documentD = sm.documentD;
        documentVector = sm.documentVector;
        gamma = sm.gamma;
        initialTemperature = sm.initialTemperature;
        iterations = sm.iterations;
        modelName = sm.modelName;
        muAffix = sm.muAffix;
        muAffixBase = sm.muAffixBase;
        muStem = sm.muStem;
        muStemBase = sm.muStemBase;
        notAffixBoundaryProb = sm.notAffixBoundaryProb;
        notStemBoundaryProb = sm.notStemBoundaryProb;
//        outputPerClass = sm.outputPerClass;
        psi = sm.psi;
        randomSeed = sm.randomSeed;
        trainDataDir = sm.trainDataDir;
        sentenceVector = sm.sentenceVector;
        splitVector = sm.splitVector;
        stateVector = sm.stateVector;
        stateS = sm.stateS;
        stemBoundaryProb = sm.stemBoundaryProb;
        switchVector = sm.switchVector;
        targetTemperature = sm.targetTemperature;
        temperatureDecrement = sm.temperatureDecrement;
        topicK = sm.topicK;
        topicSubStates = sm.topicSubStates;
        topicVector = sm.topicVector;
        wordIdx = sm.wordIdx;
        wordN = sm.wordN;
        wordVector = sm.wordVector;
        wordW = sm.wordW;
        xi = sm.xi;
    }
}
