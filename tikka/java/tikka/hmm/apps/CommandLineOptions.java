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
package tikka.hmm.apps;

import tikka.opennlp.io.DataFormatEnum;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;

import org.apache.commons.cli.*;
import tikka.utils.postags.TagSetEnum;

/**
 * Handles options from the command line. Also sets the default parameter
 * values.
 *
 * @author tsmoon
 */
public class CommandLineOptions {

    /**
     * Number of topics
     */
    protected int topics = 50;
    /**
     * Number of types (either word or morpheme) to print per state or topic
     */
    protected int outputPerClass = 10;
    /**
     * Hyperparameter of topic prior
     */
    protected double alpha = 1;
    /**
     * Hyperparameter for word/topic prior
     */
    protected double beta = 0.1;
    /**
     * Hyperparameter for state transition priors
     */
    protected double gamma = 0.1;
    /**
     * Hyperparameter for word emission priors
     */
    protected double delta = 0.0001;
    /**
     * Path for training data. Should be a full directory
     */
    protected String trainDataDir = null;
    /**
     * Path for test data. Should be a full directory
     */
    private String testDataDir = null;
    /**
     * full path of model to be loaded
     */
    protected String modelInputPath = null;
    /**
     * full path to save model to
     */
    protected String modelOutputPath = null;
    /**
     * Root of path to output annotated texts to
     */
    protected String annotatedTrainTextOutDir = null;
    /**
     * Root of path to output annotated test set texts to
     */
    private String annotatedTestTextOutDir = null;
    /**
     * Number of training iterations
     */
    protected int numIterations = 100;
    /**
     * Number of iterations for test set burnin
     */
    protected int testSetBurninIterations = 10;
    /**
     * Number to seed random number generator. This is set to -1 so it may be
     * checked during the initialization stages. If the value is -1 from the
     * calling function, it means the randomSeed must be set to 0. Otherwise,
     * use whatever value has been passed to it from the command line.
     */
    protected int randomSeed = -1;
    /**
     * Specifier of training data format.
     */
    protected DataFormatEnum.DataFormat dataFormat =
          DataFormatEnum.DataFormat.CONLL2K;
    /**
     * Specifier of training data format.
     */
    protected TagSetEnum.TagSet tagSet = TagSetEnum.TagSet.PTB;
    /**
     * Option on how much the tagset should be reduced. Default is 0 (none).
     */
    protected TagSetEnum.ReductionLevel reductionLevel = TagSetEnum.ReductionLevel.FULL;
    /**
     * Name of file to generate evaluation scores to
     */
    protected String evaluationOutputFilename = null;
    /**
     * Name of file to generate tabulated output to
     */
    protected String tabularOutputFilename = null;
    /**
     * Name of file to dump test data sample scores (perplexity) to
     */
    protected String testDataSampleScoreOutputFilename = null;
    /**
     * Name of file to dump training data sample scores (bayes factor) to
     */
    protected String trainDataSampleScoreOutputFilename = null;
    /**
     * Output buffer to write evaluation scores to
     */
    protected BufferedWriter evaluationOutput;
    /**
     * Output buffer to write normalized, tabulated data to.
     */
    protected BufferedWriter tabulatedOutput;
    /**
     * Output buffer to dump test data sample scores (perplexity) to
     */
    protected BufferedWriter testDataSampleScoreOutput;
    /**
     * Output buffer to dump training data sample scores (perplexity) to
     */
    protected BufferedWriter trainDataSampleScoreOutput;
    /**
     * Temperature at which to start annealing process
     */
    protected double initialTemperature = 1;
    /**
     * Decrement at which to reduce the temperature in annealing process
     */
    protected double temperatureDecrement = 0.1;
    /**
     * Stop changing temperature after the following temp has been reached.
     */
    protected double targetTemperature = 1;
    /**
     * Model to use for training. Use unhelpful, non-mnemonic names
     */
    protected String experimentModel = "m1";
    /**
     * Number of states in HMM.
     */
    protected int states = 15;
    /**
     * Number of samples to take
     */
    protected int samples = 100;
    /**
     * Number of iterations between samples
     */
    protected int lag = 10;

    /**
     *
     * @param cline
     * @throws IOException
     */
    public CommandLineOptions(CommandLine cline) throws IOException {

        String opt = null;

        for (Option option : cline.getOptions()) {
            String value = option.getValue();
            switch (option.getOpt().charAt(0)) {
                case 'a':
                    alpha = Double.parseDouble(value);
                    break;
                case 'b':
                    beta = Double.parseDouble(value);
                    break;
                case 'c':
                    if (value.equals("conll2k")) {
                        dataFormat = DataFormatEnum.DataFormat.CONLL2K;
                    } else if (value.equals("hashslash")) {
                        dataFormat = DataFormatEnum.DataFormat.HASHSLASH;
                    } else if (value.equals("pipesep")) {
                        dataFormat = DataFormatEnum.DataFormat.PIPESEP;
                    } else if (value.equals("raw")) {
                        dataFormat = DataFormatEnum.DataFormat.RAW;
                    } else {
                        System.err.println(
                              "\"" + value + "\" is an unknown data format option.");
                        System.exit(1);
                    }
                    break;
                case 'd':
                    if (value.endsWith("" + File.separator)) {
                        trainDataDir = value.substring(0, value.length() - 1);
                    } else {
                        trainDataDir = value;
                    }
                    break;
                case 'e':
                    experimentModel = value;
                    break;
                case 'f':
                    if (value.endsWith("" + File.separator)) {
                        testDataDir = value.substring(0, value.length() - 1);
                    } else {
                        testDataDir = value;
                    }
                    break;
                case 'g':
                    gamma = Double.parseDouble(value);
                    break;
                case 'i':
                    opt = option.getOpt();
                    if (opt.equals("itr")) {
                        numIterations = Integer.parseInt(value);
                    } else if (opt.equals("ite")) {
                        testSetBurninIterations = Integer.parseInt(value);
                    }
                    break;
                case 'j':
                    if (value.endsWith("" + File.separator)) {
                        annotatedTestTextOutDir = value.substring(0, value.length() - 1);
                    } else {
                        annotatedTestTextOutDir = value;
                    }
                    break;
                case 'k':
                    opt = option.getOpt();
                    if (opt.equals("ks")) {
                        samples = Integer.parseInt(value);
                    } else if (opt.equals("kl")) {
                        lag = Integer.parseInt(value);
                    }
                    break;
                case 'l':
                    modelInputPath = value;
                    break;
                case 'm':
                    modelOutputPath = value;
                    break;
                case 'n':
                    if (value.endsWith("" + File.separator)) {
                        annotatedTrainTextOutDir =
                              value.substring(0, value.length() - 1);
                    } else {
                        annotatedTrainTextOutDir = value;
                    }
                    break;
                case 'o':
                    opt = option.getOpt();
                    if (opt.equals("ot")) {
                        tabularOutputFilename = value;
                        tabulatedOutput = new BufferedWriter(new OutputStreamWriter(
                              new FileOutputStream(tabularOutputFilename)));
                    } else if (opt.equals("oste")) {
                        testDataSampleScoreOutputFilename = value;
                        testDataSampleScoreOutput = new BufferedWriter(new OutputStreamWriter(
                              new FileOutputStream(testDataSampleScoreOutputFilename)));
                    } else if (opt.equals("ostr")) {
                        trainDataSampleScoreOutputFilename = value;
                        trainDataSampleScoreOutput = new BufferedWriter(new OutputStreamWriter(
                              new FileOutputStream(trainDataSampleScoreOutputFilename)));
                    } else if (opt.equals("oe")) {
                        evaluationOutputFilename = value;
                        evaluationOutput = new BufferedWriter(new OutputStreamWriter(
                              new FileOutputStream(evaluationOutputFilename)));
                    }
                    break;
                case 'p':
                    opt = option.getOpt();
                    if (opt.equals("pi")) {
                        initialTemperature = Double.parseDouble(value);
                    } else if (opt.equals("pd")) {
                        temperatureDecrement = Double.parseDouble(value);
                    } else if (opt.equals("pt")) {
                        targetTemperature = Double.parseDouble(value);
                    }
                    break;
                case 'q':
                    delta = Double.parseDouble(value);
                    break;
                case 'r':
                    randomSeed = Integer.valueOf(value);
                    break;
                case 's':
                    states = Integer.parseInt(value);
                    break;
                case 't':
                    topics = Integer.parseInt(value);
                    break;
                case 'u':
                    opt = option.getOpt();
                    if (opt.equals("ut")) {
                        if (value.equals("b")) {
                            tagSet = TagSetEnum.TagSet.BROWN;
                        } else if (value.equals("p")) {
                            tagSet = TagSetEnum.TagSet.PTB;
                        } else if (value.equals("t")) {
                            tagSet = TagSetEnum.TagSet.TIGER;
                        }
                    } else if (opt.equals("ur")) {
                        int val = Integer.parseInt(value);
                        if (val == 0) {
                            reductionLevel = TagSetEnum.ReductionLevel.FULL;
                        } else {
                            reductionLevel = TagSetEnum.ReductionLevel.REDUCED;
                        }
                    }
                    break;

                case 'w':
                    outputPerClass = Integer.parseInt(value);
                    break;
            }
        }
    }

    public double getAlpha() {
        return alpha;
    }

    public double getBeta() {
        return beta;
    }

    public double getGamma() {
        return gamma;
    }

    public double getDelta() {
        return delta;
    }

    public String getTrainDataDir() {
        return trainDataDir;
    }

    public String getTestDataDir() {
        return testDataDir;
    }

    public DataFormatEnum.DataFormat getDataFormat() {
        return dataFormat;
    }

    public int getNumIterations() {
        return numIterations;
    }

    public String getTabularOutputFilename() {
        return tabularOutputFilename;
    }

    public BufferedWriter getTabulatedOutput() {
        return tabulatedOutput;
    }

    public int getOutputPerClass() {
        return outputPerClass;
    }

    public int getRandomSeed() {
        return randomSeed;
    }

    public double getInitialTemperature() {
        return initialTemperature;
    }

    public double getTargetTemperature() {
        return targetTemperature;
    }

    public double getTemperatureDecrement() {
        return temperatureDecrement;
    }

    public int getTopics() {
        return topics;
    }

    public String getExperimentModel() {
        return experimentModel;
    }

    public int getStates() {
        return states;
    }

    public String getModelInputPath() {
        return modelInputPath;
    }

    public String getModelOutputPath() {
        return modelOutputPath;
    }

    public String getAnnotatedTrainTextOutDir() {
        return annotatedTrainTextOutDir;
    }

    public String getAnnotatedTestTextOutDir() {
        return annotatedTestTextOutDir;
    }

    /**
     * @return the number of samples take
     */
    public int getSamples() {
        return samples;
    }

    /**
     * @return the number of iterations per sample
     */
    public int getLag() {
        return lag;
    }

    public BufferedWriter getTestDataSampleScoreOutput() {
        return testDataSampleScoreOutput;
    }

    public String getTestDataSampleScoreOutputFilename() {
        return testDataSampleScoreOutputFilename;
    }

    public BufferedWriter getTrainDataSampleScoreOutput() {
        return trainDataSampleScoreOutput;
    }

    public String getTrainDataSampleScoreOutputFilename() {
        return trainDataSampleScoreOutputFilename;
    }

    public int getTestSetBurninIterations() {
        return testSetBurninIterations;
    }

    /**
     * @return the evaluationOutputFilename
     */
    public String getEvaluationOutputFilename() {
        return evaluationOutputFilename;
    }

    /**
     * @return the evaluationOutput
     */
    public BufferedWriter getEvaluationOutput() {
        return evaluationOutput;
    }

    public TagSetEnum.TagSet getTagSet() {
        return tagSet;
    }

    public TagSetEnum.ReductionLevel getReductionLevel() {
        return reductionLevel;
    }
}
