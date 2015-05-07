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
package tikka.hhl.apps;

import tikka.hhl.models.base.HDPHMMLDA;
import tikka.hhl.models.m1.HDPHMMLDAm1;
import tikka.hhl.models.base.SerializableModel;
import tikka.hhl.models.m2.HDPHMMLDAm2;
import tikka.utils.math.BayesFactorEval;
import tikka.utils.math.PerplexityEval;
import tikka.utils.math.SampleEval;

import java.io.IOException;

import org.apache.commons.cli.*;
import tikka.hhl.models.m3.HDPHMM;
import tikka.hhl.models.m4.HDPLDA;

/**
 * This is a module for tagging test data sets. Parameters may be trained from
 * a training set or may be loaded from a previously trained model.
 * 
 * @author tsmoon
 */
public class Tagger extends MainBase {

    /**
     * Default main
     * 
     * @param args command line arguments
     */
    public static void main(String[] args) {
        CommandLineParser optparse = new PosixParser();
        Options options = new Options();
        setOptions(options);

        try {
            CommandLine cline = optparse.parse(options, args);

            if (cline.hasOption('h')) {
                HelpFormatter formatter = new HelpFormatter();
                formatter.printHelp("java Tag Model", options);
                System.exit(0);
            }

            CommandLineOptions modelOptions = new CommandLineOptions(cline);

            HDPHMMLDA hhl = null;
            String experimentModel = modelOptions.getExperimentModel();

            String modelInputPath = modelOptions.getModelInputPath();

            boolean normalized = false;

            /**
             * Choose whether to load from previously saved model or train on new
             */
            if (modelInputPath != null) {
                System.err.println("Loading from model:" + modelInputPath);
                SerializableModel serializableModel = new SerializableModel();
                hhl = serializableModel.loadModel(modelOptions, modelInputPath);
                hhl.initializeFromLoadedModel(modelOptions);
            } else {
                if (experimentModel.equals("m1")) {
                    System.err.println("Using model 1!");
                    hhl = new HDPHMMLDAm1(modelOptions);
                } else if (experimentModel.equals("m2")) {
                    System.err.println("Using model 2!");
                    hhl = new HDPHMMLDAm2(modelOptions);
                } else if (experimentModel.equals("m3")) {
                    System.err.println("Using HDPHMM!");
                    hhl = new HDPHMM(modelOptions);
                } else if (experimentModel.equals("m4")) {
                    System.err.println("Using HDPLDA!");
                    hhl = new HDPLDA(modelOptions);
                } else {
                    hhl = new HDPHMMLDAm1(modelOptions);
                }
                System.err.println("Randomly initializing values!");
                hhl.initializeFromTrainingData();
                System.err.println("Beginning training!");
                hhl.train();
            }

            /**
             * Save model if specified
             */
            String modelOutputPath = modelOptions.getModelOutputPath();
            if (modelOutputPath != null) {
                System.err.println("Saving model to :"
                      + modelOutputPath);
                SerializableModel serializableModel = new SerializableModel(hhl);
                serializableModel.saveModel(modelOutputPath);
            }

            /**
             * Output normalized parameters in tabular form to output if
             * specified
             */
            if (modelOptions.getTabularOutputFilename() != null) {
                System.err.println("Normalizing parameters!");
                hhl.normalize();
                System.err.println("Printing tabulated output to :"
                      + modelOptions.getTabularOutputFilename());
                hhl.printTabulatedProbabilities(modelOptions.getTabulatedOutput());
            }

            /**
             * Set the string of parameters.
             */
            hhl.setModelParameterStringBuilder();

            /**
             * Output scores for the training samples
             */
            if (modelOptions.getTrainDataSampleScoreOutputFilename() != null) {
                System.err.println("Beginning sampling train data");
                hhl.sampleFromTrain();
                System.err.println("Saving sample training scores to :"
                      + modelOptions.getTrainDataSampleScoreOutputFilename());
                sampleEval = new BayesFactorEval();
                hhl.printSampleScoreData(modelOptions.getTrainDataSampleScoreOutput(),
                      sampleEval, "Scores from TRAINING data");
            }

            /**
             * Save training text which has been tagged and segmented to output if
             * specified
             */
            String annotatedTextDir = modelOptions.getAnnotatedTrainTextOutDir();
            if (annotatedTextDir != null) {
                System.err.println("Printing annotated training text to :"
                      + annotatedTextDir);
                hhl.printAnnotatedTrainText(annotatedTextDir);
            }

            /**
             * Tag and segment test files if specified
             */
            String testDataDir = modelOptions.getTestDataDir();
            if (testDataDir != null) {
//                if (!normalized) {
//                    System.err.println("Normalizing parameters!");
//                    hhl.normalize();
//                }
//                System.err.println("Tagging test text");
//                hhl.tagTestText();

                /**
                 * Output scores for the test samples
                 */
                if (modelOptions.getTestDataSampleScoreOutputFilename() != null) {
                    System.err.println("Beginning test sampling");
                    hhl.sampleFromTest();
                    System.err.println("Saving test sample scores to :"
                          + modelOptions.getTestDataSampleScoreOutputFilename());
                    sampleEval = new PerplexityEval();
                    hhl.printSampleScoreData(modelOptions.getTestDataSampleScoreOutput(),
                          sampleEval, "Scores from TEST data");
                }

                /**
                 * Save test text which has been tagged and segmented to output
                 * if specified
                 */
                String annotatedTestTextDir = modelOptions.getAnnotatedTestTextOutDir();
                if (annotatedTestTextDir != null) {
                    System.err.println("Printing annotated test text to :"
                          + annotatedTestTextDir);
                    hhl.printAnnotatedTestText(annotatedTestTextDir);
                }
            }

        } catch (ParseException exp) {
            System.out.println("Unexpected exception parsing command line options:"
                  + exp.getMessage());
        } catch (IOException exp) {
            System.out.println("IOException:" + exp.getMessage());
            System.exit(0);
        }
    }
}
