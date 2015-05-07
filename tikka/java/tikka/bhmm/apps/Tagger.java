///////////////////////////////////////////////////////////////////////////////
// To change this template, choose Tools | Templates
// and open the template in the editor.
///////////////////////////////////////////////////////////////////////////////
package tikka.bhmm.apps;

import tikka.bhmm.model.base.*;

import java.io.IOException;

import org.apache.commons.cli.*;

/**
 *
 * @author tsmoon
 */
public class Tagger extends MainBase {

    public static void main(String[] args) {

        CommandLineParser optparse = new PosixParser();

        Options options = setOptions();

        try {
            CommandLine cline = optparse.parse(options, args);

            if (cline.hasOption('h')) {
                HelpFormatter formatter = new HelpFormatter();
                formatter.printHelp("java Tag Model", options);
                System.exit(0);
            }

            CommandLineOptions modelOptions = new CommandLineOptions(cline);

            HMMBase bhmm = null;

            String modelInputPath = modelOptions.getModelInputPath();

            /**
             * Choose whether to load from previously saved model or train on new
             */
            if (modelInputPath != null) {
                System.err.println("Loading from model:" + modelInputPath);
                SerializableModel serializableModel = new SerializableModel();
                bhmm = serializableModel.loadModel(modelOptions, modelInputPath);
                bhmm.initializeFromLoadedModel(modelOptions);
            } else {
                bhmm = ModelGenerator.generator(modelOptions);
                System.err.println("Randomly initializing values!");
                bhmm.initializeFromTrainingData();
                System.err.println("Beginning training!");
                bhmm.train();
            }

            /**
             * Save model if specified
             */
            String modelOutputPath = modelOptions.getModelOutputPath();
            if (modelOutputPath != null) {
                System.err.println("Saving model to :"
                      + modelOutputPath);
                SerializableModel serializableModel = null;

                serializableModel = new SerializableModel(bhmm);
                serializableModel.saveModel(modelOutputPath);
            }

            System.err.println("Maximum posterior decoding");
            bhmm.decode();

            /**
             * Set the string of parameters.
             */
            bhmm.setModelParameterStringBuilder();

            String evaluationOutputFilename = modelOptions.getEvaluationOutputFilename();
            if (evaluationOutputFilename != null) {
                System.err.println("Performing evaluation");
                bhmm.evaluate();
                System.err.println("Also printing evaluation results to " + evaluationOutputFilename);
                bhmm.printEvaluationScore(modelOptions.getEvaluationOutput());
                modelOptions.getEvaluationOutput().close();
            }

            /**
             * Tag and segment training files from last iteration if specified
             */
            String annotatedTextDir = modelOptions.getAnnotatedTrainTextOutDir();
            if (annotatedTextDir != null) {
                System.err.println("Printing annotated text to :"
                      + annotatedTextDir);
                bhmm.printAnnotatedTrainText(annotatedTextDir);
            }

            /**
             * Save tabulated probabilities
             */
            if (modelOptions.getTabularOutputFilename() != null) {
                System.err.println("Normalizing parameters!");
                bhmm.normalize();
                System.err.println("Printing tabulated output to :"
                      + modelOptions.getTabularOutputFilename());
                bhmm.printTabulatedProbabilities(modelOptions.getTabulatedOutput());
                modelOptions.getTabulatedOutput().close();
            }

            /**
             * Tag and segment test files if specified
             */
            String testDataDir = modelOptions.getTestDataDir();
            if (testDataDir != null) {
                /**
                 * Output scores for the test samples
                 */
                if (modelOptions.getTestEvaluationOutputFilename() != null) {
                    System.err.println("Beginning test evaluation");
                    bhmm.evaluate();
                    System.err.println("Also printing test evaluation results to "
                          + modelOptions.getTestEvaluationOutputFilename());
                    bhmm.printEvaluationScore(modelOptions.getTestEvaluationOutput());
                }
            }

        } catch (ParseException exp) {
            System.out.println("Unexpected exception parsing command line options:" + exp.getMessage());
        } catch (IOException exp) {
            System.out.println("IOException:" + exp.getMessage());
            System.exit(0);
        }
    }
}
