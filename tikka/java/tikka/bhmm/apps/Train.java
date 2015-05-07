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
package tikka.bhmm.apps;

import tikka.bhmm.model.base.*;
import tikka.bhmm.models.*;

import java.io.*;

import org.apache.commons.cli.*;

/**
 * Train and test a tagger.
 *
 * @author  Jason Baldridge
 * @version $Revision: 1.53 $, $Date: 2006/10/12 21:20:44 $
 */
public class Train extends MainBase {

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
                System.err.println("Beginning training; initialising from model:" + modelInputPath);
                SerializableModel serializableModel = new SerializableModel();
                bhmm = serializableModel.loadModel(modelOptions, modelInputPath);
                bhmm.initializeFromLoadedModel2(modelOptions);
                bhmm.setNumIterations(modelOptions.getNumIterations()); // overwrite the iter no.
                bhmm.train(false);
            } else {
                bhmm = ModelGenerator.generator(modelOptions);
                System.err.println("Beginning training; random initialisation");
                bhmm.initializeFromTrainingData();
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

            //System.err.println("Maximum posterior decoding");
            //bhmm.decode();

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

        } catch (ParseException exp) {
            System.out.println("Unexpected exception parsing command line options:" + exp.getMessage());
        } catch (IOException exp) {
            System.out.println("IOException:" + exp.getMessage());
            System.exit(0);
        }
    }
}
