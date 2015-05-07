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

import java.io.*;

import org.apache.commons.cli.*;

import tikka.hmm.model.base.HMM;
import tikka.hmm.model.base.SerializableModel;
import tikka.hmm.model.em.EMHMM;
import tikka.hmm.model.hmmlda.HMMLDA;
import tikka.hmm.model.hmmlda.SerializableModelHMMLDA;
import tikka.hmm.model.mcmc.GibbsHMM;

/**
 * Train and test a tagger.
 *
 * @author  Jason Baldridge
 * @version $Revision: 1.53 $, $Date: 2006/10/12 21:20:44 $
 */
public class Train extends MainBase {

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

            HMM hmm = null;
            HMMLDA hmmlda = null;

            String experimentModel = modelOptions.getExperimentModel();

            if (experimentModel.equals("m1")) {
                System.err.println("Using EM HMM!");
                hmm = new EMHMM(modelOptions);
            } else if (experimentModel.equals("m2")) {
                System.err.println("Using MCMC HMM!");
                hmm = new GibbsHMM(modelOptions);
            } else if (experimentModel.equals("m3")) {
                System.err.println("Using HMM LDA!");
                /**
                 * Trick to sidestep a design flaw, namely the derived class
                 * has new fields with no access through a shared class interface.
                 */
                hmm = hmmlda = new HMMLDA(modelOptions);
            }

            System.err.println("Randomly initializing values!");
            hmm.initializeFromTrainingData();
            System.err.println("Beginning training!");
            hmm.train();

            /**
             * Save model if specified
             */
            String modelOutputPath = modelOptions.getModelOutputPath();
            if (modelOutputPath != null) {
                System.err.println("Saving model to :"
                      + modelOutputPath);
                SerializableModel serializableModel = null;

                if (experimentModel.equals("m1")) {
                    serializableModel = new SerializableModel(hmm);
                } else if (experimentModel.equals("m2")) {
                    serializableModel = new SerializableModel(hmm);
                } else if (experimentModel.equals("m3")) {
                    serializableModel = new SerializableModelHMMLDA(hmmlda);
                }

                serializableModel.saveModel(modelOutputPath);
            }

            System.err.println("Maximum posterior decoding");
            hmm.decode();

            /**
             * Set the string of parameters.
             */
            hmm.setModelParameterStringBuilder();

            String evaluationOutputFilename = modelOptions.getEvaluationOutputFilename();
            if (evaluationOutputFilename != null) {
                System.err.println("Performing evaluation");
                hmm.evaluate();
                System.err.println("Also printing evaluation results to " + evaluationOutputFilename);
                hmm.printEvaluationScore(modelOptions.getEvaluationOutput());
                modelOptions.getEvaluationOutput().close();
            }

            /**
             * Tag and segment training files from last iteration if specified
             */
            String annotatedTextDir = modelOptions.getAnnotatedTrainTextOutDir();
            if (annotatedTextDir != null) {
                System.err.println("Printing annotated text to :"
                      + annotatedTextDir);
                hmm.printAnnotatedTrainText(annotatedTextDir);
            }

            /**
             * Save tabulated probabilities
             */
            if (modelOptions.getTabularOutputFilename() != null) {
                System.err.println("Normalizing parameters!");
                hmm.normalize();
                System.err.println("Printing tabulated output to :"
                      + modelOptions.getTabularOutputFilename());
                hmm.printTabulatedProbabilities(modelOptions.getTabulatedOutput());
            }

        } catch (ParseException exp) {
            System.out.println("Unexpected exception parsing command line options:" + exp.getMessage());
        } catch (IOException exp) {
            System.out.println("IOException:" + exp.getMessage());
            System.exit(0);
        }
    }
}
