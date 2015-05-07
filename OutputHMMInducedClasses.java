/*
Author: Jey Han Lau
Date: May 13

Reads a BHMM model file and outputs the induced word classes of sentences of the training data.
*/


import tikka.bhmm.models.*;
import tikka.bhmm.model.base.*;
import tikka.bhmm.apps.*;
import org.apache.commons.cli.*;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class OutputHMMInducedClasses {
    private static boolean debug = false;

    public static void main(String[] args) {
        CommandLineParser optparse = new PosixParser();
        CommandLine cline = null;
        Options options = setOptions();
        String inputModel = "";
        int numSent = 1000;
        int maxSentLen = 10;
        boolean conllFormat = false;

        try {
            cline = optparse.parse(options, args);
            if (cline.hasOption('h')) {
                HelpFormatter formatter = new HelpFormatter();
                formatter.printHelp("java OutputHMMInducedClasses", options);
                System.exit(0);
            }
            if (cline.hasOption("l")) {
                inputModel = cline.getOptionValue("l");
            }
            if (cline.hasOption("xns")) {
                numSent = Integer.parseInt(cline.getOptionValue("xns"));
            }
            if (cline.hasOption("xms")) {
                maxSentLen = Integer.parseInt(cline.getOptionValue("xms"));
            }
            if (cline.hasOption("xconll")) {
                conllFormat = true;
            }
        } catch (ParseException parseException) {
            System.err.println("Error parsing command line arguments");
            System.exit(0);
        }

        // load the model
        HMMBase bhmm = null;
        try {
            CommandLineOptions modelOptions = new CommandLineOptions(cline);
            SerializableModel serializableModel = new SerializableModel();
            bhmm = serializableModel.loadModel(modelOptions,
                modelOptions.getModelInputPath());
            bhmm.resetTrainDataDir(); // reset the previous train directory (avoid IOException)
            bhmm.initializeFromLoadedModel2(modelOptions);
        } catch (IOException e) {
            System.err.println("Error loading input model; path = " + inputModel);
            System.exit(0);
        }

        // get the sentence states
        int[] stateVector = bhmm.getStateVector();
        int[] sentenceVector = bhmm.getSentenceVector();
        int prevSent = 0;
        ArrayList<Integer> currSent = new ArrayList<Integer>();
        ArrayList<String[]> sents = new ArrayList<String[]>();
        for (int i=0; i<stateVector.length; i++) {
            int state = stateVector[i];
            int stateSent = sentenceVector[i];

            if ((stateSent != prevSent) || (i==(stateVector.length-1))) {
                if (i==(stateVector.length-1)) {
                    currSent.add(state);
                }

                if ((maxSentLen == 0) || (currSent.size() <= maxSentLen)) {
                    sents.add(convToString(currSent)); 
                }
                currSent.clear();
                if ((sents.size() % 10000) == 0) {
                    //System.err.println("Number of sentences processed = " + sents.size());
                }
            }

            currSent.add(state);
            prevSent = stateSent;
        }

        if (debug) {
            System.out.println("Sentence Vector = " + Arrays.toString(sentenceVector));
            System.out.println("State Vector    = " + Arrays.toString(stateVector));
            System.out.println("Pre-shuffle sentences = ");
            for (String[] sent : sents) {
                System.out.println(Arrays.toString(sent));
            }
        }

        // shuffle the sents and select top-N
        if (numSent != 0) {
            System.err.println("Sorting...");
            System.err.println("Number of items = " + sents.size());
            Collections.shuffle(sents);
            if (debug) {
                System.out.println("Post-shuffle sentences = ");
                for (String[] sent : sents) {
                    System.out.println(Arrays.toString(sent));
                }
            }
            System.err.println("Done sorting...");
        } else {
            numSent = sents.size();
        }
        if (sents.size() < numSent) {
            numSent = sents.size();
        }
        for (int i=0; i<numSent; i++) {
            String[] outputSent = sents.get(i);
            for (int j=0; j<outputSent.length; j++) {
                System.out.print(outputSent[j]);
                if (j == (outputSent.length-1)) {
                    if (!conllFormat) {
                        System.out.print("\n");
                    } else {
                        System.out.print("\n\n");
                    }
                } else {
                    if (!conllFormat) {
                        System.out.print(" ");
                    } else {
                        System.out.print("\n");
                    }
                }
            }
        }
    }

    private static String[] convToString(ArrayList<Integer> sent) {
        String[] strList = new String[sent.size()];
        for (int i=0; i<sent.size(); i++) {
            strList[i] = "S" + sent.get(i);
        } 
        return strList;
    }


    public static Options setOptions() {

        Options options = new Options();
        options.addOption("h", "help", false, "print help");
        options.addOption("l", "input-model", true, "trained BHMM model");
        options.addOption("xns", "num-sent", true, "number of sentences (0 = no restriction; " +
            "default = 1000)");
        options.addOption("xms", "max-sent-length", true, "maximum sentence length " +
            "(0 = no restriction; default = 10)");
        options.addOption("xconll", "conll-format", false, "use CONLL format (default = " +
            "1 line per sentence)");

        return options;
    }

}
