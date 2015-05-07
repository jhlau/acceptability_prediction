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
package tikka.bhmm.models;

import java.io.BufferedWriter;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import tikka.bhmm.apps.CommandLineOptions;
import tikka.structures.*;
import tikka.utils.annealer.Annealer;
import tikka.bhmm.model.base.HMMBase;

/**
 * This is the lda-hmm implementation
 *
 * @author tsmoon
 */
public class LDAHMM extends HMM {

    /**
     * Array of probabilities by topic
     */
    protected double[] topicProbs;
    /**
     * Counts of topics by document
     */
    protected int[] documentByTopic;
    /**
     * Table of top {@link #outputPerClass} words per topic. Used in
     * normalization and printing.
     */
    protected StringDoublePair[][] topWordsPerTopic;

    public LDAHMM(CommandLineOptions options) {
        super(options);
        topicK = options.getTopics();
    }

    // get the top words per topic
    @Override
    public StringDoublePair[][] getTopWordsPerTopic() {
        return topWordsPerTopic;
    }


    /**
     * Initializes arrays for counting occurrences. These need to be initialized
     * regardless of whether the model being trained from raw data or whether
     * it is loaded from a saved model.
     */
    @Override
    protected void initializeCountArrays() {
        super.initializeCountArrays();

        topicCounts = new int[topicK];
        topicProbs = new double[topicK];
        for (int i = 0; i < topicK; ++i) {
            topicCounts[i] = 0;
            topicProbs[i] = 0.;
        }

        topicVector = new int[wordN];
        try {
            for (int i = 0;; ++i) {
                topicVector[i] = 0;
            }
        } catch (ArrayIndexOutOfBoundsException e) {
        }

        topicByWord = new int[topicK * wordW];
        try {
            for (int i = 0;; ++i) {
                topicByWord[i] = 0;
            }
        } catch (ArrayIndexOutOfBoundsException e) {
        }

        documentByTopic = new int[documentD * topicK];
        try {
            for (int i = 0;; ++i) {
                documentByTopic[i] = 0;
            }
        } catch (ArrayIndexOutOfBoundsException e) {
        }
    }

    /**
     * Normalize the sample counts.
     */
    @Override
    public void normalize() {
        normalizeTopics();
        normalizeStates();
    }

    /**
     * Normalize the sample counts for words given topic.
     */
    protected void normalizeTopics() {
        // initialise the topic probs 
        topicProbs = new double[topicK];

        topWordsPerTopic = new StringDoublePair[topicK][];
        for (int i = 0; i < topicK; ++i) {
            topWordsPerTopic[i] = new StringDoublePair[outputPerClass];
        }


        double sum = 0.;
        for (int i = 0; i < topicK; ++i) {
            sum += topicProbs[i] = topicCounts[i] + wbeta;
            ArrayList<DoubleStringPair> topWords =
                  new ArrayList<DoubleStringPair>();
            /**
             * Start at one to leave out EOSi
             */
            for (int j = 0; j < wordW; ++j) {
                topWords.add(new DoubleStringPair(
                      topicByWord[j * topicK + i] + beta, trainIdxToWord.get(
                      j)));
            }
            Collections.sort(topWords);
            for (int j = 0; j < outputPerClass; ++j) {
                if (j < topWords.size()) {
                    topWordsPerTopic[i][j] = new StringDoublePair(
                        topWords.get(j).stringValue, topWords.get(j).doubleValue
                        / topicProbs[i]);
                } else {
                    topWordsPerTopic[i][j] =
                        new StringDoublePair("Null", 0.0);
                }

            }
        }

        for (int i = 0; i < topicK; ++i) {
            topicProbs[i] /= sum;
        }
    }

    /**
     * Normalize the sample counts for words given state. Unlike the base class,
     * it marginalizes word probabilities over the topics for the topic state,
     * i.e. state 0.
     */
    @Override
    protected void normalizeStates() {
        topWordsPerState = new StringDoublePair[stateS][];
        for (int i = 0; i < stateS; ++i) {
            topWordsPerState[i] = new StringDoublePair[outputPerClass];
        }

        double sum = 0.;
        double[] marginalwordprobs = new double[wordW];
        try {
            for (int i = 0;; ++i) {
                marginalwordprobs[i] = 0;
            }
        } catch (ArrayIndexOutOfBoundsException e) {
        }

        for (int j = 0; j < wordW; ++j) {
            int wordoff = j * topicK;
            for (int i = 0; i < topicK; ++i) {
                marginalwordprobs[j] += topicProbs[i]
                      * (topicByWord[wordoff + i] + beta)
                      / (topicCounts[i] + wbeta);
            }
        }

        // collect the top words for the content state first
        sum += stateProbs[0] = stateCounts[0] + wdelta;
        ArrayList<DoubleStringPair> topWords =
                new ArrayList<DoubleStringPair>();
        for (int j = 0; j < wordW; ++j) {
            topWords.add(new DoubleStringPair(
                    marginalwordprobs[j], trainIdxToWord.get(
                    j)));
        }
        Collections.sort(topWords);
        for (int j = 0; j < outputPerClass; ++j) {
            if (j < topWords.size()) {
                topWordsPerState[0][j] =
                    new StringDoublePair(
                    topWords.get(j).stringValue,
                    topWords.get(j).doubleValue);
            } else {
                topWordsPerState[0][j] =
                    new StringDoublePair("Null", 0.0);
            }
        }

        for (int i = 1; i < stateS; ++i) {
            sum += stateProbs[i] = stateCounts[i] + wdelta;
            topWords.clear();
            /**
             * Start at one to leave out EOSi
             */
            for (int j = 0; j < wordW; ++j) {
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

        for (int i = 0; i < stateS; ++i) {
            stateProbs[i] /= sum;
        }
    }

    /**
     * Print the normalized sample counts to out. Print only the top {@link
     * #outputPerTopic} per given state and topic.
     *
     * @param out Output buffer to write to.
     * @throws IOException
     */
    @Override
    public void printTabulatedProbabilities(BufferedWriter out) throws
          IOException {
        printStates(out);
        printNewlines(out, 4);
        printTopics(out);
        out.close();
    }

    /**
     * Print the normalized sample counts for each topic to out. Print only the top {@link
     * #outputPerTopic} per given topic.
     * 
     * @param out
     * @throws IOException
     */
    protected void printTopics(BufferedWriter out) throws IOException {
        int startt = 0, M = 4, endt = M;
        out.write("***** Word Probabilities by Topic *****\n\n");
        while (startt < topicK) {
            for (int i = startt; i < endt; ++i) {
                String header = "Topic_" + i;
                header = String.format("%25s\t%6.5f\t", header, topicProbs[i]);
                out.write(header);
            }

            out.newLine();
            out.newLine();

            for (int i = 0; i < outputPerClass; ++i) {
                for (int c = startt; c < endt; ++c) {
                    String line = String.format("%25s\t%6.5f\t",
                          topWordsPerTopic[c][i].stringValue,
                          topWordsPerTopic[c][i].doubleValue);
                    out.write(line);
                }
                out.newLine();
            }
            out.newLine();
            out.newLine();

            startt = endt;
            endt = java.lang.Math.min(topicK, startt + M);
        }
    }

    @Override
    public void initializeParametersRandom() {
        int wordid, docid, topicid, stateid;
        int prev = (stateS-1), current = (stateS-1);
        double max = 0, totalprob = 0;
        double r = 0;
        int wordtopicoff, wordstateoff, docoff, stateoff, secondstateoff;

        /**
         * Initialize by assigning random topic indices to words
         */
        for (int i = 0; i < wordN; ++i) {
            wordid = wordVector[i];
            wordstateoff = stateS * wordid;

            docid = documentVector[i];
            wordtopicoff = topicK * wordid;
            docoff = topicK * docid;

            double tmpRoll = mtfRand.nextDouble();
            if (tmpRoll > 0.5) {
                stateid = 1;
            } else {
                stateid = 0;
            }

            totalprob = 0;
            try {
                for (int j = 0;; ++j) {
                    topicProbs[j] = documentByTopic[docoff + j] + alpha;
                    if (stateid == 0) {
                        topicProbs[j] *= (topicByWord[wordtopicoff + j] + beta)
                              / (topicCounts[j] + wbeta);
                    }
                    totalprob += topicProbs[j];
                }
            } catch (java.lang.ArrayIndexOutOfBoundsException e) {
            }

            max = topicProbs[0];
            topicid = 0;
            r = mtfRand.nextDouble() * totalprob;
            while (r > max) {
                topicid++;
                max += topicProbs[topicid];
            }
            // TMP: MAKING IT JUST RANDOM
            topicid = mtfRand.nextInt(topicK);
            // END TMP
            topicVector[i] = topicid;
            max = 0;

            totalprob = 0;
            if (stateid == 1) {
                totalprob = stateProbs[0] =
                      (topicByWord[wordtopicoff + topicid] + beta)
                      / (topicCounts[topicid] + wbeta)
                      * (secondOrderTransitions[(prev*S2) + (current*S1) + 0] + gamma);
                for (int j = 1; j<(stateS-1); j++) {
                    totalprob += stateProbs[j] =
                            (stateByWord[wordstateoff + j] + delta)
                            / (stateCounts[j] + wdelta)
                            * (secondOrderTransitions[(prev*S2) + (current*S1) + j] + gamma);
                }

                r = mtfRand.nextDouble() * totalprob;
                stateid = 0;
                max = stateProbs[stateid];
                while (r > max) {
                    stateid++;
                    max += stateProbs[stateid];
                }
                // TMP: MAKING IT JUST RANDOM (and not selecting content state again)
                stateid = mtfRand.nextInt(stateS-2)+1;
                // END TMP
            }
            stateVector[i] = stateid;

            if (stateid == 0) {
                topicByWord[wordtopicoff + topicid]++;
                documentByTopic[docoff + topicid]++;
                topicCounts[topicid]++;
            } else {
                stateByWord[wordstateoff + stateid]++;
            }

            stateCounts[stateid]++;
            firstOrderTransitions[(current*S1) + stateid]++;
            secondOrderTransitions[(prev*S2) + (current*S1) + stateid]++;
            first[i] = current;
            second[i] = prev;
            prev = current;
            current = stateid;
        }
        /*
        System.out.println("After initialisation:");
        System.out.println("WordVector = " + Arrays.toString(wordVector));
        System.out.println("SentenceVector = " + Arrays.toString(sentenceVector));
        System.out.println("documentVector = " + Arrays.toString(documentVector));
        System.out.println("StateVector = " + Arrays.toString(stateVector));
        System.out.println("StateCounts = " + Arrays.toString(stateCounts));
        System.out.println("StateByWord = " + Arrays.toString(stateByWord));
        System.out.println("TopicByWord = " + Arrays.toString(topicByWord));
        System.out.println("TopicCounts = " + Arrays.toString(topicCounts));
        System.out.println("DocumentByTopic = " + Arrays.toString(documentByTopic));
        System.out.println("first = " + Arrays.toString(first));
        System.out.println("second = " + Arrays.toString(second));
        System.out.println("firstOrderTransitions = " + Arrays.toString(firstOrderTransitions));
        System.out.println("secondOrderTransitions = "+Arrays.toString(secondOrderTransitions));
        System.out.println("----------------------------------------------------------");*/
    }

    @Override
    protected void trainInnerIter(int itermax, Annealer annealer) {
        int wordid, docid, topicid, stateid;
        int prev = (stateS-1), current = (stateS-1), next = (stateS-1), nnext = (stateS-1);
        double max = 0, totalprob = 0;
        double r = 0;
        int wordtopicoff, wordstateoff, docoff;
        int pprevsentid = -1; 
        int prevsentid = -1; 
        int nextsentid = -1; 
        int nnextsentid = -1; 

        long start = System.currentTimeMillis();
        for (int iter = 0; iter < itermax; ++iter) {
            System.err.println("\n\niteration " + iter + " (Elapsed time = +" +
                (System.currentTimeMillis()-start)/1000 + "s)");
            current = stateS-1;
            prev = stateS-1;
            System.err.print("Number of words processed: ");
            for (int i = 0; i < wordN; i++) {
                if (i % 100000 == 0) {
                    System.err.print(((float)i/1000000) + "M, ");
                }
                wordid = wordVector[i];
                stateid = stateVector[i];
                wordstateoff = wordid * stateS;

                docid = documentVector[i];
                topicid = topicVector[i];
                wordtopicoff = wordid * topicK;
                docoff = docid * topicK;

                if (stateid == 0) {
                    topicByWord[wordtopicoff + topicid]--;
                    documentByTopic[docoff + topicid]--;
                    topicCounts[topicid]--;
                } else {
                    stateByWord[wordstateoff + stateid]--;
                }
                stateCounts[stateid]--;
                firstOrderTransitions[first[i] * S1 + stateid]--;
                secondOrderTransitions[(second[i]*S2) + (first[i]*S1) + stateid]--;

                try {
                    for (int j = 0;; j++) {
                        topicProbs[j] = documentByTopic[docoff + j] + alpha;
                        if (stateid == 0) {
                            topicProbs[j] *= (topicByWord[wordtopicoff + j] + beta)
                                  / (topicCounts[j] + wbeta);
                        }
                    }
                } catch (java.lang.ArrayIndexOutOfBoundsException e) {
                }
                totalprob = annealer.annealProbs(topicProbs);
                r = mtfRand.nextDouble() * totalprob;
                max = topicProbs[0];

                topicid = 0;
                while (r > max) {
                    topicid++;
                    max += topicProbs[topicid];
                }
                topicVector[i] = topicid;

                try {
                    next = stateVector[i + 1];
                    nextsentid = sentenceVector[i + 1];
                } catch (ArrayIndexOutOfBoundsException e) {
                    next = stateS-1;
                    nextsentid = -1;
                }
                try {
                    nnext = stateVector[i + 2];
                    nnextsentid = sentenceVector[i + 2];
                } catch (ArrayIndexOutOfBoundsException e) {
                    nnext = stateS-1;
                    nnextsentid = -1;
                }
                if (sentenceVector[i] != prevsentid) {
                    current = stateS-1;
                    prev = stateS-1;
                }  else if (sentenceVector[i] != pprevsentid) {
                    prev = stateS-1;
                }
                if (sentenceVector[i] != nextsentid) {
                    next = stateS-1;
                    nnext = stateS-1;
                }  else if (sentenceVector[i] != nnextsentid) {
                    nnext = stateS-1;
                }


                for (int j = 0; j < (stateS-1); j++) {
                    // see words as 'abxcd', where x is the current word
                    double x = 0.0;
                    if (j==0) {
                        x = (topicByWord[wordtopicoff + topicid] + beta) /
                            (topicCounts[topicid] + wbeta);
                    } else {
                        x = (stateByWord[wordstateoff + j] + delta) / (stateCounts[j] + wdelta);
                    }
                    double abx =
                            (secondOrderTransitions[(prev*S2+current*stateS+j)]+gamma);
                    double bxc =
                            (secondOrderTransitions[(current*S2+j*stateS+next)]+gamma) /
                            (firstOrderTransitions[current*stateS + j] + sgamma);
                    double xcd =
                            (secondOrderTransitions[(j*S2+next*stateS+nnext)]+gamma) /
                            (firstOrderTransitions[j*stateS + next] + sgamma);
                    stateProbs[j] = x*abx*bxc*xcd;
                }
                totalprob = annealer.annealProbs(stateProbs);
                r = mtfRand.nextDouble() * totalprob;
                stateid = 0;
                max = stateProbs[stateid];
                while (r > max) {
                    stateid++;
                    max += stateProbs[stateid];
                }
                stateVector[i] = stateid;

                if (stateid == 0) {
                    topicByWord[wordtopicoff + topicid]++;
                    documentByTopic[docoff + topicid]++;
                    topicCounts[topicid]++;
                } else {
                    stateByWord[wordstateoff + stateid]++;
                }

                stateCounts[stateid]++;
                firstOrderTransitions[current*stateS + stateid]++;
                secondOrderTransitions[prev*S2 + current*stateS+ stateid]++;
                first[i] = current;
                second[i] = prev;
                prev = current;
                current = stateid;
                pprevsentid = prevsentid;
                prevsentid = sentenceVector[i];
            }
        }
        /*
        System.out.println("After sampling:");
        System.out.println("WordVector = " + Arrays.toString(wordVector));
        System.out.println("SentenceVector = " + Arrays.toString(sentenceVector));
        System.out.println("documentVector = " + Arrays.toString(documentVector));
        System.out.println("StateVector = " + Arrays.toString(stateVector));
        System.out.println("StateCounts = " + Arrays.toString(stateCounts));
        System.out.println("StateByWord = " + Arrays.toString(stateByWord));
        System.out.println("TopicByWord = " + Arrays.toString(topicByWord));
        System.out.println("TopicCounts = " + Arrays.toString(topicCounts));
        System.out.println("DocumentByTopic = " + Arrays.toString(documentByTopic));
        System.out.println("first = " + Arrays.toString(first));
        System.out.println("second = " + Arrays.toString(second));
        System.out.println("firstOrderTransitions = " + Arrays.toString(firstOrderTransitions));
        System.out.println("secondOrderTransitions = "+Arrays.toString(secondOrderTransitions));
        System.out.println("----------------------------------------------------------");*/
    }

    /**
     * Creates a string stating the parameters used in the model. The
     * string is used for pretty printing purposes and clarity in other
     * output routines.
     */
    @Override
    public void setModelParameterStringBuilder() {
        super.setModelParameterStringBuilder();
        String line = null;
        line = String.format("topicK:%d", topicK) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("documentD:%d", documentD) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("alpha:%f", alpha) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("beta:%f", beta) + newline;
        modelParameterStringBuilder.append(line);
        line = String.format("wbeta:%f", wbeta) + newline;
        modelParameterStringBuilder.append(line);
    }

    /*
    @Override
    public void initializeFromLoadedModel(CommandLineOptions options) throws
          IOException {
        super.initializeFromLoadedModel(options);

        int current = 0;
        int wordid = 0, stateid = 0, docid, topicid;
        int stateoff, wordstateoff, wordtopicoff, docoff;

        for (int i = 0; i < wordN; ++i) {
            wordid = wordVector[i];
            docid = documentVector[i];
            stateid = stateVector[i];
            topicid = topicVector[i];

            stateoff = current * stateS;
            wordstateoff = wordid * stateS;
            wordtopicoff = wordid * topicK;
            docoff = docid * topicK;

            if (stateid == 0) {
                topicByWord[wordtopicoff + topicid]++;
                documentByTopic[docoff + topicid]++;
                topicCounts[topicid]++;
            } else {
                stateByWord[wordstateoff + stateid]++;
            }

            stateByWord[wordstateoff + stateid]++;
            stateCounts[stateid]++;
            firstOrderTransitions[stateoff + stateid]++;
            first[i] = current;
            current = stateid;
        }
    }*/
}
