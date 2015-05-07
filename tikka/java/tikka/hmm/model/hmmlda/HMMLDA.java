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
package tikka.hmm.model.hmmlda;

import tikka.hmm.apps.CommandLineOptions;
import tikka.hmm.model.base.HMM;

import tikka.structures.DoubleStringPair;
import tikka.structures.StringDoublePair;

import java.io.BufferedWriter;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import tikka.exceptions.IgnoreTagException;
import tikka.opennlp.io.DirReader;
import tikka.utils.annealer.Annealer;

/**
 *
 * @author tsmoon
 */
public class HMMLDA extends HMM {

    /**
     * Hyperparameter for topic-by-document prior.
     */
    protected double alpha;
    /**
     * Hyperparameter for word emission priors.
     */
    protected double beta;
    /**
     * Normalization term for word emission multinomial
     */
    protected double wbeta;
    /**
     * Number of topic types.
     */
    protected int topicK;
    /**
     * Number of documents in the test set
     */
    protected int testDocumentD;
    /**
     * Array of counts by topic
     */
    protected int[] topicCounts;
    /**
     * Array of probabilities by topic
     */
    protected double[] topicProbs;
    /**
     * Counts of topics by document
     */
    protected int[] DocumentByTopic;
    /**
     * Array of counts for words given topics.
     * This has no effect on the model. It is merely here as a bookkeeping
     * device to check how the segmentation model is doing compared to
     * normalization as if the words had been dumped into their topics.
     */
    protected int[] TopicByWord;
    /**
     * Table of top {@link #outputPerClass} words per topic. Used in
     * normalization and printing.
     */
    protected StringDoublePair[][] TopWordsPerTopic;
    /**
     * Array of topic indexes. Of length {@link #wordN}.
     */
    protected int[] topicVector;

    public HMMLDA(CommandLineOptions options) {
        super(options);
        alpha = options.getAlpha();
        beta = options.getBeta();
        topicK = options.getTopics();
    }

    /**
     * Initialize arrays that will be used to track the state, topic, split
     * position and switch of each token. The DocumentByTopic array is also
     * rewritten in sampling for test sets.
     *
     * @param dirReader Object to walk through files and directories
     * @param wordIdx   Dictionary from word to index
     * @param idxToWord Dictionary from index to word
     */
    @Override
    protected void initializeTokenArrays(DirReader dirReader,
          HashMap<String, Integer> wordIdx, HashMap<Integer, String> idxToWord) {
        documentD = 0;
        ArrayList<Integer> wordVectorT = new ArrayList<Integer>(),
              documentVectorT = new ArrayList<Integer>(),
              goldFullTagVectorT = new ArrayList<Integer>(),
              sentenceVectorT = new ArrayList<Integer>();
        while ((dataReader = dirReader.nextDocumentReader()) != null) {
            try {
                String[][] sentence;
                while ((sentence = dataReader.nextSequence()) != null) {
                    for (String[] line : sentence) {
                        try {
//                        String word = wordNormalizer.normalize(line)[0];
                            wordNormalizer.normalize(line);
                            String word = wordNormalizer.getWord();
                            String tag = wordNormalizer.getTag();
                            if (!word.isEmpty() && tag != null) {
                                if (!wordIdx.containsKey(word)) {
                                    wordIdx.put(word, wordIdx.size());
                                    idxToWord.put(idxToWord.size(), word);
                                }

                                wordVectorT.add(wordIdx.get(word));
                                documentVectorT.add(documentD);
                                sentenceVectorT.add(sentenceS);
                                goldFullTagVectorT.add(tagMap.get(tag));
                            }
                        } catch (IgnoreTagException e) {
                        }
                    }
//                    wordVectorT.add(EOSi);
//                    documentVectorT.add(documentD);
//                    sentenceVectorT.add(sentenceS);
                    sentenceS++;

                }
            } catch (IOException e) {
            }
            documentD++;
        }

        wordN = wordVectorT.size();
        wordW = wordIdx.size();
        wbeta = beta * wordW;
        wdelta = delta * wordW;
        sgamma = gamma * stateS;

        wordVector = new int[wordN];
        goldTagVector = new int[wordN];
        sentenceVector = new int[wordN];
        documentVector = new int[wordN];

        copyToArray(wordVector, wordVectorT);
        copyToArray(goldTagVector, goldFullTagVectorT);
        copyToArray(sentenceVector, sentenceVectorT);
        copyToArray(documentVector, documentVectorT);
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

        TopicByWord = new int[topicK * wordW];
        try {
            for (int i = 0;; ++i) {
                TopicByWord[i] = 0;
            }
        } catch (ArrayIndexOutOfBoundsException e) {
        }

        DocumentByTopic = new int[documentD * topicK];
        try {
            for (int i = 0;; ++i) {
                DocumentByTopic[i] = 0;
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
        TopWordsPerTopic = new StringDoublePair[topicK][];
        for (int i = 0; i < topicK; ++i) {
            TopWordsPerTopic[i] = new StringDoublePair[outputPerClass];
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
                      TopicByWord[j * topicK + i] + beta, trainIdxToWord.get(
                      j)));
            }
            Collections.sort(topWords);
            for (int j = 0; j < outputPerClass; ++j) {
                TopWordsPerTopic[i][j] = new StringDoublePair(
                      topWords.get(j).stringValue, topWords.get(j).doubleValue
                      / topicProbs[i]);
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
        TopWordsPerState = new StringDoublePair[stateS][];
        for (int i = 0; i < stateS; ++i) {
            TopWordsPerState[i] = new StringDoublePair[outputPerClass];
        }

        double sum = 0.;
        double[] marginalwordprobs = new double[wordN];
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
                      * (TopicByWord[wordoff + i] + beta)
                      / (topicCounts[i] + wbeta);
            }
        }

        {
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
                TopWordsPerState[0][j] =
                      new StringDoublePair(
                      topWords.get(j).stringValue,
                      topWords.get(j).doubleValue);
            }
        }

        for (int i = 0; i < stateS; ++i) {
            sum += stateProbs[i] = stateCounts[i] + wdelta;
            ArrayList<DoubleStringPair> topWords =
                  new ArrayList<DoubleStringPair>();
            /**
             * Start at one to leave out EOSi
             */
            for (int j = 0; j < wordW; ++j) {
                topWords.add(new DoubleStringPair(
                      StateByWord[j * stateS + i] + delta, trainIdxToWord.get(
                      j)));
            }
            Collections.sort(topWords);
            for (int j = 0; j < outputPerClass; ++j) {
                TopWordsPerState[i][j] =
                      new StringDoublePair(
                      topWords.get(j).stringValue,
                      topWords.get(j).doubleValue / stateProbs[i]);
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
                          TopWordsPerTopic[c][i].stringValue,
                          TopWordsPerTopic[c][i].doubleValue);
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
        int current = 0, prev = 0, preprev = 0, next = 0, nnext = 0, nnnext = 0;
        double max = 0, totalprob = 0;
        double r = 0;
        int wordtopicoff, wordstateoff, docoff, thirdstateoff, secondstateoff;

        /**
         * Initialize by assigning random topic indices to words
         */
        for (int i = 0; i < wordN; ++i) {
            wordid = wordVector[i];

//            if (wordid == EOSi) {
//                firstOrderTransitions[current * S1 + 0]++;
//                first[i] = current;
//                current = 0;
//            } else {
            docid = documentVector[i];
            wordtopicoff = topicK * wordid;
            wordstateoff = stateS * wordid;
            docoff = topicK * docid;

            if (mtfRand.nextDouble() > 0.5) {
                stateVector[i] = 1;
            } else {
                stateVector[i] = 0;
            }

            totalprob = 0;
            try {
                for (int j = 0;; ++j) {
                    topicProbs[j] = DocumentByTopic[docoff + j] + alpha;
                    if (stateVector[i] == 1) {
                        topicProbs[j] *= (TopicByWord[wordtopicoff + j] + beta)
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
            topicVector[i] = topicid;
            max = 0;

            totalprob = 0;
            if (stateVector[i] == 0) {
                totalprob = stateProbs[0] =
                      (TopicByWord[wordtopicoff + topicid] + delta)
                      / (topicCounts[topicid] + wdelta)
                      * (firstOrderTransitions[current * S1 + 0] + gamma);
                try {
                    for (int j = 1;; j++) {
                        totalprob += stateProbs[j] =
                              (StateByWord[wordstateoff + j] + beta)
                              / (stateCounts[j] + wbeta)
                              * (firstOrderTransitions[current * S1 + j]
                              + gamma);
                    }
                } catch (java.lang.ArrayIndexOutOfBoundsException e) {
                }
            }

            r = mtfRand.nextDouble() * totalprob;
            stateid = 0;
            max = stateProbs[stateid];
            while (r > max) {
                stateid++;
                max += stateProbs[stateid];
            }
            stateVector[i] = stateid;

            if (stateid == 0) {
                TopicByWord[wordtopicoff + topicid]++;
                DocumentByTopic[docoff + topicid]++;
                topicCounts[topicid]++;
            } else {
                StateByWord[wordstateoff + stateid]++;
            }

            stateCounts[stateid]++;
            firstOrderTransitions[current * S1 + stateid]++;
            first[i] = current;
            current = stateid;
//            }
        }
    }

    @Override
    protected void trainInnerIter(int itermax, Annealer annealer) {
        int wordid, docid, topicid, stateid;
        int current = 0, next;
        double max = 0, totalprob = 0;
        double r = 0;
        int wordtopicoff, wordstateoff, docoff, stateoff;

        for (int iter = 0; iter < itermax; ++iter) {
            System.err.println("iteration " + iter);
            current = 0;
            for (int i = 0; i < wordN; i++) {
                if (i % 100000 == 0) {
                    System.err.println("\tProcessing word " + i);
                }
                wordid = wordVector[i];

//                if (wordid == EOSi) // sentence marker
//                {
//                    firstOrderTransitions[first[i] * S1 + 0]--;
//                    firstOrderTransitions[current * S1 + 0]++;
//                    first[i] = current;
//                    current = 0;
//                } else {
                docid = documentVector[i];
                stateid = stateVector[i];
                topicid = topicVector[i];
                wordstateoff = wordid * stateS;
                wordtopicoff = wordid * topicK;
                docoff = docid * topicK;

                if (stateid == 0) {
                    TopicByWord[wordtopicoff + topicid]--;
                    DocumentByTopic[docoff + topicid]--;
                    topicCounts[topicid]--;
                } else {
                    StateByWord[wordstateoff + stateid]--;

                }
                stateCounts[stateid]--;
                firstOrderTransitions[first[i] * S1 + stateid]--;

                try {
                    for (int j = 0;; j++) {
                        topicProbs[j] = DocumentByTopic[docoff + j] + alpha;
                        if (stateid == 1) {
                            topicProbs[j] *= (TopicByWord[wordtopicoff + j] + beta)
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

                stateoff = current * stateS;
                try {
                    next = stateVector[i + 1];
                } catch (ArrayIndexOutOfBoundsException e) {
                    next = 0;
                }

                stateProbs[0] =
                      ((TopicByWord[wordtopicoff + topicid] + beta) / (topicCounts[topicid] + wbeta))
                      * (firstOrderTransitions[stateoff + 0] + gamma)
                      * ((firstOrderTransitions[S1 + next] + gamma)
                      / (stateCounts[0] + sgamma));
                for (int j = 1; j < stateS; j++) {
                    stateProbs[j] =
                          ((StateByWord[wordstateoff + j] + delta) / (stateCounts[j] + wdelta))
                          * (firstOrderTransitions[stateoff + j] + gamma)
                          * ((firstOrderTransitions[j * stateS + next] + gamma)
                          / (stateCounts[j] + sgamma));
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
                    TopicByWord[wordtopicoff + topicid]++;
                    DocumentByTopic[docoff + topicid]++;
                    topicCounts[topicid]++;
                } else {
                    StateByWord[wordstateoff + stateid]++;
                }

                stateCounts[stateid]++;
                firstOrderTransitions[current * S1 + stateid]++;
                first[i] = current;
                current = stateid;
//                }
            }
        }
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
}
