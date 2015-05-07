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
package tikka.hhl.models.m4;

import java.io.BufferedWriter;
import tikka.hhl.apps.CommandLineOptions;

import tikka.hhl.models.base.HDPHMMLDA;

import tikka.hhl.distributions.DirichletBaseDistribution;
import tikka.hhl.distributions.AffixStemStateHDP;
import tikka.hhl.distributions.StemTopicDP;

import java.io.IOException;

/**
 * This is a pure HDPLDA model. There is no HMM associated with the model
 * the state.
 *
 * @author tsmoon
 */
public class HDPLDA extends HDPHMMLDA {

    /**
     * Since there is not state sequence associated with the affixes, we
     * assign all affixes to a single state.
     */
    protected final int FIXEDSTATEID = 1;

    /**
     * Default constructor.
     *
     * @param options   Options from the command line.
     */
    public HDPLDA(CommandLineOptions options) throws IOException {
        super(options);
        stateS = 2;
        topicSubStates = stateS;
    }

    /**
     * Initialize the distributions that will be used in this model.
     */
    @Override
    protected void initalizeDistributions() {

        affixBaseDistribution = new DirichletBaseDistribution(
              affixLexicon, affixBoundaryProb, muAffix);

        stemTopicBaseDistribution = new DirichletBaseDistribution(
              stemLexicon, stemBoundaryProb, wbeta);

        affixStemStateHDP = new AffixStemStateHDP(affixBaseDistribution,
              affixLexicon, wgamma, stateS);
        stemTopicDP = new StemTopicDP(
              stemTopicBaseDistribution, stemLexicon, wbeta);
    }

    /**
     * Initializes from a pretrained, loaded model. Use this if the model has
     * been loaded from a pretrained model.
     */
    @Override
    public void initializeFromLoadedModel(CommandLineOptions options) throws
          IOException {
        super.initializeFromLoadedModel(options);

        initalizeDistributions();

        int wordid = 0, docid = 0, topicid = 0, splitid = 0, stemid =
              0, affixid = 0;
        int docoff, wordtopicoff;
        String word = "", stem = "", affix = "";

        for (int i = 0; i < wordN; ++i) {
            wordid = wordVector[i];

            if (wordid != EOSi) {
                docid = documentVector[i];
                topicid = topicVector[i];
                splitid = splitVector[i];

                docoff = topicK * docid;
                wordtopicoff = wordid * topicK;

                word = trainIdxToWord.get(wordid);

                stem = word.substring(0, splitid);
                affix = word.substring(splitid, word.length());
                stemid = stemLexicon.getOrPutIdx(stem);
                affixid = affixLexicon.getOrPutIdx(affix);
                stemVector[i] = stemid;
                affixVector[i] = affixid;

                stemTopicDP.inc(topicid, stemid);
                DocumentByTopic[docoff + topicid]++;
                topicCounts[topicid]++;
                TopicByWord[wordtopicoff + topicid]++;
                affixStemStateHDP.inc(FIXEDSTATEID, stemid, affixid);
            }
        }
    }

    /**
     * Randomly set the model parameters for use in training
     */
    @Override
    protected void randomInitializeParameters() {
        /**
         * Declaring temporary variables for training
         */
        int wordid = 0, docid = 0, topicid = 0, splitid = 0, stemid = 0, affixid = 0;
        double max = 0, totalprob = 0;
        double r = 0;
        int docoff, wordtopicoff;
        int wlength = 0;
        String word = "", stem = "", affix = "";

        double[] splitProbs = new double[MAXLEN];

        /**
         * Initialize by assigning random topic indices to words
         */
        for (int i = 0; i < wordN; ++i) {
            wordid = wordVector[i];

            if (wordid != EOSi) {
                docid = documentVector[i];
                docoff = topicK * docid;
                wordtopicoff = wordid * topicK;

                totalprob = 0;
                try {
                    for (int j = 0;; ++j) {
                        totalprob += topicProbs[j] =
                              (DocumentByTopic[docoff + j] + alpha)
                              * (TopicByWord[wordtopicoff + j] + beta)
                              / (topicCounts[j] + wbeta);
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

                word = trainIdxToWord.get(wordid);
                wlength = word.length();
                totalprob = 0;
                for (int j = 0; j < wlength + 1; ++j) {
                    stem = word.substring(0, j);
                    affix = word.substring(j, wlength);
                    stemid = stemLexicon.getIdx(stem);
                    affixid = affixLexicon.getIdx(affix);
                    totalprob += splitProbs[j] = stemTopicDP.probNumerator(
                          topicid, stem)
                          * affixStemStateHDP.prob(FIXEDSTATEID, stemid, affix);
                }
                r = mtfRand.nextDouble() * totalprob;
                max = splitProbs[0];
                splitid = 0;
                while (r > max) {
                    splitid++;
                    max += splitProbs[splitid];
                }
                stem = word.substring(0, splitid);
                affix = word.substring(splitid, wlength);
                stemid = stemLexicon.getOrPutIdx(stem);
                affixid = affixLexicon.getOrPutIdx(affix);
                stemVector[i] = stemid;
                affixVector[i] = affixid;
                splitVector[i] = splitid;

                stemTopicDP.inc(topicid, stemid);
                TopicByWord[wordtopicoff + topicid]++;
                DocumentByTopic[docoff + topicid]++;
                topicCounts[topicid]++;

                affixStemStateHDP.inc(FIXEDSTATEID, stemid, affixid);
            }
        }
    }

    /**
     * Training routine for the inner iterations
     */
    @Override
    protected void trainInnerIter(int itermax, String message) {
        /**
         * Declaring temporary variables for training
         */
        int wordid = 0, docid = 0, topicid = 0, splitid = 0, stemid = 0, affixid = 0;
        double max = 0, totalprob = 0;
        double r = 0;
        int docoff, wordtopicoff;
        int wlength = 0, splitmax = 0;
        String word = "";
        String[] stems = new String[MAXLEN], affixes = new String[MAXLEN];
        int[] stemidxes = new int[MAXLEN], affixidxes = new int[MAXLEN];

        double[] splitProbs = new double[MAXLEN];

        for (int initer = 0; initer < itermax; ++initer) {
            System.err.print("\n" + message + " " + initer);
            System.err.print("\tprocessing word ");
            for (int i = 0; i < wordN; i++) {

                if (i % 100000 == 0) {
                    System.err.print(i + ",");
                }
                wordid = wordVector[i];

                if (wordid != EOSi) // sentence marker
                {
                    docid = documentVector[i];
                    topicid = topicVector[i];
                    stemid = stemVector[i];
                    affixid = affixVector[i];

                    word = trainIdxToWord.get(wordid);
                    wlength = word.length();
                    splitmax = wlength + 1;
                    for (int k = 0; k < splitmax; ++k) {
                        stems[k] = word.substring(0, k);
                        affixes[k] = word.substring(k, wlength);
                        stemidxes[k] = stemLexicon.getIdx(stems[k]);
                        affixidxes[k] = affixLexicon.getIdx(affixes[k]);
                    }

                    docoff = docid * topicK;
                    wordtopicoff = wordid * topicK;

                    /**
                     * Decrement counts of current assignment from topics, states,
                     * switches, stems, and affixes.
                     */
                    stemTopicDP.dec(topicid, stemid);
                    DocumentByTopic[docoff + topicid]--;
                    topicCounts[topicid]--;
                    TopicByWord[wordtopicoff + topicid]--;

                    affixStemStateHDP.dec(FIXEDSTATEID, stemid, affixid);

                    /**
                     * Drawing new topicid
                     */
                    try {
                        for (int j = 0;; ++j) {
                            topicProbs[j] =
                                  DocumentByTopic[docoff + j] + alpha;
                            totalprob = 0;
                            for (int k = 0; k < splitmax; ++k) {
                                totalprob += stemTopicDP.probNumerator(j, stems[k])
                                      * affixStemStateHDP.prob(FIXEDSTATEID, stemidxes[k], affixes[k]);
                            }
                            topicProbs[j] *= totalprob;
                        }
                    } catch (java.lang.ArrayIndexOutOfBoundsException e) {
                    }
                    totalprob = annealProbs(topicProbs);
                    r = mtfRand.nextDouble() * totalprob;
                    max = topicProbs[0];

                    topicid = 0;
                    while (r > max) {
                        topicid++;
                        max += topicProbs[topicid];
                    }

                    topicVector[i] = topicid;

                    /**
                     * Drawing new stem and affix
                     */
                    for (int j = 0; j < splitmax; ++j) {
                        splitProbs[j] = stemTopicDP.probNumerator(topicid, stems[j])
                              * affixStemStateHDP.prob(FIXEDSTATEID, stemidxes[j], affixes[j]);
                    }
                    totalprob = annealProbs(splitProbs, splitmax);
                    r = mtfRand.nextDouble() * totalprob;
                    max = splitProbs[0];
                    splitid = 0;
                    while (r > max) {
                        splitid++;
                        max += splitProbs[splitid];
                    }
                    stemid = stemLexicon.getOrPutIdx(stems[splitid]);
                    affixid = affixLexicon.getOrPutIdx(affixes[splitid]);
                    stemVector[i] = stemid;
                    affixVector[i] = affixid;
                    splitVector[i] = splitid;

                    /**
                     * Increment counts of current assignment from topics, states,
                     * switches, stems, and affixes.
                     */
                    stemTopicDP.inc(topicid, stemid);
                    DocumentByTopic[docoff + topicid]++;
                    topicCounts[topicid]++;
                    TopicByWord[wordtopicoff + topicid]++;

                    affixStemStateHDP.inc(FIXEDSTATEID, stemid, affixid);
                }
            }
        }
    }

    /**
     * Normalize the sample counts for words over topics and states by summing over possible
     * segmentations. The parameters for the segmentation were learned during
     * the training stage.
     */
    @Override
    protected void normalizeWords(double[] StateByWordProbs,
          double[] TopicByWordProbs) {

        /**
         * Calculate word probability per topic and word probability per state
         * (but only for topic states)
         */
        int wlength = 0, splitmax = 0;
        String word = "";
        String[] stems = new String[MAXLEN], affixes = new String[MAXLEN];
        int[] stemidxes = new int[MAXLEN], affixidxes = new int[MAXLEN];

        for (int wordid = 1; wordid < wordW; ++wordid) {
            word = trainIdxToWord.get(wordid);
            int wordtopicoff = wordid * topicK;

            wlength = word.length();
            splitmax = wlength + 1;
            for (int k = 0; k < splitmax; ++k) {
                stems[k] = word.substring(0, k);
                affixes[k] = word.substring(k, wlength);
                stemidxes[k] = stemLexicon.getIdx(stems[k]);
                affixidxes[k] = affixLexicon.getIdx(affixes[k]);
            }

            for (int j = 0; j < topicK; ++j) {
                double tsum = 0;
                for (int k = 0; k < splitmax; ++k) {
                    double stemProb = stemTopicDP.prob(j, stems[k]);
                    double affixProb = affixStemStateHDP.prob(FIXEDSTATEID, stemidxes[k], affixes[k]);
                    tsum += stemProb * affixProb;
                }
                TopicByWordProbs[wordtopicoff + j] = tsum;
            }
        }

        setTopWordsPerTopic(TopicByWordProbs);
    }

    /**
     * Method for setting probability of tokens per sample.
     *
     * @param outiter Number of sample run
     */
    @Override
    protected void obtainSample(int outiter) {
        /**
         * Declaring temporary variables for training
         */
        int wordid = 0, topicid = 0;
        double totalprob = 0;
        int wlength = 0, splitmax = 0;
        String word = "";
        String[] stems = new String[MAXLEN], affixes = new String[MAXLEN];
        int[] stemidxes = new int[MAXLEN], affixidxes = new int[MAXLEN];

        for (int i = 0; i < wordN; i++) {

            if (i % 100000 == 0) {
                System.err.print(i + ",");
            }
            wordid = wordVector[i];

            if (wordid != EOSi) // sentence marker
            {
                word = trainIdxToWord.get(wordVector[i]);
                topicid = topicVector[i];

                wlength = word.length();
                splitmax = wlength + 1;
                for (int k = 0; k < splitmax; ++k) {
                    stems[k] = word.substring(0, k);
                    affixes[k] = word.substring(k, wlength);
                    stemidxes[k] = stemLexicon.getIdx(stems[k]);
                    affixidxes[k] = affixLexicon.getIdx(affixes[k]);
                }

                totalprob = 0;
                for (int k = 0; k < splitmax; ++k) {
                    totalprob += stemTopicDP.prob(topicid, stems[k])
                          * affixStemStateHDP.prob(FIXEDSTATEID, stemidxes[k], affixes[k]);
                }
                SampleProbs[outiter] += Math.log(totalprob);
            }
        }
    }

    /**
     * Set the arrays testWordTopicProbs and testWordStateProbs
     */
    @Override
    protected void setWordClassProbArrays() {
        double totalprob = 0;
        int wordtopicoff;
        String word = "";
        int wlength = 0, splitmax = 0;
        String[] stems = new String[MAXLEN], affixes = new String[MAXLEN];
        int[] stemidxes = new int[MAXLEN], affixidxes = new int[MAXLEN];

        /**
         * Assign probabilities to each word given either topic or state.
         * This will not change in the test run so we store this in table
         * at the beginning.
         */
        testWordTopicProbs = new double[wordW * topicK];

        for (int i = 1; i < wordW; ++i) {
            word = testIdxToWord.get(i);
            wlength = word.length();
            splitmax = wlength + 1;
            for (int k = 0; k < splitmax; ++k) {
                stems[k] = word.substring(0, k);
                affixes[k] = word.substring(k, wlength);
                stemidxes[k] = stemLexicon.getIdx(stems[k]);
                affixidxes[k] = affixLexicon.getIdx(affixes[k]);
            }
            wordtopicoff = i * topicK;

            for (int j = 0; j < topicK; ++j) {
                totalprob = 0;
                for (int k = 0; k < splitmax; ++k) {
                    totalprob += stemTopicDP.prob(j, stems[k])
                          * affixStemStateHDP.prob(FIXEDSTATEID, stemidxes[k], affixes[k]);
                }
                testWordTopicProbs[wordtopicoff + j] = totalprob;
            }
        }
    }

    /**
     * Normalize the training sample
     */
    @Override
    public void normalize() {
        stemTopicDP.normalize(0, topicK, outputPerClass, null);
        affixStemStateHDP.normalize(topicSubStates, stateS, outputPerClass,
              null, stemTopicDP);

        double[] TopicByWordProbs = new double[wordW * topicK];
        try {
            for (int i = 0;; ++i) {
                TopicByWordProbs[i] = 0;
            }
        } catch (java.lang.ArrayIndexOutOfBoundsException e) {
        }

        normalizeWords(null, TopicByWordProbs);
        normalizeRawTopics();
    }

    /**
     * Print normalized probabilities for each category to out. Print only
     * the top {@link #outputPerClass} per category.
     *
     * @param out   Buffer to write to.
     * @throws IOException
     */
    @Override
    public void printTabulatedProbabilities(BufferedWriter out) throws
          IOException {
        affixStemStateHDP.print(topicSubStates, stateS, outputPerClass, stateProbs,
              out);
        printNewlines(out, 4);
        stemTopicDP.print(0, topicK, outputPerClass, topicProbs, out);
        printNewlines(out, 4);
        super.printTabulatedProbabilities(out);
    }

    /**
     * Sample word segmentations. This is in the last stage of sampling
     * after all classes have been sampled. It is only needed to print the
     * annotated text
     */
    @Override
    protected void sampleTestWordSplitLocations() {
        int wordid = 0, topicid = 0, splitid = 0;
        double max = 0, totalprob = 0;
        double r = 0;
        String word = "";
        int wlength = 0, splitmax = 0;
        String[] stems = new String[MAXLEN], affixes = new String[MAXLEN];
        int[] stemidxes = new int[MAXLEN], affixidxes = new int[MAXLEN];

        double[] splitProbs = new double[MAXLEN];

        System.err.println("\nSampling split locations");
        for (int i = 0; i < wordN; i++) {
            wordid = wordVector[i];
            if (wordid != EOSi) // sentence marker
            {
                topicid = topicVector[i];

                word = testIdxToWord.get(wordid);
                wlength = word.length();
                splitmax = wlength + 1;
                for (int k = 0; k < splitmax; ++k) {
                    stems[k] = word.substring(0, k);
                    affixes[k] = word.substring(k, wlength);
                    stemidxes[k] = stemLexicon.getIdx(stems[k]);
                    affixidxes[k] = affixLexicon.getIdx(affixes[k]);
                }

                for (int j = 0; j < splitmax; ++j) {
                    splitProbs[j] = stemTopicDP.prob(topicid, stems[j])
                          * affixStemStateHDP.prob(FIXEDSTATEID, stemidxes[j],
                          affixes[j]);
                }
                totalprob = annealProbs(splitProbs, splitmax);
                r = mtfRand.nextDouble() * totalprob;
                max = splitProbs[0];
                splitid = 0;
                while (r > max) {
                    splitid++;
                    max += splitProbs[splitid];
                }
                splitVector[i] = splitid;
            }
        }
    }

    /**
     * Sample model output on test.
     *
     * @see HDPHMMLDA#setWordClassProbArrays()
     * @see HDPHMMLDA#sampleTestWordSplitLocations()
     */
    @Override
    public void sampleFromTest() {
        /**
         * This is a workaround because wgamma and wbeta are overwritten in the
         * following method. Global variables documentD, wordW, wordN are
         * overwritten.
         */
        double twbeta = wbeta, twgamma = wgamma;
        initializeTokenArrays(testDirReader, testWordIdx, testIdxToWord);
        wbeta = twbeta;
        wgamma = twgamma;
        double talpha = alpha * topicK;

        DocumentCounts = new double[documentD];
        try {
            for (int i = 0;; ++i) {
                DocumentCounts[i] = talpha;
            }
        } catch (java.lang.ArrayIndexOutOfBoundsException e) {
        }

        int wordid = 0, docid = 0, topicid = 0;
        double max = 0, totalprob = 0;
        double r = 0;
        int docoff, wordtopicoff;

        setWordClassProbArrays();

        /**
         * Gibbs sample initial parameters for test set
         */
        System.err.print("\nSample initial document parameters for test set");
        temperature = 1;
        temperatureReciprocal = 1;
        for (int i = 0; i < wordN; ++i) {
            wordid = wordVector[i];

            if (wordid != EOSi) {
                docid = documentVector[i];
                docoff = topicK * docid;
                wordtopicoff = wordid * topicK;

                try {
                    for (int j = 0;; ++j) {
                        topicProbs[j] =
                              (DocumentByTopic[docoff + j] + alpha)
                              * testWordTopicProbs[wordtopicoff + j];
                    }
                } catch (java.lang.ArrayIndexOutOfBoundsException e) {
                }

                totalprob = annealProbs(topicProbs);
                r = mtfRand.nextDouble() * totalprob;
                max = topicProbs[0];
                topicid = 0;
                while (r > max) {
                    topicid++;
                    max += topicProbs[topicid];
                }
                topicVector[i] = topicid;

                DocumentByTopic[docoff + topicid]++;
                DocumentCounts[docid]++;
            }
        }

        /**
         * Burn in then sample for test set regarding document vectors.
         */
        samples = 1; // set to one since perplexity is only sampled once
        SampleProbs = new double[samples];
        System.err.print("\nBurn in sample of initial document parameters "
              + "for test set");
        temperature = MAPTEMP;
        temperatureReciprocal = 1 / temperature;
        for (int iter = 0; iter < testSetBurninIterations + samples; ++iter) {
            System.err.print("\niteration " + iter);
            System.err.print("\tprocessing word ");
            for (int i = 0; i < wordN; ++i) {
                if (i % 100000 == 0) {
                    System.err.print(i + ",");
                }

                wordid = wordVector[i];

                if (wordid != EOSi) {
                    docid = documentVector[i];
                    docoff = topicK * docid;
                    topicid = topicVector[i];
                    wordtopicoff = wordid * topicK;

                    DocumentByTopic[docoff + topicid]--;

                    try {
                        for (int j = 0;; ++j) {
                            topicProbs[j] = (DocumentByTopic[docoff + j] + alpha)
                                  * testWordTopicProbs[wordtopicoff + j];
                        }
                    } catch (java.lang.ArrayIndexOutOfBoundsException e) {
                    }

                    totalprob = annealProbs(topicProbs);
                    r = mtfRand.nextDouble() * totalprob;
                    max = topicProbs[0];
                    topicid = 0;
                    while (r > max) {
                        topicid++;
                        max += topicProbs[topicid];
                    }
                    topicVector[i] = topicid;

                    DocumentByTopic[docoff + topicid]++;
                }
            }

            /**
             * Sample words and probabilities. We are going for perplexity in this
             * situation, not the bayes factor
             */
            int samplenum = iter - testSetBurninIterations;
            if (samplenum > -1) {
                System.err.print("\nObtain final test sample");
                System.err.print("\nSample " + samplenum + ":");
                stabilizeTemperature();
                System.err.print("\tprocessing word ");

                for (int i = 0; i < wordN; i++) {

                    if (i % 100000 == 0) {
                        System.err.print(i + ",");
                    }
                    wordid = wordVector[i];

                    if (wordid != EOSi) // sentence marker
                    {
                        docid = documentVector[i];
                        topicid = topicVector[i];
                        wordtopicoff = wordid * topicK;

                        totalprob = testWordTopicProbs[wordtopicoff + topicid]
                              * (DocumentByTopic[docid * topicK + topicid] + alpha)
                              / (DocumentCounts[docid]); // The doc counts array already contains the talpha so don't mess with this
                    }
                    SampleProbs[samplenum] += Math.log(totalprob);
                }
            }
        }

        sampleTestWordSplitLocations();
    }
}
