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
package tikka.hhl.distributions;

import tikka.structures.DoubleStringPair;
import tikka.structures.StringDoublePair;
import tikka.hhl.lexicons.Lexicon;
import tikka.hhl.lexicons.ThreeDimProbLexicon;
import tikka.hhl.lexicons.TwoDimLexicon;
import tikka.hhl.lexicons.TwoDimProbLexicon;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

/**
 * 
 * @author tsmoon
 */
public class StemAffixStateDP extends FourDimDirichletProcess {

    /**
     * Array of probabilities for the stems. Allocated and populated in
     * {@link #normalize(int)}.
     */
//    double[] stemProbs;
    /**
     * 2D array of {@link Infltrait.structures.StringDoublePair}. Defined
     * for {@code Z} states and {@code N} stems which have {@code N} highest
     * likelihoods in the given topic. Allocated and populated in
     * {@link #normalize(int, int, double[])}.
     */
    protected StringDoublePair[][] TopStemsPerState;
    /**
     * Table of top {@code N} words per state. Used in
     * normalization and printing. Calculated from the bookkeeping table
     * of {@link #StateByWord}. Is not related to counts derived from
     * segmentations.
     */
    protected StringDoublePair[][] TopWordsPerStateFromRaw;

    /**
     *
     * @param baseDistribution
     * @param lexicon
     * @param hyper
     */
    public StemAffixStateDP(
            DirichletBaseDistribution baseDistribution,
            Lexicon lexicon, double hyper, int states) {
        super(baseDistribution, lexicon, hyper, states);
    }

    /**
     * Normalize sample counts for future printing.
     *
     * @param topicS    Number of topic states.
     * @param stateS    Total number of topics.
     * @param outputPerTopic    How many affixes to print per state in
     *                  the output.
     * @param affixStateDP  State by affix DP
     * @param affixLexicon  Affix-to-idx and idx-to-affix lexicon
     */
    public void normalize(int topicS, int stateS, int outputPerTopic,
            ThreeDimDirichletProcess affixStateDP, Lexicon affixLexicon) {
        int maxid = 0;
        for (int stemid : lexicon.keySet()) {
            if (stemid > maxid) {
                maxid = stemid;
            }
        }
        maxid++;
        double[] stemProbs = new double[maxid];

        TopStemsPerState = new StringDoublePair[stateS][];
        for (int i = topicS; i < stateS; ++i) {
            TopStemsPerState[i] = new StringDoublePair[outputPerTopic];
        }

        for (int i = topicS; i < stateS; ++i) {
            try {
                for (int j = 0;; ++j) {
                    stemProbs[j] = 0;
                }
            } catch (ArrayIndexOutOfBoundsException e) {
            }
            ArrayList<DoubleStringPair> topStems =
                    new ArrayList<DoubleStringPair>();
            ThreeDimProbLexicon stemAffixProbLexicon = new ThreeDimProbLexicon();
            stemAffixClsProbs.put(i, stemAffixProbLexicon);

            for (int affixid : stemAffixClsCounts.get(i).keySet()) {
                double affixStateProb = affixStateDP.getConstProb(i, affixid);
                TwoDimLexicon stemCounts =
                        stemAffixClsCounts.get(i).get(affixid);
                TwoDimProbLexicon stemProbLexicon = new TwoDimProbLexicon();
                stemAffixProbLexicon.put(affixid, stemProbLexicon);

                for (int stemid : stemCounts.keySet()) {
                    double p = affixStateProb *
                            prob(i, affixid, stemid);
                    stemProbs[stemid] += p;
                    stemProbLexicon.put(stemid, p);
                }
            }
            try {
                for (int j = 0;; ++j) {
                    double d = stemProbs[j];
                    if (d > 0) {
                        topStems.add(new DoubleStringPair(d,
                                lexicon.getString(j)));
                    }
                }
            } catch (ArrayIndexOutOfBoundsException e) {
            }
            Collections.sort(topStems);
            for (int j = 0; j < outputPerTopic; ++j) {
                try {
                    TopStemsPerState[i][j] =
                            new StringDoublePair(topStems.get(j).stringValue,
                            topStems.get(j).doubleValue);
                } catch (IndexOutOfBoundsException e) {
//                    e.printStackTrace();
                }
            }
        }
    }

    /**
     * Print top {@code N} stems by {@code C} states to output buffer.
     *
     * @param topicS    Number of topic states.
     * @param stateS    Total number of topics.
     * @param outputPerTopic    How many stems to print per topic in
     *                  the output.
     * @param stateProbs    Array of probabilities for each topic.
     * @param out   Output buffer
     * @throws IOException
     */
    public void print(int topicS, int stateS, int outputPerTopic, double[] stateProbs,
            BufferedWriter out) throws IOException {
        int startt = topicS, M = 4, endt = M;

        out.write("***** Stem Probabilities by Topic *****\n\n");
        while (startt < stateS) {
            for (int i = startt; i < endt; ++i) {
                String header = "Topic_" + i;
                header = String.format("%25s\t%6.5f\t", header,
                        stateProbs[i]);
                out.write(header);
            }

            out.newLine();
            out.newLine();

            for (int i = 0; i < outputPerTopic; ++i) {
                for (int c = startt; c < endt; ++c) {
                    String line = String.format("%25s\t%7s\t", "", "");
                    try {
                        line = String.format("%25s\t%6.5f\t",
                                TopStemsPerState[c][i].stringValue,
                                TopStemsPerState[c][i].doubleValue);
                    } catch (NullPointerException e) {
//                        e.printStackTrace();
                    }
                    out.write(line);
                }
                out.newLine();
            }

            out.newLine();
            out.newLine();

            startt = endt;
            endt = java.lang.Math.min(stateS, startt + M);
        }
    }
}
