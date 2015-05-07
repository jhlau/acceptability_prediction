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

import tikka.hhl.lexicons.ThreeDimLexicon;
import tikka.hhl.lexicons.ThreeDimProbLexicon;

import java.io.BufferedWriter;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Collections;

/**
 * The class of the hierarchical Dirichlet process for stems over topics.
 * The counts are maintained in {@link #morphClsCounts}.
 *
 * @author tsmoon
 */
public class StemTopicDP extends ThreeDimDirichletProcess {

    /**
     * Alias for {@link #morphClsCounts}.
     */
    protected ThreeDimLexicon stemClsCounts;
    /**
     * Alias for {@link #morphClsProbs}.
     */
    protected ThreeDimProbLexicon stemClsProbs;
    /**
     * 2D array of {@link Infltrait.structures.StringDoublePair}. Defined
     * for {@code C} states and {@code N} affixes which have {@code N} highest
     * likelihoods in the given state. Allocated and populated in
     * {@link #normalize(int)}.
     */
    protected StringDoublePair[][] TopStemsPerTopic;

    /**
     * 
     * @param baseDistribution
     * @param lexicon
     * @param hyper
     */
    public StemTopicDP(DirichletBaseDistribution baseDistribution,
          Lexicon lexicon, double hyper) {
        super(baseDistribution, lexicon, hyper);
        stemClsCounts = morphClsCounts;
        stemClsProbs = morphClsProbs;
    }

    /**
     * Normalize sample counts.
     *
     * @param topicS    Dummy variable. Not used at all.
     * @param topicK    Total number of states. This excludes the first
     *                  state 0, which is the sentence boundary.
     * @param outputPerTopic    How many affixes to print per state in
     *                  the output.
     * @param topicProbs    Dummy variable. Not used at all.
     */
    @Override
    public void normalize(int topicS, int topicK, int outputPerState,
          double[] topicProbs) {

        TopStemsPerTopic = new StringDoublePair[topicK][];
        for (int i = 0; i < topicK; ++i) {
            TopStemsPerTopic[i] = new StringDoublePair[outputPerState];
        }

        for (int i = 0; i < topicK; ++i) {
            ArrayList<DoubleStringPair> topStems =
                  new ArrayList<DoubleStringPair>();
            if (stemClsCounts.containsKey(i)) {
                for (int stemid : stemClsCounts.get(i).keySet()) {
                    double p = prob(i, stemid);
                    topStems.add(new DoubleStringPair(p, lexicon.getString(
                          stemid)));
                }
                Collections.sort(topStems);
                for (int j = 0; j < outputPerState; ++j) {
                    try {
                        TopStemsPerTopic[i][j] = new StringDoublePair(
                              topStems.get(j).stringValue,
                              topStems.get(j).doubleValue);
                    } catch (IndexOutOfBoundsException e) {
                    }
                }
            } else {
                for (int j = 0; j < outputPerState; ++j) {
                    TopStemsPerTopic[i][j] = new StringDoublePair("", 0);
                }
            }
        }
    }

    /**
     * Print normalized probability tables for affixes by topic.
     *
     * @param topicS    Dummy variable. Not used at all.
     * @param topicK    Number of topics
     * @param outputPerTopic    How many affixes to print per state in
     *                  the output.
     * @param topicProbs Array of probabilties for each state. The first
     *                  cell should be ignored.
     * @param out   Destination of output
     */
    @Override
    public void print(int topicS, int topicK, int outputPerTopic,
          double[] topicProbs, BufferedWriter out)
          throws IOException {
        int startt = 0, M = 4, endt = Math.min(M, topicProbs.length);

        out.write("***** Stem Probabilities by Topic *****\n\n");
        while (startt < topicK) {
            for (int i = startt; i < endt; ++i) {
                String header = "Topic_" + i;
                header = String.format("%25s\t%6.5f\t", header,
                      topicProbs[i]);
                out.write(header);
            }

            out.newLine();
            out.newLine();

            for (int i = 0; i < outputPerTopic; ++i) {
                for (int c = startt; c < endt; ++c) {
                    String line = String.format("%25s\t%7s\t", "", "");
                    try {
                        line = String.format("%25s\t%6.5f\t",
                              TopStemsPerTopic[c][i].stringValue,
                              TopStemsPerTopic[c][i].doubleValue);
                    } catch (NullPointerException e) {
//                        e.printStackTrace();
                    } catch (ArrayIndexOutOfBoundsException e) {
                    }
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
}
