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
//  You should have received affix copy of the GNU Lesser General Public
//  License along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
///////////////////////////////////////////////////////////////////////////////
package tikka.hhl.distributions;

import tikka.exceptions.EmptyCountException;
import tikka.exceptions.EmptyTwoDimLexiconException;
import tikka.structures.DoubleStringPair;
import tikka.structures.StringDoublePair;
import tikka.hhl.lexicons.FourDimLexicon;
import tikka.hhl.lexicons.FourDimProbLexicon;
import tikka.hhl.lexicons.Lexicon;
import tikka.hhl.lexicons.ThreeDimLexicon;
import tikka.hhl.lexicons.ThreeDimProbLexicon;
import tikka.hhl.lexicons.TwoDimLexicon;
import tikka.hhl.lexicons.TwoDimProbLexicon;

import java.io.BufferedWriter;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Collections;

/**
 * Distribution for affixes conditioned on stems and affixes. The base distribution
 * for the affixes is not the usual base distribution but another conditional
 * DP.
 * 
 * @author tsmoon
 */
public class AffixStemStateHDP extends AffixStemStateDP {

    /**
     * Alias for stemAffixClsCounts
     */
    protected FourDimLexicon affixStemClsCounts;
    /**
     * Aliases for stemAffixClsProbs
     */
    protected FourDimProbLexicon affixStemClsProbs;
    /**
     * Distribution for maintaining prob of stem given state.
     */
    protected AffixStateDP affixStateDP;

    /**
     *
     * @param baseDistribution
     * @param lexicon
     * @param hyper
     */
    public AffixStemStateHDP(DirichletBaseDistribution baseDistribution,
          Lexicon lexicon, double hyper, int states) {
        super(baseDistribution, lexicon, hyper, states);
        /**
         * Making the hyperparameter 5 times bigger here. Good? Bad?
         */
        affixStateDP = new AffixStateDP(baseDistribution, lexicon, hyper * 5);
        affixStemClsCounts = stemAffixClsCounts;
        affixStemClsProbs = stemAffixClsProbs;
    }

    /**
     * Decrement counts for an stem given affix and class.
     *
     * @param cls
     * @param stem
     * @param affix
     * @return
     */
    @Override
    public int dec(int cls, int stem, int affix) {
        int val = 0;

        try {
            val = affixStemClsCounts.dec(cls, stem, affix);
        } catch (EmptyCountException e) {
            e.printMessage(lexicon.getString(affix), affix);
            System.exit(1);
        } catch (EmptyTwoDimLexiconException e) {
            affixStemClsCounts.get(cls).remove(stem);
        }

        affixStateDP.dec(cls, affix);

        return val;
    }

    /**
     * Increment counts for stem given affix and class.
     *
     * @param cls
     * @param stem
     * @param affix
     * @return
     */
    @Override
    public int inc(int cls, int stem, int affix) {
        affixStateDP.inc(cls, affix);
        return affixStemClsCounts.inc(cls, stem, affix);
    }

    /**
     * Get probabilties. Only used in the final normalization stage
     * after training is complete. Do NOT call this when training!
     *
     * @param cls
     * @param affix
     * @param stem
     * @return
     */
    @Override
    public double prob(int cls, int stem, int affix) {
        return (getCount(cls, stem, affix) + hyper * affixStateDP.prob(cls, affix))
              / (getCumCount(cls, stem) + hyper);
    }

    /**
     *
     * @param cls
     * @param stem
     * @param affix
     * @return
     */
    @Override
    public double prob(int cls, int stem, String affix) {
        int affixStemClsCount = 0, affixClsCount = 0;
        if (stem != -1) {
            affixClsCount = getCumCount(cls, stem);
            affixStemClsCount = getCount(cls, stem, lexicon.getIdx(affix));
        }
        return (affixStemClsCount + hyper * affixStateDP.prob(cls, affix))
              / (affixClsCount + hyper);
    }

    /**
     * Normalize sample counts for future printing.
     *
     * @param topicS    
     * @param stateS    Total number of topics.
     * @param outputPerTopic    How many affixes to print per state in
     *                  the output.
     * @param affixStateDP  State by affix DP
     */
    public void normalize(int topicS, int stateS, int outputPerTopic,
          StemStateDP stemStateDP, StemTopicDP stemTopicDP) {

        int maxid = 0;
        for (int affixid : lexicon.keySet()) {
            if (affixid > maxid) {
                maxid = affixid;
            }
        }
        maxid++;
        double[] affixProbs = new double[maxid];

        TopAffixesPerState = new StringDoublePair[stateS][];
        for (int i = 1; i < stateS; ++i) {
            TopAffixesPerState[i] = new StringDoublePair[outputPerTopic];
        }

        for (int i = 1; i < stateS; ++i) {
            try {
                for (int j = 0;; ++j) {
                    affixProbs[j] = 0;
                }
            } catch (ArrayIndexOutOfBoundsException e) {
            }

            ArrayList<DoubleStringPair> topAffixes =
                  new ArrayList<DoubleStringPair>();
            ThreeDimProbLexicon affixStemProbLexicon = new ThreeDimProbLexicon();
            ThreeDimLexicon affixStemCounts = affixStemClsCounts.get(i);
            affixStemClsProbs.put(i, affixStemProbLexicon);

            for (int stemid : affixStemCounts.keySet()) {
                double mult = 0;
                if (i < topicS) {
                    mult = stemTopicDP.prob(i, stemid);
                } else {
                    mult = stemStateDP.prob(i, stemid);
                }
                TwoDimLexicon affixCounts =
                      affixStemCounts.get(stemid);
                TwoDimProbLexicon stemProbLexicon = new TwoDimProbLexicon();
                affixStemProbLexicon.put(stemid, stemProbLexicon);
                for (int affixid : affixCounts.keySet()) {
                    double p = mult * prob(i, stemid, affixid);
                    stemProbLexicon.put(affixid, p);
                    affixProbs[affixid] += p;
                }
            }
            try {
                for (int j = 0;; ++j) {
                    double d = affixProbs[j];
                    if (d > 0) {
                        topAffixes.add(new DoubleStringPair(d,
                              lexicon.getString(j)));
                    }
                }
            } catch (ArrayIndexOutOfBoundsException e) {
            }

            Collections.sort(topAffixes);
            for (int j = 0; j < outputPerTopic; ++j) {
                try {
                    TopAffixesPerState[i][j] =
                          new StringDoublePair(topAffixes.get(j).stringValue,
                          topAffixes.get(j).doubleValue);
                } catch (IndexOutOfBoundsException e) {
//                    e.printStackTrace();
                }
            }
        }
    }

    /**
     * Print top {@code N} stems by {@code Z} topics to output buffer.
     *
     * @param topicS    Number of topic states
     * @param stateS    Total number of states
     * @param outputPerTopic    How many stems to print per topic in
     *                  the output.
     * @param stateProbs    Array of probabilities for each state.
     * @param out   Output buffer
     * @throws IOException
     */
    public void print(int topicS, int stateS, int outputPerTopic,
          double[] stateProbs,
          BufferedWriter out) throws IOException {
        int startt = 1, M = 4,
              endt = Math.min(M + 1, stateProbs.length);

        out.write("***** Affix Probabilities by State *****\n\n");
        while (startt < stateS) {
            for (int i = startt; i < endt; ++i) {
                String header = "State_" + i;
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
                              TopAffixesPerState[c][i].stringValue,
                              TopAffixesPerState[c][i].doubleValue);
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
