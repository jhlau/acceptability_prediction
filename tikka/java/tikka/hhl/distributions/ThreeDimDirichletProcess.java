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

import tikka.exceptions.EmptyCountException;
import tikka.exceptions.KeyRemovedException;
import tikka.exceptions.EmptyTwoDimLexiconException;
import tikka.hhl.lexicons.Lexicon;
import tikka.hhl.lexicons.ThreeDimLexicon;
import tikka.hhl.lexicons.ThreeDimProbLexicon;
import java.io.BufferedWriter;
import java.io.IOException;

/**
 * The class of the hierarchical Dirichlet process for affixes over states.
 * The counts are maintained in {@link #morphClsCounts}.
 *
 * @author tsmoon
 */
public abstract class ThreeDimDirichletProcess extends DirichletProcess {

    /**
     * {@link java.util.HashMap} which keeps counts of morphs per class. This
     * is used in the training as well as normalization stage.
     */
    protected ThreeDimLexicon morphClsCounts;
    /**
     * {@link java.util.HashMap} which keeps probs of morphs per class. This
     * is used only in the normalization stage and is solely intended for fast
     * lookup.
     */
    protected ThreeDimProbLexicon morphClsProbs;
    /**
     * Array of probabilities for the affixes to be used in computing probabilities for
     * stems and words conditioned only on the topics. Allocated and populated in
     * {@link #normalize(int)}.
     */
    protected double[] affixTopicStateProbs;
    /**
     * For use in the word probability normalization stage. If an affix has
     * not been observed in the sampling process, and the affix is to be used
     * in conditioning stems conditioned on syntactic states, its probability will
     * be retrieved from this array based on the length of the affix.
     */
    protected double[] nonexistentStateAffixProbs;
    /**
     * For use in the word probability normalization stage. If an affix has
     * not been observed in the sampling process, and the affix is to be used
     * in conditioning stems conditioned on topic states, its probability will
     * be retrieved from this array based on the length of the affix.
     */
    protected double[] nonexistentTopicStateAffixProbs;

    /**
     * Class constructor. Assigns {@link #baseDistribution}, {@link #lexicon},
     * {@link #hyper}. Allocates {@link #morphClsCounts}.
     *
     * @param baseDistribution  The pre-allocated base distribution
     * @param lexicon   The pre-allocated lexicon
     * @param hyper Hyperparamter for this distribution
     */
    public ThreeDimDirichletProcess(DirichletBaseDistribution baseDistribution,
          Lexicon lexicon, double hyper) {
        this.lexicon = lexicon;
        this.hyper = hyper;
        this.baseDistribution = baseDistribution;
        morphClsCounts = new ThreeDimLexicon();
        morphClsProbs = new ThreeDimProbLexicon();
        stringProbs = baseDistribution.getStringProbs();
    }

    /**
     * Decrements the count of an morph in some class.
     * <p>
     * This decrements counts of the morph in {@link #baseDistribution},
     * the counts of the morph in {@link #morphClsCounts}, and the counts
     * of the morph in {@link #lexicon}.
     * 
     * @param cls   Index of the class
     * @param morph Index of the morph
     * @return  Count of the morph given the class
     */
    public int dec(int cls, int morph) {
        int val = 0;

        try {
            val = morphClsCounts.dec(cls, morph);
        } catch (EmptyCountException e) {
            e.printMessage(lexicon.getString(morph), morph);
            System.exit(1);
        } catch (EmptyTwoDimLexiconException e) {
        }

        try {
            lexicon.dec(morph);
        } catch (KeyRemovedException e) {
        }

        return val;
    }

    /**
     * Increment the count of an morph in some class.
     * <p>
     * This increments counts of the morph in {@link #baseDistribution},
     * the counts of the morph in {@link #morphClsCounts}, and the counts
     * of the morph in {@link #lexicon}.
     * 
     * @param cls   Index of class
     * @param morph Index of morph
     * @return  The count of an morph given a state after increment.
     */
    public int inc(int cls, int morph) {
        lexicon.inc(morph);
        return morphClsCounts.inc(cls, morph);
    }

    /**
     *
     * Get the probability of an morph given a state. Only used in the final
     * normalization stage after training is complete. Do NOT call this when
     * training!
     * <p>
     * This function calculates and returns:
     * <pre>
     * (N_{a|c} + hyper * P(a)) /
     * (N_c + hyper)
     * </pre>
     * 
     * @param cls   Index of the state
     * @param morph Index of the morph
     * @return      The probability of {@code P(a|c)}.
     */
    public double prob(int cls, int morph) {
        return (getCount(cls, morph) + baseDistribution.prob(morph))
              / (getCumCount(cls) + hyper);
    }

    /**
     * Get the probability of an morph given a state.
     * 
     * <p>
     * This function calculates and returns:
     * <pre>
     * (N_{a|c} + hyper * P(a)) /
     * (N_c + hyper)
     * </pre>
     * 
     * @param cls   Index of the state
     * @param morph Affix string
     * @return      The probability of {@code P(a|c)}.
     */
    public double prob(int cls, String morph) {
        return (getCount(cls, lexicon.getIdx(morph))
              + baseDistribution.prob(morph))
              / (getCumCount(cls) + hyper);
    }

    /**
     *
     * Get the numerator of the probabilty for an morph string given the
     * state. Used in the training stage.
     * <p>
     * This function calculates and returns:
     * <pre>
     * N_{a|c} + hyper * P(a)
     * </pre>
     *
     * @param cls   Index of the state
     * @param morph Affix string
     * @return      The value of {@code N_{a|c} + hyper * B_0(a)}
     */
    public double probNumerator(int cls, String morph) {

        return getCount(cls, lexicon.getIdx(morph)) + baseDistribution.prob(
              morph);
    }

    /**
     *
     * Get the numerator of the probabilty for an morph index given the
     * state. Only used in the final normalization stage
     * after training is complete. Do NOT call this when training!
     * <p>
     * This function calculates and returns:
     * <pre>
     * N_{a|c} + hyper * P(a)
     * </pre>
     *
     * @param cls   Index of the state
     * @param morph Index of the morph
     * @return  The value of {@code N_{a|c} + hyper * P(a)}
     * @see #normalize(int, int, double[]) 
     */
    public double probNumerator(int cls, int morph) {
        return getCount(cls, morph) + baseDistribution.prob(morph);
    }

    /**
     * Get the cumulative count of a given class.
     * 
     * @param cls   Index of the state
     * @return  The count of the state
     */
    public int getCumCount(int cls) {
        return morphClsCounts.getCumCount(cls);
    }

    /**
     * Get the count of an morph given a class.
     * 
     * @param cls   Index of the state
     * @param morph Index of the morph
     * @return  The count of the morph
     */
    public int getCount(int cls, int morph) {
        return morphClsCounts.get(cls, morph);
    }

    /**
     * Return probabilities of the affixes.
     * <p>
     * The probabilities were calculated by
     * <pre>
     * p(a) = \sum_c p(a|c) * p(c)
     * </pre>
     *
     * @return  Array of affix probabilties
     */
    public double[] getAffixTopicStateProbs() {
        return affixTopicStateProbs;
    }

    /**
     * Normalize sample counts.
     *
     * @param topicS    Number of topic states.
     * @param stateS    Total number of states. This excludes the first
     *                  state 0 (which is the sentence boundary) up to topicS
     *                  (which is the number of topic states).
     * @param outputPerState    How many affixes to print per state in
     *                  the output.
     * @param stateProbs    Array of probabilties for each state. The first
     *                  cell should be ignored.
     */
    public abstract void normalize(int topicS, int stateS, int outputPerState,
          double[] stateProbs);

    /**
     * Print normalized probability tables for affixes by topic.
     *
     * @param topicS    Number of topic states.
     * @param stateS    Total number of states. This excludes the first
     *                  state 0 (which is the sentence boundary) up to topicS
     *                  (which is the number of topic states).
     * @param outputPerState    How many affixes to print per state in
     *                  the output.
     * @param stateProbs Array of probabilties for each state. The first
     *                  cell should be ignored.
     * @param out   Destination of output
     */
    public abstract void print(int topicS, int stateS, int outputPerState,
          double[] stateProbs,
          BufferedWriter out) throws IOException;

    /**
     * Returns the probability of a morph given a class. This is only to be
     * called after training and normalization have completed.
     *
     * @param cls   Index of the class
     * @param morph Index of the morph
     * @return  Probability of morph given class
     */
    public double getConstProb(int cls, int morph) {
        return morphClsProbs.get(cls).get(morph);
    }

//    /**
//     * Probability of a given string.
//     *
//     * @param idx   Index of the string in the lexicon
//     * @return      The probability
//     */
//    public double prob(int idx) {
//        return prob(lexicon.getString(idx));
//    }
//
//    /**
//     * Probability of a given string. The probability of the string <pre>s</pre>
//     * is calculated to be
//     * <pre>
//     * P(#) * (1-P(#))^|s| * \prod_i^{|s|} P(s_i)
//     * </pre>
//     *
//     * @param s The string
//     * @return  The probability
//     */
//    public double prob(String s) {
//        double val = 0;
//        try {
//            val = stringProbs[s.length()];
//        } catch (ArrayIndexOutOfBoundsException e) {
//            System.err.println("\"" + s + "\" is longer than " + maxlen);
//            val = hyper * Math.exp(Math.log(morphBoundaryProb) + s.length() *
//                    (Math.log(notMorphBoundaryProb) + Math.log(ALPHAPROB)));
//        }
//        return val;
//    }
    /**
     * Sets array of probabilities for unobserved affixes conditioned on
     * syntactic states.
     * For use in the word probability normalization stage. If an affix has
     * not been observed in the sampling process, and the affix is to be used
     * in conditioning stems conditioned on syntactic states, its probability will
     * be retrieved from this array based on the length of the affix.
     *
     * @param clsProbs  Array of sample class probabilities
     * @param starti    Start index of array
     * @param endi      End index of array
     */
    public void setNonexistentStateAffixProbs(double[] clsProbs, int starti,
          int endi) {
        nonexistentStateAffixProbs = new double[maxlen];
        for (int i = 0; i < maxlen; ++i) {
            nonexistentStateAffixProbs[i] = 0;
        }
        for (int i = starti; i < endi; ++i) {
            double cprob = clsProbs[i];
            double cdenom = getCumCount(i) + hyper;
            double cmult = cprob / cdenom;
            for (int j = 0; j < maxlen; ++j) {
                double val = 0;
                double prob = stringProbs[j];
                val = prob * cmult;
                nonexistentStateAffixProbs[j] += val;
            }
        }
    }

    /**
     * Get array of probabilities for unobserved affixes conditioned on
     * syntactic states.
     *
     * @return Array of probabilities for unobserved affixes.
     */
    public double[] getNonexistentStateAffixProbs() {
        return nonexistentStateAffixProbs;
    }

    /**
     * Sets array of probabilities for unobserved affixes conditioned on
     * topic states.
     * For use in the word probability normalization stage. If an affix has
     * not been observed in the sampling process, and the affix is to be used
     * in conditioning stems conditioned on topic states, its probability will
     * be retrieved from this array based on the length of the affix.
     *
     * @param clsProbs Array of sample class probabilities
     * @param
     */
    public void setNonexistentTopicStateAffixProbs(double[] clsProbs, int starti,
          int endi) {
        nonexistentTopicStateAffixProbs = new double[maxlen];
        for (int i = 0; i < maxlen; ++i) {
            nonexistentTopicStateAffixProbs[i] = 0;
        }
        for (int i = starti; i < endi; ++i) {
            double cprob = clsProbs[i];
            double cdenom = getCumCount(i) + hyper;
            double cmult = cprob / cdenom;
            for (int j = 0; j < maxlen; ++j) {
                double val = 0;
                double prob = stringProbs[j];
                val = prob * cmult;
                nonexistentTopicStateAffixProbs[j] += val;
            }
        }
    }

    /**
     * Get array of probabilities for unobserved affixes conditioned on
     * topic states.
     *
     * @return Array of probabilities for unobserved affixes.
     */
    public double[] getNonexistentTopicStateAffixProbs() {
        return nonexistentTopicStateAffixProbs;
    }
}
