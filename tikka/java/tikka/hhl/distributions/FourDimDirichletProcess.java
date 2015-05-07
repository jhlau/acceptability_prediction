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
import tikka.hhl.lexicons.FourDimLexicon;
import tikka.hhl.lexicons.FourDimProbLexicon;
import tikka.hhl.lexicons.Lexicon;

/**
 *
 * @author tsmoon
 */
public class FourDimDirichletProcess extends DirichletProcess {

    /**
     * 
     */
    protected FourDimLexicon stemAffixClsCounts;
    /**
     * 
     */
    protected FourDimProbLexicon stemAffixClsProbs;

    /**
     * Default constructor.
     * 
     * @param baseDistribution
     * @param lexicon
     * @param hyper
     */
    public FourDimDirichletProcess(DirichletBaseDistribution baseDistribution,
            Lexicon lexicon, double hyper, int classes) {
        this.lexicon = lexicon;
        this.hyper = hyper;
        this.baseDistribution = baseDistribution;
        stemAffixClsCounts = new FourDimLexicon(classes);
        stemAffixClsProbs = new FourDimProbLexicon();
        stringProbs = baseDistribution.getStringProbs();
    }

    /**
     * Decrement counts for stem given affix and class.
     * 
     * @param cls
     * @param affix
     * @param stem
     * @return
     */
    public int dec(int cls, int affix, int stem) {
        int val = 0;

        try {
            val = stemAffixClsCounts.dec(cls, affix, stem);
        } catch (EmptyCountException e) {
            e.printMessage(lexicon.getString(stem), stem);
            System.exit(1);
        } catch (EmptyTwoDimLexiconException e) {
            stemAffixClsCounts.get(cls).remove(affix);
        }

        try {
            lexicon.dec(stem);
        } catch (KeyRemovedException e) {
        }

        return val;
    }

    /**
     * Increment counts for stem given affix and class.
     * 
     * @param cls
     * @param affix
     * @param stem
     * @return
     */
    public int inc(int cls, int affix, int stem) {
        lexicon.inc(stem);
        return stemAffixClsCounts.inc(cls, affix, stem);
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
    public double prob(int cls, int affix, int stem) {
        return (getCount(cls, affix, stem) + baseDistribution.prob(stem)) /
                (getCumCount(cls, affix) + hyper);
    }

    /**
     * 
     * @param cls
     * @param affix
     * @param stem
     * @return
     */
    public double prob(int cls, int affix, String stem) {
        int clsAffixStemCount = 0, clsAffixCount = 0;
        if (affix != -1) {
            clsAffixCount = getCumCount(cls, affix);
            clsAffixStemCount = getCount(cls, affix, lexicon.getIdx(stem));
        }
        return (clsAffixStemCount + baseDistribution.prob(stem)) /
                (clsAffixCount + hyper);
    }

    /**
     * Returns the probability of a stem given a class and affix. This is
     * only to be called after training and normalization have completed.
     *
     * @param cls   Index of the class
     * @param affix Index of the affix
     * @param stem  Index of the stem
     * @return  Probability of the stem
     */
    public double getConstProb(int cls, int affix,
            int stem) {
        return stemAffixClsProbs.get(cls).get(affix).get(stem);
    }

    /**
     * For use in the normalization stage
     * <p>
     * Calculates and returns
     * <pre>
     * N_{a|c} + hyper
     * </pre>
     *
     * @param cls   Index of class, either state or topic
     * @param affix Index of affix
     * @return  Denominator in a probability
     */
    public double probDenominator(
            int cls, int affix) {
        return getCumCount(cls, affix) + hyper;
    }

    /**
     * Cumulative count of affix and class (which can be either topic or stem).
     * <p>
     * Returns the following value
     * <pre>
     * N_{a,c}
     * </pre>
     * 
     * @param cls   Index of class, either state or topic
     * @param affix Index of affix
     * @return Count of affix and class
     */
    public int getCumCount(int cls, int affix) {
        return stemAffixClsCounts.getCumCount(cls, affix);
    }

    /**
     * Count of stem given affix and class (which can be either topic or
     * stem).
     * <p>
     * Returns the following value
     * <pre>
     * N_{s|a,c}
     * </pre>
     *
     * @param cls   Index of class, either state or topic
     * @param affix Index of affix
     * @param stem  Index of stem
     * @return  Count of stem given affix and class
     */
    public int getCount(int cls, int affix, int stem) {
        return stemAffixClsCounts.get(cls, affix, stem);
    }

    /**
     * Probability of a given string. This assumes that the index is
     * already in the lexicon.
     * 
     * @param idx   Index of the string in the lexicon
     * @return      The probability
     */
    public double prob(int idx) {
        return prob(lexicon.getString(idx));
    }

    /**
     * Probability of a given string. The probability of the string <pre>s</pre>
     * is calculated to be
     * <pre>
     * P(#) * (1-P(#))^|s| * \prod_i^{|s|} P(s_i)
     * </pre>
     * 
     * @param s The string
     * @return  The probability
     */
    public double prob(String s) {
        double val = 0;
        try {
            val = stringProbs[s.length()];
        } catch (ArrayIndexOutOfBoundsException e) {
            System.err.println("\"" + s + "\" is longer than " + maxlen);
            val = baseDistribution.prob(s);
        }
        return val;
    }
}
