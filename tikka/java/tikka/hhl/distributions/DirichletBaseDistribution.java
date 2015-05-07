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
import tikka.hhl.lexicons.Lexicon;
import java.util.HashMap;

/**
 * The base distribution for a dirichlet process.
 * 
 * @author tsmoon
 */
public class DirichletBaseDistribution extends HashMap<String, Integer> {

    /**
     * Cumulative count of all items in dictionary/distribution. Used in
     * normalization
     */
    protected int cumCount = 0;
    /**
     * Assumed maximum length of string types that will be observed in corpus
     */
    protected int maxlen = 100;
    /**
     * Array of probabilities for strings given the length. Initialized at
     * construction and never changed.
     */
    protected double[] stringProbs;
    /**
     * Probability of a morpheme boundary. Equivalent to probability of empty
     * string.
     */
    protected double morphBoundaryProb;
    /**
     * Equivalent to <pre>1-morphBoundaryProb</pre>
     */
    protected double notMorphBoundaryProb;
    /**
     * Assumed size of the alphabet.
     */
    protected static final int ALPHASIZE = 26;
    /**
     * Uniform probability of alphabet
     */
    protected static final double ALPHAPROB = 1.0 / ALPHASIZE;
    /**
     * Dictionary of string types to indexes and back. Also maintains counts.
     */
    protected Lexicon lexicon;
    /**
     * Hyperparameter for the dirichlet distribution
     */
    protected double hyper;

    /**
     * Default constructor. Since the pmf for strings is uninformative,
     * an array of probabilities given string length is initialized here.
     * 
     * @param lexicon   Dictionary of strings and indexes
     * @param morphBoundaryProb Constant probability of a morpheme boundary
     * @param hyper Hyperparameter for base distribution
     */
    public DirichletBaseDistribution(Lexicon lexicon, double morphBoundaryProb,
          double hyper) {
        this.morphBoundaryProb = morphBoundaryProb;
        notMorphBoundaryProb = 1 - morphBoundaryProb;
        this.hyper = hyper;
        this.lexicon = lexicon;
        stringProbs = new double[maxlen];
        for (int i = 0; i < maxlen; ++i) {
            double stringProb = Math.log(morphBoundaryProb) + i
                  * (Math.log(notMorphBoundaryProb) + Math.log(ALPHAPROB));
            stringProbs[i] = this.hyper * Math.exp(stringProb);
        }
    }

    public int dec(String s) throws EmptyCountException {
        throw new UnsupportedOperationException("Don't use this!");
    }

    public int inc(String s) {
        throw new UnsupportedOperationException("Don't use this!");
    }

//    /**
//     * Do not call this.
//     *
//     * @param count
//     * @return
//     */
//    @Override
//    public double setDenominator(int count) {
//        return denominator = count - 1 + hyper;
//    }
    /**
     * The probability of a string as indentified by its index. This assumes
     * that the string is already in the lexicon
     *
     * @param idx   Index of the string
     * @return  The probability of the string
     */
    public double prob(int idx) {
        return prob(lexicon.getString(idx));
    }

    /**
     * Get the likelihood of a string. It is a geometric distribution on the length of the string
     * and the morpheme boundary probability.
     * 
     * @param s The string input
     * @return  The likelihood of the string
     */
    public double prob(String s) {
        double stringProb = 0;
        try {
            stringProb = stringProbs[s.length()];
        } catch (ArrayIndexOutOfBoundsException e) {
            System.err.println("\"" + s + "\" is longer than " + maxlen);
            stringProb = hyper * Math.exp(Math.log(morphBoundaryProb) + s.length()
                  * (Math.log(notMorphBoundaryProb) + Math.log(ALPHAPROB)));
        }
        return stringProb;
    }

    /**
     * Get the probability array of strings given length after initialization.
     *
     * @return Probability array of strings given length
     */
    public double[] getStringProbs() {
        return stringProbs;
    }
}
