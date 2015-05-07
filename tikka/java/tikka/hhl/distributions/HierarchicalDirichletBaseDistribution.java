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

/**
 * The base distribution in a hierarchical dirichlet process
 * 
 * @author tsmoon
 */
public class HierarchicalDirichletBaseDistribution extends DirichletBaseDistribution {

    /**
     * Default constructor. Since the pmf for strings is uninformative,
     * an array of probabilities given string length is initialized here.
     * 
     * @param lexicon   Dictionary of strings and indexes
     * @param morphBoundaryProb Constant probability of a morpheme boundary
     * @param hyper Hyperparameter for base distribution
     */
    public HierarchicalDirichletBaseDistribution(Lexicon lexicon,
            double morphBoundaryProb,
            double hyper) {
        super(lexicon, morphBoundaryProb, hyper);

        this.morphBoundaryProb = morphBoundaryProb;
        notMorphBoundaryProb = 1 - morphBoundaryProb;
        this.hyper = hyper;
        this.lexicon = lexicon;
        stringProbs = new double[maxlen];
        for (int i = 0; i < maxlen; ++i) {
            double stringProb = Math.log(morphBoundaryProb) + i *
                    (Math.log(notMorphBoundaryProb) + Math.log(ALPHAPROB));
            stringProbs[i] = this.hyper * Math.exp(stringProb);
        }
    }

    /**
     * Decrements the count for a given string. Also decrements the cumulative
     * count. If count is zero, remove the key. This method assumes that the
     * key already exists in this table
     * 
     * @param s String to decrement
     * @return  Count after decrement
     * @throws EmptyCountException  Throw if count is negative
     */
    @Override
    public int dec(String s) throws EmptyCountException {
        int val = get(s) - 1;
        if (val > 0) {
            put(s, val);
        } else if (val == 0) {
            remove(s);
        } else {
            throw new EmptyCountException();
        }

        cumCount--;
        return val;
    }

    /**
     * Increments the count for a given string. Also increments the cumulative
     * count. If string does not exist in table, create entry and put count 1.
     * 
     * @param s String to increment
     * @return  Count after increment
     */
    @Override
    public int inc(String s) {
        if (!containsKey(s)) {
            put(s, 0);
        }
        int val = get(s) + 1;
        put(s, val);
        cumCount++;
        return val;
    }

//    /**
//     * Set the normalization factor for all probability calculations. This is
//     * a constant value in the base distribution.
//     *
//     * @param count The total count of all word types in the corpus
//     * @return
//     */
//    public double setDenominator(int count) {
//        return denominator = count - 1 + hyper;
//    }
    /**
     * Probability of a given string index. Assumes that the index
     * already exits in the lookup table.
     *
     * @param idx   Index of string
     * @return  Probability of string
     */
    @Override
    public double prob(int idx) {
        return prob(lexicon.getString(idx));
    }

    /**
     * Probability of a given string. Does not assume that the string exits
     * in the lookup table. If the string does not exist in the table, the prior
     * probability of the string will be:
     * <pre>
     * (hyper*P(#)*(1-P(#))^|s| \prod_i^{|s|} P(s_i)) / (cumCount + hyper)
     * </pre>
     *
     * @param s
     * @return
     */
    @Override
    public double prob(String s) {
        double stringProb = 0;
        try {
            stringProb = stringProbs[s.length()];
        } catch (ArrayIndexOutOfBoundsException e) {
            System.err.println("\"" + s + "\" is longer than " + maxlen);
            stringProb = hyper * Math.exp(Math.log(morphBoundaryProb) + s.length() *
                    (Math.log(notMorphBoundaryProb) + Math.log(ALPHAPROB)));
        }

        int sCount = 0;
        try {
            sCount = get(s);
        } catch (NullPointerException e) {
            sCount = 0;
        }
        return (sCount + stringProb) / (cumCount + hyper);
    }
}
