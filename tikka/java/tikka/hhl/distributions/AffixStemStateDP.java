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

import tikka.structures.StringDoublePair;
import tikka.hhl.lexicons.Lexicon;

/**
 * Dirichlet process that models dependency from affix to stem. Affixes are
 * dependent on stems and the states.
 *
 * @author tsmoon
 */
public class AffixStemStateDP extends FourDimDirichletProcess {

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
    protected StringDoublePair[][] TopAffixesPerState;
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
    public AffixStemStateDP(
            DirichletBaseDistribution baseDistribution,
            Lexicon lexicon, double hyper, int states) {
        super(baseDistribution, lexicon, hyper, states);
    }
}
