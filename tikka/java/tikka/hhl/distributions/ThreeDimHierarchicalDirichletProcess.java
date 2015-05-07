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
import tikka.structures.StringDoublePair;
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
public abstract class ThreeDimHierarchicalDirichletProcess extends ThreeDimDirichletProcess {

    /**
     * Class constructor. Assigns {@link #baseDistribution}, {@link #lexicon},
     * {@link #hyper}. Allocates {@link #morphClsCounts}.
     *
     * @param baseDistribution  The pre-allocated base distribution
     * @param lexicon   The pre-allocated lexicon
     * @param hyper Hyperparamter for this distribution
     */
    public ThreeDimHierarchicalDirichletProcess(HierarchicalDirichletBaseDistribution baseDistribution,
            Lexicon lexicon, double hyper) {
        super(baseDistribution, lexicon, hyper);
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
    @Override
    public int dec(int cls, int morph) {
        int val = 0;
        try {
            baseDistribution.dec(lexicon.getString(morph));
        } catch (EmptyCountException e) {
            e.printMessage(lexicon.getString(morph), morph);
            System.exit(1);
        }

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
    @Override
    public int inc(int cls, int morph) {
        lexicon.inc(morph);
        baseDistribution.inc(lexicon.getString(morph));
        return morphClsCounts.inc(cls, morph);
    }
}
