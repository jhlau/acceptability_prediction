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

/**
 * 
 * @author tsmoon
 */
public class StemAffixTopicHDP extends StemAffixTopicDP {

    /**
     *
     * @param baseDistribution
     * @param lexicon
     * @param hyper
     */
    public StemAffixTopicHDP(
            HierarchicalDirichletBaseDistribution baseDistribution,
            Lexicon lexicon, double hyper, int topics) {
        super(baseDistribution, lexicon, hyper, topics);
    }

    /**
     * Decrement counts for stem given affix and class.
     *
     * @param cls
     * @param affix
     * @param stem
     * @return
     */
    @Override
    public int dec(int cls, int affix, int stem) {
        int val = 0;
        try {
            baseDistribution.dec(lexicon.getString(stem));
        } catch (EmptyCountException e) {
            e.printMessage(lexicon.getString(stem), stem);
            System.exit(1);
        }

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
    @Override
    public int inc(int cls, int affix, int stem) {
        lexicon.inc(stem);
        baseDistribution.inc(lexicon.getString(stem));
        return stemAffixClsCounts.inc(cls, affix, stem);
    }
}
