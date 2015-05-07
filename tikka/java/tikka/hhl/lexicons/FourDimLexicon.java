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
package tikka.hhl.lexicons;

import tikka.exceptions.EmptyCountException;
import tikka.exceptions.EmptyTwoDimLexiconException;
import java.util.HashMap;

/**
 * A table from a class index (either state or topic) to a subtable of affix
 * type indexes that again point to a subtable of counts for stem type
 * indexes. Suitable increment, decrement, and get functions are defined.
 *
 * @author tsmoon
 */
public class FourDimLexicon extends HashMap<Integer, ThreeDimLexicon> {

    /**
     * Create a sublexicon for every class
     * 
     * @param classes Number of classes to hand over
     */
    public FourDimLexicon(int classes) {
        for (int cls = 0; cls < classes; ++cls) {
            ThreeDimLexicon lex = new ThreeDimLexicon();
            put(cls, lex);
        }
    }

    /**
     * Increment counts of a stem index given an affix index and a class index.
     * If the class has not been observed previously, create new subtable
     * for the class. Then increment.
     * 
     * @param cls   Index of class
     * @param affix Index of affix
     * @param stem  Index of stem
     * @return  Count of stem given affix and class after increment
     */
    public int inc(int cls, int affix, int stem) {
        if (!containsKey(cls)) {
            ThreeDimLexicon lex = new ThreeDimLexicon();
            put(cls, lex);
        }
        return get(cls).inc(affix, stem);
    }

    /**
     * Decrement counts of a stem index given an affix index and a class index.
     * If the class has not been observed previously, create new subtable
     * for the class. Then increment.
     * 
     * @param cls
     * @param affix
     * @param stem
     * @return
     * @throws EmptyCountException
     * @throws EmptyTwoDimLexiconException
     */
    public int dec(int cls, int affix, int stem) throws
          EmptyCountException, EmptyTwoDimLexiconException {
        return get(cls).dec(affix, stem);
    }

    /**
     * Get joint count of classes (either state or topic) and affixes. The
     * counts are used in normalizing counts of stems conditioned affixes
     * and classes. This function returns the cumulative count of the subtable conditioned
     * on the class and the affix. Returns 0 if either the class or the affix
     * has no subtable
     *
     * @param cls   Index of class
     * @param affix Index of affix
     * @return  Cumulative count of subtable conditioned on class and affix
     */
    public int getCumCount(int cls, int affix) {
        if (!containsKey(cls)) {
            return 0;
        } else {
            return get(cls).getCumCount(affix);
        }
    }

    /**
     * Get count of stem conditioned on affix and class (either state or topic).
     * Returns 0 if either the class or the affix or the stem have previously
     * not been sampled.
     *
     * @param cls   Index of class
     * @param affix Index of affix
     * @param stem  Index of stem
     * @return  Count of stem conditioned on affix and class
     */
    public int get(int cls, int affix, int stem) {
        try {
            return get(cls).get(affix).getCount(stem);
        } catch (NullPointerException e) {
            return 0;
        }
    }
}
