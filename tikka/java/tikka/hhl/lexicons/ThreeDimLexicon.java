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
import tikka.exceptions.KeyRemovedException;
import tikka.exceptions.EmptyTwoDimLexiconException;
import java.util.HashMap;

/**
 * Table for of type indexes (either state, topic, or affix) to
 * TwoDimLexicon (or subtables). 
 *
 * @author tsmoon
 */
public class ThreeDimLexicon extends HashMap<Integer, TwoDimLexicon> {

    /**
     * Cumulative count of all items in table as well as subtables (i.e. the
     * TwoDimLexicons)
     */
    protected int cumCount = 0;

    /**
     * Increment counts of a conditioned type index (a stem or affix) given another
     * conditioning string type index (a class or affix). If the index for the conditioning type does not
     * exist, create a new subtable for it and increment all counts. Increment
     * the cumulative count as well.
     * 
     * @param cls Index of conditioning type
     * @param idx  Index of conditioned string
     * @return  Count of conditioned string after increment
     */
    public int inc(int cls, int idx) {
        if (!containsKey(cls)) {
            TwoDimLexicon lex = new TwoDimLexicon();
            put(cls, lex);
        }
        cumCount++;
        return get(cls).inc(idx);

    }

    /**
     * Decrement counts of a conditioned string type index (a stem or affix) given
     * another conditioning string type index (a class or affix). Decrement
     * the cumulative count as well. If the subtable throws an exception, catch
     * it and pass it on to the calling method.
     *
     * @param cls Index of conditioning type
     * @param idx  Index of conditioned string
     * @return  Count of conditioned string after decrement
     * @throws EmptyCountException  Thrown if count of a conditioned string is negative
     * @throws EmptyTwoDimLexiconException  Thrown if cumulative count of subtable is zero
     */
    public int dec(int cls, int idx) throws EmptyCountException,
            EmptyTwoDimLexiconException {
        cumCount--;
        return get(cls).dec(idx);
    }

    /**
     * Get cumulative count of subtable for conditioning type index. Return 0 if
     * index has no entry
     * 
     * @param cls Index of conditioning type
     * @return  Cumulative count of subtable
     */
    public int getCumCount(int cls) {
        if (!containsKey(cls)) {
            return 0;
        } else {
            return get(cls).getCumCount();
        }
    }

    /**
     * Get count of conditioned string index given conditioning type index.
     * Return 0 if either conditioned string index
     * has no entry in the subtable or conditioning type index
     * has no entry in this table.
     * 
     * @param cls Index of the conditioning type
     * @param idx  Index of the conditioned string
     * @return  Count of conditioned string
     */
    public int get(int cls, int idx) {
        if (!containsKey(cls)) {
            return 0;
        } else {
            return get(cls).getCount(idx);
        }
    }
}
