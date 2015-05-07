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
 * Table of string type indexes and their counts. The indexes maybe affixes or
 * stems.
 * 
 * @author tsmoon
 */
public class TwoDimLexicon extends HashMap<Integer, Integer> {

    /**
     * Cumulative count of all items in this table
     */
    protected int cumCount = 0;

    /**
     * Increment counts of an index. Create key if index does not exist and
     * put count at one. Increment cumulative count as well.
     *
     * @param idx Index of type to increment
     * @return  Count of index
     */
    public int inc(int idx) {
        if (!containsKey(idx)) {
            put(idx, 0);
        }
        cumCount++;
        int val = get(idx) + 1;
        put(idx, val);
        return val;
    }

    /**
     * Decrement counts of an index. Remove key if index count is zero. 
     * Decrement cumulative count as well. Throw
     * {@link #EmptyTwoDimLexiconException} if cumulative count is zero so entire
     * table may be removed.
     *
     * @param idx   Index of type to decrement
     * @return  Count of index
     * @throws EmptyCountException Thrown if count of an index is negative
     * @throws EmptyTwoDimLexiconException  Thrown if cumulative count is zero
     */
    public int dec(int idx) throws EmptyCountException,
            EmptyTwoDimLexiconException {
        cumCount--;
        int val = get(idx) - 1;
        if (val > 0) {
            put(idx, val);
        } else if (val == 0) {
            remove(idx);
        } else {
            throw new EmptyCountException();
        }
        if (cumCount == 0) {
            throw new EmptyTwoDimLexiconException();
        }
        return val;
    }

    /**
     * Get count of index. Return 0 if index does not exist
     *
     * @param idx   Index of type count to retrive
     * @return  Count of index
     */
    public int getCount(int idx) {
        if (!containsKey(idx)) {
            return 0;
        } else {
            return get(idx);
        }
    }

    /**
     * Get cumulative count of table
     * 
     * @return  cumulative count of table
     */
    public int getCumCount() {
        return cumCount;
    }
}
