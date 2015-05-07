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

import tikka.exceptions.KeyRemovedException;

import java.util.HashMap;
import java.util.Stack;

/**
 * A data structure which keeps track of indexes, their counts, the strings
 * that they represent, and available slots for new indexes.
 * 
 * @author tsmoon
 */
public class Lexicon extends HashMap<Integer, StringCountPair> {

    /**
     * A stack of available indexes when new strings are encountered. If old
     * indexes are vacated, they are pushed on to the stack. They are popped
     * when previously unknown strings occur. If the stack is empty, the size
     * of the hashmap is popped onto the stack.
     */
    protected Stack<Integer> availableIdx;
    /**
     * A map from strings to indexes.
     */
    protected HashMap<String, Integer> reverseMap;

    /**
     * Default constructor. The map from strings to indexes is shared throughout
     * the model and is allocated outside this class.
     * 
     * @param reverseMap    A map from strings to indexes
     */
    public Lexicon(HashMap<String, Integer> reverseMap) {
        this.reverseMap = reverseMap;
        availableIdx = new Stack<Integer>();
        availableIdx.push(0);
    }

    /**
     * Removes key when index counts are zero. Pushes the vacating index onto
     * the stack.
     * 
     * @param idx   The vacating index
     * @return  The vacating index
     */
    public int removeKey(int idx) {
        availableIdx.push(idx);
        reverseMap.remove(getString(idx));
        remove(idx);
        return availableIdx.peek();
    }

    /**
     * Decrements the key count. If the count is zero, removes the key. Throws
     * an exception so that additional removal procedures can be taken
     * in other structures.
     *
     * @param idx   Key to decrement
     * @return  Count of key after decrement
     * @throws KeyRemovedException
     */
    public int dec(int idx) throws KeyRemovedException {
        StringCountPair sc = get(idx);
        if (sc.count == 1) {
            removeKey(idx);
            throw new KeyRemovedException();
        }
        sc.count--;
        return sc.count;
    }

    /**
     * Increments key count
     *
     * @param idx   Key to increment
     * @return  Count of key after increment
     */
    public int inc(int idx) {
        StringCountPair sc = get(idx);
        try {
            sc.count++;
        } catch (NullPointerException e) {
            e.printStackTrace();
        }
        return sc.count;
    }

    /**
     * Get index value of string. If string is not in lexicon, return
     * negative one.
     * 
     * @param s String to look up.
     * @return  Index value of string.
     */
    public int getIdx(String s) {
        if (!reverseMap.containsKey(s)) {
            return -1;
        } else {
            return reverseMap.get(s);
        }
    }

    /**
     * Get or put index value of string. If string does no exist in lexicon, 
     * pop available index value from stack and place in lexicon. If stack
     * is empty after pop, push size of lexicon.
     *
     * @param s String to look up or deposit.
     * @return  Index value of string.
     */
    public int getOrPutIdx(String s) {
        if (!reverseMap.containsKey(s)) {
            int thisIdx = availableIdx.pop();
            put(thisIdx, new StringCountPair(s, 0));
            if (availableIdx.size() == 0) {
                availableIdx.push(size());
            }
            reverseMap.put(s, thisIdx);
            return thisIdx;
        } else {
            return reverseMap.get(s);
        }
    }

    /**
     * Get string corresponding to index. Assumes given index always exists
     * in lexicon.
     * 
     * @param idx   Index to look up
     * @return  Corresponding string
     */
    public String getString(int idx) {
        try {
            return get(idx).string;
        } catch (NullPointerException e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Occurrence count of index
     * 
     * @param idx   Index to look up
     * @return  Count of index
     */
    public int getCount(int idx) {
        return get(idx).count;
    }
}
