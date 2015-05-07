///////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2010 Taesun Moon <tsunmoon@gmail.com>
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
///////////////////////////////////////////////////////////////////////////////
package tikka.utils.postags;

import java.util.HashMap;
import java.util.HashSet;

/**
 * Class is needed to facilitate evaluation in Evaluator class
 *
 * @author tsmoon
 */
public class PennTagsCE extends PennTags {

    public PennTagsCE(int _modelTagSize) {
        contentTagSet = pennContentTagSet;
        functionTagSet = pennFunctionTagSet;
        fullTagSet = pennFullTagSet;

        initializeReduced(_modelTagSize);
        reduceTag();
    }

    @Override
    public int getTagSetSize() {
        return reducedTagSet.size();
    }

    @Override
    protected HashSet<String> reduceTag() {
        super.reduceTag();
        idxToReducedTag = new HashMap<Integer, String>();
        setIdxMap(reducedTagSet, idxToReducedTag);
        return reducedTagSet;
    }
}
