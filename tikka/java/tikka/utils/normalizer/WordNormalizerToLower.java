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
package tikka.utils.normalizer;

import tikka.exceptions.IgnoreTagException;
import tikka.utils.postags.TagMap;

/**
 *
 * @author tsmoon
 */
public class WordNormalizerToLower extends WordNormalizer {

    public WordNormalizerToLower(TagMap tagMap) {
        super(tagMap);
    }

    @Override
    public String[] normalize(String[] strings) throws IgnoreTagException {
        this.strings = new String[strings.length];

        try {
            fullTag = strings[1];
            reducedTag = tagMap.getReducedTag(strings[1]);

            word = strings[0].toLowerCase();

            if (tagMap.isIgnoreTag(fullTag)) {
                throw new IgnoreTagException(word, fullTag);
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            reducedTag = null;
            word = "";
        }
        return this.strings;
    }
}
