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

import java.util.regex.Pattern;
import tikka.exceptions.IgnoreTagException;

/**
 *
 * @author tsmoon
 */
public class WordNormalizerToLowerNoNum extends WordNormalizer {

    @Override
    public String[] normalize(String[] strings) throws IgnoreTagException {
        this.strings = new String[strings.length];
//        String reducedTag = "", word = "";
        try {
            reducedTag = strings[1];
            pattern = Pattern.compile("^\\w.*$");
            matcher = pattern.matcher(reducedTag);
            if (!matcher.find()) {
                reducedTag = "";
            }
            this.strings[1] = reducedTag;
//            this.strings[1] = tagMap.getTag(reducedTag);
        } catch (ArrayIndexOutOfBoundsException e) {
            reducedTag = null;
        }

        if (reducedTag == null || !reducedTag.isEmpty()) {
            word = strings[0].toLowerCase();
            pattern = Pattern.compile("^\\W*$");
            matcher = pattern.matcher(word);
            if (!matcher.find()) {
                pattern = Pattern.compile("\\d\\d*");
                matcher = pattern.matcher(word);
                if (matcher.find()) {
                    word = "#";
                }
            } else {
                word = "";
            }
        } else {
            word = "";
        }
        this.strings[0] = word;

        if (tagMap.isIgnoreTag(fullTag)) {
            throw new IgnoreTagException(word, fullTag);
        }

        return this.strings;
    }
}
