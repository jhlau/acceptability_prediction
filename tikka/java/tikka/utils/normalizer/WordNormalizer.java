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

import tikka.utils.postags.TagMap;

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import tikka.exceptions.IgnoreTagException;

/**
 *
 * @author tsmoon
 */
public class WordNormalizer {

    protected Pattern pattern;
    protected Matcher matcher;
    protected String[] strings;
    protected String word, reducedTag, fullTag;
    protected TagMap tagMap;

    public WordNormalizer(TagMap tagMap) {
        this.tagMap = tagMap;
    }

    public WordNormalizer() {
    }

    public String[] normalize(String[] strings) throws IgnoreTagException {
        this.strings = new String[strings.length];
        try {
            fullTag = strings[1];
            this.strings[1] = reducedTag = tagMap.getReducedTag(fullTag);
        } catch (ArrayIndexOutOfBoundsException e) {
            reducedTag = null;
        }

        if (reducedTag == null || !reducedTag.isEmpty()) {
            word = strings[0];
            pattern = Pattern.compile("^\\W*$");
            matcher = pattern.matcher(word);
            if (!matcher.find()) {
                pattern = Pattern.compile("(^\\W*(\\w.*\\w)\\W*$|(^\\w+$)|.*)");
                matcher = pattern.matcher(word);
                word = matcher.replaceAll("$2$3").toLowerCase();
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

    static public String normalize(String s) {
        Pattern p = Pattern.compile("^\\W*$");
        Matcher m = p.matcher(s);
        if (!m.find()) {
            p = Pattern.compile("(^\\W*(\\w.*\\w)\\W*$|(^\\w+$)|.*)");
            m = p.matcher(s);
            return m.replaceAll("$2$3");
        } else {
            return "";
        }
    }

    public String getWord() {
        return word;
    }

    public String getTag() {
        return fullTag;
    }

    public String getReducedTag() {
        return tagMap.getReducedTag(fullTag);
    }
}
