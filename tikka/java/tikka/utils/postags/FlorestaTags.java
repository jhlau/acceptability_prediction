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

import java.util.Arrays;
import java.util.HashSet;

/**
 *
 * @author Taesun Moon <tsunmoon@gmail.com>
 */
public class FlorestaTags extends TagMap {

    protected final HashSet<String> florestaFullTagSet = new HashSet<String>(Arrays.asList(
          ",",
          ".", //
          "adj", // adjective
          "adv", // adverb
          "art", // article
          // "artd", // ** two occurrences of "as" . map this to art
          "conj-c", // coordinating conjunction
          "conj-s", // subordinating conjunction
          // "cuv-inf", // ** one occurrence of "ter". map this to v-inf
          // "det", // ** one occurrence of "os". map this to art
          // "ec", // ** five occurrences, one each of "anti", "ex", "pós", "ex", "pr?". map to n
          "in", // interjection
          "n", // noun
          "num", // numberal
          // "pp", // ** map this to adv
          // "prob-indp", // ** one occurrence of "que". map to pron-indp
          "pron-det", // determiner pronoun
          // "pron-ind", // ** one occurrence of "que". map to pron-indp
          "pron-indp", // independent pronoun.
          "pron-pers", // personal pronoun
          // "pron-rel", // ** one occurrence of "que". map to pron-indp
          "prop", // proper noun
          "prp", // preposition
          // "um", // ** one occurrence of "um". map to art
          "v-fin", // finite verb
          "v-ger", // gerund
          "v-inf", // infinitive 
          "v-pcp" // participle
          // "vfin", // ** three occurrences. map to v-fin
          // "vp" // ** two occurrences of "existente" and "subordinanda". map to adj
          ));

    /**
     *
     * @param _modelTagSize
     */
    public FlorestaTags(int _modelTagSize) {
        super(_modelTagSize);
        contentTagSet = florestaFullTagSet;
        functionTagSet = florestaFullTagSet;
        fullTagSet = florestaFullTagSet;
        initializeFull(_modelTagSize);
    }

    protected FlorestaTags() {
    }

    @Override
    protected HashSet<String> reduceTag() {

        for (String tag : fullTagSet) {
            fullTagToReducedTag.put(tag, tag);
        }
        return reducedTagSet;
    }
}
