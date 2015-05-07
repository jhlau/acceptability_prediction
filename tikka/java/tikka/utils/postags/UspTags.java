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
public class UspTags extends TagMap {

    protected final HashSet<String> uspFullTagSet = new HashSet<String>(Arrays.asList(
          "***",
          "3S",
          "???",
          "A1P",
          "A1S", // singular first person absolutive
          "A2S",
          "ABS",
          "ADJ", // adjective
          "ADV", // adverb
          "ADVC",
          "AFE",
          "AFI",
          "AGT",
          "AP", // antipassive
          "APELL",
          "APLI",
          "ART",
          "B'IK",
          "CARD",
          "CAU",
          "CLAS",
          "COM",
          "COND",
          "CONJ",
          "DEM", // demonstrative
          "DIM",
          "DIR",
          "E1",
          "E1P",
          "E1S", // singular first person ergative
          "E2",
          "E2P",
          "E2S",
          "E3",
          "E3P",
          "E3S", // singular third person ergative
          "ENF",
          "ESP", // spanish loan word
          "EXS", // existential
          "GEN",
          "GNT", // gentilicio (demonym)
          "IMP",
          "INC", // incompletive
          "INS",
          "INT",
          "ITR",
          "ITS",
          "MED",
          "MOV",
          "NC",
          "NEG", // negation,
          "NOM",
          "NUM",
          "PART", // particle
          "PAS",
          "PERS", // person marking
          "PL",
          "POS",
          "PP",
          "PRE",
          "PREP", // preposition
          "PRG",
          "PROG",
          "PRON", // pronoun
          "PRONA",
          "REC",
          "REL",
          "RFX",
          "S", // sustantivo (noun)
          "SAB",
          "SC", // category suffix
          "SR", // relational noun
          "SREL", // relational noun
          "SUB",
          "SUF", // suffix
          "SV",
          "TAM", // tense/aspect/mood
          "TOP",
          "TR",
          "TRN",
          "VI", // intransitive verb
          "VOC",
          "VT" // transitive verb
          ));

    /**
     *
     * @param _modelTagSize
     */
    public UspTags(int _modelTagSize) {
        super(_modelTagSize);
        contentTagSet = uspFullTagSet;
        functionTagSet = uspFullTagSet;
        fullTagSet = uspFullTagSet;
        initializeFull(_modelTagSize);
    }

    protected UspTags() {
    }

    @Override
    protected HashSet<String> reduceTag() {

        for (String tag : fullTagSet) {
            fullTagToReducedTag.put(tag, tag);
        }
        return reducedTagSet;
    }
}
