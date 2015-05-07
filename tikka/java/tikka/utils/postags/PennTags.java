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
import java.util.HashMap;
import java.util.HashSet;

/**
 *
 * @author tsmoon
 */
public class PennTags extends TagMap {

    protected final HashSet<String> pennContentTagSet = new HashSet<String>(Arrays.asList(
          "FW", //Foreign word
          "JJ", //Adjective
          "JJR", //Adjective, comparative
          "JJS", //Adjective, superlative
          "NN", //Noun, singular or mass
          "NNS", //Noun, plural
          "NNP", //Proper noun, singular
          "NNPS", //Proper noun, plural
          "RB", //Adverb
          "RBR", //Adverb, comparative
          "RBS", //Adverb, superlative
          "VB", //Verb, base form
          "VBD", //Verb, past tense
          "VBG", //Verb, gerund or present participle
          "VBN", //Verb, past participle
          "VBP", //Verb, non-3rd person singular present
          "VBZ" //Verb, 3rd person singular present
          ));
    protected final HashSet<String> pennFunctionTagSet = new HashSet<String>(Arrays.asList(
          //              "$", //dollar $ -$ --$ A$ C$ HK$ M$ NZ$ S$ U.S.$ US$
          "``", //opening quotation mark ` ``
          "''", //closing quotation mark ' ''
          //              "(", //opening parenthesis ( [ {
          //              ")", //closing parenthesis ) ] }
          ",", //comma ,
          //              "-- ", //dash --
          ".", //sentence terminator . ! ?
          ":", //colon or ellipsis : ; ...
          "$", //dollar sign
          "-RRB-", // Right braces/brackets
          "-LRB-", // Left braces/brackets
          "CC", //Coordinating conjunction
          "CD", //Cardinal number
          "DT", //Determiner
          "EX", //Existential there
          "IN", //Preposition or subordinating conjunction
          "LS", //List item marker
          "MD", //Modal
          "PDT", //Predeterminer
          "POS", //Possessive ending
          "PRP", //Personal pronoun
          "PRP$", //Possessive pronoun
          "RP", //Particle
          "SYM", //Symbol
          "TO", //to
          "UH", //Interjection
          "WDT", //Wh-determiner
          "WP", //Wh-pronoun
          "WP$", //Possessive wh-pronoun
          "WRB" //Wh-adverb
          ));
    protected final HashSet<String> pennFullTagSet = new HashSet<String>(Arrays.asList(
          //              "$", //dollar $ -$ --$ A$ C$ HK$ M$ NZ$ S$ U.S.$ US$
          "``", //opening quotation mark ` ``
          "''", //closing quotation mark ' ''
          //              "(", //opening parenthesis ( [ {
          //              ")", //closing parenthesis ) ] }
          ",", //comma ,
          //              "-- ", //dash --
          ".", //sentence terminator . ! ?
          ":", //colon or ellipsis : ; ...
          "$", //dollar sign
          "-RRB-", // Right braces/brackets
          "-LRB-", // Left braces/brackets
          "CC", //Coordinating conjunction
          "CD", //Cardinal number
          "DT", //Determiner
          "EX", //Existential there
          "FW", //Foreign word
          "IN", //Preposition or subordinating conjunction
          "JJ", //Adjective
          "JJR", //Adjective, comparative
          "JJS", //Adjective, superlative
          "LS", //List item marker
          "MD", //Modal
          "NN", //Noun, singular or mass
          "NNS", //Noun, plural
          "NNP", //Proper noun, singular
          "NNPS", //Proper noun, plural
          "PDT", //Predeterminer
          "POS", //Possessive ending
          "PRP", //Personal pronoun
          "PRP$", //Possessive pronoun
          "RB", //Adverb
          "RBR", //Adverb, comparative
          "RBS", //Adverb, superlative
          "RP", //Particle
          "SYM", //Symbol
          "TO", //to
          "UH", //Interjection
          "VB", //Verb, base form
          "VBD", //Verb, past tense
          "VBG", //Verb, gerund or present participle
          "VBN", //Verb, past participle
          "VBP", //Verb, non-3rd person singular present
          "VBZ", //Verb, 3rd person singular present
          "WDT", //Wh-determiner
          "WP", //Wh-pronoun
          "WP$", //Possessive wh-pronoun
          "WRB" //Wh-adverb
          ));
    protected final HashSet<String> pennIgnoreSet = new HashSet<String>(Arrays.asList(
          "-NONE-"
          ));

    public PennTags(int _modelTagSize) {
        super(_modelTagSize);
        contentTagSet = pennContentTagSet;
        functionTagSet = pennFunctionTagSet;
        fullTagSet = pennFullTagSet;
        initializeFull(_modelTagSize);
        ignoreSet = pennIgnoreSet;
    }

    protected PennTags() {
    }

    @Override
    protected HashSet<String> reduceTag() {
        fullTagToReducedTag.put("CD", "ADJ");
        fullTagToReducedTag.put("JJ", "ADJ");
        fullTagToReducedTag.put("JJR", "ADJ");
        fullTagToReducedTag.put("JJS", "ADJ");
        fullTagToReducedTag.put("PRP$", "ADJ");
        fullTagToReducedTag.put("CC", "CONJ");
        fullTagToReducedTag.put(".", "ENDPUNC");
        fullTagToReducedTag.put("-LRB-", "LPUNC");
        fullTagToReducedTag.put("``", "LPUNC");
        fullTagToReducedTag.put("$", "LPUNC");
        fullTagToReducedTag.put("POS", "POS");
        fullTagToReducedTag.put("RP", "PRT");
        fullTagToReducedTag.put("TO", "TO");
        fullTagToReducedTag.put("MD", "V");
        fullTagToReducedTag.put("VBD", "V");
        fullTagToReducedTag.put("VBP", "V");
        fullTagToReducedTag.put("VB", "V");
        fullTagToReducedTag.put("VBZ", "V");
        fullTagToReducedTag.put("VBG", "VBG");
        fullTagToReducedTag.put("RB", "ADV");
        fullTagToReducedTag.put("RBR", "ADV");
        fullTagToReducedTag.put("RBS", "ADV");
        fullTagToReducedTag.put("DT", "DET");
        fullTagToReducedTag.put("PDT", "DET");
        fullTagToReducedTag.put(",", "INPUNC");
        fullTagToReducedTag.put(":", "INPUNC");
        fullTagToReducedTag.put("LS", "INPUNC");
        fullTagToReducedTag.put("SYM", "INPUNC");
        fullTagToReducedTag.put("UH", "INPUNC");
        fullTagToReducedTag.put("EX", "N");
        fullTagToReducedTag.put("FW", "N");
        fullTagToReducedTag.put("NN", "N");
        fullTagToReducedTag.put("NNP", "N");
        fullTagToReducedTag.put("NNPS", "N");
        fullTagToReducedTag.put("NNS", "N");
        fullTagToReducedTag.put("PRP", "N");
        fullTagToReducedTag.put("IN", "PREP");
        fullTagToReducedTag.put("-RRB-", "RPUNC");
        fullTagToReducedTag.put("''", "RPUNC");
        fullTagToReducedTag.put("WDT", "W");
        fullTagToReducedTag.put("WP$", "W");
        fullTagToReducedTag.put("WP", "W");
        fullTagToReducedTag.put("WRB", "W");
        fullTagToReducedTag.put("VBN", "VBN");

        return reducedTagSet;
    }
}
