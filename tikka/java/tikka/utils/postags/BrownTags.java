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
package tikka.utils.postags;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

/**
 * EnglishTagMap for handling full Brown corpus tagset. Does not reduce any of the tags.
 * 
 * @author tsmoon
 */
public class BrownTags extends TagMap {

    protected final HashSet<String> brownContentTagSet = new HashSet<String>(Arrays.asList(
          "FW", //foreign word (hyphenated before regular tag)
          "JJ", //adjective
          "JJ$", //possessive adjective ????
          "JJR", //comparative adjective
          "JJS", //semantically superlative adjective (chief, top)
          "JJT", //morphologically superlative adjective (biggest)
          "NN", //singular or mass noun
          "NN$", //possessive singular noun
          "NNS", //plural noun
          "NNS$", //possessive plural noun
          "NP", //proper noun or part of name phrase
          "NP$", //possessive proper noun
          "NPS", //plural proper noun
          "NPS$", //possessive plural proper noun
          "NR", //adverbial noun (home, today, west)
          "NRS", //plural adverbial noun (Sundays)
          "NR$",
          "QL", //qualifier (very, fairly)
          "QLP", //post-qualifier (enough, indeed)
          "RB", //adverb
          "RB$", //possessive adverb (else's)
          "RBR", //comparative adverb
          "RBT", //superlative adverb
          "RN", //nominal adverb (here, then, indoors)
          "VB", //verb, base form
          "VBD", //verb, past tense
          "VBG", //verb, present participle/gerund
          "VBN", //verb, past participle
          "VBZ" //verb, 3rd. singular present
          ));
    protected final HashSet<String> brownFunctionTagSet = new HashSet<String>(Arrays.asList(
          ".", //sentence closer (. ; ? *)
          "(", //left paren
          ")", //right paren
          "*", //not, n't
          //              "--", //dash
          ",", //comma
          ":", //colon
          "ABL", //pre-qualifier (quite, rather)
          "ABN", //pre-quantifier (half, all)
          "ABX", //pre-quantifier (both)
          "AP", //post-determiner (many, several, next)
          "AP$", //possessive post-determiner (other's)
          "AT", //article (a, the, no)
          "BE", //be
          "BED", //were
          "BED*", //weren't
          "BEDZ", //was
          "BEDZ*", //wasn't
          "BEG", //being
          "BEM", //am
          "BEM*", //ain't
          "BEN", //been
          "BER", //are, art
          "BER*", //aren't
          "BEZ", //is
          "BEZ*", //isn't
          "CC", //coordinating conjunction (and, or)
          "CD", //cardinal numeral (one, two, 2, etc.)
          "CD$", //possessive cardinal numeral (1960's)
          "CS", //subordinating conjunction (if, although)
          "DO", //do
          "DO*", //don't
          "DOD", //did
          "DOD*",
          "DOZ", //does
          "DOZ*", //doesn't
          "DT", //singular determiner/quantifier (this, that)
          "DT$", //possessive singular determiner/quantifier (another's)
          "DTI", //singular or plural determiner/quantifier (some, any)
          "DTS", //plural determiner (these, those)
          "DTX", //determiner/double conjunction (either)
          "EX", //existential there
          "HV", //have
          "HV*", //haven't
          "HVD", //had (past tense)
          "HVD*", //hadn't (past tense)
          "HVZ", //has
          "HVZ*", //hasn't
          "HVG", //having
          "HVN", //had (past participle)
          "IN", //preposition
          "MD", //modal auxiliary (can, should, will)
          "MD*", //negated modal auxiliary (can't, shouldn't, won't)
          //              "NC", //cited word (hyphenated after regular tag)
          "OD", //ordinal numeral (first, 2nd)
          "PN", //nominal pronoun (everybody, nothing)
          "PN$", //possessive nominal pronoun
          "PP$", //possessive personal pronoun (my, our)
          "PP$$", //second (nominal) possessive pronoun (mine, ours)
          "PPL", //singular reflexive/intensive personal pronoun (myself)
          "PPLS", //plural reflexive/intensive personal pronoun (ourselves)
          "PPO", //objective personal pronoun (me, him, it, them)
          "PPS", //3rd. singular nominative pronoun (he, she, it, one)
          "PPSS", //other nominative personal pronoun (I, we, they, you)
          "RP", //adverb/particle (about, off, up)
          "TO", //infinitive marker to
          "UH", //interjection, exclamation
          "WDT", //wh- determiner (what, which)
          "WP$", //possessive wh- pronoun (whose)
          "WPO", //objective wh- pronoun (whom, which, that)
          "WPS", //nominative wh- pronoun (who, which, that)
          "WQL", //wh- qualifier (how)
          "WRB" //wh- adverb (how, where, when)
          ));
    protected final HashSet<String> brownFullTagSet = new HashSet<String>(Arrays.asList(
          ".", //sentence closer (. ; ? *)
          "(", //left paren
          ")", //right paren
          "*", //not, n't
          //              "--", //dash
          ",", //comma
          ":", //colon
          "ABL", //pre-qualifier (quite, rather)
          "ABN", //pre-quantifier (half, all)
          "ABX", //pre-quantifier (both)
          "AP", //post-determiner (many, several, next)
          "AP$", //possessive post-determiner (other's)
          "AT", //article (a, the, no)
          "BE", //be
          "BED", //were
          "BED*", //weren't
          "BEDZ", //was
          "BEDZ*", //wasn't
          "BEG", //being
          "BEM", //am
          "BEM*", //ain't
          "BEN", //been
          "BER", //are, art
          "BER*", //aren't
          "BEZ", //is
          "BEZ*", //isn't
          "CC", //coordinating conjunction (and, or)
          "CD", //cardinal numeral (one, two, 2, etc.)
          "CD$", //possessive cardinal numeral (1960's)
          "CS", //subordinating conjunction (if, although)
          "DO", //do
          "DO*", //don't
          "DOD", //did
          "DOD*", //didn't
          "DOZ", //does
          "DOZ*", //doesn't
          "DT", //singular determiner/quantifier (this, that)
          "DT$", //possessive singular determiner/quantifier (another's)
          "DTI", //singular or plural determiner/quantifier (some, any)
          "DTS", //plural determiner (these, those)
          "DTX", //determiner/double conjunction (either)
          "EX", //existential there
          "FW", //foreign word (hyphenated before regular tag)
          "HV", //have
          "HV*", //haven't
          "HVD", //had (past tense)
          "HVD*", //hadn't (past tense)
          "HVG", //having
          "HVN", //had (past participle)
          "HVZ", //has
          "HVZ*", //hasn't
          "IN", //preposition
          "JJ", //adjective
          "JJ$", //possessive adjective ????
          "JJR", //comparative adjective
          "JJS", //semantically superlative adjective (chief, top)
          "JJT", //morphologically superlative adjective (biggest)
          "MD", //modal auxiliary (can, should, will)
          "MD*", //negated modal auxiliary (can't, shouldn't, won't)
          //              "NC", //cited word (hyphenated after regular tag)
          "NN", //singular or mass noun
          "NN$", //possessive singular noun
          "NNS", //plural noun
          "NNS$", //possessive plural noun
          "NP", //proper noun or part of name phrase
          "NP$", //possessive proper noun
          "NPS", //plural proper noun
          "NPS$", //possessive plural proper noun
          "NR", //adverbial noun (home, today, west)
          "NRS", //plural adverbial noun (Sundays)
          "NR$",
          "OD", //ordinal numeral (first, 2nd)
          "PN", //nominal pronoun (everybody, nothing)
          "PN$", //possessive nominal pronoun
          "PP$", //possessive personal pronoun (my, our)
          "PP$$", //second (nominal) possessive pronoun (mine, ours)
          "PPL", //singular reflexive/intensive personal pronoun (myself)
          "PPLS", //plural reflexive/intensive personal pronoun (ourselves)
          "PPO", //objective personal pronoun (me, him, it, them)
          "PPS", //3rd. singular nominative pronoun (he, she, it, one)
          "PPSS", //other nominative personal pronoun (I, we, they, you)
          "QL", //qualifier (very, fairly)
          "QLP", //post-qualifier (enough, indeed)
          "RB", //adverb
          "RB$", //possessive adverb (else's)
          "RBR", //comparative adverb
          "RBT", //superlative adverb
          "RN", //nominal adverb (here, then, indoors)
          "RP", //adverb/particle (about, off, up)
          "TO", //infinitive marker to
          "UH", //interjection, exclamation
          "VB", //verb, base form
          "VBD", //verb, past tense
          "VBG", //verb, present participle/gerund
          "VBN", //verb, past participle
          "VBZ", //verb, 3rd. singular present
          "WDT", //wh- determiner (what, which)
          "WP$", //possessive wh- pronoun (whose)
          "WPO", //objective wh- pronoun (whom, which, that)
          "WPS", //nominative wh- pronoun (who, which, that)
          "WQL", //wh- qualifier (how)
          "WRB" //wh- adverb (how, where, when)
          ));
    protected final HashSet<String> brownIgnoreSet = new HashSet<String>(Arrays.asList(
          "NIL"
          ));

    public BrownTags(int _modelTagSize) {
        super(_modelTagSize);
        contentTagSet = brownContentTagSet;
        functionTagSet = brownFunctionTagSet;
        fullTagSet = brownFullTagSet;
        initializeFull(_modelTagSize);
        ignoreSet = brownIgnoreSet;
    }

    protected BrownTags() {
    }

    /**
     *
     * @param tag
     * @return
     */
    @Override
    public Integer get(String tag) {
        String[] tags = tag.split("[+\\-]");
        return super.get(tags[0]);
    }

    @Override
    public String getReducedTag(String tag) {
        String[] tags = tag.split("[+\\-]");
        return super.getReducedTag(tags[0]);
    }

    @Override
    protected HashSet<String> reduceTag() {
        fullTagToReducedTag.put(".", "ENDPUNC"); //sentence closer (. ; ? *)
        fullTagToReducedTag.put("(", "LPUNC"); //left paren
        fullTagToReducedTag.put(")", "RPUNC"); //right paren
        fullTagToReducedTag.put("*", "ADV"); //not, n't
        fullTagToReducedTag.put(",", "INPUNC"); //comma
        fullTagToReducedTag.put(":", "INPUNC"); //colon
        fullTagToReducedTag.put("ABL", "ADV"); //pre-qualifier (quite, rather)
        fullTagToReducedTag.put("ABN", "DET"); //pre-quantifier (half, all)
        fullTagToReducedTag.put("ABX", "DET"); //pre-quantifier (both)
        fullTagToReducedTag.put("AP", "ADJ"); //post-determiner (many, several, next)
        fullTagToReducedTag.put("AP$", "ADJ"); 
        fullTagToReducedTag.put("AT", "DET"); //article (a, the, no)
        fullTagToReducedTag.put("BE", "V"); //be
        fullTagToReducedTag.put("BED", "V"); //were
        fullTagToReducedTag.put("BED*", "V");
        fullTagToReducedTag.put("BEDZ", "V"); //was
        fullTagToReducedTag.put("BEDZ*", "V"); //wasn't
        fullTagToReducedTag.put("BEG", "VBG"); //being
        fullTagToReducedTag.put("BEM", "V"); //am
        fullTagToReducedTag.put("BEM*", "V");
        fullTagToReducedTag.put("BEN", "VBN"); //been
        fullTagToReducedTag.put("BER", "V"); //are, art
        fullTagToReducedTag.put("BER*", "V"); 
        fullTagToReducedTag.put("BEZ", "V"); //is
        fullTagToReducedTag.put("BEZ*", "V"); //isn't
        fullTagToReducedTag.put("CC", "CONJ"); //coordinating conjunction (and, or)
        fullTagToReducedTag.put("CD", "ADJ"); //cardinal numeral (one, two, 2, etc.)
        fullTagToReducedTag.put("CD$", "ADJ"); 
        fullTagToReducedTag.put("CS", "PREP"); //subordinating conjunction (if, although)
        fullTagToReducedTag.put("DO", "V"); //do
        fullTagToReducedTag.put("DO*", "V"); //don't
        fullTagToReducedTag.put("DOD", "V"); //did
        fullTagToReducedTag.put("DOD*", "V"); //did
        fullTagToReducedTag.put("DOZ", "V"); //does
        fullTagToReducedTag.put("DOZ*", "V"); //does
        fullTagToReducedTag.put("DT", "DET"); //singular determiner/quantifier (this, that)
        fullTagToReducedTag.put("DT$", "DET");
        fullTagToReducedTag.put("DTI", "DET"); //singular or plural determiner/quantifier (some, any)
        fullTagToReducedTag.put("DTS", "DET"); //plural determiner (these, those)
        fullTagToReducedTag.put("DTX", "DET"); //determiner/double conjunction (either)
        fullTagToReducedTag.put("EX", "N"); //existential there
        fullTagToReducedTag.put("FW", "N"); //foreign word (hyphenated before regular tag)
        fullTagToReducedTag.put("HV", "V"); //have
        fullTagToReducedTag.put("HV*", "V");
        fullTagToReducedTag.put("HVD", "V"); //had (past tense)
        fullTagToReducedTag.put("HVD*", "V");
        fullTagToReducedTag.put("HVG", "VBG"); //having
        fullTagToReducedTag.put("HVZ", "V");
        fullTagToReducedTag.put("HVZ*", "V");
        fullTagToReducedTag.put("HVN", "V"); //had (past participle)
        fullTagToReducedTag.put("IN", "PREP"); //preposition
        fullTagToReducedTag.put("JJ", "ADJ"); //adjective
        fullTagToReducedTag.put("JJ$", "ADJ");
        fullTagToReducedTag.put("JJR", "ADJ"); //comparative adjective
        fullTagToReducedTag.put("JJS", "ADJ"); //semantically superlative adjective (chief, top)
        fullTagToReducedTag.put("JJT", "ADJ"); //morphologically superlative adjective (biggest)
        fullTagToReducedTag.put("MD", "V"); //modal auxiliary (can, should, will)
        fullTagToReducedTag.put("MD*", "V");
        fullTagToReducedTag.put("NN", "N"); //singular or mass noun
        fullTagToReducedTag.put("NN$", "ADJ"); //possessive singular noun
        fullTagToReducedTag.put("NNS", "N"); //plural noun
        fullTagToReducedTag.put("NNS$", "ADJ"); //possessive plural noun
        fullTagToReducedTag.put("NP", "N"); //proper noun or part of name phrase
        fullTagToReducedTag.put("NP$", "ADJ"); //possessive proper noun
        fullTagToReducedTag.put("NPS", "N"); //plural proper noun
        fullTagToReducedTag.put("NPS$", "ADJ"); //possessive plural proper noun
        fullTagToReducedTag.put("NR", "N"); //adverbial noun (home, today, west)
        fullTagToReducedTag.put("NRS", "N");
        fullTagToReducedTag.put("NR$", "N");
        fullTagToReducedTag.put("OD", "ADJ"); //ordinal numeral (first, 2nd)
        fullTagToReducedTag.put("PN", "N"); //nominal pronoun (everybody, nothing)
        fullTagToReducedTag.put("PN$", "ADJ"); //possessive nominal pronoun
        fullTagToReducedTag.put("PP$", "ADJ"); //possessive personal pronoun (my, our)
        fullTagToReducedTag.put("PP$$", "ADJ"); //second (nominal) possessive pronoun (mine, ours)
        fullTagToReducedTag.put("PPL", "N"); //singular reflexive/intensive personal pronoun (myself)
        fullTagToReducedTag.put("PPLS", "N"); //plural reflexive/intensive personal pronoun (ourselves)
        fullTagToReducedTag.put("PPO", "N"); //objective personal pronoun (me, him, it, them)
        fullTagToReducedTag.put("PPS", "N"); //3rd. singular nominative pronoun (he, she, it, one)
        fullTagToReducedTag.put("PPSS", "N"); //other nominative personal pronoun (I, we, they, you)
        fullTagToReducedTag.put("QL", "ADV"); //qualifier (very, fairly)
        fullTagToReducedTag.put("QLP", "ADV"); //post-qualifier (enough, indeed)
        fullTagToReducedTag.put("RB", "ADV"); //adverb
        fullTagToReducedTag.put("RB$", "ADV");
        fullTagToReducedTag.put("RBR", "ADV"); //comparative adverb
        fullTagToReducedTag.put("RBT", "ADV"); //superlative adverb
        fullTagToReducedTag.put("RN", "ADV"); //nominal adverb (here, then, indoors)
        fullTagToReducedTag.put("RP", "PRT"); //adverb/particle (about, off, up)
        fullTagToReducedTag.put("TO", "TO"); //infinitive marker to
        fullTagToReducedTag.put("UH", "INPUNC"); //interjection, exclamation
        fullTagToReducedTag.put("VB", "V"); //verb, base form
        fullTagToReducedTag.put("VBD", "V"); //verb, past tense
        fullTagToReducedTag.put("VBG", "VBG"); //verb, present participle/gerund
        fullTagToReducedTag.put("VBN", "VBN"); //verb, past participle
        fullTagToReducedTag.put("VBZ", "V"); //verb, 3rd. singular present
        fullTagToReducedTag.put("WDT", "W"); //wh- determiner (what, which)
        fullTagToReducedTag.put("WP$", "W"); //possessive wh- pronoun (whose)
        fullTagToReducedTag.put("WPO", "W"); //objective wh- pronoun (whom, which, that)
        fullTagToReducedTag.put("WPS", "W"); //nominative wh- pronoun (who, which, that)
        fullTagToReducedTag.put("WQL", "W"); //wh- qualifier (how)
        fullTagToReducedTag.put("WRB", "W"); //wh- adverb (how, where, when)

        /**
         * The reduced tag set for brown does not use the POS label
         */
        reducedTagSet.remove("POS");

        return reducedTagSet;
    }
}
