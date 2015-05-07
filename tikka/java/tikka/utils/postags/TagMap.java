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

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

/**
 *
 * @author Taesun Moon <tsunmoon@gmail.com>
 */
public abstract class TagMap extends HashMap<String, Integer> implements
      Serializable {

    static private final long serialVersionUID = 100L;
    protected HashSet<String> contentTagSet, functionTagSet, fullTagSet, ignoreSet;
    protected final HashSet<String> reducedTagSet = new HashSet<String>(Arrays.asList(
          "ADJ", //CD JJ JJR JJS PRP$
          "CONJ",//CC
          "ENDPUNC",//.
          "LPUNC",//“ -LRB
          "POS",//POS
          "PRT",//RP
          "TO",//TO
          "V",//MD VBD VBP VB VBZ
          "VBG",//VBG
          "ADV",//RB RBR RBS
          "DET",//DT PDT
          "INPUNC",//,: LS SYM UH
          "N",//EX FW NN NNP NNPS NNS PRP
          "PREP",//IN
          "RPUNC",//” -RRB-
          "W",//WDT WP$ WP WRB
          "VBN"//VBN
          ));
    protected HashMap<Integer, String> idxToFullTag, idxToReducedTag;
    protected HashMap<String, String> fullTagToReducedTag;
    protected IntTagMap oneToOneTagMap;
    protected IntTagMap manyToOneTagMap;
    protected IntTagMap goldToModelTagMap;
    protected TagSetEnum.TagSet tagSet;
    protected TagSetEnum.ReductionLevel level;

    protected TagMap() {
    }

    public TagMap(int _modelTagSize) {
    }

    protected void initialize(int _modelTagSize) {
        ignoreSet = new HashSet<String>();

        oneToOneTagMap = new IntTagMap();
        manyToOneTagMap = new IntTagMap();
        for (int i = 0; i < _modelTagSize; ++i) {
            oneToOneTagMap.put(i, -1);
            manyToOneTagMap.put(i, -1);
        }
        goldToModelTagMap = new IntTagMap();

        fullTagToReducedTag = new HashMap<String, String>();
        for (String tag : fullTagSet) {
            fullTagToReducedTag.put(tag, tag);
        }
    }

    protected void initializeFull(int _modelTagSize) {
        initialize(_modelTagSize);
        idxToFullTag = new HashMap<Integer, String>();
        setIdxMap(fullTagSet, idxToFullTag);
        reduceTag();

        idxToReducedTag = new HashMap<Integer, String>();
        int idx = 0;
        for (String tag : fullTagToReducedTag.keySet()) {
            idx = get(tag);
            String reducedTag = fullTagToReducedTag.get(tag);
            idxToReducedTag.put(idx, reducedTag);
        }
    }

    protected void initializeReduced(int _modelTagSize) {
        initialize(_modelTagSize);
    }

    /**
     *
     * @param _tagSet
     */
    protected void setIdxMap(HashSet<String> _tagSet,
          HashMap<Integer, String> _idxToTag) {
        int idx = 0;
        for (String tag : _tagSet) {
            _idxToTag.put(idx, tag);
            put(tag, idx++);
        }
    }

    /**
     *
     * @return
     */
    protected abstract HashSet<String> reduceTag();

    /**
     * If the _tag exists
     * @param _tag
     * @return
     */
    public String getReducedTag(String _tag) {
        String tag = null;
        if (fullTagSet.contains(_tag)) {
            tag = fullTagToReducedTag.get(_tag);
        }

        return tag;
    }

    public boolean isIgnoreTag(String _tag) {
        if (ignoreSet.contains(_tag)) {
            return true;
        }
        return false;
    }

    /**
     *
     * @param tag
     * @return
     */
    public Integer get(String tag) {
        if (!containsKey(tag)) {
            System.err.println("\"" + tag + "\" did not exist in the postag lexicon");
            System.exit(1);
        }
        return super.get(tag);
    }

    /**
     *
     * @param _tag
     * @return
     */
    public String getFullTag(String tag) {
        if (fullTagSet.contains(tag)) {
            return tag;
        } else {
            return "";
        }
    }

    public int getTagSetSize() {
        return fullTagSet.size();
    }

    public String getTagSetName() {
        return tagSet.toString();
    }

    public TagSetEnum.ReductionLevel getReductionLevel() {
        return level;
    }

    public int getReducedGoldTagSize() {
        return reducedTagSet.size();
    }

    public int getFullGoldTagSize() {
        return fullTagSet.size();
    }

    public int getModelTagSize() {
        return oneToOneTagMap.size();
    }

    public String getOneToOneTagString(int stateid) {
        return idxToFullTag.get(oneToOneTagMap.get(stateid));
    }

    public String getManyToOneTagString(int stateid) {
        return idxToFullTag.get(manyToOneTagMap.get(stateid));
    }

    public String getGoldTagString(int goldid) {
        return idxToFullTag.get(goldid);
    }

    public String getGoldReducedTagString(int goldid) {
        return idxToReducedTag.get(goldid);
    }
}
