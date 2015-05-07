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

/**
 *
 * @author tsmoon
 */
public class TagMapGenerator {

    public static TagMap generate(TagSetEnum.TagSet tagSet,
          TagSetEnum.ReductionLevel level, int modelTagSize) {
        TagMap tagMap = null;
        switch (tagSet) {
            case BROWN:
                switch (level) {
                    case FULL:
                        tagMap = new BrownTags(modelTagSize);
                        break;
                    case REDUCED:
                        tagMap = new BrownTagsCE(modelTagSize);
                        break;
                }
                break;
            case PTB:
                switch (level) {
                    case FULL:
                        tagMap = new PennTags(modelTagSize);
                        break;
                    case REDUCED:
                        tagMap = new PennTagsCE(modelTagSize);
                        break;
                }
                break;
            case TIGER:
                switch (level) {
                    case FULL:
                        tagMap = new TigerTags(modelTagSize);
                        break;
                    case REDUCED:
                        tagMap = new TigerTagsCE(modelTagSize);
                        break;
                }
                break;
            case USP:
                switch (level) {
                    case FULL:
                        tagMap = new UspTags(modelTagSize);
                        break;
                    case REDUCED:
                        tagMap = new UspTagsCE(modelTagSize);
                        break;
                }
                break;
            case FLORESTA:
                switch (level) {
                    case FULL:
                        tagMap = new FlorestaTags(modelTagSize);
                        break;
                    case REDUCED:
                        tagMap = new FlorestaTagsCE(modelTagSize);
                        break;
                }
                break;
            case NONE:
                tagMap = new NoneTags();
                break;
        }
        tagMap.tagSet = tagSet;
        tagMap.level = level;
        return tagMap;
    }
}
