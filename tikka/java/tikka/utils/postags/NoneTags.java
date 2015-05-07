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

import java.util.HashSet;

/**
 * EnglishTagMap for handling full Brown corpus tagset. Does not reduce any of the tags.
 * 
 * @author tsmoon
 */
public class NoneTags extends TagMap {

    public NoneTags(int _modelTagSize) {
        super(_modelTagSize);
    }

    protected NoneTags() {
    }

    /**
     *
     * @param tag
     * @return
     */
    @Override
    public Integer get(String tag) {
        return -1;
    }

    @Override
    public String getReducedTag(String tag) {
        return null;
    }

    @Override
    public boolean isIgnoreTag(String _tag) {
        return false;
    }

    @Override
    protected HashSet<String> reduceTag() {
        return null;
    }

    @Override
    public String getOneToOneTagString(int stateid) {
        return null;
    }

    @Override
    public String getManyToOneTagString(int stateid) {
        return null;
    }

    @Override
    public String getGoldTagString(int goldid) {
        return null;
    }

    @Override
    public String getGoldReducedTagString(int goldid) {
        return null;
    }
}
