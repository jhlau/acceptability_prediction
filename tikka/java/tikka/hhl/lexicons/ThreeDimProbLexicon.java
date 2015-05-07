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
package tikka.hhl.lexicons;

import java.util.HashMap;

/**
 * A table from a type index (this could be a state, a topic, or an affix)
 * to a subtable of probabilities. The probabilities
 * in the subtable will either be conditioned on the index in this table or
 * be conditioned on the index in this table and the index of an enclosing table
 * (see {@link BadMetaphor.structures.lexicons.FourDimProbLexicon}).
 *
 * @author tsmoon
 */
public class ThreeDimProbLexicon extends HashMap<Integer, TwoDimProbLexicon> {
}
