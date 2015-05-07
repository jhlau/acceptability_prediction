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

/**
 * Jaccard measure for populating cost matrix in Evaluator
 * 
 * @author tsmoon
 */
public class JaccardMeasure extends DistanceMeasure {

    public JaccardMeasure(int[] _cooccurrenceMatrix, int[] _modelTagCounts,
          int[] _goldTagCounts, int _N) {
        super(_cooccurrenceMatrix, _modelTagCounts, _goldTagCounts, _N);
    }

    @Override
    public double cost(int _i, int _j) {
        return 1 - cooccurrenceMatrix[_i * N + _j]
              / (0. + modelTagCounts[_i] + goldTagCounts[_j] - cooccurrenceMatrix[_i * N + _j]);
    }
}
