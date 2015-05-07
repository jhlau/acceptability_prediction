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
package tikka.hhl.distributions;

import tikka.hhl.lexicons.Lexicon;

/**
 * Class skeleton for a dirichlet process.
 * 
 * @author tsmoon
 */
public abstract class DirichletProcess {

    /**
     * Assumed maximum length of string types that will be observed in corpus
     */
    protected int maxlen = 100;
    /**
     * Array of probabilities for strings given the length. Initialized at
     * construction and never changed.
     */
    protected double[] stringProbs;
    /**
     * Dictionary of string types to indexes and back. Also maintains counts.
     */
    protected Lexicon lexicon;
    /**
     * Hyperparameter for the dirichlet distribution
     */
    protected double hyper;
    /**
     * Base distribution for the dirichlet process. This is instantiated as
     * a HierarchicalDirichletBaseDistribution when a member of a
     * hierarchical dirichlet process. In a dirichlet process, simply a mapping
     * from the length of a string to some value in [0,1].
     */
    protected DirichletBaseDistribution baseDistribution;
}
