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
package tikka.hmm.model.mcmc;

import tikka.hmm.apps.CommandLineOptions;
import tikka.hmm.model.base.HMM;
import tikka.utils.annealer.Annealer;

/**
 *
 *
 * @author tsmoon
 */
public class GibbsHMM extends HMM {

    public GibbsHMM(CommandLineOptions options) {
        super(options);
    }

    /**
     * Randomly initialize learning parameters
     */
    @Override
    public void initializeParametersRandom() {
        int wordid, stateid;
        int current = 0;
        double max = 0, totalprob = 0;
        double r = 0;
        int wordstateoff, stateoff;

        /**
         * Initialize by assigning random topic indices to words
         */
        for (int i = 0; i < wordN; ++i) {
            wordid = wordVector[i];

//            if (wordid == EOSi) {
//                firstOrderTransitions[current * stateS + 0]++;
//                first[i] = current;
//                current = 0;
//            } else {
            wordstateoff = stateS * wordid;

            totalprob = 0;
            stateoff = current * stateS;
            try {
                for (int j = 0;; j++) {
                    totalprob += stateProbs[j] =
                          (StateByWord[wordstateoff + j] + delta)
                          / (stateCounts[j] + wdelta)
                          * (firstOrderTransitions[stateoff + j] + gamma);
                }
            } catch (java.lang.ArrayIndexOutOfBoundsException e) {
            }

            r = mtfRand.nextDouble() * totalprob;
            stateid = 0;
            max = stateProbs[stateid];
            while (r > max) {
                stateid++;
                max += stateProbs[stateid];
            }
            stateVector[i] = stateid;
            StateByWord[wordstateoff + stateid]++;
            stateCounts[stateid]++;
            firstOrderTransitions[stateoff + stateid]++;
            first[i] = current;
            current = stateid;
//            }
        }
    }

    /**
     * Training routine for the inner iterations
     */
    @Override
    protected void trainInnerIter(int itermax, Annealer annealer) {
        int wordid, stateid;
        int current = 0, next = 0;
        double max = 0, totalprob = 0;
        double r = 0;
        int wordstateoff, stateoff;

        for (int iter = 0; iter < itermax; ++iter) {
            System.err.println("iteration " + iter);
            current = 0;
            for (int i = 0; i < wordN; i++) {
                if (i % 100000 == 0) {
                    System.err.println("\tProcessing word " + i);
                }
                wordid = wordVector[i];

//                if (wordid == EOSi) // sentence marker
//                {
//                    firstOrderTransitions[first[i] * stateS + 0]--;
//                    firstOrderTransitions[current * stateS + 0]++;
//                    first[i] = current;
//                    current = 0;
//                } else {
                stateid = stateVector[i];
                wordstateoff = wordid * stateS;

                StateByWord[wordstateoff + stateid]--;
                stateCounts[stateid]--;
                firstOrderTransitions[first[i] * stateS + stateid]--;

                stateoff = current * stateS;
                try {
                    next = stateVector[i + 1];
                } catch (ArrayIndexOutOfBoundsException e) {
                    next = 0;
                }

                try {
                    for (int j = 0;; j++) {
                        stateProbs[j] =
                              ((StateByWord[wordstateoff + j] + delta) / (stateCounts[j] + wdelta))
                              * (firstOrderTransitions[stateoff + j] + gamma) / (stateCounts[j] + sgamma)
                              * (firstOrderTransitions[j * stateS + next] + gamma);
                    }
                } catch (ArrayIndexOutOfBoundsException e) {
                }
                totalprob = annealer.annealProbs(stateProbs);
                r = mtfRand.nextDouble() * totalprob;
                stateid = 0;
                max = stateProbs[stateid];
                while (r > max) {
                    stateid++;
                    max += stateProbs[stateid];
                }
                stateVector[i] = stateid;

                StateByWord[wordstateoff + stateid]++;
                stateCounts[stateid]++;
                firstOrderTransitions[stateoff + stateid]++;
                first[i] = current;
                current = stateid;
//                }
            }
        }
    }
}
