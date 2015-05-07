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
package tikka.bhmm.models;

import java.io.IOException;
import tikka.bhmm.model.base.HMMBase;
import tikka.bhmm.apps.CommandLineOptions;
import tikka.utils.annealer.Annealer;

/**
 * The CDHMM-d model in the paper
 *
 * @author tsmoon
 */
public class CDHMMD extends HMMBase {

    public CDHMMD(CommandLineOptions options) {
        super(options);
    }

    /**
     * Training routine for the inner iterations
     */
    @Override
    protected void trainInnerIter(int itermax, Annealer annealer) {
        int wordid, stateid, docid;
        int prev = (stateS-1), current = (stateS-1), next = (stateS-1), nnext = (stateS-1);
        double max = 0, totalprob = 0;
        double r = 0;
        int wordstateoff, docoff;
        int pprevsentid = -1; 
        int prevsentid = -1; 
        int nextsentid = -1;
        int nnextsentid = -1;

        long start = System.currentTimeMillis();
        for (int iter = 0; iter < itermax; ++iter) {
            System.err.println("\n\niteration " + iter + " (Elapsed time = +" +
                (System.currentTimeMillis()-start)/1000 + "s)");
            current = stateS-1;
            prev = stateS-1;
            System.err.print("Number of words processed: ");
            for (int i = 0; i < wordN; ++i) {
                if (i % 100000 == 0) {
                    System.err.print(((float)i/1000000) + "M, ");
                }
                wordid = wordVector[i];
                docid = documentVector[i];
                stateid = stateVector[i];
                wordstateoff = wordid * stateS;
                docoff = docid * stateC;

                if (stateid < stateC) {
                    contentStateByDocument[docoff + stateid]--;
                    documentCounts[docid]--;
                }
                stateByWord[wordstateoff + stateid]--;
                stateCounts[stateid]--;
                firstOrderTransitions[first[i] * stateS + stateid]--;
                secondOrderTransitions[(second[i]*S2) + (first[i]*stateS) + stateid]--;

                try {
                    next = stateVector[i + 1];
                    nextsentid = sentenceVector[i + 1];
                } catch (ArrayIndexOutOfBoundsException e) {
                    next = stateS-1;
                    nextsentid = -1;
                }

                try {
                    nnext = stateVector[i + 2];
                    nnextsentid = sentenceVector[i + 2];
                } catch (ArrayIndexOutOfBoundsException e) {
                    nnext = stateS-1;
                    nnextsentid = -1;
                }

                if (sentenceVector[i] != prevsentid) {
                    current = stateS-1;
                    prev = stateS-1;
                }  else if (sentenceVector[i] != pprevsentid) {
                    prev = stateS-1;
                }

                if (sentenceVector[i] != nextsentid) {
                    next = stateS-1;
                    nnext = stateS-1;
                }  else if (sentenceVector[i] != nnextsentid) {
                    nnext = stateS-1;
                }

                for (int j=0; j < (stateS-1); j++) {
                    // see words as 'abxcd', where x is the current word
                    double x = 0.0;
                    if (j<stateC) {
                        x = (stateByWord[wordstateoff + j] + beta) /
                            (stateCounts[j] + wbeta) *
                            ((contentStateByDocument[docoff + j] + alpha) /
                            (documentCounts[docid] + calpha));
                    } else {
                        x = (stateByWord[wordstateoff + j] + delta) /
                            (stateCounts[j] + wdelta);
                    }
                    double abx =
                            (secondOrderTransitions[(prev*S2+current*stateS+j)]+gamma);
                    double bxc =
                            (secondOrderTransitions[(current*S2+j*stateS+next)]+gamma) /
                            (firstOrderTransitions[current*stateS + j] + sgamma);
                    double xcd =
                            (secondOrderTransitions[(j*S2+next*stateS+nnext)]+gamma) /
                            (firstOrderTransitions[j*stateS + next] + sgamma);
                    stateProbs[j] = x*abx*bxc*xcd;
                }

                totalprob = annealer.annealProbs(stateProbs);
                r = mtfRand.nextDouble() * totalprob;
                max = stateProbs[0];
                stateid = 0;
                while (r > max) {
                    stateid++;
                    max += stateProbs[stateid];
                }
                stateVector[i] = stateid;

                if (stateid < stateC) {
                    contentStateByDocument[docoff + stateid]++;
                    documentCounts[docid]++;
                }

                stateByWord[wordstateoff + stateid]++;
                stateCounts[stateid]++;
                firstOrderTransitions[current*stateS + stateid]++;
                secondOrderTransitions[prev*S2 + current*stateS+ stateid]++;
                first[i] = current;
                second[i] = prev;
                prev = current;
                current = stateid;
                pprevsentid = prevsentid;
                prevsentid = sentenceVector[i];
            }
        }
    }

    /**
     * Randomly initialize learning parameters
     */
    @Override
    public void initializeParametersRandom() {

        int wordid, docid, stateid;
        int prev = (stateS-1), current = (stateS-1);
        double max = 0, totalprob = 0;
        double r = 0;
        int wordstateoff, docoff, stateoff, secondstateoff;

        /**
         * Initialize by assigning random topic indices to words
         */
        for (int i = 0; i < wordN; ++i) {
            wordid = wordVector[i];
            wordstateoff = wordid * stateS;

            docid = documentVector[i];
            stateoff = current * stateS;
            secondstateoff = (prev*S2) + (current*stateS);
            docoff = docid * stateC;

            totalprob = 0;

            if (mtfRand.nextDouble() > 0.5) {
                for (int j = 0; j < stateC; j++) {
                    totalprob += stateProbs[j] = 1.0;
                }
                stateid = 0;
            } else {
                for (int j = stateC; j < (stateS-1); j++) {
                    totalprob += stateProbs[j] = 1.0;
                }
                r = mtfRand.nextDouble() * totalprob;
                stateid = stateC;
            }

            r = mtfRand.nextDouble() * totalprob;
            max = stateProbs[0];
            while (r > max) {
                stateid++;
                max += stateProbs[stateid];
            }
            stateVector[i] = stateid;

            if (stateid < stateC) {
                contentStateByDocument[docoff + stateid]++;
                documentCounts[docid]++;
            }

            firstOrderTransitions[stateoff + stateid]++;
            secondOrderTransitions[secondstateoff + stateid]++;
            stateByWord[wordstateoff + stateid]++;
            stateCounts[stateid]++;
            first[i] = current;
            second[i] = prev;
            prev = current;
            current = stateid;
        }
    }

    /*
    @Override
    public void initializeFromLoadedModel(CommandLineOptions options) throws
          IOException {
        super.initializeFromLoadedModel(options);

        int current = 0;
        int wordid = 0, stateid = 0, docid;
        int stateoff, wordstateoff, docoff;

        for (int i = 0; i < wordN; ++i) {
            wordid = wordVector[i];
            docid = documentVector[i];
            stateid = stateVector[i];

            stateoff = current * stateS;
            wordstateoff = wordid * stateS;
            docoff = docid * stateC;

            if (stateid < stateC) {
                contentStateByDocument[docoff + stateid]++;
                documentCounts[docid]++;
            }
            stateByWord[wordstateoff + stateid]++;
            stateCounts[stateid]++;
            firstOrderTransitions[stateoff + stateid]++;
            first[i] = current;
            current = stateid;
        }
    }*/
}
