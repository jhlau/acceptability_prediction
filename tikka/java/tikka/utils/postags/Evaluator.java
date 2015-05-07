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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;

/**
 * Class for evaluating model tags based on gold tags
 *
 * @author tsmoon
 */
public class Evaluator {

    protected final static double EPSILON = 1e-12;
    protected int[] modelTags;//, modelTagCounts;
    protected int[] fullGoldTags;//, goldTagCounts;
    protected int[] reducedGoldTags;//, reducedGoldTagCounts;
    protected double fullOneToOneAccuracy, fullManyToOneAccuracy;
    protected double reducedOneToOneAccuracy, reducedManyToOneAccuracy;
    protected TagMap fullTagMap;
    protected TagMap reducedTagMap;
    protected IntTagMap fullOneToOneTagMap;
    protected IntTagMap fullManyToOneTagMap;
    protected IntTagMap reducedOneToOneTagMap;
    protected IntTagMap reducedManyToOneTagMap;
    protected IntTagMap goldToModelTagMap;
    protected DistanceMeasureEnum.Measure measure;
    protected DistanceMeasure distanceMeasure;
    protected ClusterEvalScore fullClusterEvalScore, reducedClusterEvalScore;

    /**
     * 
     * @param _tagMap
     * @param _measure
     */
    public Evaluator(TagMap _tagMap, DistanceMeasureEnum.Measure _measure) {
        fullTagMap = _tagMap;
        reducedTagMap = TagMapGenerator.generate(fullTagMap.tagSet, TagSetEnum.ReductionLevel.REDUCED, fullTagMap.oneToOneTagMap.size());

        fullOneToOneTagMap = fullTagMap.oneToOneTagMap;
        fullManyToOneTagMap = fullTagMap.manyToOneTagMap;

        reducedOneToOneTagMap = reducedTagMap.oneToOneTagMap;
        reducedManyToOneTagMap = reducedTagMap.manyToOneTagMap;

        measure = _measure;

        fullClusterEvalScore = new ClusterEvalScore();
        reducedClusterEvalScore = new ClusterEvalScore();
    }

    /**
     * 
     * @param _modelTags
     * @param _goldTags
     */
    public void evaluateTags(int[] _modelTags, int[] _goldTags) {
        modelTags = _modelTags;
        fullGoldTags = _goldTags;

        reducedGoldTags = new int[fullGoldTags.length];
        for (int i = 0; i < fullGoldTags.length; ++i) {
            int fullid = fullGoldTags[i];
            String fulltag = fullTagMap.idxToFullTag.get(fullid);
            reducedGoldTags[i] = reducedTagMap.get(reducedTagMap.getReducedTag(fulltag));
        }

        matchTags(_modelTags, fullGoldTags, fullTagMap, fullOneToOneTagMap, fullManyToOneTagMap);
        fullOneToOneAccuracy = measureAccuracy(_modelTags, fullGoldTags, fullOneToOneTagMap);
        fullManyToOneAccuracy = measureAccuracy(_modelTags, fullGoldTags, fullManyToOneTagMap);
        matchTags(_modelTags, reducedGoldTags, reducedTagMap, reducedOneToOneTagMap, reducedManyToOneTagMap);
        reducedOneToOneAccuracy = measureAccuracy(_modelTags, reducedGoldTags, reducedOneToOneTagMap);
        reducedManyToOneAccuracy = measureAccuracy(_modelTags, reducedGoldTags, reducedManyToOneTagMap);

        clusterEvaluation(modelTags, fullGoldTags, fullTagMap.getModelTagSize(), fullTagMap.getTagSetSize(), fullClusterEvalScore);
        clusterEvaluation(modelTags, reducedGoldTags, reducedTagMap.getModelTagSize(), reducedTagMap.getTagSetSize(), reducedClusterEvalScore);
    }

    /**
     * 
     * @param _modelTags
     * @param _goldTags
     * @param _tagMap
     * @param _oneToOneTagMap
     * @param _manyToOneTagMap
     */
    public void matchTags(int[] _modelTags, int[] _goldTags, TagMap _tagMap,
          IntTagMap _oneToOneTagMap, IntTagMap _manyToOneTagMap) {
        int M = _tagMap.oneToOneTagMap.size();
        int N = _tagMap.getTagSetSize();
        int[] cooccurrenceMatrix = new int[M * N];
        double[] costMatrix = new double[M * N];
        int[] modelTagCounts = new int[M];
        int[] goldTagCounts = new int[N];

        for (int i = 0; i < M * N; ++i) {
            cooccurrenceMatrix[i] = 0;
            costMatrix[i] = 0;
        }
        for (int i = 0; i < M; ++i) {
            modelTagCounts[i] = 0;
        }
        for (int i = 0; i < N; ++i) {
            goldTagCounts[i] = 0;
        }

        for (int i = 0; i < _modelTags.length; ++i) {
            int j = _modelTags[i];
            int k = _goldTags[i];
            modelTagCounts[j]++;
            goldTagCounts[k]++;
            cooccurrenceMatrix[j * N + k]++;
        }

        switch (measure) {
            case JACCARD:
                distanceMeasure = new JaccardMeasure(cooccurrenceMatrix, modelTagCounts, goldTagCounts, N);
                break;
            case JENSEN_SHANNON:
                distanceMeasure = new JensenShannonMeasure(cooccurrenceMatrix, modelTagCounts, goldTagCounts, N);
                break;
        }

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                costMatrix[i * N + j] = distanceMeasure.cost(i, j);
            }
        }

        HashSet<Integer> rows = new HashSet<Integer>();
        HashSet<Integer> cols = new HashSet<Integer>();
        setTickers(rows, cols, M, N);

        double[] tmpCostMatrix = costMatrix.clone();
        buildOneToOneMap(tmpCostMatrix, rows, cols, N, _oneToOneTagMap);

        rows = new HashSet<Integer>();
        cols = new HashSet<Integer>();
        setTickers(rows, cols, M, N);
        buildManyToOneMap(costMatrix, rows, cols, N, _manyToOneTagMap);
    }

    /**
     * 
     * @param _rows
     * @param _cols
     * @param _M
     * @param _N
     */
    protected void setTickers(HashSet<Integer> _rows,
          HashSet<Integer> _cols, int _M, int _N) {
        for (int i = 0; i < _M; ++i) {
            _rows.add(i);
        }
        for (int i = 0; i < _N; ++i) {
            _cols.add(i);
        }
    }

    /**
     * 
     * @param _costMatrix
     * @param _rows
     * @param _cols
     * @param _N
     * @param _oneToOneTagMap
     */
    protected void buildOneToOneMap(double[] _costMatrix, HashSet<Integer> _rows,
          HashSet<Integer> _cols, int _N, IntTagMap _oneToOneTagMap) {
        double min = 1;
        int imin = 0, jmin = 0;
        for (int i : _rows) {
            for (int j : _cols) {
                if (_costMatrix[i * _N + j] < min) {
                    imin = i;
                    jmin = j;
                    min = _costMatrix[i * _N + j];
                }
            }
        }

        if (min == 1) {
            ArrayList<Integer> tl = new ArrayList<Integer>(_rows);
            Collections.shuffle(tl);
            imin = tl.get(0);

            tl = new ArrayList<Integer>(_cols);
            Collections.shuffle(tl);
            jmin = tl.get(0);
        }

        _rows.remove(imin);
        _cols.remove(jmin);

        _oneToOneTagMap.put(imin, jmin);

        if (!_rows.isEmpty() && !_cols.isEmpty()) {
            buildOneToOneMap(_costMatrix, _rows, _cols, _N, _oneToOneTagMap);
        }
    }

    /**
     * 
     * @param _costMatrix
     * @param _rows
     * @param _cols
     * @param _N
     * @param _manyToOneTagMap
     */
    protected void buildManyToOneMap(double[] _costMatrix,
          HashSet<Integer> _rows, HashSet<Integer> _cols, int _N,
          IntTagMap _manyToOneTagMap) {
        for (int i : _rows) {
            double min = 1;
            int jmin = 0;
            for (int j : _cols) {
                if (_costMatrix[i * _N + j] < min) {
                    jmin = j;
                    min = _costMatrix[i * _N + j];
                }
            }
            _manyToOneTagMap.put(i, jmin);
        }
    }

    /**
     * 
     * @param _modelTags
     * @param _goldTags
     * @param _tagMap
     * @return
     */
    protected double measureAccuracy(int[] _modelTags, int[] _goldTags,
          IntTagMap _tagMap) {
        int total = _modelTags.length;
        int correct = 0;
        for (int i = 0; i < total; ++i) {
            int j = _modelTags[i];
            if (_tagMap.get(j) == _goldTags[i]) {
                correct++;
            }
        }
        return correct / (double) total;
    }

    protected void clusterEvaluation(int[] _modelTags, int[] _goldTags,
          int _modelK, int _goldK, ClusterEvalScore _clusterEvalScore) {
        int[] confusionMatrix = new int[_modelK * _goldK];
        int[] rowSum = new int[_modelK];
        int[] columnSum = new int[_goldK];
        int N = _modelTags.length;

        for (int i = 0; i < _modelK * _goldK; ++i) {
            confusionMatrix[i] = 0;
        }
        for (int i = 0; i < _modelK; ++i) {
            rowSum[i] = 0;
        }
        for (int i = 0; i < _goldK; ++i) {
            columnSum[i] = 0;
        }

        for (int i = 0; i < N; ++i) {
            int modelidx = _modelTags[i];
            int goldidx = _goldTags[i];
            confusionMatrix[modelidx * _goldK + goldidx] += 1;
        }

        double tp = 0, fp = 0, fn = 0;

        for (int i = 0; i < _modelK; ++i) {
            for (int j = 0; j < _goldK; ++j) {
                int val = confusionMatrix[i * _goldK + j];
                tp += Math.pow(val, 2);
                columnSum[j] += val;
                rowSum[i] += val;
            }
        }

        for (int i = 0; i < _modelK; ++i) {
            fp += Math.pow(rowSum[i], 2);
        }

        for (int j = 0; j < _goldK; ++j) {
            fn += Math.pow(columnSum[j], 2);
        }

        tp -= N;
        tp /= 2;
        fn -= N;
        fn /= 2;
        fp -= N;
        fp /= 2;
        fn -= tp;
        fp -= tp;

        double precision = tp / (tp + fp);
        double recall = tp / (tp + fn);
        double fscore = 2 * precision * recall
              / (precision + recall);

        _clusterEvalScore.pairwisePrecision = precision;
        _clusterEvalScore.pairwiseRecall = recall;
        _clusterEvalScore.pairwiseFScore = fscore;

        double modelEntropy = 0;
        for (int i = 0; i < _modelK; ++i) {
            double p = rowSum[i] / (double) N;
            if (p > EPSILON) {
                modelEntropy -= p * Math.log(p);
            }
        }

        double goldEntropy = 0;
        for (int i = 0; i < _goldK; ++i) {
            double p = columnSum[i] / (double) N;
            if (p > EPSILON) {
                goldEntropy -= p * Math.log(p);
            }
        }

        double mutualInformation = 0;
        for (int i = 0; i < _modelK; ++i) {
            for (int j = 0; j < _goldK; ++j) {
                int val = confusionMatrix[i * _goldK + j];
                double jointClusterProb = val / (double) N;
                double goldClusterProb = columnSum[j] / (double) N;
                double modelClusterProb = rowSum[i] / (double) N;
                double ratio = jointClusterProb / (goldClusterProb * modelClusterProb);
                if (ratio > EPSILON) {
                    mutualInformation += jointClusterProb * Math.log(ratio);
                }
            }
        }

        double variationOfInformation = modelEntropy + goldEntropy - 2 * mutualInformation;
        _clusterEvalScore.variationOfInformation = variationOfInformation;
    }

    /**
     * @return the fullOneToOneAccuracy
     */
    public double getFullOneToOneAccuracy() {
        return fullOneToOneAccuracy;
    }

    /**
     * @return the fullManyToOneAccuracy
     */
    public double getFullManyToOneAccuracy() {
        return fullManyToOneAccuracy;
    }

    /**
     * @return the fullOneToOneAccuracy
     */
    public double getReducedOneToOneAccuracy() {
        return reducedOneToOneAccuracy;
    }

    /**
     * @return the fullManyToOneAccuracy
     */
    public double getReducedManyToOneAccuracy() {
        return reducedManyToOneAccuracy;
    }

    public double getFullPairwiseFScore() {
        return fullClusterEvalScore.pairwiseFScore;
    }

    public double getFullPairwisePrecision() {
        return fullClusterEvalScore.pairwisePrecision;
    }

    public double getFullPairwiseRecall() {
        return fullClusterEvalScore.pairwiseRecall;
    }

    public double getFullVariationOfInformation() {
        return fullClusterEvalScore.variationOfInformation;
    }

    public double getReducedPairwiseFScore() {
        return reducedClusterEvalScore.pairwiseFScore;
    }

    public double getReducedPairwisePrecision() {
        return reducedClusterEvalScore.pairwisePrecision;
    }

    public double getReducedPairwiseRecall() {
        return reducedClusterEvalScore.pairwiseRecall;
    }

    public double getReducedVariationOfInformation() {
        return reducedClusterEvalScore.variationOfInformation;
    }

    protected class ClusterEvalScore {

        public double pairwisePrecision = 0,
              pairwiseRecall = 0,
              pairwiseFScore = 0,
              variationOfInformation = 0;
    }
}
