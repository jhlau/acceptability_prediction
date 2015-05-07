"""
Stdin:          N/A
Stdout:         N/A
Author:         Jey Han Lau
Date:           Jul 13
"""

import argparse
import sys
from scipy.stats.mstats import pearsonr, spearmanr

#parser arguments
desc = "Calculates correlation of model scoring functions and gug ratings."
parser = argparse.ArgumentParser(description=desc)

#####################
#positional argument#
#####################
parser.add_argument("test_csv", help="csv file that contain the scoring functions")
parser.add_argument("rating_file", help="file that contains human ratings")

###################
#optional argument#
###################

args = parser.parse_args()

#parameters
debug = False


###########
#functions#
###########

######
#main#
######

#process the rating file
ratings = []
for line in open(args.rating_file):
    ratings.append(float(line.strip()))

if debug:
    print "Ratings", len(ratings), "=", ratings[:10]

#process the test.csv file
metrics = []
probs = []
for line_id, line in enumerate(open(args.test_csv)):
    data = line.strip().split(",")
    if line_id == 0:
        metrics = data[3:]
        if debug:
            print "\nmetrics =", metrics
    else:
        for i, score in enumerate(data[3:]):
            if len(probs) == i:
                probs.append([])
            if score == "":
                score = 0
            probs[i].append(float(score))

#print "\n".join(metrics), "\n"
print "METRICS\tCORRELATION"
for i, prob in enumerate(probs):
    if debug:
        print "\nmetric =", metrics[i]
        print "\tprob", len(prob), "=", prob[:5]
    corr = pearsonr(ratings, prob)[0]
    print metrics[i] + "\t" + str(corr)
        
        
