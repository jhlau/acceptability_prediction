"""
Stdin:          N/A
Stdout:         N/A
Author:         Jey Han Lau
Date:           Oct 14
"""

import argparse
import sys
from scipy.stats.mstats import pearsonr, spearmanr
import numpy
from collections import defaultdict
import math


#parser arguments
desc = "Calculates the correlation of grammaticality measures produced by the RNNLM model."
parser = argparse.ArgumentParser(description=desc)

#####################
#positional argument#
#####################
parser.add_argument("logprob_file", \
    help="text file that contains the sentence logprobs output by the system")
parser.add_argument("gold", help="gold standard mean rating file (e.g. ratings_4cat.txt)")

###################
#optional argument#
###################
parser.add_argument("-t", "--test_csv_output", help="test.csv output (for SVR)");
group = parser.add_argument_group("train_test", "train/test corpora; include these " + \
    "arguments if the model does not collect unigram probabilities (e.g. rwthlm)")
group.add_argument("-e", "--test_file", help="test corpus")
group.add_argument("-r", "--train_file", help="train corpus")

args = parser.parse_args()

#parameters
debug = True


###########
#functions#
###########
def mean_of_percentile (data, n):
    """Gets the mean of the values lower or equal to the Nth percentile """
    p = percentile(data, n)
    low_values = filter(lambda x : x <= p, data)
    if len(low_values) == 0:
        mean = data[0]
    else:
        mean = sum(low_values) / len(low_values)
    return mean

def percentile(values,p):
    """
    Computes the percentile of a list of values using the NIST method.
    
    @parameter values - a list of values
    @parameter p - the percentage, a float value from in the [0.0,100.0) interval
    
    @return - the percentile
    """
    sorted_values = sorted(values)
    N = len(sorted_values) - 1
    rank = p / 100. * (N + 1)
    k = int(rank)
    d = rank - k
    if k == 0:
        return sorted_values[0]
    elif k == N:
        return sorted_values[-1]
    else:
        return sorted_values[k-1] + d * (sorted_values[k] - sorted_values[k-1])

######
#main#
######

#train and test files are mutually inclusive
if (args.test_file and not args.train_file) or (not args.test_file and args.train_file):
    parser.error("-e and -r arguments are mutually inclusive.")

#globals
sent_lens = []
unigram_lp = []
mean_lp = []
norm_lp_div = []
norm_lp_sub = []
slor = []
wlp_min1 = []
wlp_min2 = []
wlp_min3 = []
wlp_min4 = []
wlp_min5 = []
wlp_mean = []
wlp_m1q = []
wlp_m2q = []

#get the system logprobs and gold standard mean ratings
gold = [float(item) for item in open(args.gold)]
lps =[]
wordlps = []
for line in open(args.logprob_file):
    data = line.strip().split()
    lps.append(float(data[0]))
    sent_lens.append(int(data[1]))
    unigram_lp.append(float(data[2]))
    wordlps.append([float(item) for item in data[3].split(",")])

#reprocess the unigram probabilities if train/test files are given
unigram_freq = defaultdict(int)
total_sents = 0
total_words = 0
#get unigram frequency
if args.train_file:
    for line in open(args.train_file):
        for w in line.strip().split():
            unigram_freq[w] += 1
            total_words += 1
        total_sents += 1
    #compute sentence boundary marker logprob
    logprob_sb = math.log( (float(total_sents)/total_words), 10 )
    for i, line in enumerate(open(args.test_file)):
        sent_unigram_logprob = 0.0
        #check the number of words in test sentence matches
        words = line.strip().split()    
        if len(wordlps[i]) - len(words) != 1:
            print "wordlps[i] and len(test_sentence) mismatch"
            print i, line
            print "len(wordlps[i]) =", len(wordlps[i])
            print "len(test_sentence) =", len(words)
            raise SystemExit
        for j, w in enumerate(words):
            logprob_w = math.log( (float(unigram_freq[w])/total_words), 10 )
            sent_unigram_logprob += logprob_w
            wordlps[i][j] = wordlps[i][j] / logprob_w * -1.0
        #sentence boundary marker
        wordlps[i][len(words)] = wordlps[i][len(words)] / logprob_sb * -1.0  
        #update unigram_lp
        unigram_lp[i] = sent_unigram_logprob

#test.csv output
header = "id,ppl,sent_length,logprob,unigram_logprob,mean_logprob,norm_logprob_div,norm_logprob_sub,slor,wlogprob-bot-1,wlogprob-bot-2,wlogprob-bot-3,wlogprob-bot-4,wlogprob-bot-5,wlogprob_mean,wlogprob_m1q,wlogprob_m2q"
if (args.test_csv_output):
    test_out = open(args.test_csv_output, "w")
    #header
    test_out.write(header + "\n")

#calculate the grammaticality measures
for i in range(0, len(sent_lens)):
    lp = lps[i]
    mean_lp.append(lp / sent_lens[i])
    norm_lp_div.append((-1.0 * lp) / unigram_lp[i])
    norm_lp_sub.append(lp - unigram_lp[i])
    slor.append( (lp - unigram_lp[i]) / sent_lens[i] )

    #bottom 5 lowest word logprobs, mean, m1q and m2q
    wordlp = wordlps[i]
    wordlp_min5 = sorted(wordlp)[:5]
    wlp_min1.append(wordlp_min5[0])
    wlp_min2.append(wordlp_min5[1])
    wlp_min3.append(wordlp_min5[2])
    wlp_min4.append(wordlp_min5[3])
    wlp_min5.append(wordlp_min5[4])
    wlp_mean.append(numpy.mean(wordlp))
    wlp_m1q.append(mean_of_percentile(wordlp, 25.0))
    wlp_m2q.append(mean_of_percentile(wordlp, 50.0))

    if (args.test_csv_output):
        test_out.write(str(i) + ",," + str(sent_lens[i]) + "," + str(lp) + "," + str(unigram_lp[i]))
        test_out.write("," + str(mean_lp[-1]) + "," + str(norm_lp_div[-1]) + ",")
        test_out.write(str(norm_lp_sub[-1]) + "," + str(slor[-1]) + ",")
        test_out.write(",".join([str(item) for item in wordlp_min5]) + ",")
        test_out.write(str(wlp_mean[-1]) + "," + str(wlp_m1q[-1]) + "," + str(wlp_m2q[-1]) + "\n")

metrics_list = header.split(",")[3:]
results = [lps, unigram_lp, mean_lp, norm_lp_div, norm_lp_sub, slor, wlp_min1, wlp_min2, wlp_min3, wlp_min4, wlp_min5, wlp_mean, wlp_m1q, wlp_m2q]

#print the results
print "METRICS\tCORRELATION"
for i, m in enumerate(metrics_list):
    print m + "\t" + str(pearsonr(results[i], gold)[0])
