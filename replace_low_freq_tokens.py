"""
Stdin:          N/A
Stdout:         N/A
Other Input:    N/A
Other Output:   N/A
Author:         Jey Han Lau
Date:           Nov 13
"""

import argparse
import sys
import pickle
import re
import shutil
import os
import codecs
from collections import defaultdict

#parser arguments
desc = "Replaces low frequency token using Berkeley style signatures. In training mode it " + \
    "additionally outputs lexicon; in testing mode it accepts an input lexicon for substitution."
parser = argparse.ArgumentParser(description=desc)

#####################
#positional argument#
#####################
parser.add_argument("input", help="input file")
parser.add_argument("output", help="output file/dir which has low frequency tokens replaced")

###################
#optional argument#
###################
parser.add_argument("-of", "--output_format", type=str, choices=["normal", "conll1", "conll2"], \
    help="normal style or CONLL1 (one file for all documents) or CONLL2 (one file for each " + \
    "document)", default="normal")
parser.add_argument("-ft", "--frequency_threshold", help="minimum frequency to accept a token " + \
    "(only used for TRAIN option", type=int, default=4)
parser.add_argument("-d", "--disable_en_unk_sig", \
    help="disable the use of English signatures for UNK tokens", action="store_true")
parser.add_argument("-u", "--utf8", help="use utf-8 encoding for reading/writing files", \
    action="store_true")

###########################
#mutually exclusive groups#
###########################
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-tr", "--train", help="destination to save the lexicon")
group.add_argument("-te", "--test", help="input lexicon for token subtistution")

args = parser.parse_args()

#parameters
debug = False


###########
#functions#
###########
#replace token with berkeley style signature
def replace_token_with_signature(token):
    digit_matcher = re.compile("\d")

    new_token = "UNK"
    if token.istitle():
        new_token += "-TC"
    if "-" in token:
        new_token += "-DASH"
    if digit_matcher.search(token) != None:
        new_token += "-NUM"

    #suffix signatures
    if not args.disable_en_unk_sig:
        if token.endswith("s") and (len(token) >= 3):
            new_token += "-s"
        elif (len(token) >= 5):
            if token.endswith("ed"):
                new_token += "-ed"
            elif token.endswith("ing"):
                new_token += "-ing"
            elif token.endswith("ion"):
                new_token += "-ion"
            elif token.endswith("er"):
                new_token += "-er"
            elif token.endswith("est"):
                new_token += "-est"
            elif token.endswith("ly"):
                new_token += "-ly"
            elif token.endswith("ity"):
                new_token += "-ity"
            elif token.endswith("y"):
                new_token += "-y"
            elif token.endswith("al"):
                new_token += "-al"

    if debug:
        print "\nOld Token =", token
        print "New Token =", new_token

    return new_token

######
#main#
######
lexicon = defaultdict(int)
#if train, generate the lexicon
if args.train:
    lines = []
    if args.utf8:
        lines = codecs.open(args.input, "r", "utf-8")
    else:
        lines = open(args.input)

    for line in lines:
        for token in line.strip().split():
            token = token.lower()
            lexicon[token] += 1

    #remove tokens that has frequency less than N
    for token,freq in lexicon.items():
        if freq < args.frequency_threshold:
            del lexicon[token]

            if debug:
                print "Token =", token,
                print "Freq =", freq,
                print "Removed"

    #output the lexicon
    pickle.dump(lexicon, open(args.train, "w"))

    if debug:
        print "\nLexicon Token Frequency:"
        for token, freq in sorted(lexicon.items()):
            print token, "=", freq
    
#if test, read the lexicon
elif args.test:
    lexicon = pickle.load(open(args.test))


#now that we have the lexicon, replace the tokens
#create the output file / directory
if (args.output_format == "normal") or (args.output_format == "conll1"):
    if args.utf8:
        output_file = codecs.open(args.output, "w", "utf-8")
    else:
        output_file = open(args.output, "w")
elif args.output_format == "conll2":
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)
token_dist = defaultdict(int)
doc_id = 0
new_doc = True

lines = []
if args.utf8:
    lines = codecs.open(args.input, "r", "utf-8")
else:
    lines = open(args.input)

for line_id, line in enumerate(lines):
    if line.strip() == "<PAGEBOUNDARY>":
        new_doc = True
        doc_id += 1
        continue

    if args.output_format == "conll2" and new_doc:
        if args.utf8:
            output_file = codecs.open(args.output + "/" + str(doc_id).zfill(15) + ".txt", "w", \
                "utf-8")
        else:
            output_file = open(args.output + "/" + str(doc_id).zfill(15) + ".txt", "w")
        new_doc = False

    token_list = []
    for token in line.strip().split():
        token = token.lower()
        new_token = token
        if token not in lexicon:
            new_token = replace_token_with_signature(token)
        token_list.append(new_token)
        token_dist[new_token] += 1

    if args.output_format == "normal":
        output_file.write(" ".join(token_list) + "\n")
    elif (args.output_format == "conll1") or (args.output_format == "conll2"):
        output_file.write("\n".join(token_list) + "\n\n")
    output_file.flush()

if debug:
    print "\nToken Frequency (Post Substitution):"
    for token, freq in sorted(token_dist.items()):
        print token, "=", freq
