#! /usr/bin/python

import os, sys, re

def sufstat(array):
    ave = 0
    stdev = 0
    if len(array) > 0:
        ave = sum(array) / len(array)
        stdev = (sum([(i - ave)**2 for i in array]) / (len(array) - 1)) ** 0.5
    return (ave, stdev, len(array), len(array) == 10)

score_finder=re.compile(r"^(0\.\d+\s*)+$")

models=("m1","m2","m3","m4","m6")
evals=("f1to1","fmto1","r1to1","rmto1")

inpath=os.path.abspath(os.path.expanduser(sys.argv[1]))

content_states=0
function_states=0
model_id=""
data_id=""
corpus=""

table = {"brown":{}, "wsj":{}}
for corp, subt in table.iteritems():
    for model in models:
        subt[model] = {20:{}, 30:{}, 40:{}, 50:{}}
        for states,scores in subt[model].iteritems():
            for ev in evals:
                scores[ev] = []

for fi in os.listdir(inpath):
    fullpath=os.path.join(inpath,fi)
    if os.path.isfile(fullpath):
        labs = fi.split(".")
        corpus=labs[0]
        if labs[-2] != "full":
                continue
        model_id=labs[-3]
        function_states=int(labs[-5])
        content_states=int(labs[-4])
        states=function_states+content_states
        handle = open(fullpath)
        scores = ""
        for line in handle:
            m = score_finder.search(line)
            if m:
                scores=line
                break
        if len(scores) > 0:
            scores = scores.split()
            f1to1 = float(scores[0])
            fmto1 = float(scores[1])
            r1to1 = float(scores[2])
            rmto1 = float(scores[3])
            table[corpus][model_id][states]["f1to1"].append(f1to1)
            table[corpus][model_id][states]["fmto1"].append(fmto1)
            table[corpus][model_id][states]["r1to1"].append(r1to1)
            table[corpus][model_id][states]["rmto1"].append(rmto1)

addendum = "The following exps are not yet done:\n"

stringscore = {}
for mod in models:
    stringscore[mod] = {20:[0]*8,30:[0]*8,40:[0]*8,50:[0]*8}
    
mapper = {}
mapper["brown"] = {}
mapper["wsj"] = {}
mapper["wsj"]["f1to1"] = 0
mapper["wsj"]["fmto1"] = 1
mapper["wsj"]["r1to1"] = 2
mapper["wsj"]["rmto1"] = 3
mapper["brown"]["f1to1"] = 4
mapper["brown"]["fmto1"] = 5
mapper["brown"]["r1to1"] = 6
mapper["brown"]["rmto1"] = 7

for corpus, subt in table.iteritems():
    for mid, subsubt in subt.iteritems():
        for states, subsubsubt in subsubt.iteritems():
            for evals, scores in subsubsubt.iteritems():
                score = sufstat(scores)
                stringscore[mid][states][mapper[corpus][evals]] = "%.2f (%.2f)" % score[:2]
                if not score[3]:
                    addendum += "\t%s, %s, %d, %s\n" % (corpus, mid, states, evals)

for mid, subt in stringscore.iteritems():
    print mid, ":"
    print "%%"
    for states in [20, 30, 40, 50]:
        line = "\t & %d & " % states
        line += " & ".join(subt[states]) + r" \\ \cline{2-10}"
        print line
        print "%%"

print addendum
