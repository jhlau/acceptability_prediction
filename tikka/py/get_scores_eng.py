#! /usr/bin/python

import os, sys, re

BROWN_LCURVE=( "learningcurve0008", "learningcurve0016", "learningcurve0032",\
                   "learningcurve0064", "learningcurve0128", "learningcurve0256", "full" )
WSJ_LCURVE=( "learningcurve0008", "learningcurve0016", "learningcurve0032", \
                 "learningcurve0064", "learningcurve0128", "learningcurve0256", \
                 "learningcurve0512", "learningcurve1024", "full" )

score_finder=re.compile(r"^(0\.\d+\s*)+$")

brown_map={}
counter = 1
for c in BROWN_LCURVE:
    brown_map[c]="l%d" % counter
    counter += 1

wsj_map={}
counter = 1
for c in WSJ_LCURVE:
    wsj_map[c] = "l%d" % counter
    counter += 1

models=("m1","m2","m3","m4","m6")

inpath=os.path.abspath(os.path.expanduser(sys.argv[1]))

content_states=0
function_states=0
model_id=""
data_id=""
corpus=""

dataline = "model.id,corpus,data.id,function.states,content.states,states,f1to1,fmto1,r1to1,rmto1"
print dataline

for fi in os.listdir(inpath):
    fullpath=os.path.join(inpath,fi)
    if os.path.isfile(fullpath):
        labs = fi.split(".")
        corpus=labs[0]
        if corpus == "brown":
            data_id=brown_map[labs[-2]]
        else:
            data_id=wsj_map[labs[-2]]
        model_id=labs[-3]
        function_states=labs[-5]
        content_states=labs[-4]
        states = "%d" % (int(function_states) + int(content_states))
        handle = open(fullpath)
        scores = ""
        for line in handle:
            m = score_finder.search(line)
            if m:
                scores=line
                break
        if len(scores) > 0:
            scores = scores.split()
            f1to1 = scores[0]
            fmto1 = scores[1]
            r1to1=scores[2]
            rmto1=scores[3]

            datam = {"model_id":model_id, "data_id":data_id, "corpus":corpus, \
                         "function_states":function_states, "content_states":content_states, \
                         "f1to1":f1to1, "fmto1":fmto1, "r1to1":r1to1, "rmto1":rmto1,"states":states}

            dataline = "%(model_id)s,%(corpus)s,%(data_id)s,%(function_states)s,%(content_states)s,%(states)s,%(f1to1)s,%(fmto1)s,%(r1to1)s,%(rmto1)s" % datam
            print dataline
