#! /usr/bin/python

import os, sys, re
from tikka_out_processor import *

data_id_map = get_data_id_map(sys.argv[1])
score_finder = get_score_finder()
inpath = get_abs_path(sys.argv[1])

dataline = "model.id,corpus,data.id,function.states,content.states,states,f1to1,fmto1,r1to1,rmto1,fprecision,frecall,ffscore,fvi,rprecision,rrecall,rfscore,rvi,is.last"
print dataline

for fi in os.listdir(inpath):
    fullpath=os.path.join(inpath,fi)
    if os.path.isfile(fullpath):
        labs = fi.split(".")
        corpus=labs[0]
        data_id = data_id_map[corpus][labs[-2]]
        model_id=labs[-3]
        function_states=labs[-4]
        content_states=labs[-5]
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
            if len(scores) == 12:
                if corpus == "usp" and function_states == "95":
                    continue
                datam = {"model_id":model_id, "data_id":data_id, "corpus":corpus, \
                             "function_states":function_states, "content_states":content_states, \
                             "states":states}
                islast = "0"
                if data_id_map[corpus]["full"] == data_id:
                    islast = "1"
                scores.append(islast)
                dataline = "%(model_id)s,%(corpus)s,%(data_id)s,%(function_states)s,%(content_states)s,%(states)s" % datam
                dataline= ",".join([dataline] + scores)
                print dataline
