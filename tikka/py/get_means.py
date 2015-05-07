#! /usr/bin/python

import os, sys, re
from tikka_out_processor import *

data_id_map = get_data_id_map(sys.argv[1])
score_finder = get_score_finder()
inpath = get_abs_path(sys.argv[1])
stat_table = {}

for fi in os.listdir(inpath):
    fullpath=os.path.join(inpath,fi)
    if os.path.isfile(fullpath):
        labs = fi.split(".")
        corpus=labs[0]
        iter_id = int(labs[1])
        data_id = data_id_map[corpus][labs[-2]]
        model_id=labs[-3]
        function_states=int(labs[-4])
        content_states=int(labs[-5])
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
                if not stat_table.has_key(corpus):
                    stat_table[corpus] = { \
                        data_id : {\
                            model_id: { \
                                function_states : { \
                                    content_states :
                                        []
                                    } \
                                    } \
                                } \
                            }
                elif not stat_table[corpus].has_key(data_id):
                    stat_table[corpus][data_id] = { \
                        model_id: { \
                            function_states : { \
                                content_states :
                                    []
                                } \
                                } \
                            } 
                elif not stat_table[corpus][data_id].has_key(model_id):
                    stat_table[corpus][data_id][model_id] = { \
                        function_states : { \
                            content_states :
                                []
                            } \
                            }
                elif not stat_table[corpus][data_id][model_id].has_key(function_states):
                    stat_table[corpus][data_id][model_id][function_states] = { \
                        content_states :
                            []
                        }
                elif not stat_table[corpus][data_id][model_id][function_states].has_key(content_states):
                    stat_table[corpus][data_id][model_id][function_states][content_states] = []

                stat_table[corpus][data_id][model_id][function_states][content_states].append([float(x) for x in scores])

for corpus, data_id_table in stat_table.iteritems():
    for data_id, model_id_table in data_id_table.iteritems():
        for model_id, function_state_table in model_id_table.iteritems():
            for function_states, content_state_table in function_state_table.iteritems():
                for content_states, array_array in content_state_table.iteritems():
                    content_state_table[content_states] = sufstat(array_array)

for corpus, data_id_table in stat_table.iteritems():
    for data_id, model_id_table in data_id_table.iteritems():
        for model_id, function_state_table in model_id_table.iteritems():
            for function_states, content_state_table in function_state_table.iteritems():
                for content_states, ave_stdev in content_state_table.iteritems():
                    ave = ave_stdev[0]
                    stdev = ave_stdev[1]
                    if data_id == data_id_map[corpus]["full"] and corpus == "floresta":
#                    if data_id == data_id_map[corpus]["full"] and (function_states+content_states) >= 50:
#                        try:
                        print corpus, model_id, content_states, function_states,
                        print " & ".join("%.2f (%.2f)" % tup for tup in zip(ave, stdev))
#                        except TypeError:
#                            print ave
#                            print stdev
