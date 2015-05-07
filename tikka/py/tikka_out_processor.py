#! /usr/bin/python

import os, sys, re

def get_abs_path(root):
    return os.path.abspath(os.path.expanduser(root))

def get_score_finder():
    return re.compile(r"^(\d\.\d+\s*)+$")

def get_data_id_map(root):
    LCURVE_TUPLE=[ "learningcurve0008", "learningcurve0016", "learningcurve0032", \
                       "learningcurve0064", "learningcurve0128", "learningcurve0256", \
                       "learningcurve0512", "learningcurve1024" ]

    models=("m1","m2","m3","m4","m6")

    data_id_map = {}

    inpath=get_abs_path(root)

    for fi in os.listdir(inpath):
        fullpath=os.path.join(inpath,fi)
        if os.path.isfile(fullpath):
            labs = fi.split(".")
            corpus=labs[0]
            if not data_id_map.has_key(corpus):
                data_id_map[corpus] = set([])
            data_id = labs[-2]
            if data_id != "full":
                data_id_map[corpus].add(labs[-2])

    for corpus in data_id_map.iterkeys():
        idset = data_id_map[corpus]
        data_id_map[corpus] = {}
        maxid = 0
        for id in idset:
            curid = LCURVE_TUPLE.index(id)
            data_id_map[corpus][id] = curid
            if curid > maxid:
                maxid = curid
        maxid += 1
        data_id_map[corpus]["full"] = maxid

    return data_id_map

def sufstat(array_array):
    rowDim = len(array_array)
    colDim = len(array_array[0])
    ave = [0.0] * colDim
    stdev = [0.0] * colDim
    
    for array in array_array:
        for i in xrange(colDim):
            ave[i] += array[i] / float(rowDim)

    for array in array_array:
        for i in xrange(colDim):
            stdev[i] += (array[i] - ave[i])**2 / float(rowDim - 1)
            
    for i in xrange(colDim):
        stdev[i] = stdev[i] ** 0.5
            
    return ave, stdev

