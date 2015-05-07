#! /usr/bin/python

import os, sys

root=os.path.abspath(os.path.expanduser(sys.argv[1]))

files = os.listdir(root)
files.remove("PARAMETERS")
files=[os.path.join(root,fi) for fi in files]

unmatched_gold = set()
mappings = {}

perfect_matches = []
ttoo_matches=[]
thtoo_matches=[]
mtoo_matches=[]

for fi in files:
    handle = open(fi)
    for line in handle:
        if len(line.strip()) != 0:
            tokes = line.split()
            otoo_tag = tokes[1].split(":")[1]
            mtoo_tag = tokes[2].split(":")[1]
            gold_tag = tokes[3].split(":")[1]
            try:
                mappings[mtoo_tag].add(otoo_tag)
            except KeyError:
                mappings[mtoo_tag] = set([otoo_tag])


for m,o in mappings.iteritems():
    if len(o)==1:
        perfect_matches.append((m,"".join(o)))
    elif len(o)==2:
        ttoo_matches.append((m,", ".join(o)))
    elif len(o)==3:
        thtoo_matches.append((m,", ".join(o)))
    else:
        mtoo_matches.append((m,", ".join(o)))
    
total_mtags = len(mappings)

print "Perfect matches (%d/%d):" % (len(perfect_matches),total_mtags)
for p in perfect_matches:
    print "\t%s\t%s" % p

print "2 to 1 matches (%d/%d):" % (len(ttoo_matches),total_mtags)
for p in ttoo_matches:
    print "\t%s\t%s" % p

print "3 to 1 matches (%d/%d):" % (len(thtoo_matches),total_mtags)
for p in thtoo_matches:
    print "\t%s\t%s" % p

print "M to 1 matches (%d/%d):" % (len(mtoo_matches),total_mtags)
for p in mtoo_matches:
    print "\t%s\t%s" % p
