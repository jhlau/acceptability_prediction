# A simple script to produce ngram counts

import argparse
from bncpylib.util import safe_open_with_encoding, tag_separator, Timer
import sys
from bncpylib.ngrams import LaplaceSmoothingModel, KneserNeySmoothingModel, ContextNgramKneserNey, TagBasedKneserNeySmoothingModel, BasicCountModel, StingyKneserNeySmoothingModel

smoothing_methods = ['laplace', 'kneser-ney', 'tag-kneser-ney', 'context-kneser-ney', 'none', 'sniff','stingy-kneser-ney']

def process_arguments():
    parser = argparse.ArgumentParser(description='This script trains a n-gram language model from a training corpus and saves it to disk.')
    parser.add_argument('-i', '--input', type=str, metavar='CORPUS',required=True,
                        help='reads input from CORPUS. The corpus is a list of sentences, one per line')
    parser.add_argument('-ts', type=str, metavar='CORPUS',required=False,
                        help='Used only by the stingy version of Kneser-Ney')
    parser.add_argument('-lm', type=str, metavar='LANGUAGE_MODEL',required=True,
                        help='write the language model to OUTPUT (must be a file) instead of stdout')
    parser.add_argument('-o','--order',type=int, metavar='ORDER',
                        help='the order of the generated model. It defaults to 3 (trigrams).',
                        default=3)
    parser.add_argument('-s', '--smoothing', type=str, metavar='SMOOTHING_METHOD',
                        help='applies the specified smoothing method. Currently supported smoothing methods: {0} . If no technique is supplied (or if an unrecognized method is specified) it defaults to a Kneser-Ney model. sniff is a special flag that tells the program to pick the most appropriate one. Practically it checks if the corpus is tagged, in which case it uses cluster based Kneser-Ney otherwise it defaults to Kneser-Ney'.format(' | '.join(smoothing_methods)),
                        default='kneser-ney')
    parser.add_argument('-c', '--cutoff', type=int, help='Ngrams with count equal to or smaller than CUTOFF are not recorded',
                        default=0)
    kn_estimate_group = parser.add_mutually_exclusive_group()
    kn_estimate_group.add_argument('--knestimate', type=file, metavar='CORPUS',
                                   help='Estimate the discount parameter from CORPUS. The file CORPUS is read multiple times which makes the process extremely slow')
    kn_estimate_group.add_argument('--knestimate_in_memory', type=file, metavar='CORPUS',
                                   help='Estimate the discount parameter from CORPUS. The file CORPUS is entirely load in memory which makes the optimization process faster but that may be a problem if CORPUS is too big.',dest='knestimate_in_memory')
    parser.add_argument('-t', '--tell_me_when', help='When passed this option the script tries to estimate the time it will take to end the construction of the model (it will always underestimate the true time as it does not take into account the time needed to save the model)', action='store_true')
    args = parser.parse_args()
    args.input = safe_open_with_encoding(args.input,'r')
    if not args.ts is None:
        args.ts = safe_open_with_encoding(args.ts,'r')
                
    # remember to add more as we add techniques
    args.smoothing = args.smoothing.strip()
    if not (args.smoothing in smoothing_methods):
        sys.stderr.write('Warning: unrecognized smoothing method {0}, generating a Kneser-Ney model'.format(args.smoothing))
        args.smoothing = 'kneser-ney'

    return args

def main(args):

    # train the model
    lm = None

    if args.tell_me_when:
        tot_lines = 0
        for line in args.input:
            tot_lines += 1
        args.input.seek(0)
        timer = Timer(tot_lines)
    else:
        timer = None        
    # check if we need to sniff the smoothing method
    if args.smoothing == 'sniff':
        line = args.input.readline()
        while (line == ''):
            line = args.input.readline()
        args.input.seek(0) # we reset the corpus reading position
        tokens = line.split()
        if tag_separator in tokens[0]:
            args.smoothing = 'tag-kneser-ney'
        else:
            args.smoothing = 'kneser-ney'
    
    if args.smoothing == 'laplace':
        lm = LaplaceSmoothingModel(train_corpus=args.input, order=args.order,cutoff=args.cutoff,timer=timer)
    elif args.smoothing == 'none':
        lm = BasicCountModel(train_corpus=args.input, order=args.order,cutoff=args.cutoff,timer=timer)
    elif args.smoothing == 'kneser-ney':
        lm = KneserNeySmoothingModel(train_corpus=args.input, order=args.order,timer=timer)
        if args.knestimate != None:
            lm.estimate_discount_parameter(args.knestimate)
        elif args.knestimate_in_memory:
            lm.estimate_discount_parameter(args.knestimate_in_memory,load_text_in_memory=True)
        else:
            lm.discount = 0.7 # this is always good
    elif args.smoothing == 'stingy-kneser-ney':
        lm = StingyKneserNeySmoothingModel(train_corpus=args.input, test_corpus=args.ts, order=args.order,timer=timer)
        if args.knestimate != None:
            lm.estimate_discount_parameter(args.knestimate)
        elif args.knestimate_in_memory:
            lm.estimate_discount_parameter(args.knestimate_in_memory,load_text_in_memory=True)
        else:
            lm.discount = 0.7 # this is always good            
    elif args.smoothing == 'context-kneser-ney':
        lm = ContextNgramKneserNey(train_corpus=args.input, order=args.order,timer=timer)
        if args.knestimate != None:
            lm.estimate_discount_parameter(args.knestimate)
        elif args.knestimate_in_memory:
            lm.estimate_discount_parameter(args.knestimate_in_memory,load_text_in_memory=True)
        else:
            lm.discount = 0.7 # this is always good
    elif args.smoothing == 'tag-kneser-ney':
        lm = TagBasedKneserNeySmoothingModel(train_corpus=args.input, order=args.order,timer=timer)
        if args.knestimate != None:
            lm.estimate_discount_parameter(args.knestimate)
        elif args.knestimate_in_memory:
            lm.estimate_discount_parameter(args.knestimate_in_memory,load_text_in_memory=True)
        else:
            lm.discount = 0.7 # this is always good

    if not timer is None:
        timer.done()

    # save the model
    lm.save(args.lm)
    
if __name__ == '__main__':
    args = process_arguments()
    main(args)
