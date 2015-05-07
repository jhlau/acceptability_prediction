#!/bin/bash
#script to replace unseen or low frequency tokens with UNK tokens.

#parameters
#input
train_file="example_dataset/raw/train.txt" #no document boundary information
train_file_doc="example_dataset/raw/train.doc.txt" #document boundary information contained in file
test_file="example_dataset/raw/test.txt"
#output
#3 different output formats for the training file:
#'normal' = one sentence per line (used by n-gram and RNNLM models)
#'conll1' = one word per line (used by the Bayesian models)
#'conll2' = one word per line, one file per document (used by LDAHMM)
output_normal="example_dataset/cleaned-normal"
output_conll1="example_dataset/cleaned-conll1"
output_conll2="example_dataset/cleaned-conll2"


#create output directories
mkdir -p $output_normal/train_dir 2>/dev/null
mkdir -p $output_conll1/train_dir 2>/dev/null
mkdir -p $output_conll2/train_dir 2>/dev/null

#process the train file and save the vocabulary, and then process the test file
#'normal' format
python replace_low_freq_tokens.py $train_file $output_normal/train_dir/train.txt -of normal -tr $output_normal/lexicon.pickle
python replace_low_freq_tokens.py $test_file $output_normal/test.txt -of normal -te $output_normal/lexicon.pickle
#'conll1' format
python replace_low_freq_tokens.py $train_file $output_conll1/train_dir/train.txt -of conll1 -tr $output_conll1/lexicon.pickle
python replace_low_freq_tokens.py $test_file $output_conll1/test.txt -of conll1 -te $output_conll1/lexicon.pickle
#'conll2' format (test still uses the conll1 format as it has no document boundary information)
python replace_low_freq_tokens.py $train_file_doc $output_conll2/train_dir -of conll2 -tr $output_conll2/lexicon.pickle
python replace_low_freq_tokens.py $test_file $output_conll2/test.txt -of conll1 -te $output_conll2/lexicon.pickle

#test file for rnnlm
sed -e 's/^/1 /' $output_normal/test.txt > $output_normal/test_rnnlm.txt
