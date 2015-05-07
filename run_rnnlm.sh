#!/bin/bash

#script to run RNNLM model 
#Note: parallelising the computation appears to decrease the performance fairly significantly.
#Best to use 1 thread to get the best performance

############
#parameters#
############

#training parameters
hidden=30
class=100
bptt=4
bpttblk=10
direct=0
dorder=3
seed=1

#input/output
train_corpus="example_dataset/cleaned-normal/train_dir/train.txt"
valid_corpus="example_dataset/cleaned-normal/test.txt" #use for stopping the RNNLM; data not seen by the model itself
test_corpus="example_dataset/cleaned-normal/test_rnnlm.txt"
gs="example_dataset/raw/test_gold_ratings.txt" #gold standard ratings
output_dir="output/rnnlm"



#############
#main script#
#############
mkdir -p $output_dir 2>/dev/null

#train model
echo 'Training the language model...'
rnnlm/rnnlm -train $train_corpus -valid $valid_corpus -rnnlm $output_dir/model.bin -hidden $hidden \
    -rand-seed $seed -debug 2 -class $class -bptt $bptt -bptt-block $bpttblk -binary \
    -direct-order $dorder -direct $direct -threads 1

#compute sentence logprobs
echo 'Computing the logprob scores for the test...'
rnnlm/rnnlm -rnnlm $output_dir/model.bin -test $test_corpus -nbest -debug 0 > $output_dir/scores.txt

#calculate correlation
echo 'Computing the correlation of the computed scores and the gold standard ratings...'
python rnnlm/calc_corr.py $output_dir/scores.txt $gs -t $output_dir/test.csv > $output_dir/test_correlation.txt
