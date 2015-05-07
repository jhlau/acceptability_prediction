#!/bin/bash
#script to run the lexical n-gram models for predicting acceptability

#parameters
order=3 #ngram order
train_corpus="example_dataset/cleaned-normal/train_dir/train.txt"
test_corpus="example_dataset/cleaned-normal/test.txt"
#gold standard ratings
gs="example_dataset/raw/test_gold_ratings.txt"
output_dir="output/ngram"

#create output directory if necessary
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

echo 'Training the language model...'
python ngram_count.py -o $order -i $train_corpus -lm $output_dir/train.lm -t

echo 'Computing the logprob scores for the test...'
python ngram.py -o $order -lm $output_dir/train.lm -test $test_corpus -r $output_dir/test.csv -t

echo 'Computing the correlation of the computed scores and the gold standard ratings...'
python calc_correlation.py $output_dir/test.csv $gs > $output_dir/test_correlation.txt
