#!/bin/bash

#script to run faster RNNLM model 
#before running the code, compile faster-rnnlm, e.g.:
#cd faster-rnnlm; ./build.sh
#note that faster-rnnlm requires eigen (the build script will download it), and for larger datasets you want to run the program on a GPU-enabled machine with CUDA installed
#for more information on faster-rnnlm, check out the original repository: https://github.com/yandex/faster-rnnlm
#our version of faster-rnnlm has a few changes: (1) test inference now yields sentence unigram logprob and (smoothed) word logprobs; (2) word generation function (SampleFromLM function) now listens to stdin in a loop (note that this does not affect the acceptability prediction task)

############
#parameters#
############

#training parameters
reverse_sentence=0 #reverse the order of the sentence
char_lm=0 #turn this on to do character-level language model (default = word-level)

hidden_type="sigmoid" #sigmoid, tanh, relu, gru, gru-bias, gru-insyn, gru-full
hidden=10 #number of neurons/units in the hidden layer
hidden_count=1 #number of hidden layers
direct=0
direct_order=0

bptt=0
bptt_skip=0

seed=1
alpha=0.1 #initial learning rate
beta=0

threads=10
diagonal_initialization=0.9

#nce options
nce=10
use_cuda=0 #use cuda (GPU) for entropy calculation; turn this on only if the machine is GPU-enabled and has CUDA installed
nce_unigram_power=1.0


#input/output
train_corpus="example_dataset/cleaned-normal/train_dir/train.txt"
valid_corpus="example_dataset/cleaned-normal/test.txt" #use for stopping the RNNLM; data not seen by the model itself
test_corpus="example_dataset/cleaned-normal/test_rnnlm.txt"
gs="example_dataset/raw/test_gold_ratings.txt" #gold standard ratings
output_dir="output/faster-rnnlm"

#############
#main script#
#############
rm -rf $output_dir 2>/dev/null
mkdir -p $output_dir 2>/dev/null

#train model
echo 'Training the language model...'
ts=`python -c "print '0-' + str($threads-1)"`
taskset -c $ts faster-rnnlm/faster-rnnlm/rnnlm -train $train_corpus -valid $valid_corpus -rnnlm $output_dir/model.bin \
    -reverse-sentence $reverse_sentence \
    -hidden $hidden -hidden-type $hidden_type -hidden-count $hidden_count \
    -direct $direct -direct-order $direct_order \
    -bptt $bptt -bptt-skip $bptt_skip \
    -seed $seed -alpha $alpha -threads $threads  -diagonal-initialization $diagonal_initialization \
    -beta $beta \
    -nce $nce -use-cuda $use_cuda -nce-unigram-power $nce_unigram_power -char-lm $char_lm

#compute sentence logprobs
echo 'Computing the logprob scores for the test...'
faster-rnnlm/faster-rnnlm/rnnlm -rnnlm $output_dir/model.bin -test $test_corpus -nce-accurate-test 1 > $output_dir/scores.txt

#calculate correlation
echo 'Computing the correlation of the computed scores and the gold standard ratings...'
python rnnlm/calc_corr.py $output_dir/scores.txt $gs -t $output_dir/test.csv > $output_dir/test_correlation.txt
scores=`tail -n 1 $output_dir/test_correlation.txt`
metrics=`tail -n 2 $output_dir/test_correlation.txt | head -n 1`

cat $output_dir/test_correlation.txt

