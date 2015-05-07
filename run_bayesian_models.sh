#!/bin/bash
#script to run the bayesian models (tikka)

############
#parameters#
############

#m2 = bhmm, m4 = ldahmm, m7 = two-tier bhmm, m10 = bayesian chunker
model_type="m2"

#input/output
#Note1: remember to use cleaned-conll2 (conll2 format) for m4 (ldahmm)
#Note2: as m10 takes word classes as input for training, use
#OutputHMMInducedClasses.java (gen_bchunker_train.sh) to generate the training data from 
#previously trained model (e.g. m7)
train_dir="example_dataset/cleaned-conll1/train_dir/"
test_corpus="example_dataset/cleaned-conll1/test.txt"
output_dir="output/$model_type"

#gold standard ratings
gs="example_dataset/raw/test_gold_ratings.txt"

#for bayesian chunker (m10) it needs previously trained model to generate classes during test inference
hmm2t_model="output/m7/train.model"

#training parameters
seed=1
num_threads=8 #number of threads for parallelisation
num_itr=100 #number of iterations

#model parameters
#BHMM parameters: num_states, gamma, delta
#LDAHMM paramters: num_states, num_topics, alpha, beta, gamma, delta
#2T-BHMM parameters: num_chunks, num_states, alpha, gamma, delta
#B. Chunker parameters: alpha, beta, phash
#Note: for unused parameters, you can use any value (e.g. 0)
num_states=30
num_topics=0
num_chunks=0
gamma=1.0 
alpha=10.0
beta=0.1
delta=0.01
phash=0.5


######
#main#
######

#create output directory
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

#run training
echo 'Training the model...'
cd tikka/bin
./cdhmm-train.sh -e $model_type -m ../../$output_dir/train.model \
    -d ../../$train_dir -ut n \
    -itr $num_itr -pt 1 -pd 0.1 -pi 1 -sc 0 -sf $num_states \
    -r $seed --gamma $gamma --delta $delta --alpha $alpha --beta $beta --topics $num_topics \
    -xthread $num_threads -xc $num_chunks -xp $phash -xt 

#compute log probabilities for test
echo 'Computing the logprob scores for the test...'
cd ../../tikka_test_inference
if [ $model_type != "m10" ]
then
    javac TestProbSimple.java
    java TestProbSimple -l ../$output_dir/train.model -z  \
        ../$test_corpus > ../$output_dir/test.csv
else
    javac TestProbBayesianChunker.java
    java -Xmx4G TestProbBayesianChunker -l ../$hmm2t_model -z ../$test_corpus \
    -xchunker ../$output_dir/train.model > ../$output_dir/test.csv
fi

#compute correlation
echo 'Computing the correlation of the computed scores and the gold standard ratings...'
cd ..
python calc_correlation.py $output_dir/test.csv $gs > $output_dir/test_correlation.txt
