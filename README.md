This package contains scripts and tools for doing unsupervised 
acceptability prediction.  For a full description of the software, 
please refer to the publication listed at the bottom of this document.
Datasets are hosted on our project website.

Project website: http://www.dcs.kcl.ac.uk/staff/lappin/smog/

Description
===========
In acceptability prediction, the task is to predict the acceptability of 
a given sentence. To do this, our methodology first trains unsupervised 
language models on raw text corpus, and then computes 'test' sentence 
probabilities using the trained models. To map these sentence 
probabilities to acceptability, several functions are used to normalise 
sentence length and word frequency.

Implementation of the following models are provided:
* Lexical N-gram
* Bayesian HMM
* LDAHMM
* Two-tier BHMM
* Bayesian Chunker
* RNNLM

General Workflow
================
* Prepare training data and test data in the specified format (below)
* Run clean_dataset.sh to replace unseen/low frequency tokens in both train and
test data
* Run run_ngram.sh to train/test N-gram models
* Run run_bayesian_models.sh to train/test Bayesian models
* Run run_rnnlm.sh to train/test RNNLM
* For Bayesian Chunker, since it takes word classes as input, use gen_bchunker_train.sh
to generate train input. Note that you'll need provide a previously trained model to
do this; the two-tier BHMM is a good model to use.

Format
======
For the original train data (input to clean_dataset.sh), you'll need to provide 
two formats:
* one sentence per line (used by all models except LDAHMM)
* one sentence per line with an additional document boundary token (\<PAGEBOUNDARY\>)
to denote document boundary (used by LDAHMM only)

For test data, you'll only need to provide the first format.

An example dataset is provided under directory example_dataset/raw

Installation
============
###tikka
The source code of the original Bayesian models are provided by tikka-postagger; full
documentation can be found on: http://code.google.com/p/tikka-postagger/

Modifications: implemented a few new models, and parallelised the original models.

tikka requires Java 1.6 and ant. To build the source, set the environment variables 
JAVA_HOME and TIKKA_DIR to the appropriate directories. Also, update the CLASSPATH variable so
you won't need to specify the class path every time when compiling code that uses tikka.

Example commands:
* export JAVA_HOME="/usr/lib/jvm/java-6-openjdk-amd64"
* export TIKKA_DIR="/home/exampleuser/acceptability_prediction/tikka"
* export CLASSPATH=".:/home/exampleuser/acceptabiliy_prediction/tikka/build/classes"

Once the environment variables are set, run "ant" to build tikka

Note: edit tikka/bin/tikka-env to specify the maximum amount of memory for Java to use. 

###RNNLM
The RNNLM model is provided by the RNNLM toolkit; full documentation can be found on:
http://www.fit.vutbr.cz/~imikolov/rnnlm/

Modifications were made to map the sentence probabilities to acceptability scores.

RNNLM requires c++ compiler (g++4.6 or newer) to compile the source.

To compile RNNLM, run "make"


Publication
-----------
* Jey Han Lau, Shalom Lappin and Alexander Clark (2015). 
Unsupervised Prediction of Acceptability Judgements. In Proceedings of
the 52nd Annual Meeting of the Association for Computational Linguistics 
and the 7th International Joint Conference on Natural Language 
Processing (ACL-IJCNLP 2015), Beijing, China.
