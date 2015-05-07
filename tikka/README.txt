Requirements
============

* Version 1.6 of the Java 2 SDK (http://java.sun.com)


Configuring your environment variables
======================================

The easiest thing to do is to set the environment variables JAVA_HOME
and TIKKA_DIR to the relevant locations on your system. Set JAVA_HOME
to match the top level directory containing the Java installation you
want to use.

Building the system from source
===============================

The tikka build system is based on Apache Ant.  Ant is a little but very
handy tool that uses a build file written in XML (build.xml) as building
instructions.  Building the Java portion of tikka is accomplished using the
script `ant'; this works under Windows and Unix, but requires that
you run it from the top-level directory (where the build.xml file is
located).

Trying it out
=============

There are two main executable shells in $TIKKA_DIR/bin for training
the model from unlabeled text and tagging this text with model
tags. The script for training is cdhmm-train.sh and the script for
tagging is cdhmm-tag.sh.

The system can run with a raw untagged corpus. In such cases, each
token (including punctuation) should be on a separate
line. Furthermore, for models where document boundaries are relevant,
each file should correspond to a single document.

It can also run with tagged texts of specific format to allow direct
evaluation. In such cases the following corpora can be used with the
model:

* Penn Treebank WSJ
* Brown corpus
* Tiger corpus
* Floresta

These texts must be converted to CONLL format (i.e. one token per
line, token tab separated by tag on each line, and empty lines between
sentences) and separate documents must be kept in separate
files.

Let's call the directory where all the converted documents have been
placed $TRAIN. Then the model used in the paper (see "Reference"
below) can be run as follows:

cdhmm-train.sh -e $MODEL_TYPE -m $PATH_TO_MODEL -d $TRAIN -ut $TAG_SET

Besides -d $TRAIN, the commandline options that need to be set are:

* $MODEL_TYPE: This determines whether the HMM, HMM+, LDAHMM or CDHMM
  will be run. The values are "m2" for HMM, "m3" for HMM+, "m4" for
  LDAHMM and "m6" for CDHMM.
* $PATH_TO_MODEL: Path to store the binary model file.
* $TAG_SET: The tagset of the data you are using. This can be "n" for
  none/raw, "b" for Brown, "p" for Penn Treebank, "t" for Tiger or "f"
  for floresta.

To replicate the experiments in the paper, run:

cdhmm-train.sh -e $MODEL_TYPE -m $PATH_TO_MODEL -d $TRAIN -ut $TAG_SET \
-itr 1000 -pt 1 -pd 0.1 -pi 1 -sc 5 -sf 45 -t 50  -r 0

See "Full command line options" below for details of the arguments.

To tag the text with the model, run

cdhmm-tag.sh -l $PATH_TO_MODEL -oe $PATH_TO_EVAL_OUTPUT \
-n $PATH_TO_OUTPUT -d $TRAIN

The arguments are:

* $PATH_TO_MODEL: Path to the trained binary model file
* $PATH_TO_EVAL_OUTPUT: Path to the evaluation score. This will be 
  stored in a file
* $PATH_TO_OUTPUT: This must be a directory. Path to store the model
  tagged output

Interpreting the Output
=======================

If the model is run on a text, it will generate output such as follows
(the Brown corpus in this case):

it      N:0     F:nn    R:nn    GF:pps  	GR:N
wasn't  N:0     F:nn    R:nn    GF:bedz*        GR:V
just    N:0     F:nn    R:nn    GF:rb   	GR:ADV
the     N:7     F:at    R:at    GF:at   	GR:DET

The first column (labeled N) is the index of the state assigned by the
(CD)HMM model. If it has been specified that there are to be C content
tags and U function word tags, then the indexes 0 to C-1 are content
words and the remaining C to C+U-1 are function words. The F column is
based on the "one-to-one mapping" in the field (see Mark Johnson's
2007 paper "Why doesn't EM find good HMM POS-taggers") and it's the
tag assigned to the numerical index in N during the eval process. The
R column is based on the many-to-one mapping and again, it's the tag
assigned to the numerical index in N. The GF is the original Brown tag
and GR is a reduced tag based on the 17 tag set reduction used by Noah
Smith in "Contrastive estimation: Training log-linear models on
unlabeled data" (2005). If you're curious about the details of the
mappings between some English tag set and the reduced tags, the
relevant portion of the code is in
$TIKKA_DIR/java/tikka/utils/postags/BrownTags.java and
$TIKKA_DIR/java/tikka/utils/postags/PennTags.java.

Questions and bug reports
=========================

If you have any questions or bugs to report, contact the author
tsunmoon@gmail.com.

Reference
=========

Please refer to the following document for a full description and
refer to this should you use it in your experiments.

@InProceedings{moon-erk-baldridge:2010:EMNLP,
  author    = {Moon, Taesun  and  Erk, Katrin  and  Baldridge, Jason},
  title     = {Crouching Dirichlet, Hidden Markov Model: Unsupervised
  {POS} Tagging with Context Local Tag Generation},
  booktitle = {Proceedings of the 2010 Conference on Empirical Methods
  in Natural Language Processing},
  month     = {October},
  year      = {2010},
  address   = {Cambridge, MA},
  publisher = {Association for Computational Linguistics},
  pages     = {196--206},
  url       = {http://www.aclweb.org/anthology/D/D10/D10-1020}
}

Full command line options for cdhmm-train.sh
============================================

 -a,--alpha <arg>                             alpha value
                                              (default=50/topics)
 -b,--beta <arg>                              beta value (default=0.1)
 -c,--data-format <arg>                       format of input data
                                              [conll2k, hashslash, pipesep, raw; default=conll2k]
 -d,--train-data-dir <arg>                    full path to directory
                                              containing training documents
 -e,--experiment-model <arg>                  model to use [m1,m2,m3;
                                              default=m1]
 -f,--test-data-dir <arg>                     full path to directory
                                              containing test documents
 -g,--gamma <arg>                             set gamma value
                                              (default=0.1)
 -h,--help                                    print help
 -ite,--test-iterations <arg>                 number of test set burn in
                                              iterations (default=10)
 -itr,--training-iterations <arg>             number of training
                                              iterations (default=100)
 -j,--annotated-test-text <arg>               full path to save annotated
                                              test set text to
 -kl,--lag <arg>                              number of iterations between
                                              samples (default=100)
 -ks,--samples <arg>                          number of samples to take
                                              (default=100)
 -l,--model-input-path <arg>                  full path of model to be
                                              loaded
 -m,--model-output-path <arg>                 full path to save model to
 -n,--annotated-text <arg>                    full path to save annotated
                                              text to
 -oe,--output-evaluation-score <arg>          path of output for
                                              evaluation results
 -oste,--output-test-sample-score <arg>       path of output for
                                              perplexity measures for samples taken for the test data
 -ostr,--output-train-sample-score <arg>      path of output for bayes
                                              factors for samples taken for the training data
 -ot,--output-tabulated-probabilities <arg>   path of tabulated
                                              probability output
 -pd,--temperature-decrement <arg>            temperature decrement steps
                                              (default=0.1)
 -pi,--initial-temperature <arg>              initial temperature for
                                              annealing regime (default=0.1)
 -pt,--target-temperature <arg>               temperature at which to stop
                                              annealing (default=1)
 -q,--delta <arg>                             set delta value
                                              (default=0.0001
 -r,--random-seed <arg>                       seed random number generator
                                              (default=false)
 -sc,--content-states <arg>                   number of content states in
                                              HMM (default=4)
 -sf,--function-states <arg>                  number of function states in
                                              HMM (default=7)
 -t,--topics <arg>                            number of topics in LDAHMM
                                              (default=50)
 -ur,--tag-reduction-level <arg>              how much the tagset should
                                              be reduced [0=none, 1=CE; default=0]
 -ut,--tagset <arg>                           tagset used in the data
                                              [b=brown, p=penntreebank, t=tiger; default=b]
 -w,--words-class <arg>                       number of words to print per
                                              class (default=50)

Full command line options for cdhmm-tag.sh
============================================

 -a,--alpha <arg>                             alpha value
                                              (default=50/topics)
 -b,--beta <arg>                              beta value (default=0.1)
 -c,--data-format <arg>                       format of input data
                                              [conll2k, hashslash, pipesep, raw; default=conll2k]
 -d,--train-data-dir <arg>                    full path to directory
                                              containing training documents
 -e,--experiment-model <arg>                  model to use [m1,m2,m3;
                                              default=m1]
 -f,--test-data-dir <arg>                     full path to directory
                                              containing test documents
 -g,--gamma <arg>                             set gamma value
                                              (default=0.1)
 -h,--help                                    print help
 -ite,--test-iterations <arg>                 number of test set burn in
                                              iterations (default=10)
 -itr,--training-iterations <arg>             number of training
                                              iterations (default=100)
 -j,--annotated-test-text <arg>               full path to save annotated
                                              test set text to
 -kl,--lag <arg>                              number of iterations between
                                              samples (default=100)
 -ks,--samples <arg>                          number of samples to take
                                              (default=100)
 -l,--model-input-path <arg>                  full path of model to be
                                              loaded
 -m,--model-output-path <arg>                 full path to save model to
 -n,--annotated-text <arg>                    full path to save annotated
                                              text to
 -oe,--output-evaluation-score <arg>          path of output for
                                              evaluation results
 -oste,--output-test-sample-score <arg>       path of output for
                                              perplexity measures for samples taken for the test data
 -ostr,--output-train-sample-score <arg>      path of output for bayes
                                              factors for samples taken for the training data
 -ot,--output-tabulated-probabilities <arg>   path of tabulated
                                              probability output
 -pd,--temperature-decrement <arg>            temperature decrement steps
                                              (default=0.1)
 -pi,--initial-temperature <arg>              initial temperature for
                                              annealing regime (default=0.1)
 -pt,--target-temperature <arg>               temperature at which to stop
                                              annealing (default=1)
 -q,--delta <arg>                             set delta value
                                              (default=0.0001
 -r,--random-seed <arg>                       seed random number generator
                                              (default=false)
 -sc,--content-states <arg>                   number of content states in
                                              HMM (default=4)
 -sf,--function-states <arg>                  number of function states in
                                              HMM (default=7)
 -t,--topics <arg>                            number of topics in LDAHMM
                                              (default=50)
 -ur,--tag-reduction-level <arg>              how much the tagset should
                                              be reduced [0=none, 1=CE; default=0]
 -ut,--tagset <arg>                           tagset used in the data
                                              [b=brown, p=penntreebank, t=tiger; default=b]
 -w,--words-class <arg>                       number of words to print per
                                              class (default=50)
