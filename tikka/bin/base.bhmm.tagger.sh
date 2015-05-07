#! /bin/bash -x

L=$1
shift
M=$1
shift
D=$1
shift
P=$1
shift

#remaining parameters
R=''
while (( "$#" )); do
    R="$R $1"
    shift
done

date
# bhmm-train.sh  -itr 100 -pt 0.1 -pd 0.1 -pi 10 -e $M -sc 4 -sf 7 \
#     -m $TIKKA_DIR/models/bhmm.$D.$L.$M.model -ot $TIKKA_DIR/out/bhmm.$D.$L.$M.out \
#     -d $P/train/learningcurve0032
# bhmm-train.sh  -itr 100 -pt 0.5 -pd 0.1 -pi 1.5 -e $M -sc 4 -sf 7 \
#     -m $TIKKA_DIR/models/bhmm.$D.$L.$M.model -ot $TIKKA_DIR/out/bhmm.$D.$L.$M.out \
#     -d $P/train/learningcurve0032 $R
bhmm-tagger.sh -l $TIKKA_DIR/models/bhmm.$D.$L.$M.model -ot $TIKKA_DIR/out/bhmm.$D.$L.$M.out \
    -oe $TIKKA_DIR/out/bhmm.eval.$D.$L.$M.out -n $TIKKA_DIR/out/bhmm.$D.$L.$M.anno.out \
    -d $P/train/learningcurve0032 $R
date
