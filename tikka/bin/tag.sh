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
cdhmm-tag.sh -l $TIKKA_DIR/models/$D.$M.$L.model -ot $TIKKA_DIR/tab/$D.$M.$L.tab \
    -oe $TIKKA_DIR/eval/$D.$M.$L.eval -n $TIKKA_DIR/out/$D.$M.$L \
    -d $P/train/$L $R
date