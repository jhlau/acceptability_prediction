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
cdhmm-train.sh -e $M -m $TIKKA_DIR/models/$D.$M.$L.model \
    -d $P/train/$L $R
date