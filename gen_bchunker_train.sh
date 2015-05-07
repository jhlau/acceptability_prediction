#!/bin/bash

#script to generate training data for bchunker. Takes a previously trained model (e.g. two-tier bhmm) and output the induced word classes of the original train data.
trained_model="output/m7/train.model"
output_dir="example_dataset/bchunker"

mkdir $output_dir 2>/dev/null

javac OutputHMMInducedClasses.java
java OutputHMMInducedClasses -l $trained_model -xconll -xms 0 -xns 0 > $output_dir/train.txt
