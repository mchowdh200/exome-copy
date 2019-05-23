#!/bin/bash

EXON_BED=../data/BED/20120518.consensus_add50bp.bed
OUT_DIR=../data/BED

TRAIN="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
TEST="21 22 X Y"

awk -v chromosomes="$TRAIN" \ '
    BEGIN {
    split(chromosomes, x); # array of chromosomes
    for (i in x) 
        chr[x[i]] = "" # make a dict keyed by chromosome
     }
    {
        if ($1 in chr) { # if the line is from our set of chromosomes
            print $0     # then print the line
        }
    } 
' $EXON_BED > $OUT_DIR/train_regions.bed

awk -v chromosomes="$TEST" \ '
    BEGIN {
    split(chromosomes, x); 
    for (i in x) 
        chr[x[i]] = "" 
    } 
    {
        if ($1 in chr) {
            print $0
        }
    } 
' $EXON_BED > $OUT_DIR/test_regions.bed
