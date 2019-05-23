#!/bin/bash

BEDFILE=$1
# take a bed file and window the regions to be 
# of width 500 with a 400bp overlap
bedtools makewindows -b $BEDFILE -w 500 -s 400
