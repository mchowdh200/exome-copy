#!/bin/bash

# get file name from command line arg
bam_file=$1
bai_file=${bam_file}.bai

# split by '/' to get the batch number
batch_number="$(echo $bam_file | cut -d'/' -f2)"

# then split by '.' to the date(?)
date="$(echo $bam_file | cut -d'.' -f7)"

# call mosdepth to generate output file
