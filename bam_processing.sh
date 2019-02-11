#!/bin/bash
storage_dir=/mnt/local

# get file name from command line arg
bam_file=$1
# echo $bam_file
bai_file=${bam_file}.bai

# get just the filenames without directory prefix
# bam_no_dir="$(echo $bam_file | cut -d'/' f3)"
bam_no_dir="$(echo $bam_file | cut -d'/' -f4)"
bai_no_dir="$(echo $bai_file | cut -d'/' -f4)"

# split by '/' to get the batch number
batch_number="$(echo $bam_file | cut -d'/' -f2)"

# then split by '.' to the date(?)
date="$(echo $bam_file | cut -d'.' -f7)"

# copy over the bam and bai file
aws s3 cp "s3://1000genomes/phase1/$bam_file" .
aws s3 cp "s3://1000genomes/phase1/$bai_file" .

# call mosdepth to generate output file
