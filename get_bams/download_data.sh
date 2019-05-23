#!/bin/bash


# input bam file and output directory
BAM_FILE_WITH_DIR=$1
DATA_DIR=$2

# get the sample number and bam filename from the file index
SAMPLE=$(echo $BAM_FILE_WITH_DIR | cut -d '/' -f2 )
BAM_FILE=$(echo $BAM_FILE_WITH_DIR | cut -d '/' -f4)

if [ ! -f "$DATA_DIR/$BAM_FILE" ]; then
    aws s3 cp "s3://1000genomes/phase1/$BAM_FILE_WITH_DIR" $DATA_DIR
fi

if [ ! -f "$DATA_DIR/$BAM_FILE.bai" ]; then
    aws s3 cp "s3://1000genomes/phase1/$BAM_FILE_WITH_DIR.bai" $DATA_DIR
fi
