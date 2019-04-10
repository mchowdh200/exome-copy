#!/bin/bash

# input bam file and output directory
BAM_FILE_WITH_DIR=$1
DATA_DIR=$2

# get the sample number and bam filename from the file index
SAMPLE=$(echo $BAM_FILE_WITH_DIR | cut -d '/' -f2 )
BAM_FILE=$(echo $BAM_FILE_WITH_DIR | cut -d '/' -f4)

if [ -f "$DATA_DIR/labels/$SAMPLE.del.bed" ] || [ -f "$DATA_DIR/labels/$SAMPLE.dup.bed" ]; then
    # get bam/bai only if we have corresponding SV bed files
    aws s3 cp "s3://1000genomes/phase1/$BAM_FILE_WITH_DIR" $DATA_DIR
    aws s3 cp "s3://1000genomes/phase1/$BAM_FILE_WITH_DIR.bai" $DATA_DIR

    # get pileups sequence from bam files
    if [ -f "$DATA_DIR/labels/$SAMPLE.del.bed" ]; then
        python get_pileups.py \
            "$DATA_DIR/labels/$SAMPLE.del.bed" \
            "$DATA_DIR/$BAM_FILE" > "$DATA_DIR/pileups/$SAMPLE.pileups.del.bed"
    fi

    if [ -f "$DATA_DIR/labels/$SAMPLE.dup.bed" ]; then
        python get_pileups.py \
            "$DATA_DIR/labels/$SAMPLE.dup.bed" \
            "$DATA_DIR/$BAM_FILE" > "$DATA_DIR/pileups/$SAMPLE.pileups.dup.bed"
    fi

    # clean up
    rm "$DATA_DIR/$BAM_FILE"
    rm "$DATA_DIR/$BAM_FILE.bai"

fi


