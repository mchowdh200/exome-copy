#!/bin/bash

# input bam file and output directory
BAM_FILE_WITH_DIR=$1
DATA_DIR=$2

# get the sample number and bam filename from the file index
SAMPLE=$(echo $BAM_FILE_WITH_DIR | cut -d '/' -f2 )
BAM_FILE=$(echo $BAM_FILE_WITH_DIR | cut -d '/' -f4)

# If we've got the BAM's and if we've not already extracted SV's
if [ -f "$DATA_DIR/$BAM_FILE" ] && [ -f "$DATA_DIR/$BAM_FILE.bai" ]; then

    # get pileups sequence from bam files
    if [ -f "$DATA_DIR/labels/$SAMPLE.del.bed" ] && \
       [ ! -f "$DATA_dir/pileups/$SAMPLE.pileups.del.bed" ]; then
        python get_pileup.py \
            "$DATA_DIR/labels/$SAMPLE.del.bed" \
            "$DATA_DIR/$BAM_FILE" > "$DATA_DIR/pileups/$SAMPLE.pileups.del.bed"
    fi

    if [ -f "$DATA_DIR/labels/$SAMPLE.dup.bed" ] && \
       [ ! -f "$DATA_DIR/pileups/$SAMPLE.pileups.dup.bed" ]; then
        python get_pileup.py \
            "$DATA_DIR/labels/$SAMPLE.dup.bed" \
            "$DATA_DIR/$BAM_FILE" > "$DATA_DIR/pileups/$SAMPLE.pileups.dup.bed"
    fi

fi


