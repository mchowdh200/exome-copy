#!/bin/bash

# Where to find data and where to put results
BAM_FILE_WITH_DIR=$1
DATA_DIR=$2
LABEL_DIR=$3
OUT_DIR=$4

# get the sample number and bam filename from the file index
SAMPLE=$(echo $BAM_FILE_WITH_DIR | cut -d '/' -f2 )
BAM_FILE=$(echo $BAM_FILE_WITH_DIR | cut -d '/' -f4)

# If we've got the BAM's and if we've not already extracted SV's
if [ -f "$DATA_DIR/$BAM_FILE" ] && [ -f "$DATA_DIR/$BAM_FILE.bai" ]; then

    # get pileups sequence from bam files
    if [ -f "$LABEL_DIR/$SAMPLE.del.bed" ] && \
       [ ! -f "$OUT_DIR/$SAMPLE.pileups.del.bed" ]; then
        python get_pileup.py \
            "$LABEL_DIR/$SAMPLE.del.bed" \
            "$DATA_DIR/$BAM_FILE" > "$OUT_DIR/$SAMPLE.pileups.del.bed"
    fi

    if [ -f "$LABEL_DIR/$SAMPLE.dup.bed" ] && \
       [ ! -f "$OUT_DIR/$SAMPLE.pileups.dup.bed" ]; then
        python get_pileup.py \
            "$LABEL_DIR/$SAMPLE.dup.bed" \
            "$DATA_DIR/$BAM_FILE" > "$OUT_DIR/$SAMPLE.pileups.dup.bed"
    fi

fi


