#!/bin/bash

# input bam file and output directory
BAM_FILE_WITH_DIR=$1
DATA_DIR=$2
LABEL_DIR=$3
OUT_DIR=$4

# number of regions to sample per bed file
N=100

# get the sample number and bam filename from the file index
SAMPLE=$(echo $BAM_FILE_WITH_DIR | cut -d '/' -f2 )
BAM_FILE=$(echo $BAM_FILE_WITH_DIR | cut -d '/' -f4)


# do we have the bam file with index?
if [ -f "$DATA_DIR/$BAM_FILE" ] && [ -f "$DATA_DIR/$BAM_FILE.bai" ]; then
    
    # get nonsv pileups from bam file
    if [ -f "$LABEL_DIR/$SAMPLE.nosv.bed" ] && \
       [ ! -f "$OUT_DIR/$SAMPLE.pileups.nosv.bed" ]; then
        # shuffle the bed file, then take the first N
        shuf "$LABEL_DIR/$SAMPLE.nosv.bed" | head -$N > "$LABEL_DIR/$SAMPLE.subset.bed"
        python get_pileup_nogtype.py \
            "$LABEL_DIR/$SAMPLE.subset.bed" \
            "$DATA_DIR/$BAM_FILE" > "$OUT_DIR/$SAMPLE.pileups.nosv.bed"

        rm "$LABEL_DIR/$SAMPLE.subset.bed"
    fi
fi
