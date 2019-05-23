#!/bin/bash

# input bam file and output directory
BAM_FILE_WITH_DIR=$1
DATA_DIR=$2

# number of regions to sample per bed file
N=100

# get the sample number and bam filename from the file index
SAMPLE=$(echo $BAM_FILE_WITH_DIR | cut -d '/' -f2 )
BAM_FILE=$(echo $BAM_FILE_WITH_DIR | cut -d '/' -f4)


# do we have the bam file with index?
if [ -f "$DATA_DIR/$BAM_FILE" ] && [ -f "$DATA_DIR/$BAM_FILE.bai" ]; then
    
    # get nonsv pileups from bam file
    if [ -f "$DATA_DIR/labels/$SAMPLE.nosv.bed" ] && \
       [ ! -f "$DATA_DIR/pileups/$SAMPLE.pileups.nosv.bed" ]; then
        # shuffle the bed file, then take the first N
        shuf "$DATA_DIR/labels/$SAMPLE.nosv.bed" | head -$N > "$DATA_DIR/labels/$SAMPLE.subset.bed"
        python get_pileup_nogtype.py \
            "$DATA_DIR/labels/$SAMPLE.subset.bed" \
            "$DATA_DIR/$BAM_FILE" > "$DATA_DIR/pileups/$SAMPLE.pileups.nosv.bed"

        rm "$DATA_DIR/labels/$SAMPLE.subset.bed"
    fi
fi
