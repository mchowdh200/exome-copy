#!/bin/bash

SAMPLE=$1
REGIONS_BED="$SAMPLE.regions.bed.gz"
PERBASE_BED="$SAMPLE.per-base.bed.gz"
PERBASE_CSI="$PERBASE_BED.csi"
SIGNALS_BED="$SAMPLE.signals.bed"


# Don't redownload if I don't have to
if [ ! -f $REGIONS_BED ]; then
    aws s3 cp "s3://layerlab/exome/$REGIONS_BED" .
    aws s3 cp "s3://layerlab/exome/$PERBASE_BED" .
    aws s3 cp "s3://layerlab/exome/$PERBASE_CSI" .
fi


# don't recalculate if I don't have to
if [ ! -f $SIGNALS_BED ]; then
    python3 extract_signals.py $SAMPLE
    # output=$(python3 extract_signals.py $SAMPLE)
    # echo "$output" > $SIGNALS_BED
    # python3 extract_signals.py $SAMPLE > $SIGNALS_BED
fi

aws s3 cp $SIGNALS_BED "s3://layerlab/exome/signals/$SIGNALS_BED"

rm $SAMPLE.*

