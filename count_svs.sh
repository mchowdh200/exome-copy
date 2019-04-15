#!/bin/bash

DATA_DIR=$1

count=$((0))

for SAMPLE in $(cat data/samples.txt); do
    if [ -f "$DATA_DIR/labels/$SAMPLE.del.bed" ]; then
        count=$((count + $(wc -l < "$DATA_DIR/labels/$SAMPLE.del.bed")))
    fi

    if [ -f "$DATA_DIR/labels/$SAMPLE.dup.bed" ]; then
        count=$((count + $(wc -l < "$DATA_DIR/labels/$SAMPLE.dup.bed")))
    fi
done

echo $count
