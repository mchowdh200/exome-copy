#!/bin/bash

count=$((0))

for SAMPLE in $(cat data/samples.txt); do
    if [ -f "data/labels/$SAMPLE.del.bed" ]; then
        count=$((count + $(wc -l < "data/labels/$SAMPLE.del.bed")))
    fi

    if [ -f "data/labels/$SAMPLE.dup.bed" ]; then
        count=$((count + $(wc -l < "data/labels/$SAMPLE.dup.bed")))
    fi
done

echo $count
