#!/bin/bash
for TRAIN_TEST in train test val; do
    for SVTYPE in nosv del dup; do
        cat ../data/pileups/${TRAIN_TEST}/*.${SVTYPE}.bed > ../data/pileups/${TRAIN_TEST}/${TRAIN_TEST}_${SVTYPE}.bed
        python create_dataset.py $TRAIN_TEST $SVTYPE
    done
done
