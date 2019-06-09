#!/bin/bash
sbatch --job-name=train_lables get_labels.sbatch train
sbatch --job-name=test_labels get_labels.sbatch test
sbatch --job-name=val_labels get_labels.sbatch val
