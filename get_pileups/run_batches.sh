#!/bin/bash
sbatch --job-name=train_pileups process_bams.sbatch train
sbatch --job-name=test_pileups process_bams.sbatch test
sbatch --job-name=val_pileups process_bams.sbatch val

sbatch --job-name=train_nonsv sample_nonsv.sbatch train
sbatch --job-name=test_nonsv sample_nonsv.sbatch test
sbatch --job-name=val_nonsv sample_nonsv.sbatch val

