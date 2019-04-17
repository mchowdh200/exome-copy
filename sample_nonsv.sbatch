#!/bin/bash    
#SBATCH -p short # use short for less than 24hrs
#SBATCH --job-name=sample_nonsv
#SBATCH --mail-type=ALL
#SBATCH --mail-user=much8161@colorado.edu
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --output=/Users/much8161/Repositories/exome-copy/out/sample_nonsv.out
#SBATCH --error=/Users/much8161/Repositories/exome-copy/out/sample_nonsv.err

BAM_FILE_INDEX=data/phase1.exome.alignment.index
DATA_DIR=/scratch/Shares/layer/projects/exome-copy/data

cat $BAM_FILE_INDEX | gargs -p 64 "bash sample_nonsv.sh {0} $DATA_DIR"



