#!/bin/bash    
#SBATCH -p short # use short for less than 24hrs
#SBATCH --job-name=process_bams
#SBATCH --mail-type=ALL
#SBATCH --mail-user=much8161@colorado.edu
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --output=/Users/much8161/Repositories/exome-copy/out/process_bams.out
#SBATCH --error=/Users/much8161/Repositories/exome-copy/out/process_bams.err

BAM_FILE_INDEX=data/phase1.exome.alignment.index
DATA_DIR=/scratch/Shares/layer/projects/exome-copy/data

# $1 is {0}: data/$SAMPLE/exome_alignment/$BAM_FILE
# $2 is $DATA_DIR
cat $BAM_FILE_INDEX | gargs -p 64 "bash process_bams.sh {0} $DATA_DIR"

