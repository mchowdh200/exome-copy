#!/bin/bash    
#SBATCH -p short              # use short for less than 24hrs
#SBATCH --job-name=structural_variant_extraction
#SBATCH --mail-type=ALL
#SBATCH --mail-user=much8161@colorado.edu
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=16gb
#SBATCH --time=24:00:00
#SBATCH --output=/Users/much8161/Repositories/exome-copy/out/structural_variant_extraction.out
#SBATCH --error=/Users/much8161/Repositories/exome-copy/out/structural_variant_extraction.err

OUT_DIR=/scratch/Shares/layer/projects/exome-copy/data/labels
SAMPLES=/Users/much8161/Repositories/exome-copy/data/samples.txt

cat $SAMPLES | gargs -p 32 "bash get_labels.sh {} $OUT_DIR"
