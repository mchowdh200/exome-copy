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

# TODO use bcftools query -l <input.vcf> to get the list of samples
# SAMPLES=/Users/much8161/Repositories/exome-copy/data/samples.txt

# get the intersection of samples from our exome bam set and the samples
# in the structural variant vcf file.
SAMPLES=$(sort <(cat data/samples.txt) <(bcftools query -l \
    data/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf.gz) | \
    uniq -d)

cat $SAMPLES | gargs -p 32 "bash get_labels.sh {} $OUT_DIR"
