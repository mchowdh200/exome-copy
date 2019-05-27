#!/bin/bash    
#SBATCH -p short              # use short for less than 24hrs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=much8161@colorado.edu
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --output=/Users/much8161/Repositories/exome-copy/out/structural_variant_extraction%j.out
#SBATCH --error=/Users/much8161/Repositories/exome-copy/out/structural_variant_extraction%j.err


TRAIN_TEST=$1
SAMPLES_LIST="../data/sample_lists/${TRAIN_TEST}_samples.txt"
VCF="../data/VCF/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf.gz"
BED="../data/BED/${TRAIN_TEST}_regions.windowed.bed"

OUT_DIR="/scratch/Shares/layer/projects/exome-copy/data/labels/$TRAIN_TEST"

# get the intersection of samples from our exome bam set and the samples
# in the structural variant vcf file.
SAMPLES=$(sort <(cat $SAMPLES_LIST) <(bcftools query -l $VCF) | uniq -d)

echo "$SAMPLES" | gargs -p 64 "bash get_labels.sh {} $VCF $BED $OUT_DIR"
