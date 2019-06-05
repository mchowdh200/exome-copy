#!/bin/bash

SAMPLE_INDEX="./phase1.exome.alignment.index"
VCF="../VCF/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf.gz"

# only keep samples sequenced with illumina then get the sample ID from index
# and intersect with samples that exist in the VCF.  Let's shuffle afterwards
# for good measure
ILLUMINA_SAMPLES=$(grep "illumina" $SAMPLE_INDEX | cut -d '/' -f2)
sort <(echo "$ILLUMINA_SAMPLES") \
     <(bcftools query -l $VCF) | \
     uniq -d | shuf > all_samples.txt

# NOTE: why does that give us the intersection?
# uniq -d gives us duplicate lines if they appear one after another.
# Therefore we must sort first.
