#!/bin/bash

SAMPLE=$1
VCF=data/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf.gz

# get deletions
bcftools view -s $SAMPLE -c 1 -i 'SVTYPE="DEL"' $VCF | \
bcftools query -f '%CHROM\t%POS\t%INFO/END\t[%GT]\n' | \
bedtools intersect -wao -b stdin -a $SAMPLE.signals.bed -f 0.25 | \
awk '{OFS="\t"; label=$8; if(label==".") {label="0|0";} print $1,$2,$3,label,$4}' > $SAMPLE.del.bed

# get duplications
bcftools view -s $SAMPLE -c 1 -i 'SVTYPE="DUP"' $VCF | \
bcftools query -f '%CHROM\t%POS\t%INFO/END\t[%GT]\n' | \
bedtools intersect -wao -b stdin -a $SAMPLE.signals.bed -f 0.25 | \
awk '{OFS="\t"; label=$8; if(label==".") {label="0|0";} print $1,$2,$3,label,$4}' > $SAMPLE.dup.bed

# get CNV?
bcftools view -s $SAMPLE -c 1 -i 'SVTYPE="CNV"' $VCF | \
bcftools query -f '%CHROM\t%POS\t%INFO/END\t[%GT]\n' | \
bedtools intersect -wao -b stdin -a $SAMPLE.signals.bed -f 0.25 | \
awk '{OFS="\t"; label=$8; if(label==".") {label="0|0";} print $1,$2,$3,label,$4}' > $SAMPLE.cnv.bed

