#!/bin/bash
# For a given sample and structural variants VCF, find all DEL/DUP SVtypes and
# output a $SAMPLE.$SVTYPE.bed file
# Format: CHROM    START    END    GENOTYPE

# sample to extract
SAMPLE=$1

# Variants for all samples are contained in here
VCF=data/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf.gz

# Exon locations
EXON_BED=data/20120518.consensus_add50bp.bed

# get deletions
bcftools view -s $SAMPLE -c 1 -i 'SVTYPE="DEL"' $VCF | \
bcftools query -f '%CHROM\t%POS\t%INFO/END\t[%GT]\n' | \
bedtools intersect -wao -b stdin -a $EXON_BED -f 0.25 | \
awk '{OFS="\t"; label=$7; if(label==".") {next} print $1,$2,$3,label}' > $SAMPLE.del.bed

# get duplications
bcftools view -s $SAMPLE -c 1 -i 'SVTYPE="DUP"' $VCF | \
bcftools query -f '%CHROM\t%POS\t%INFO/END\t[%GT]\n' | \
bedtools intersect -wao -b stdin -a $EXON_BED -f 0.25 | \
awk '{OFS="\t"; label=$7; if(label==".") {next} print $1,$2,$3,label}' > $SAMPLE.dup.bed

# get CNV TODO This may need more than just genotype to interpret
# bcftools view -s $SAMPLE -c 1 -i 'SVTYPE="CNV"' $VCF | \
# bcftools query -f '%CHROM\t%POS\t%INFO/END\t[%GT]\n' | \
# bedtools intersect -wao -b stdin -a $SAMPLE.signals.bed -f 0.25 | \
# awk '{OFS="\t"; label=$7; if(label==".") {next} print $1,$2,$3,label}' > $SAMPLE.del.bed

