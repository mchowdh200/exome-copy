#!/bin/bash
# For a given sample and structural variants VCF, find all DEL/DUP SVtypes and
# output a $SAMPLE.$SVTYPE.bed file
# Format: CHROM    START    END    GENOTYPE

# sample to extract and where to output results
SAMPLE=$1
OUT_DIR=$2

# Variants for all samples are contained in here
VCF=data/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf.gz

# Exon locations
EXON_BED=data/20120518.consensus_add50bp.bed

# get deletions
if [ ! -f "$OUT_DIR/$SAMPLE.del.bed" ]; then
    bcftools view -s $SAMPLE -c 1 -i 'SVTYPE="DEL"' $VCF | \
    bcftools query -f '%CHROM\t%POS\t%INFO/END\t[%GT]\n' | \
    bedtools intersect -wao -b stdin -a $EXON_BED -f 0.25 | \
    awk '{OFS="\t"; label=$7; if(label==".") {next} print $1,$2,$3,label}' > "$OUT_DIR/$SAMPLE.del.bed"

    # remove empty result
    if [ -s "$OUT_DIR/$SAMPLE.del.bed" ]; then
        rm "$OUT_DIR/$SAMPLE.del.bed"
    fi
fi

# get duplications
if [ ! -f "$OUT_DIR/$SAMPLE.dup.bed" ]; then
    bcftools view -s $SAMPLE -c 1 -i 'SVTYPE="DUP"' $VCF | \
    bcftools query -f '%CHROM\t%POS\t%INFO/END\t[%GT]\n' | \
    bedtools intersect -wao -b stdin -a $EXON_BED -f 0.25 | \
    awk '{OFS="\t"; label=$7; if(label==".") {next} print $1,$2,$3,label}' > "$OUT_DIR/$SAMPLE.dup.bed"

    if [ -s "$OUT_DIR/$SAMPLE.dup.bed" ]; then
        rm "$OUT_DIR/$SAMPLE.dup.bed"
    fi
fi

# get non-structural variants
if [ ! -f "$OUT_DIR/$SAMPLE.nosv.bed" ]; then
    bcftools view -s $SAMPLE -c 1 -i 'SVTYPE="DUP"|SVTYPE="DEL"|SVTYPE="CNV"' $VCF | \
    bcftools query -f '%CHROM\t%POS\t%INFO/END\n' | \
    bedtools intersect -v -b stdin -a $EXON_BED -f 0.25 > "$OUT_DIR/$SAMPLE.nosv.bed"

    if [ -s "$OUT_DIR/$SAMPLE.nosv.bed" ]; then
        rm "$OUT_DIR/$SAMPLE.nosv.bed"
    fi
fi



# get CNV TODO This may need more than just genotype to interpret
