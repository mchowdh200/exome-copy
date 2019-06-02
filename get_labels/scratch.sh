
# sample to extract and where to output results
SAMPLE=$1
# OUT_DIR=$2

# Variants for all samples are contained in here
VCF=data/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf.gz

# Exon locations
EXON_BED=data/exons_tiled.bed


# These two lines will give you the regions in the source
# sample of where deletions occur in BED format (with genotype)
bcftools view -s $SAMPLE -c 1 -i 'SVTYPE="DEL"' $VCF | \
bcftools query -f '%CHROM\t%POS\t%INFO/END\t[%GT]\n' | \
bedtools intersect -wao -b stdin -a $EXON_BED -f 0.25


