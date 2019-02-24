
import sys
import gzip
import pysam

REGIONS_BED = sys.argv[1]+'.regions.bed.gz'
PERBASE_BED = sys.argv[1]+'.per-base.bed.gz'
PERBASE_CSI = PERBASE_BED+'.csi'

tabixfile = pysam.TabixFile(PERBASE_BED, 
                            index=PERBASE_CSI,
                            parser=pysam.asTuple())

with gzip.open(REGIONS_BED, mode='rt') as regions:
    for line in regions:
        A = line.split()
        chrom = A[0]
        start = int(A[1])
        end= int(A[2])
        tabix_out = [(int(row[1]), int(row[2]), row[3]) 
                     for row in tabixfile.fetch(chrom, start, end)]

        region_start = tabix_out[0][0]
        region_end = tabix_out[-1][1]
        depths = ''.join((row[2]+',')*(row[1] - row[0]) for row in tabix_out)

        # print("OUTPUT:")
        print('{0}\t{1}\t{2}\t{3}'.format(chrom, region_start, region_end, depths))

