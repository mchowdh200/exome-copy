
import sys
import gzip
import pysam

REGIONS_BED = sys.argv[1]+'.regions.bed.gz'
PERBASE_BED = sys.argv[1]+'.per-base.bed.gz'
PERBASE_CSI = PERBASE_BED+'.csi'

tabixfile = pysam.TabixFile(PERBASE_BED, index=PERBASE_CSI,)

with gzip.open(REGIONS_BED, mode='rt') as regions:
    for line in regions:
        A = line.rstrip().split()
        chrom = A[0]
        start = int(A[1])
        end= int(A[2])
        depths = []
        for row in tabixfile.fetch(chrom, start, end):
            B = row.rstrip().split()
            row_start = int(B[1])
            if row_start < start:
                row_start = start
            row_end = int(B[2])
            if row_end > end:
                row_end = end
            depth = B[3]
            depths += [depth] * (row_end - row_start)

        print('{0}\t{1}\t{2}\t{3}'.format(chrom, 
                                          start,
                                          end,
                                          ','.join(depths)))

