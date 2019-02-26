
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
            A = row.rstrip().split()
            region_start = int(A[1])
            if region_start < start:
                region_start = start
            region_end = int(A[2])
            if region_end > end:
                region_end = end
            depth = A[3]
            depths += [depth] * (region_end - region_start)

        print('{0}\t{1}\t{2}\t{3}'.format(chrom, 
                                          region_start,
                                          region_end,
                                          ','.join(depths)))


# for l in sys.stdin:
#     A = l.rstrip().split()
#     chrm = A[0]
#     start = int(A[1])
#     end = int(A[2])

#     depths = []
#     for row in tabixfile.fetch(chrm, start, end):
#         _A = row.rstrip().split()
#         row_start = int(_A[1])
#         if row_start < start:
#             row_start = start
#         row_end = int(_A[2])
#         if row_end > end:
#             row_end = end
#         depth = _A[3]
#         depths += [depth]*(row_end-row_start)

#     print '\t'.join(A[:3]) + '\t' + ','.join(depths)
