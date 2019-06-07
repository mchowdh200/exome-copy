import sys
from collections import Counter
import pysam
import pandas as pd

REGIONS_BED = sys.argv[1]
BAM_FILE = sys.argv[2]

# Used to filter out info in idxstats
chromosomes = {
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', 
    '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y'
}

# Get the total number of reads/million in the BAM ----------------------------
# line format: chr  sequence-length  num-mapped-reads  num-unmapped-reads
idxstats = pysam.idxstats(BAM_FILE) # pylint: disable=no-member
num_reads = 0
for chrm in idxstats.rstrip().split('\n'):
    chrm = chrm.rstrip().split()
    if chrm[0] in chromosomes: 
        num_reads += int(chrm[2])
num_reads /= 1e6

# Get pileup sequences --------------------------------------------------------
with open(REGIONS_BED, 'r') as regions, \
        pysam.AlignmentFile(BAM_FILE, 'rb') as samfile:
    data = []
    for region in regions:
        chrom, start, end, genotype = region.rstrip().split()
        pileups = samfile.pileup(chrom, int(start), int(end), truncate=True)
        data.append((chrom, start, end, genotype,
                     pd.DataFrame.from_records(
                         [Counter(column.get_query_sequences())
                         for column in pileups],
                         columns=['A', 'T', 'C', 'G',
                                  'a', 't', 'c', 'g']
                     )))

# Write results ---------------------------------------------------------------
    for chrom, start, end, genotype, df in data:
        values = df.fillna(value=0).values/num_reads
        print(chrom, start, end, genotype,
              ','.join(str(n) for n in values.flatten().tolist()),
              sep='\t')
