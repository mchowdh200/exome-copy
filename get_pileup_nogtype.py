"""
Same as get_pileup.py, but without the genotype field.
Inelegant, I know, but this way was quicker than to generalize
the original script to handle files with/without genotype
"""
import sys
from collections import Counter
import pysam
import pandas as pd

REGIONS_BED = sys.argv[1]
BAM_FILE = sys.argv[2]

with open(REGIONS_BED, 'r') as regions, \
        pysam.AlignmentFile(BAM_FILE, 'rb') as samfile:
    data = []
    for region in regions:
        chrom, start, end = region.rstrip().split()
        pileups = samfile.pileup(chrom, int(start), int(end), truncate=True)
        data.append((chrom, start, end,
                     pd.DataFrame.from_records(
                         [Counter(column.get_query_sequences())
                         for column in pileups],
                         columns=['A', 'T', 'C', 'G',
                                  'a', 't', 'c', 'g']
                     )))

    for chrom, start, end, df in data:
        print(chrom, start, end,
              ','.join(str(n) for n in df.fillna(value=0).values.astype(int).flatten().tolist()),
              sep='\t')
