import sys
from collections import Counter
import pysam
# import numpy as np
import pandas as pd

REGIONS_BED = sys.argv[1]
BAM_FILE = sys.argv[2]

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

    for chrom, start, end, genotype, df in data:
        print(chrom, start, end, genotype,
              ','.join(str(n) for n in df.fillna(value=0).values.astype(int).flatten().tolist()),
              sep='\t')
