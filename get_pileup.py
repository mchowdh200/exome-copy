import sys
from collections import Counter
import pysam
import numpy as np
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)
REGIONS_BED = sys.argv[1]
BAM_FILE = sys.argv[2]

<<<<<<< HEAD
def get_pileups(reg, file):
    """
    Get the pileup distributions for each position in the region.  Result
    is a DataFrame where each column is a nucleotide (ATCGatcg) and each
    row is a read position.
    """
    chrom, start, end = reg.rstrip().split()
    with pysam.AlignmentFile(file, 'rb') as samfile:
        pileups = samfile.pileup(chrom, int(start), int(end), truncate=True,)
        return reg, pd.DataFrame.from_records([Counter(column.get_query_sequences())
                                               for column in pileups],
                                               columns=['A', 'T', 'C', 'G',
                                                        'a', 't', 'c', 'g'])


with open(REGIONS_BED, 'r') as regions:
    data = Parallel(n_jobs=1)(delayed(get_pileups)(region, BAM_FILE) 
                      for region, _ in zip(regions, range(1000)))

    for region, df in data:
        print(region.rstrip(), end='\t')
        print(','.join(str(n) for n in 
                       df.fillna(value=0).values.astype(int).flatten().tolist()))
=======
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
>>>>>>> 80f3f6ab36482501dff0733b22bdf3f2b95d6400
