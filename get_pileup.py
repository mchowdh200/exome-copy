import sys
from collections import Counter
import pysam
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

np.set_printoptions(threshold=sys.maxsize)
REGIONS_BED = sys.argv[1]
BAM_FILE = sys.argv[2]

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
