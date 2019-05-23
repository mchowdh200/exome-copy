"""
Testing out the mechanics of the pysam pileup method
"""

import sys
from collections import Counter
import pysam
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

REGIONS_BED = './data/labels/HG00096.del.bed'
BAM_FILE = './data/HG00096.mapped.illumina.mosaik.GBR.exome.20110411.bam'

with open(REGIONS_BED, 'r') as regions, \
        pysam.AlignmentFile(BAM_FILE, 'rb') as samfile:
    for region in regions:
        chrom, start, end, genotype = region.rstrip().split()
        pileups = samfile.pileup(chrom, int(start), int(end), 
                                 truncate=True, min_base_quality=13)

        # test to see if the counter is properly converted into dataframe
        print(chrom, start, end, genotype, sep='\t')
        for column in pileups:
            x = Counter(column.get_query_sequences(add_indels=False))
            print(x)
        pileups = samfile.pileup(chrom, int(start), int(end), 
                                 truncate=True, min_base_quality=13)
        y = pd.DataFrame.from_records(
            [Counter(c.get_query_sequences())
             for c in pileups],
            columns=['A', 'T', 'C', 'G',
                     'a', 't', 'c', 'g']
        ).fillna(value=0).astype(int)
        print(y)

        # make sure that when you reshape back into the original matrix format
        # we will have to reshape into (8, -1) THEN transpose
        z = ','.join(str(n) for n in y.values.flatten().tolist())
        print(z)

        print(np.fromstring(z, sep=',').reshape((8, -1)).tolist())
