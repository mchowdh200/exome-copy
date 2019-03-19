"""
Demonstrate how to get the pileup sequence for a given region
and plot the sequence/read depth as a colorcoded bar plot
"""

import sys
import pysam
import pandas as pd
# import matplotlib.pyplot as plt

REGIONS_BED = sys.argv[1]
BAM_FILE = sys.argv[2]

with open(REGIONS_BED, 'r') as regions, \
        pysam.AlignmentFile(BAM_FILE, 'rb') as samfile:
    data = {'chrom': [], 'start': [], 'end': [], 'signal': [], 'sequence': []}

    for line in regions:
        A = line.rstrip().split()
        chrom = A[0]
        start = int(A[1])
        end = int(A[2])
        
        signal = []
        sequence = []
        for x in samfile.pileup(chrom, start, end):
            seq = x.get_query_sequences()
            if seq:
                # modify this to get a distribution over bases?
                signal.append(len(seq))
                sequence.append(seq[0])
            else:
                signal.append(0)
                sequence.append('_')
        data['chrom'].append(chrom)
        data['start'].append(start)
        data['end'].append(end)
        data['signal'].append(signal)
        data['sequence'].append(''.join(sequence))
    df = pd.DataFrame(data)
    print(df.head())
# with pysam.AlignmentFile(sys.argv[1], 'rb') as samfile:

#     # just hardcode one of the exon locations for now
#     signal = []
#     sequence = []
#     # for x in samfile.pileup('1', 861271, 861443):
#     for x in samfile.pileup('1', 866368, 866519, max_depth=100000):
#         # print(x.get_mapping_qualities())
#         seq = x.get_query_sequences()
#         print(x.pileups)
#         if seq:
#             signal.append(len(x.get_query_sequences()))
#             sequence.append(x.get_query_sequences()[0])
#         else:
#             signal.append(0)
#             sequence.append('_')

#     # visualize a region by read depth color coded by base
#     base2color = {'A': 'red', 'a': 'red', 'T': 'green',
#                   't': 'green', 'C': 'blue', 'c': 'blue',
#                   'G': 'black', 'g': 'black', '_': 'cyan'}
#     plt.bar(range(len(signal)), signal, width=1,
#             color=[base2color[i] for i in sequence])
#     plt.show()
