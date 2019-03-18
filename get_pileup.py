"""
Demonstrate how to get the pileup sequence for a given region
and plot the sequence/read depth as a colorcoded bar plot
"""

import sys
import pysam

import matplotlib.pyplot as plt

samfile = pysam.AlignmentFile(sys.argv[1], 'rb')

# just hardcode one of the exon locations for now
signal = []
sequence = []
# for x in samfile.pileup('1', 861271, 861443):
for x in samfile.pileup('1', 866368, 866519, max_depth=100000):
    # print(x.get_mapping_qualities())
    seq = x.get_query_sequences(add_indels=True)
    print(x.pileups)
    if seq:
        signal.append(len(x.get_query_sequences()))
        sequence.append(x.get_query_sequences()[0])
    else:
        signal.append(0)
        sequence.append('_')

# visualize a region by read depth color coded by base
base2color = {'A': 'red', 'a': 'red', 'T': 'green',
              't': 'green', 'C': 'blue', 'c': 'blue',
              'G': 'black', 'g': 'black', 'N': 'cyan'}
plt.bar(range(len(signal)), signal, width=1,
        color=[base2color[i] for i in sequence])
plt.show()
