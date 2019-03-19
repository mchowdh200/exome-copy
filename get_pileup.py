"""
Demonstrate how to get the pileup sequence for a given region
and plot the sequence/read depth as a colorcoded bar plot
"""
import sys
from io import StringIO
import pysam
# import matplotlib.pyplot as plt

REGIONS_BED = sys.argv[1]
BAM_FILE = sys.argv[2]

out = StringIO()

with open(REGIONS_BED, 'r') as regions, \
        pysam.AlignmentFile(BAM_FILE, 'rb') as samfile:
    for line in regions:
        A = line.rstrip().split()
        chrom = A[0]
        start = int(A[1])
        end = int(A[2])
        
        signal = []
        sequence = []
        for x in samfile.pileup(chrom, start, end, truncate=True):
            seq = x.get_query_sequences()
            if seq:
                # modify this to get a distribution over bases?
                signal.append(len(seq))
                sequence.append(seq[0])
            else:
                signal.append(0)
                sequence.append('_')
        # print('{0}\t{1}\t{2}\t{3}\t{4}'.format(
        #     chrom, start, end, signal, ''.join(sequence)))
        out.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(
            chrom, start, end, signal, ''.join(sequence)))
    print(out.getvalue())

#     # visualize a region by read depth color coded by base
#     base2color = {'A': 'red', 'a': 'red', 'T': 'green',
#                   't': 'green', 'C': 'blue', 'c': 'blue',
#                   'G': 'black', 'g': 'black', '_': 'cyan'}
#     plt.bar(range(len(signal)), signal, width=1,
#             color=[base2color[i] for i in sequence])
#     plt.show()
