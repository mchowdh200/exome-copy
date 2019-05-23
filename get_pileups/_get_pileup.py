import multiprocessing
import sys
import pysam
import pandas as pd


def get_pileup(region):
    """
    For a given genomic region, get the pileup at each position and
    return the read depth and sequence.

    INPUTS: 
        - bed region string i.e. "chrom    start    end"
        - AlignmentFile object used to get the pileups

    """
    A = region.rstrip().split()
    chrom = A[0]
    start = int(A[1])
    end = int(A[2])
    
    signal = []
    sequence = []
    # QUESTION: will it be possible for me to to multiprocessing pileup using
    # just a single samfile object? or will I have to open the file in here?
    # ANSWER: no, just open a new one here
    with pysam.AlignmentFile(BAM_FILE, 'rb') as samfile:
        for x in samfile.pileup(chrom, start, end, truncate=True):
            seq = x.get_query_sequences()
            if seq:
                # modify this to get a distribution over bases
                signal.append(len(seq))
                sequence.append(seq[0])
            else:
                # if there was no pileup at this position
                signal.append(0)
                sequence.append('N')
    return chrom, start, end, signal, sequence


if __name__ == '__main__':
    # Get filenames from command line args
    REGIONS_BED = sys.argv[1]
    BAM_FILE = sys.argv[2]

    with multiprocessing.Pool() as pool, \
            open(REGIONS_BED, 'r') as regions:
        # for each line in the regions bed file, this will in a multiprocessing
        # fashion return each output of get_pileup to a list.
        data = pool.imap_unordered(get_pileup, (line for line in regions))
        df = pd.DataFrame(list(data), columns=['chr', 'start', 'end', 'signal', 'sequence'])
        df.to_csv('out.txt', sep='\t', header=False, index=False)

    #     # visualize a region by read depth color coded by base
    #     base2color = {'A': 'red', 'a': 'red', 'T': 'green',
    #                   't': 'green', 'C': 'blue', 'c': 'blue',
    #                   'G': 'black', 'g': 'black', '_': 'cyan'}
    #     plt.bar(range(len(signal)), signal, width=1,
    #             color=[base2color[i] for i in sequence])
    #     plt.show()


