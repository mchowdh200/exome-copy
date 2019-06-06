import sys
import numpy as np
import pandas as pd

TRAIN_TEST = sys.argv[1]
SVTYPE = sys.argv[2]

df = pd.read_csv(f'../data/pileups/{TRAIN_TEST}/{TRAIN_TEST}_{SVTYPE}.bed',
                 sep='\t', 
                 names=['chrom', 'start', 'end', 'data'] if SVTYPE == 'nosv'
                 else ['chrom', 'start', 'end', 'genotype', 'data'])
df.dropna(axis=0, inplace=True)
df.data = df.data.apply(lambda x: np.fromstring(x, sep=',').reshape((-1, 8)).T)
pd.to_pickle(df, f'../data/DataFrames/{TRAIN_TEST}/{TRAIN_TEST}_{SVTYPE}.pkl')
