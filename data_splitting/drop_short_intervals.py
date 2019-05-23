import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BED_FILE = './data/exons_tiled.bed'
df = pd.read_csv(BED_FILE, sep='\t', names=['chr', 'start', 'end'])
df = df[df['end'] - df['start'] > 100]
widths = df['end'] - df['start']

# plt.hist(widths.values, bins=20)
# plt.show()

# write back to file
df.to_csv('./data/exons_tiled.bed', header=False, index=False, sep='\t')

