
import sys
import pandas as pd

def expand_read(x):
    return (x['end'] - x['start']) * (str(x['depth']) + ',')

df = pd.read_csv(sys.stdin, names=['chr', 'start', 'end', 'depth'], sep='\t')
print(df.apply(expand_read, axis=1).str.cat(sep=None))

