
#!/bin/bash

BEDFILE=$1

# take a bed file and window the regions to be 
# of width 500 with a no overlap
bedtools makewindows -b $BEDFILE -w 500
