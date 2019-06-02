#!/bin/bash
# given a set of regions a set of per-base reads (bed),
# write each region as:
# chrom    start    end     sig_1,sig_2,...

region_chr=$1
region_start=$2
region_end=$3

# the per base bed doesn't have 'chr' in the chromosome number
per_base_reads=$(tabix $4 "${region_chr: -1}:$region_start-$region_end")

first_read=$(echo "$per_base_reads" | head -n1)
last_read=$(echo "$per_base_reads" | tail -n1)

# if first and/or last read depth is zero skip that read
if [ $(echo "$first_read" | cut -f4) -eq 0 ]
then
    per_base_reads=$(echo "$per_base_reads" | tail -n+2)
fi

if [ $(echo "$last_read" | cut -f4) -eq 0 ]
then
    per_base_reads=$(echo "$per_base_reads" | sed \$d)
fi

if [ -z "$per_base_reads" ]; then
    exit 0
fi

# print the chr and the interval of the signal
region_chr=$(echo "$per_base_reads" | head -n1 | cut -f1)
region_start=$(echo "$per_base_reads" | head -n1 | cut -f2)
region_end=$(echo "$per_base_reads" | tail -n1 | cut -f3)
printf "$region_chr\t$region_start\t$region_end\t"


# expand the signal to match the width of the interval
# and add the the end of the line
python3 region2signal.py <<< "$per_base_reads"


# done <<< "$regions"

