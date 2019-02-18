#!/bin/bash
# given a set of regions a set of per-base reads (bed),
# write each region as $chr $start $end $(read depths, ...)
regions=$(cat $1)

while read -r region; do
    # separate the columns of the bed file and get the reads
    region_chr=$(echo $region | cut -d ' ' -f1)
    region_start=$(echo $region | cut -d ' ' -f2)
    region_end=$(echo $region | cut -d ' ' -f3)

    # per_base_reads=$(tabix HG00096.per-base.bed.gz \
    #     "${region_chr: -1}:$region_start-$region_end")
    per_base_reads=$(tabix $2 "${region_chr: -1}:$region_start-$region_end")

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
        continue
    fi

    region_chr=$(echo "$per_base_reads" | head -n1 | cut -f1)
    region_start=$(echo "$per_base_reads" | head -n1 | cut -f2)
    region_end=$(echo "$per_base_reads" | tail -n1 | cut -f3)

    printf "$region_chr\t$region_start\t$region_end\t"
    python3 region2signal.py <<< "$per_base_reads"


    # while read -r reading; do
    #     read_start=$(echo "$reading" | cut -f2)              
    #     read_end=$(echo "$reading" | cut -f3)              
    #     read_depth=$(echo "$reading" | cut -f4)              

    #     for i in $(seq 1 $(( $read_end-$read_start ))); do
    #         printf "$read_depth\t"
    #     done
    # done <<< "$per_base_reads"
    # printf "\n"

done <<< "$regions"

