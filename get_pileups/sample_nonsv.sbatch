#!/bin/bash    
#SBATCH -p short # use short for less than 24hrs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=much8161@colorado.edu
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --output=/Users/much8161/Repositories/exome-copy/out/sample_nonsv.out
#SBATCH --error=/Users/much8161/Repositories/exome-copy/out/sample_nonsv.err

TRAIN_TEST=$1 # train|test|val
if [ -z $TRAIN_TEST ]; then
    printf "TRAIN_TEST variable unset!\n" >&2
    exit 1
fi


SAMPLE_LIST="../data/sample_lists/${TRAIN_TEST}_samples.txt"

# take our list of samples and get the listings in the index 
# that inersect with them.
BAM_FILE_INDEX=$(grep -f $SAMPLE_LIST "../data/sample_lists/phase1.exome.alignment.index")

DATA_DIR="/scratch/Shares/layer/projects/exome-copy/data"
LABEL_DIR="$DATA_DIR/labels/$TRAIN_TEST"
OUT_DIR="$DATA_DIR/pileups/$TRAIN_TEST"

echo "$BAM_FILE_INDEX" | gargs -p 64 \
    "bash sample_nonsv.sh {0} $DATA_DIR $LABEL_DIR $OUT_DIR"



