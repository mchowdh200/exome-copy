#!/bin/bash    
#SBATCH -p short # use short for less than 24hrs
#SBATCH --job-name=count_svs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=much8161@colorado.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2gb
#SBATCH --time=24:00:00
#SBATCH --output=/Users/much8161/Repositories/exome-copy/out/count_svs.out
#SBATCH --error=/Users/much8161/Repositories/exome-copy/out/count_svs.err

DATA_DIR="/scratch/Shares/layer/projects/exome-copy/data/"
bash count_svs.sh $DATA_DIR
