#!/bin/bash
#SBATCH --job-name=preprocess_wiki
#SBATCH --output=preprocess_wiki_%j.out
#SBATCH --error=preprocess_wiki_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G

# Load any required modules (if needed)

# Activate your virtual environment if you have one
source activate wiki310
# Run the preprocessing script
python preprocess_wiki.py \
    --dump_path ../enwiki-20181220-pages-articles.xml.bz2 \
    --save_path ../test_sample.jsonl \
    --chunk_by sentence \
    --chunk_size 512 \
    --num_workers 1