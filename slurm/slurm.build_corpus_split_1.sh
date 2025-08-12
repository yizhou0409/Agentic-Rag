#!/bin/bash
#SBATCH --job-name=build-corpus-split-1
#SBATCH --output=build-corpus-split-1_%j.out
#SBATCH --error=build-corpus-split-1_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=sfscai

# Load modules
module load cuda/12.4.1-gcc-10.1.0
# Activate conda environment
source /scratch/yl9038/anaconda3/etc/profile.d/conda.sh
conda activate agent

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Navigate to project directory
cd /gpfsnyu/scratch/yl9038/longcot-rag-inference

# Run split 1 processing
echo "Starting corpus split 1 processing..."
python build_wikipedia_corpus.py \
    --data_dir enwiki-20171001-pages-meta-current-withlinks-processed \
    --output_dir wikipedia_e5_index \
    --split_id 1 \
    --total_splits 2 \
    --batch_size 1000 \
    --embedding_batch_size 32 \
    --chunk_size 512 \
    --chunk_overlap 50 \
    --model_name /scratch/yl9038/models/e5-large-v2 \
    --device cuda

echo "Corpus split 1 processing completed!"
