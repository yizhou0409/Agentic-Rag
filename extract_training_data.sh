#!/bin/bash
#SBATCH --job-name=extract-training-data
#SBATCH --partition=sfscai
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h20:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=extract-training-data-%j.out
#SBATCH --error=extract-training-data-%j.err

# Load modules
module load cuda/12.4.1-gcc-10.1.0

# Activate conda environment
source activate agent

# Set working directory
cd /gpfsnyu/scratch/yl9038/longcot-rag-inference

# Run the extraction script
echo "Starting training data extraction with single GPU..."

python extract_training_data.py \
    --dataset_paths "./data/hotpotqa/trajectory_train_200.jsonl,./data/2wikimultihop/trajectory_train_200.jsonl" \
    --base_model_path /scratch/yl9038/models/Qwen3-32B \
    --judge_model_path /scratch/yl9038/models/Qwen3-32B \
    --output_path ./extracted_training_data.json \
    --max_examples_per_dataset 10000 \
    --use_quantization

echo "Extraction completed!" 