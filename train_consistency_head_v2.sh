#!/bin/bash
#SBATCH --job-name=train-consistency-head
#SBATCH --partition=sfscai
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h20:2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=train-consistency-head-%j.out
#SBATCH --error=train-consistency-head-%j.err

# Load modules
module load cuda/12.4.1-gcc-10.1.0

# Activate conda environment
source activate agent

# Set working directory
cd /gpfsnyu/scratch/yl9038/longcot-rag-inference

# Run the training script
echo "Starting consistency head training on 32B model..."

python train_consistency_head_v2.py \
    --model_path /scratch/yl9038/models/Qwen3-32B \
    --training_data_path ./extracted_training_data.json \
    --save_path ./trained_consistency_head_v2.pt \
    --num_epochs 10 \
    --batch_size 8 \
    --learning_rate 3e-4 \
    --use_quantization \
    --use_multi_gpu

echo "Training completed!" 