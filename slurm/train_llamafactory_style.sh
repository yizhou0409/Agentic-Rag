#!/bin/bash
#SBATCH --job-name=llamafactory_style_train
#SBATCH --output=llamafactory_style_train_%j.out
#SBATCH --error=llamafactory_style_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h20:4
#SBATCH --partition=sfscai
#SBATCH --mem=128G
#SBATCH --requeue


# Load modules (adjust based on your cluster)
module load cuda/12.4.1-gcc-10.1.0

# Activate conda environment
source activate agent

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

# Change to project directory
cd /gpfsnyu/scratch/yl9038/longcot-rag-inference

# Run training with distributed training across 4 GPUs
torchrun --nproc_per_node=4 --master_port=29500 train_supervised_llamafactory_style.py \
    --model_name_or_path /scratch/yl9038/models/Qwen3-8B \
    --data_paths data/hotpotqa/trajectory_train_5.jsonl data/2wikimultihop/trajectory_train_5.jsonl \
    --dataset_dir data \
    --max_samples 100000 \
    --val_size 0.15 \
    --output_dir ./output/llamafactory_style_sft \
    --num_train_epochs 10.0 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_steps 50 \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --eval_steps 100 \
    --save_total_limit 2 \
    --cutoff_len 1024 \
    --compute_type pure_bf16 \
    --report_to none \
    --dataloader_num_workers 1 \
    --seed 42 \
    --ddp_timeout 180000000 