#!/bin/bash
#SBATCH --job-name=llamafactory_sft
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h20:4
#SBATCH --partition=sfscai
#SBATCH --time=12:00:00

# Load modules if needed (uncomment and edit if your cluster uses module load)
# module load cuda/11.7
module load cuda/12.4.1-gcc-10.1.0

# Activate conda environment
source /gpfsnyu/scratch/yl9038/anaconda3/etc/profile.d/conda.sh
conda activate llama-train

# Ensure dataset_info.json is present and print its contents
echo "Checking dataset_info.json on compute node:"
ls -l LLaMA-Factory/data/dataset_info.json || cp /gpfsnyu/scratch/yl9038/longcot-rag-inference/LLaMA-Factory/data/dataset_info.json LLaMA-Factory/data/dataset_info.json
cat LLaMA-Factory/data/dataset_info.json | head -20

# Set environment variables for distributed training
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Print some useful information
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
nvidia-smi

# Navigate to LLaMA-Factory directory

# Run the LLaMA-Factory training using CLI with torchrun
echo "Starting LLaMA-Factory training..."
torchrun --nproc_per_node=4 LLaMA-Factory/src/train.py \
    --model_name_or_path /scratch/yl9038/models/Qwen3-4B \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --dataset 2wikimultihop_baseline_alpaca,hotpotqa_baseline_alpaca \
    --dataset_dir LLaMA-Factory/data \
    --template default \
    --cutoff_len 2048 \
    --max_samples 100000 \
    --overwrite_cache \
    --preprocessing_num_workers 4 \
    --dataloader_num_workers 2 \
    --output_dir trained_baseline_models/qwen3-4b \
    --logging_steps 5 \
    --save_steps 100 \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model \
    --report_to none \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 10.0 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --warmup_steps 50 \
    --bf16 \
    --ddp_timeout 180000000 \
    --gradient_checkpointing \
    --deepspeed LLaMA-Factory/examples/deepspeed/ds_z3_config.json \
    --ddp_backend nccl \
    --ddp_find_unused_parameters false \
    --prompt_template_path /gpfsnyu/scratch/yl9038/longcot-rag-inference/prompts/default_QA.yaml

# Print completion message
echo "Job completed at $(date)" 