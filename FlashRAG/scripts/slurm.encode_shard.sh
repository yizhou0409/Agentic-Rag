#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=encode
#SBATCH --output=/gpfsnyu/scratch/bc3088/longcot-rag/FlashRAG/logs%x_%j_%A_%a.out
#SBATCH --error=/gpfsnyu/scratch/bc3088/longcot-rag/FlashRAG/logs%x_%j_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:h20:1
#SBATCH --partition=sfscai
#SBATCH --mem=128GB

#SBATCH --mail-type=ALL
#SBATCH --mail-user=bale.chen@nyu.edu

source /scratch/bc3088/env.sh
conda activate agent

cd $SLURM_SUBMIT_DIR

CUDA_VISIBLE_DEVICES=0 python -m flashrag.retriever.encode_shard \
    --retrieval_method bge \
    --model_path /gpfsnyu/scratch/bc3088/models/bge-large-en-v1.5 \
    --corpus_path ../wiki-20231120-chunk256.jsonl \
    --save_dir ../index/bge-2023 \
    --use_fp16 \
    --max_length 512 \
    --batch_size 256 \
    --sentence_transformer \
    --faiss_type Flat \
    --save_embedding \
    --shard_id $SLURM_ARRAY_TASK_ID \
    --num_shards $SLURM_ARRAY_TASK_COUNT | tee ../logs/encode_shard.log