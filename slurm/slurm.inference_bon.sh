#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=rar
#SBATCH --output=/scratch/bc3088/longcot-rag/log/%x_%j.out
#SBATCH --error=/scratch/bc3088/longcot-rag/log/%x_%j.err
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --requeue

#SBATCH --mail-type=ALL
#SBATCH --mail-user=bale.chen@nyu.edu

source /scratch/bc3088/env.sh
conda activate agent

cd $SLURM_SUBMIT_DIR

DATASET_NAME=$1
DATASET_SPLIT=$2

python rar_inference_bon.py data.dataset_name=$DATASET_NAME data.dataset_split=$DATASET_SPLIT inference.max_turns=10 

echo "Done"
