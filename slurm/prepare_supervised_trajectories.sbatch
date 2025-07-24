#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=search-o1
#SBATCH --output=./%x_%j.out
#SBATCH --error=./%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h20:2
#SBATCH --partition=sfscai
#SBATCH --mem=200G
#SBATCH --requeue

# Load conda environment
source activate agent
module load cuda/12.4.1-gcc-10.1.0
# Move to project root (assumes slurm/ is in project root)
cd /scratch/yl9038/longcot-rag-inference

# Print start time
echo "Job started at $(date)"

# Run the script
python prepare_supervised_trajectories.py

# Print end time
echo "Job finished at $(date)" 