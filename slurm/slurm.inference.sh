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

#SBATCH --mail-type=ALL
#SBATCH --mail-user=yl9038@nyu.edu
module load cuda/12.4.1-gcc-10.1.0

source activate agent

cd /scratch/yl9038/longcot-rag-inference
export HYDRA_FULL_ERROR=1

python rar_inference.py --config-name inference_int4.yaml inference.max_turns=5
    #  data.subset_size=1000
echo "Done"