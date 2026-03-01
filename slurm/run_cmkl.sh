#!/bin/bash --login
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=32G
#SBATCH -J mcgl_cmkl
#SBATCH -o mcgl_cmkl_%J.out

# Run CMKL experiment on IBEX
# Usage: sbatch slurm/run_cmkl.sh [decoder]

source ~/miniconda3/bin/activate
conda activate mcgl

mkdir -p results checkpoints

DECODER=${1:-DistMult}

echo "Job ID: $SLURM_JOB_ID"
echo "Decoder: $DECODER"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python scripts/run_cmkl.py \
    --decoder $DECODER \
    --embedding-dim 256 \
    --num-epochs 100 \
    --batch-size 512 \
    --device cuda \
    --seeds 42 123 456 789 1024 \
    --output-dir results

echo "End: $(date)"
