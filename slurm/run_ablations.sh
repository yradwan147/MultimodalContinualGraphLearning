#!/bin/bash --login
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=32G
#SBATCH -J mcgl_abl
#SBATCH -o mcgl_abl_%J.out

# Run a SINGLE ablation study on IBEX
# Usage:
#   sbatch slurm/run_ablations.sh struct_only
#   sbatch slurm/run_ablations.sh buffer_size_sweep
#
# DO NOT use "all" — submit each ablation as a separate job via submit_all.sh

source ~/miniconda3/bin/activate
conda activate mcgl

mkdir -p results

ABLATION=${1:?Usage: sbatch run_ablations.sh <ablation_name>}

echo "Job ID: $SLURM_JOB_ID"
echo "Ablation: $ABLATION"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python scripts/run_ablations.py \
    --ablation $ABLATION \
    --embedding-dim 256 \
    --num-epochs 100 \
    --batch-size 512 \
    --device cuda \
    --seeds 42 123 456 789 1024 \
    --output-dir results

echo "End: $(date)"
