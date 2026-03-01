#!/bin/bash --login
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=32G
#SBATCH -J mcgl_bl
#SBATCH -o mcgl_bl_%J.out

# Run a SINGLE baseline on IBEX
# Usage:
#   sbatch slurm/run_baseline.sh naive_sequential TransE
#   sbatch slurm/run_baseline.sh ewc TransE

source ~/miniconda3/bin/activate
conda activate mcgl

mkdir -p results

BASELINE=${1:?Usage: sbatch run_baseline.sh <baseline_name> [model]}
MODEL=${2:-TransE}

echo "Job ID: $SLURM_JOB_ID"
echo "Baseline: $BASELINE, Model: $MODEL"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python scripts/run_baselines.py \
    --baseline $BASELINE \
    --model $MODEL \
    --embedding-dim 256 \
    --num-epochs 100 \
    --batch-size 512 \
    --device cuda \
    --seeds 42 123 456 789 1024 \
    --output-dir results

echo "End: $(date)"
