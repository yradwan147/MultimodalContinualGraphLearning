#!/bin/bash
#SBATCH --job-name=mcgl-baseline
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Run baseline experiments on IBEX
# Usage: sbatch slurm/run_baseline.sh <baseline_name> <config_file>
# Example: sbatch slurm/run_baseline.sh ewc configs/ewc.yaml

module load cuda/11.8
module load conda
conda activate mcgl
mkdir -p slurm_logs results

BASELINE=${1:-naive_sequential}
CONFIG=${2:-configs/base.yaml}

echo "Job ID: $SLURM_JOB_ID"
echo "Running baseline: $BASELINE"
echo "Config: $CONFIG"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python scripts/run_baselines.py \
    --baseline $BASELINE \
    --config $CONFIG \
    --seeds 42 123 456 789 1024 \
    --output_dir results/${BASELINE}_$(date +%Y%m%d_%H%M%S)

echo "End: $(date)"
