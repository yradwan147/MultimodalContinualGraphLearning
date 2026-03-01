#!/bin/bash
#SBATCH --job-name=mcgl-ablation
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Run ablation studies on IBEX
# Usage: sbatch slurm/run_ablations.sh [ablation_name]
# Example: sbatch slurm/run_ablations.sh struct_only

module load cuda/11.8
module load conda
conda activate mcgl
mkdir -p slurm_logs results

ABLATION=${1:-all}

echo "Job ID: $SLURM_JOB_ID"
echo "Ablation: $ABLATION"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python scripts/run_ablations.py \
    --ablation $ABLATION \
    --seeds 42 123 456 789 1024 \
    --output_dir results/ablation_${ABLATION}_$(date +%Y%m%d_%H%M%S)

echo "End: $(date)"
