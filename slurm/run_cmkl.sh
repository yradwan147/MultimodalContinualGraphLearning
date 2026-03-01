#!/bin/bash
#SBATCH --job-name=mcgl-cmkl
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Run CMKL experiments on IBEX
# Usage: sbatch slurm/run_cmkl.sh [config_file]
# Example: sbatch slurm/run_cmkl.sh configs/cmkl.yaml

module load cuda/11.8
module load conda
conda activate mcgl
mkdir -p slurm_logs results checkpoints

CONFIG=${1:-configs/cmkl.yaml}

echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python scripts/run_cmkl.py \
    --config $CONFIG \
    --seeds 42 123 456 789 1024 \
    --output_dir results/cmkl_$(date +%Y%m%d_%H%M%S) \
    --checkpoint_dir checkpoints/

echo "End: $(date)"
