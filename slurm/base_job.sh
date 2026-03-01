#!/bin/bash
#SBATCH --job-name=mcgl
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# MCGL Base SLURM Job Template for IBEX
# Usage: sbatch slurm/base_job.sh

# Load modules (adjust to IBEX's available modules)
module load cuda/11.8
module load conda

# Activate environment
conda activate mcgl

# Create log directory
mkdir -p slurm_logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# ---- REPLACE THIS WITH YOUR COMMAND ----
# python scripts/run_baselines.py --config configs/base.yaml
# python scripts/run_cmkl.py --config configs/cmkl.yaml
# ----------------------------------------

echo "End time: $(date)"
