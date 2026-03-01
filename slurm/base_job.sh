#!/bin/bash --login
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=32G
#SBATCH -J mcgl
#SBATCH -o mcgl_%J.out

# MCGL Base SLURM Job Template for IBEX

source ~/miniconda3/bin/activate
conda activate mcgl

mkdir -p results checkpoints

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time: $(date)"

# ---- REPLACE THIS WITH YOUR COMMAND ----
# python scripts/run_baselines.py --baseline naive_sequential
# python scripts/run_cmkl.py --seeds 42 123 456 789 1024
# ----------------------------------------

echo "End time: $(date)"
