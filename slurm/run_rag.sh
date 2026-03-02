#!/bin/bash --login
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH -J mcgl_rag
#SBATCH -o slurm/slurm_logs/mcgl_rag_%J.out

# Run RAG agent baseline for KGQA on IBEX
# Usage: sbatch slurm/run_rag.sh

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate mcgl

mkdir -p results slurm/slurm_logs

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python scripts/run_rag.py \
    --questions-per-task 200 \
    --seeds 42 123 456 789 1024 \
    --output-dir results

echo "End: $(date)"
