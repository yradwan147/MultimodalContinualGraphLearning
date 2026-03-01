#!/bin/bash --login
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=32G
#SBATCH -J mcgl_nc
#SBATCH -o slurm/slurm_logs/mcgl_nc_%J.out

# Run node classification experiments on IBEX
# Usage: sbatch slurm/run_nc.sh <method>
#   Methods: naive_sequential, joint_training, ewc, experience_replay, cmkl

source ~/miniconda3/bin/activate
conda activate mcgl

mkdir -p results slurm/slurm_logs

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

METHOD=${1:?Usage: sbatch run_nc.sh <method>}

echo "Job ID: $SLURM_JOB_ID"
echo "NC Method: $METHOD"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python scripts/run_nc.py \
    --method $METHOD \
    --embedding-dim 256 \
    --num-epochs 100 \
    --batch-size 512 \
    --device cuda \
    --seeds 42 123 456 789 1024 \
    --output-dir results

echo "End: $(date)"
