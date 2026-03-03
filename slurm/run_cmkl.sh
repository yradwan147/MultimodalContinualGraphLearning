#!/bin/bash --login
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=48G
#SBATCH -o slurm/slurm_logs/%x_%J.out

# Run CMKL with a single seed on IBEX
# Usage:
#   sbatch -J cmkl_s42 slurm/run_cmkl.sh DistMult 42

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate mcgl

mkdir -p results checkpoints slurm/slurm_logs

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

DECODER=${1:-DistMult}
SEED=${2:?Must provide seed}

echo "Job ID: $SLURM_JOB_ID"
echo "Decoder: $DECODER, Seed: $SEED"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python scripts/run_cmkl.py \
    --decoder $DECODER \
    --embedding-dim 256 \
    --num-epochs 100 \
    --batch-size 512 \
    --device cuda \
    --seeds $SEED \
    --output-dir results \
    --output-suffix _seed${SEED} \
    --eval-multihop

echo "End: $(date)"
