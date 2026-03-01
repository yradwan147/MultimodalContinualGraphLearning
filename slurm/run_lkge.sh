#!/bin/bash --login
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=32G
#SBATCH -J mcgl_lkge
#SBATCH -o slurm/slurm_logs/mcgl_lkge_%J.out

# Run LKGE baseline on IBEX
# Usage: sbatch slurm/run_lkge.sh [model]

source ~/miniconda3/bin/activate
conda activate mcgl

mkdir -p results slurm/slurm_logs

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

MODEL=${1:-TransE}

echo "Job ID: $SLURM_JOB_ID"
echo "LKGE Model: $MODEL"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

# Clone LKGE if not already present
if [ ! -d "external/LKGE" ]; then
    echo "Cloning LKGE repository..."
    mkdir -p external
    git clone https://github.com/nju-websoft/LKGE.git external/LKGE
    cd external/LKGE && pip install -r requirements.txt && cd ../..
fi

python scripts/run_lkge.py \
    --model $MODEL \
    --num-epochs 100 \
    --seeds 42 123 456 789 1024 \
    --output-dir results

echo "End: $(date)"
