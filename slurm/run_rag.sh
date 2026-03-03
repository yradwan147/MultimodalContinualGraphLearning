#!/bin/bash --login
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH -o slurm/slurm_logs/%x_%J.out

# Run RAG agent (retrieval-only, no LLM/GPU needed) with a single seed
# Usage:
#   sbatch -J rag_s42 slurm/run_rag.sh 42

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate mcgl

mkdir -p results slurm/slurm_logs

SEED=${1:?Must provide seed}

echo "Job ID: $SLURM_JOB_ID"
echo "RAG Seed: $SEED (retrieval-only, no LLM)"
echo "Start: $(date)"

python scripts/run_rag.py \
    --no-llm \
    --questions-per-task 200 \
    --seeds $SEED \
    --output-dir results \
    --output-suffix _seed${SEED} \
    --eval-multihop

echo "End: $(date)"
