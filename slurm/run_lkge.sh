#!/bin/bash --login
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=128G
#SBATCH -J mcgl_lkge
#SBATCH -o slurm/slurm_logs/mcgl_lkge_%J.out

# Run LKGE baseline on IBEX
# Usage: sbatch slurm/run_lkge.sh [model]

eval "$(~/miniconda3/bin/conda shell.bash hook)"
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
    # LKGE has no requirements.txt — deps from README: pytorch, pyg, prettytable, quadprog
    # pytorch and pyg are already in the mcgl conda env
    pip install prettytable quadprog
fi

# Remove stale LKGE format data from previous runs
rm -rf results/lkge_format

# Skip task_0_base (8.1M triples) — too large for LKGE's in-memory GCN.
# Use only the incremental CL tasks.
python scripts/run_lkge.py \
    --model $MODEL \
    --num-epochs 100 \
    --seeds 42 123 456 789 1024 \
    --output-dir results \
    --task-names task_1_disease_related task_1_drug_related \
        task_2_disease_related task_2_gene_protein \
        task_3_gene_protein task_3_phenotype_related \
        task_4_biological_process task_4_phenotype_related \
        task_5_anatomy_pathway

echo "End: $(date)"
