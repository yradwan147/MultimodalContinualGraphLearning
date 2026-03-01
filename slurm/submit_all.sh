#!/bin/bash
# Submit ALL experiment jobs in parallel on IBEX
# Usage: bash slurm/submit_all.sh
#
# This submits 20 independent jobs across 3 tasks:
#
# Link Prediction (14 jobs):
#   4 baselines + 1 LKGE + 1 CMKL + 8 ablations = 14
#
# KGQA (1 job):
#   1 RAG agent = 1
#
# Node Classification (5 jobs):
#   4 baselines + 1 CMKL = 5
#
# Each job: 32G RAM (64G for RAG), 1 V100 GPU, up to 24 hours
# All jobs have PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

mkdir -p slurm/slurm_logs

echo "Submitting all MCGL experiment jobs..."
echo "======================================="
echo ""

# -----------------------------------------------------------
# Link Prediction: 4 KGE Baselines (1 job each)
# -----------------------------------------------------------
echo "=== Link Prediction: Baselines ==="
sbatch slurm/run_baseline.sh naive_sequential TransE
sbatch slurm/run_baseline.sh joint_training TransE
sbatch slurm/run_baseline.sh ewc TransE
sbatch slurm/run_baseline.sh experience_replay TransE

# -----------------------------------------------------------
# Link Prediction: LKGE (Baseline 5)
# -----------------------------------------------------------
echo "=== Link Prediction: LKGE ==="
sbatch slurm/run_lkge.sh TransE

# -----------------------------------------------------------
# Link Prediction: CMKL
# -----------------------------------------------------------
echo "=== Link Prediction: CMKL ==="
sbatch slurm/run_cmkl.sh DistMult

# -----------------------------------------------------------
# Link Prediction: 8 Ablation Studies (1 job each)
# -----------------------------------------------------------
echo "=== Link Prediction: Ablations ==="
sbatch slurm/run_ablations.sh struct_only
sbatch slurm/run_ablations.sh text_only
sbatch slurm/run_ablations.sh concat_fusion
sbatch slurm/run_ablations.sh global_ewc
sbatch slurm/run_ablations.sh random_replay
sbatch slurm/run_ablations.sh buffer_size_sweep
sbatch slurm/run_ablations.sh lambda_sweep
sbatch slurm/run_ablations.sh distillation

# -----------------------------------------------------------
# KGQA: RAG Agent (Baseline 6)
# -----------------------------------------------------------
echo "=== KGQA: RAG Agent ==="
sbatch slurm/run_rag.sh

# -----------------------------------------------------------
# Node Classification: 4 baselines + CMKL (1 job each)
# -----------------------------------------------------------
echo "=== Node Classification ==="
sbatch slurm/run_nc.sh naive_sequential
sbatch slurm/run_nc.sh joint_training
sbatch slurm/run_nc.sh ewc
sbatch slurm/run_nc.sh experience_replay
sbatch slurm/run_nc.sh cmkl

echo ""
echo "======================================="
echo "Submitted 20 jobs. Monitor with: squeue -u \$USER"
echo "Results will appear in results/*.json"
echo ""
echo "Job breakdown:"
echo "  Link Prediction:      14 jobs (4 baselines + LKGE + CMKL + 8 ablations)"
echo "  KGQA:                  1 job  (RAG agent)"
echo "  Node Classification:   5 jobs (4 baselines + CMKL)"
echo "  Total:                20 jobs"
