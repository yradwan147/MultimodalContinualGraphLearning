#!/bin/bash
# Submit ALL experiment jobs in parallel on IBEX
# Usage: bash slurm/submit_all.sh
#
# This submits 12 independent jobs:
#   4 baselines + 1 CMKL + 7 ablations = 12 jobs
# Each job: 32G RAM, 1 GPU, up to 6 hours
# Total GPU-hours: ~72 (but all run in parallel)

echo "Submitting all MCGL experiment jobs..."
echo ""

# --- 4 Baselines (1 job each) ---
echo "=== Baselines ==="
sbatch slurm/run_baseline.sh naive_sequential TransE
sbatch slurm/run_baseline.sh joint_training TransE
sbatch slurm/run_baseline.sh ewc TransE
sbatch slurm/run_baseline.sh experience_replay TransE

# --- CMKL ---
echo "=== CMKL ==="
sbatch slurm/run_cmkl.sh DistMult

# --- 7 Ablations (1 job each) ---
echo "=== Ablations ==="
sbatch slurm/run_ablations.sh struct_only
sbatch slurm/run_ablations.sh text_only
sbatch slurm/run_ablations.sh concat_fusion
sbatch slurm/run_ablations.sh global_ewc
sbatch slurm/run_ablations.sh random_replay
sbatch slurm/run_ablations.sh buffer_size_sweep
sbatch slurm/run_ablations.sh lambda_sweep

echo ""
echo "Submitted 12 jobs. Monitor with: squeue -u \$USER"
echo "Results will appear in results/*.json"
