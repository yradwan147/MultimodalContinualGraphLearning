# MCGL Project Worklog

This file tracks all work done across Claude sessions. Updated after every session.
Used for: cross-session context recovery, progress reporting to Prof. Zhang.

---

## 2026-03-01 Session 1 - Project Scaffolding (Phase 0)

### Changes Made
- Initialized git repository
- Created full project directory structure (src/, scripts/, slurm/, configs/, notebooks/, papers/, docs/)
- Created `CLAUDE.md` with mandatory worklog rules, documentation standards, compute rules
- Created `.claude-plans/` with 7 phase plan files covering entire project lifecycle
- Created `.gitignore` for Python, data, checkpoints, wandb
- Created `requirements.txt` and `environment.yml` for conda environment
- Created `README.md` with project overview and setup instructions
- Created 4 SLURM template scripts for IBEX (base_job.sh, run_baseline.sh, run_cmkl.sh, run_ablations.sh)
- Created 4 config YAML files (base.yaml, ewc.yaml, replay.yaml, cmkl.yaml)
- Created all `__init__.py` files for src/ subpackages
- Created .gitkeep files in data/, checkpoints/, results/
- Created `docs/architecture.md` and `docs/experiment-results.md` (placeholder templates)

## 2026-03-01 Session 2 - Complete Scaffolding

### Changes Made
- Created all 29 placeholder Python implementation modules with proper docstrings, type hints, and `NotImplementedError` stubs:
  - `src/data/`: download.py, temporal_diff.py, task_sequence.py, splits.py, features.py
  - `src/baselines/`: naive_sequential.py, joint_training.py, ewc.py, experience_replay.py, lkge.py, rag_agent.py
  - `src/models/`: encoders.py, fusion.py, cmkl.py, decoders.py
  - `src/continual/`: modality_ewc.py, multimodal_replay.py, distillation.py
  - `src/evaluation/`: metrics.py, statistical.py, visualization.py
  - `src/utils/`: config.py, logging.py, io.py
- Created 6 runnable scripts with argparse CLI:
  - `scripts/`: download_primekg.py, build_benchmark.py, run_baselines.py, run_cmkl.py, run_ablations.py, generate_tables.py
- Created 4 Jupyter notebooks (placeholder .ipynb):
  - `notebooks/`: 01_explore_primekg, 02_benchmark_stats, 03_results_analysis, 04_paper_figures
- Created paper LaTeX templates:
  - `papers/paper_a_benchmark/`: main.tex, references.bib, figures/
  - `papers/paper_b_method/`: main.tex, references.bib, figures/
- Created `docs/literature-review.md` with initial bibliography structure (8 categories)
- Fixed `.gitignore` - removed overly broad `*.csv` rule that would have blocked results tracking
- Initial git commit with all scaffolding

### Phase Plans Created
| Phase | File | Description | Status |
|-------|------|-------------|--------|
| 0 | (this session) | Project scaffolding | DONE |
| 1 | phase1-setup-and-literature.md | Environment setup, PrimeKG exploration | PENDING |
| 2 | phase2-benchmark-construction.md | Temporal benchmark from PrimeKG snapshots | PENDING |
| 3 | phase3-baseline-implementation.md | 6 baseline methods | PENDING |
| 4 | phase4-cmkl-development.md | Novel CMKL method | PENDING |
| 5 | phase5-experiments-and-ablations.md | Full experiments + ablations | PENDING |
| 6 | phase6-paper-a-benchmark.md | Benchmark paper writing | PENDING |
| 7 | phase7-paper-b-method.md | Method paper writing | PENDING |

### File Count Summary
- Total scaffold files created: ~80 files across 15+ directories
- Implementation modules: 29 (all placeholder stubs with `NotImplementedError`)
- Config YAML files: 4
- SLURM scripts: 4
- Notebooks: 4
- Paper LaTeX: 2 papers with templates
- Phase plans: 7 + index

### Next Steps
- Phase 1: Set up conda environment, install dependencies, download and explore PrimeKG t0
