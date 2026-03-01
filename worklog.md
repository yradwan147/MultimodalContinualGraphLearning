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

---

## 2026-03-01 Session 3 - Phase 1: Environment Setup & PrimeKG Exploration

### Changes Made

#### Step 1.1: Conda Environment Created
- Created `mcgl` conda environment with Python 3.10 on Apple M3 Pro (MPS backend)
- Installed all dependencies via pip (PyTorch CPU/MPS for local, CUDA for IBEX):
  - PyTorch 2.10.0, PyG 2.7.0, PyKEEN 1.11.1
  - Transformers 4.50.3 (downgraded by TDC from 5.2.0), NumPy 1.26.4 (downgraded by TDC from 2.2.6)
  - RDKit 2025.09.5, pandas 2.3.3, scikit-learn 1.7.2
  - LangChain 1.2.10, ChromaDB 1.5.2, SentenceTransformers 5.2.3
  - W&B 0.25.0, Hydra 1.3.2
  - PyTDC 1.1.15 (for PrimeKG download)
- All imports verified successfully
- MPS (Apple Silicon GPU) available for local development

#### Step 1.2: Literature Review
- SKIPPED per user preferences (PDF provides sufficient coverage)

#### Step 1.3: Implemented Utility Modules
- `src/utils/config.py`: load_config(), merge_configs(), save_config() with YAML support and dot-notation overrides
- `src/utils/io.py`: save_json(), load_json(), save_tensor(), load_tensor(), ensure_dir()
- Both modules tested and verified with round-trip tests

#### Step 1.4: Implemented src/data/download.py
- `download_primekg_t0()`: Downloads via TDC (preferred) or Harvard Dataverse, with fallback
- `download_primekg_tdc()`: Uses TDC PrimeKG resource, also saves drug features
- `download_primekg_dataverse()`: Direct download from Harvard Dataverse
- `load_primekg()`: CSV loading with optional chunked reading
- `verify_primekg()`: Data integrity checks (row count, column names, node/relation types)

#### Step 1.5: Downloaded and Explored PrimeKG t0
- Downloaded via TDC: 888MB download, saved as `data/benchmark/snapshots/kg_t0.csv` (847MB)
- Also obtained `drug_features_t0.csv` (9.3MB, 7957 drugs, 18 feature columns)

### PrimeKG t0 Exploration Results
| Metric | Value |
|--------|-------|
| Total edges | 8,100,498 |
| Total columns | 10 (TDC version lacks x_index, y_index) |
| Node types | 10 |
| Relation types | 30 |
| Unique nodes | 129,312 |
| Drug-Disease edges | 85,262 |
| Drug-Disease relation types | contraindication (61,350), indication (18,776), off-label use (5,136) |

#### Node counts by type:
| Node Type | Count |
|-----------|-------|
| biological_process | 28,642 |
| gene/protein | 27,610 |
| disease | 17,080 |
| effect/phenotype | 15,311 |
| anatomy | 14,033 |
| molecular_function | 11,169 |
| drug | 7,957 |
| cellular_component | 4,176 |
| pathway | 2,516 |
| exposure | 818 |

#### Top 5 edge types:
| Relation | Count |
|----------|-------|
| anatomy_protein_present | 3,036,406 |
| drug_drug | 2,672,628 |
| protein_protein | 642,150 |
| disease_phenotype_positive | 300,634 |
| bioprocess_protein | 289,610 |

#### Drug Features Coverage (18 columns):
- description: 57.7% non-null (4,591/7,957)
- indication: 42.6% non-null
- mechanism_of_action: 40.7% non-null
- pharmacodynamics: 33.4% non-null
- molecular_weight: 35.2% non-null (numeric, for molecular fingerprints)
- state: 81.9% non-null

#### Key Observations:
1. **TDC version lacks `x_index`/`y_index` columns** - need to create integer node mapping in Phase 2
2. **Highly skewed edge distribution** - anatomy_protein_present and drug_drug dominate (>70% of edges)
3. **Drug-disease edges (85K)** are a small fraction (~1%) of total edges - good for link prediction
4. **Drug features have variable coverage** - description (58%), indication (43%), molecular properties (~35%)
5. **129,312 unique nodes** matches expected ~129K from PrimeKG paper

### Files Modified/Created
- `src/utils/config.py`: Fully implemented (was stub)
- `src/utils/io.py`: Fully implemented (was stub)
- `src/data/download.py`: Fully implemented (was stub)
- `notebooks/01_explore_primekg.ipynb`: Full EDA notebook with 7 sections
- `results/primekg_t0_relation_dist.png`: Relation type distribution plot
- `results/primekg_t0_node_dist.png`: Node type distribution plot
- `results/primekg_t0_edge_heatmap.png`: Node type cross-tabulation heatmap

### Phase Status Update
| Phase | Status |
|-------|--------|
| 0 - Scaffolding | DONE |
| 1 - Setup & Exploration | DONE (literature review skipped per user pref) |
| 2 - Benchmark Construction | PENDING (next) |

### Next Steps
- Phase 2: Build temporal benchmark from PrimeKG snapshots
  - Create integer node index mapping for TDC data
  - Download/rebuild PrimeKG t1 (July 2023)
  - Compute temporal diffs between snapshots
  - Create CL task sequences
  - Extract multimodal features
