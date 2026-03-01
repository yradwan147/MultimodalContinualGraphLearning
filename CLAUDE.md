# MCGL Project - Claude Workspace Rules

## Project Overview
**Multimodal Continual Graph Learning (MCGL)** for evolving biomedical knowledge graphs.
- Core question: How can graph-based models continually learn from evolving, multimodal biomedical KGs without catastrophically forgetting previously learned knowledge?
- Primary KG: PrimeKG (129K+ nodes, 8.1M+ edges, 10 node types, 30 edge types)
- Two papers: Paper A (Benchmark), Paper B (Method - CMKL)
- Supervisor: Prof. Zhang

## Source Document
All project details are in `MCGL Full Project Guide.pdf` (50 pages) in the project root. Code suggestions in that PDF are **guidance only** and may not be correct - always verify, test, and fix before using.

---

## MANDATORY: Worklog Rules

**Every Claude session MUST update `worklog.md`** with:

### For every code change:
```
## [DATE] Session - [Brief Description]
### Changes Made
- `path/to/file.py`: [What was changed and why]
- `path/to/other.py`: [What was changed and why]
```

### For every experiment run:
```
### Experiment: [Name]
- **Config:** [key hyperparameters]
- **Command:** `[exact command run]`
- **Results:**
  - Metric 1: value
  - Metric 2: value
- **Observations:** [what the results mean]
- **Next steps:** [what to do based on these results]
```

### For every error/bug encountered:
```
### Issue: [Description]
- **Error:** [error message or unexpected behavior]
- **Root cause:** [what caused it]
- **Fix:** [how it was resolved]
```

### Session start protocol:
1. Read this CLAUDE.md file
2. Read `worklog.md` to recover context from previous sessions
3. Read the relevant phase plan in `.claude-plans/`
4. Continue from where the last session left off

### Session end protocol:
1. Update `worklog.md` with everything done this session
2. Note any incomplete work or next steps
3. Commit changes with descriptive message

---

## Code Documentation Standards

### Every Python file must have:
- Module-level docstring explaining purpose
- All public functions/classes have docstrings with Args, Returns, and usage examples where helpful
- Inline comments for non-obvious logic (not every line, just the tricky parts)
- Type hints for function signatures

### README files:
- `README.md` (root): Project overview, setup, quickstart
- `src/README.md`: Source code architecture
- Each major module directory should have brief docs in docstrings or a README if complex

### Experiment documentation:
- All experiment configs saved to `configs/`
- All results saved to `results/` with descriptive filenames
- Key results also recorded in `docs/experiment-results.md`

---

## Directory Structure

```
.
├── CLAUDE.md               # THIS FILE - workspace rules
├── README.md               # Project overview + setup
├── worklog.md              # Running log of all work (MANDATORY)
├── .claude-plans/          # Phase-by-phase implementation plans
├── docs/                   # Documentation
│   ├── architecture.md     # CMKL architecture notes
│   └── experiment-results.md
├── src/                    # Source code
│   ├── data/               # Data loading, processing, benchmark construction
│   ├── baselines/          # 6 baseline implementations
│   ├── models/             # CMKL model components
│   ├── continual/          # Continual learning mechanisms
│   ├── evaluation/         # Metrics, stats, visualization
│   └── utils/              # Config, logging, I/O
├── scripts/                # Top-level runnable scripts
├── slurm/                  # SLURM job scripts for IBEX cluster
├── configs/                # Experiment configuration YAML files
├── notebooks/              # Jupyter notebooks for exploration
├── papers/                 # LaTeX paper drafts
│   ├── paper_a_benchmark/
│   └── paper_b_method/
├── data/                   # .gitignored - downloaded data
├── checkpoints/            # .gitignored - model weights
├── results/                # Experiment results (git tracked)
├── environment.yml         # Conda environment
└── requirements.txt        # Pip requirements
```

---

## GPU / Compute Rules

- **Local machine:** Data exploration, small tests, code development, plotting
- **IBEX (KAUST):** Model training, large experiments, hyperparameter sweeps, ablation studies
  - Partition: `gpu` (general)
  - GPU: V100
  - SLURM scripts in `slurm/` directory
  - Workflow: Claude prepares scripts -> User runs on IBEX -> User pastes outputs -> Claude analyzes

---

## Key Technical Details

### Benchmark: 3 temporal PrimeKG snapshots
- t0: June 2021 (original PrimeKG, Harvard Dataverse)
- t1: July 2023 (rebuilt with updated databases)
- t2: Feb 2026 (PrimeKG-U, optional)

### Tasks:
1. **Continual Drug-Disease Link Prediction** (MRR, Hits@K, AUPRC)
2. **Continual Biomedical KGQA** (EM, F1, accuracy)
3. **Continual Disease Node Classification** (Macro-F1, accuracy) - optional

### Baselines (6):
1. Naive Sequential (lower bound)
2. Joint Training (upper bound)
3. EWC for KGE
4. Experience Replay (BER-style)
5. LKGE (external framework)
6. RAG Agent (for KGQA)

### Proposed Method: CMKL
- Modality-specific encoders (R-GCN, BiomedBERT, Morgan fingerprint)
- Cross-modal attention fusion
- Modality-aware EWC (per-modality Fisher matrices)
- Multimodal memory replay (K-means diverse selection)

### CL Metrics:
- Average Performance (AP), Average Forgetting (AF)
- Backward Transfer (BWT), Forward Transfer (FWT), Remembering (REM)

### Statistical Rigor:
- 5 random seeds per experiment: [42, 123, 456, 789, 1024]
- Report mean +/- std
- Paired t-tests or Wilcoxon signed-rank at p < 0.05
