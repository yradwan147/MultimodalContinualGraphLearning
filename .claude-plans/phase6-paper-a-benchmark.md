# Phase 6: Paper A - Benchmark Paper
**Estimated time:** Weeks 22-26 | **PDF Section:** 9.1
**Depends on:** Phase 5 (all experiments done) | **Blocks:** None (can run in parallel with Phase 7)

---

## Objectives
- Write Paper A: A temporal, multimodal biomedical KG benchmark for continual learning evaluation
- Target venues: NeurIPS Datasets & Benchmarks, Nature Scientific Data, WWW

---

## Paper Structure

### 1. Introduction (1 page)
- Motivation: biomedical KGs evolve as new drugs, diseases, and relationships are discovered
- Gap: existing CGL benchmarks use synthetic splits on generic KGs (FB15k-237, WN18RR), not real temporal evolution on domain-specific KGs
- Contribution: first CL benchmark on a real biomedical KG with genuine temporal evolution and multimodal features

### 2. Related Work (1 page)
- Existing CGL benchmarks: PS-CKGE, LKGE datasets, ICEWS temporal splits
- Biomedical KGs: PrimeKG, TarKG, BioMedKG
- Temporal KGs: distinguish from event-based temporal KGs (ICEWS) - ours captures knowledge evolution

### 3. Benchmark Construction (2 pages)
- PrimeKG temporal snapshots: t0 (June 2021), t1 (July 2023), optionally t2 (Feb 2026)
- Snapshot strategy: real database version differences, not artificial splits
- Task sequence design: entity_type strategy and rationale
- Multimodal features: text (DrugBank, Mayo Clinic), molecular (SMILES fingerprints), structural
- Statistics: node/edge counts per snapshot, diff statistics, task sizes

### 4. Evaluation Protocol (1 page)
- Metrics: AP, AF, BWT, FWT, REM, MRR, Hits@K
- Experimental setup: 5 seeds, significance testing
- Task descriptions: LP, KGQA, optional NC

### 5. Baseline Experiments (2 pages)
- Results of all 6 baselines on the benchmark
- Main results table
- Analysis of forgetting patterns
- Per-task and per-relation-type breakdown

### 6. Analysis (1 page)
- How multimodal features affect forgetting
- Domain-specific forgetting patterns (drug-related vs. disease-related)
- Comparison with standard CGL benchmarks
- Distribution shift analysis between snapshots

### 7. Conclusion (0.5 page)
- Summary of contributions
- Limitations and future directions

---

## Key Selling Points
- **First** continual learning benchmark on a real biomedical KG with genuine temporal evolution
- Multimodal features (text, molecular, structural) enable studying modality-specific forgetting
- Comprehensive baselines provide reference points for future research
- Real-world utility: drug repurposing under evolving knowledge

---

## Writing Process

### Step 6.1: Set up LaTeX template
```
papers/paper_a_benchmark/
├── main.tex          # NeurIPS template: \usepackage{neurips_2026}
├── references.bib    # BibTeX entries from Google Scholar / Semantic Scholar
├── figures/
│   ├── benchmark_overview.pdf    # Architecture/pipeline diagram
│   ├── temporal_evolution.pdf    # KG evolution visualization
│   ├── results_heatmap.pdf       # Results matrix heatmaps
│   └── forgetting_curves.pdf     # Forgetting over task sequence
└── tables/
    ├── dataset_stats.tex         # Benchmark statistics
    ├── main_results.tex          # All methods x all metrics
    └── per_task_results.tex      # Detailed per-task breakdown
```

### Step 6.2: Write draft sections
Write each section, pulling results from `results/` and `docs/experiment-results.md`.

### Step 6.3: Generate all figures and tables
Use `scripts/generate_tables.py` and `notebooks/04_paper_figures.ipynb`.

### Step 6.4: Internal review
- Check all numbers match actual experiment results
- Verify no data leakage in benchmark design
- Ensure reproducibility: all code, data, and configs are documented

**Log to worklog:** Draft completion dates per section, figure/table generation, review feedback.

---

## Completion Criteria
- [ ] Complete LaTeX draft in `papers/paper_a_benchmark/`
- [ ] All figures and tables generated from actual results
- [ ] BibTeX references complete (30+ papers)
- [ ] Draft reviewed for consistency and clarity
- [ ] Code and data release plan documented
- [ ] All activities logged in `worklog.md`
