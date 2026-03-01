# Phase 6: Paper A — Benchmark Paper (NeurIPS Datasets & Benchmarks)
**Estimated time:** Weeks 22-26 | **PDF Section:** 9.1
**Depends on:** Phase 5 (all experiments done) | **Blocks:** None (parallel with Phase 7)

**Writing methodology:** Follows `~/.claude/skills/write-research-paper/SKILL.md` exactly.
**Style file:** `neurips_2025.sty` from `~/.claude/skills/write-research-paper/references/ReefNet_NeurIPS_2025/`
**Reference papers:** ReefNet (NeurIPS 2025 D&B), FishNet (NeurIPS 2025 D&B) in `~/.claude/skills/write-research-paper/references/`

---

## PHASE 0: Context Validation (Skip — already confirmed)

Project has code (`src/`), results (`results/*.json`), configs (`configs/`), documentation (`docs/`, `CLAUDE.md`, `worklog.md`), and figures (`results/*.png`). Passes the 2-of-4 gate.

---

## PHASE 1: Deep Project Scan (Adapted — skip repo exploration since we know the codebase)

### 1.0 Recency & Relevance Triage
**Active files for Paper A (benchmark paper):**
- `src/data/download.py` — PrimeKG t0 download
- `src/data/kg_builder.py` — Real t1 construction (~1200 lines)
- `scripts/build_real_t1.py` — t1 build CLI
- `scripts/build_benchmark.py` — End-to-end benchmark pipeline
- `src/data/temporal_diff.py` — Temporal diff computation
- `src/data/task_sequence.py` — CL task sequence creation
- `src/data/splits.py` — Train/val/test splits with leakage prevention
- `src/data/features.py` — Multimodal feature extraction
- `src/data/kgqa.py` — KGQA dataset generation
- `src/data/node_classification.py` — NC dataset construction
- `src/baselines/` — All 6 baselines (naive_seq, joint, ewc, replay, lkge, rag)
- `src/evaluation/metrics.py` — CL metrics (AP, AF, BWT, FWT, REM)
- `src/evaluation/statistical.py` — Statistical tests
- `configs/t1_sources.yaml` — Database configuration

**Legacy/excluded:** None — all code is active for this paper.

### 1.1 Methodology Extraction (for Section 3: Benchmark Construction)
- Read `src/data/kg_builder.py` for t1 build pipeline
- Read `scripts/build_benchmark.py` for full pipeline
- Read `src/data/temporal_diff.py` for diff computation
- Read `src/data/task_sequence.py` for CL task design

### 1.2 Results Collection
- Read `results/*.json` — all baseline results from IBEX
- Read `worklog.md` — benchmark statistics (t0: 8.1M edges, t1: 13M edges, diff stats)
- Read `data/benchmark/statistics.json` — benchmark stats

### 1.3 Dataset Understanding
- t0: PrimeKG June 2021 (Harvard Dataverse/TDC) — 129K nodes, 8.1M edges, 10 node types, 30 relation types
- t1: Rebuilt July 2023 from 9 free databases — 134K nodes, 13M edges, 25 relation types
- Temporal diff: +5.7M added, -889K removed, 7.2M persistent
- 6 CL tasks via entity_type strategy
- 3 evaluation tasks: Link Prediction, KGQA, Node Classification

### 1.4 Figures Needed
- `fig:benchmark_overview` — Pipeline diagram (download → diff → tasks → splits → features)
- `fig:temporal_evolution` — t0 vs t1 statistics comparison
- `fig:task_distribution` — Task size distribution bar chart
- `fig:results_heatmap` — Results matrix heatmaps for baselines
- `fig:forgetting_curves` — Forgetting over task sequence

---

## PHASE 2: Paper Planning — Section Mapping

### Core contribution
First continual learning benchmark on a real biomedical KG (PrimeKG) with genuine temporal evolution (not synthetic splits) and multimodal features.

### Problem statement
Existing CGL benchmarks use artificial temporal splits on generic KGs (FB15k-237, WN18RR). No benchmark captures real knowledge evolution in domain-specific biomedical KGs.

### Prior work contrast
- PS-CKGE, LKGE: synthetic splits on generic KGs
- ICEWS: event-based temporal, not knowledge evolution
- PrimeKG, TarKG: static biomedical KGs, no CL evaluation

### Key results (from IBEX experiments)
- All 6 baselines evaluated on 3 tasks with 5 seeds
- Naive sequential shows significant forgetting
- Joint training as upper bound
- EWC and replay reduce forgetting

### Section mapping
| Section | Source files |
|---------|-------------|
| Abstract | Aggregate from all sections |
| Introduction | `CLAUDE.md`, `docs/architecture.md`, `worklog.md` |
| Related Work | PDF guide sections 3-4, literature |
| Benchmark Construction | `src/data/`, `scripts/build_*.py`, `configs/t1_sources.yaml` |
| Evaluation Protocol | `src/evaluation/metrics.py`, `src/evaluation/statistical.py` |
| Baseline Experiments | `results/*.json`, `src/baselines/` |
| Analysis | Results analysis, forgetting patterns |
| Conclusion | Summary + limitations |

---

## PHASE 3: Output Directory Structure

```
papers/paper_a_benchmark/
├── main.tex                # Main document — \input{} all sections
├── neurips_2025.sty        # Copy from references/ReefNet_NeurIPS_2025/
├── sec/
│   ├── 0_abstract.tex      # 150-250 words, 6-part structure
│   ├── 1_intro.tex         # 5-paragraph structure with wrapfigure teaser
│   ├── 2_related.tex       # \paragraph{} topics, 20+ citations
│   ├── 3_method.tex        # Benchmark construction (dataset paper → \section{\ours})
│   ├── 4_experiments.tex   # Baseline results, per-task breakdown
│   ├── 5_conclusion.tex    # ~200 words + data availability
│   └── supplementary.tex   # S-numbered sections
├── tables/
│   ├── comparison.tex      # Table 1: ours vs prior CGL benchmarks (\cmark/\xmark)
│   ├── dataset_stats.tex   # t0/t1 statistics
│   ├── main_results.tex    # All baselines × all metrics
│   └── per_task.tex        # Per-task breakdown
├── figs/                   # Copy from results/, generate new ones
└── refs.bib                # BibTeX, \bibliographystyle{unsrt}
```

---

## PHASE 4: Section-by-Section Writing (following SKILL.md spec exactly)

### 4.1 Abstract (`sec/0_abstract.tex`)
Follow the **mandatory 6-part structure**:
1. Problem context (biomedical KGs evolve, CL needed)
2. Introduce contribution ("We introduce **PrimeKG-CL**,...")
3. Technical details with bold numbers (**129K** nodes, **8.1M** edges, **3** temporal snapshots)
4. Unique properties (real temporal evolution, multimodal features, 3 evaluation tasks)
5. Key quantitative results (baseline AP, AF numbers)
6. Release statement

### 4.2 Introduction (`sec/1_intro.tex`)
Follow the **5-paragraph structure**:
1. Broad motivation: biomedical KGs, drug repurposing, evolving knowledge (cite PrimeKG, TxGNN, CGL surveys)
2. Existing approaches and gaps: synthetic splits, no domain-specific benchmarks
3. Our contribution with wrapfigure teaser + bold numbered settings pattern
4. Key findings from baseline experiments
5. Contribution list (inline numbered format)

Include **Comparison Table 1** (Table 1: ours vs prior CGL benchmarks) after intro or related work using `\cmark`/`\xmark` columns.

### 4.3 Related Work (`sec/2_related.tex`)
Use `\paragraph{}` topics:
- `\paragraph{Continual Graph Learning.}` — LKGE, EWC for KG, PS-CKGE, BER
- `\paragraph{Biomedical Knowledge Graphs.}` — PrimeKG, TarKG, BioMedKG, Hetionet
- `\paragraph{Temporal Knowledge Graphs.}` — ICEWS, YAGO, distinguish event-based vs evolution
- `\paragraph{CGL Benchmarks.}` — Existing benchmark limitations
- `\paragraph{Multimodal Graph Learning.}` — MSCGL, OMG-NAS, MoDE
- Final paragraph positioning our work

### 4.4 Benchmark Construction (`sec/3_method.tex`)
Title: `\section{PrimeKG-CL Benchmark}`

Subsections:
- `\subsection{Temporal Snapshots}` — t0, t1 construction, database sources
- `\subsection{Temporal Difference Computation}` — Added/removed/persistent analysis
- `\subsection{Continual Learning Task Sequence}` — Entity-type strategy, task definitions
- `\subsection{Multimodal Features}` — Text (BiomedBERT), molecular (Morgan FP), structural (R-GCN)
- `\subsection{Evaluation Tasks}` — LP, KGQA, NC definitions
- `\subsection{Dataset Statistics}` — Comprehensive stats with table

### 4.5 Experiments (`sec/4_experiments.tex`)
- Opening: Overview of 6 baselines, 3 tasks, 5 seeds
- `\subsection{Experimental Setup}` — Hardware, hyperparameters, metrics
- `\subsection{Link Prediction Results}` — Main results table with wraptable
- `\subsection{KGQA Results}` — RAG agent results
- `\subsection{Node Classification Results}` — NC results
- `\subsection{Analysis}` — Forgetting patterns, domain-specific analysis

### 4.6 Conclusion (`sec/5_conclusion.tex`)
Follow spec: Restate → Findings → Limitations → Release/impact + Data Availability Statement

### 4.7 Supplementary (`sec/supplementary.tex`)
- S-numbered sections/figures/tables
- Experimental details (all hyperparameters)
- Additional results (per-relation breakdown, all seeds)
- Limitations discussion

---

## PHASE 5: Post-Generation

### 5.1 LaTeX Compilation (MANDATORY)
```bash
cd papers/paper_a_benchmark && \
/opt/homebrew/bin/pdflatex -interaction=nonstopmode main.tex && \
/opt/homebrew/bin/bibtex main && \
/opt/homebrew/bin/pdflatex -interaction=nonstopmode main.tex && \
/opt/homebrew/bin/pdflatex -interaction=nonstopmode main.tex
```

### 5.2 Verification
- Zero errors in main.log
- All citations resolved
- All cross-references valid
- PDF > 500KB, ~10 pages main + refs + supplementary

### 5.3 NeurIPS Checklist
Generate all 15 questions with `\answerYes{}`/`\answerNo{}`/`\answerNA{}` and justifications.

---

## Custom Commands for Paper A

```latex
\newcommand{\ours}{PrimeKG-CL\xspace}
\newcommand{\totalentities}{129,312\xspace}
\newcommand{\totaledgest}{8,100,498\xspace}
\newcommand{\totalnodetypes}{10\xspace}
\newcommand{\totalreltypes}{30\xspace}
\newcommand{\numtasks}{6\xspace}
\newcommand{\numbaselines}{6\xspace}
```

---

## Key References to Include (minimum 20+)

### Biomedical KGs
- Chandak et al. 2023 — PrimeKG
- Huang et al. 2024 — TxGNN
- Himmelstein et al. 2017 — Hetionet

### Continual Graph Learning
- Daruna et al. 2021 — CGL survey
- Cui et al. 2023 — LKGE
- Kirkpatrick et al. 2017 — EWC
- Rolnick et al. 2019 — Experience Replay

### KG Embedding
- Bordes et al. 2013 — TransE
- Yang et al. 2015 — DistMult
- Sun et al. 2019 — RotatE
- Ali et al. 2021 — PyKEEN

### Temporal KGs
- Garcia-Duran et al. 2018 — ICEWS
- Jin et al. 2020 — RE-NET

### Multimodal
- Liang et al. 2022 — Multimodal learning survey

---

## Completion Criteria
- [ ] Complete LaTeX draft following SKILL.md spec
- [ ] All sections use correct formatting (booktabs, \cmark/\xmark, \cref, etc.)
- [ ] Comparison Table 1 with checkmarks
- [ ] Main results table with bold best
- [ ] 20+ verified BibTeX entries
- [ ] NeurIPS checklist completed
- [ ] Paper compiles to clean PDF
- [ ] All activities logged in worklog.md
