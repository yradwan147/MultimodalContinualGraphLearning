# Phase 7: Paper B — Method Paper (ICML / NeurIPS / KDD)
**Estimated time:** Weeks 26-30 | **PDF Section:** 9.2
**Depends on:** Phase 5 (experiments), Phase 6 (parallel) | **Blocks:** None

**Writing methodology:** Follows `~/.claude/skills/write-research-paper/SKILL.md` exactly.
**Style file:** `neurips_2025.sty` from `~/.claude/skills/write-research-paper/references/ReefNet_NeurIPS_2025/`
**Reference papers:** ReefNet, FishNet in `~/.claude/skills/write-research-paper/references/`

---

## PHASE 0: Context Validation (Skip — already confirmed)

Project has code (`src/`), results (`results/*.json`), configs, docs, figures. Passes gate.

---

## PHASE 1: Deep Project Scan (Adapted — we know the codebase)

### 1.0 Recency & Relevance Triage
**Active files for Paper B (method paper — CMKL):**
- `src/models/cmkl.py` — Full CMKL assembly (~850 lines)
- `src/models/encoders.py` — StructuralEncoder (R-GCN), TextualEncoder (BiomedBERT), MolecularEncoder
- `src/models/fusion.py` — CrossModalAttentionFusion, ConcatenationFusion
- `src/models/decoders.py` — TransE, DistMult, Bilinear decoders
- `src/continual/modality_ewc.py` — Modality-Aware EWC (~250 lines)
- `src/continual/multimodal_replay.py` — Multimodal Memory Buffer (K-means)
- `src/continual/distillation.py` — Knowledge Distillation (optional)
- `scripts/run_cmkl.py` — CMKL experiment runner
- `scripts/run_ablations.py` — 8 ablation studies
- `src/baselines/` — All 6 baselines for comparison
- `src/evaluation/` — All metrics and statistical tests
- `configs/cmkl.yaml` — CMKL hyperparameters

**Legacy/excluded:** None — all code is active.

### 1.1 Methodology Extraction (for Section 4: Method)
- Read `src/models/cmkl.py` — Full CMKL training pipeline
- Read `src/models/encoders.py` — 3 modality encoders
- Read `src/models/fusion.py` — Cross-modal attention mechanism
- Read `src/continual/modality_ewc.py` — Per-modality Fisher computation
- Read `src/continual/multimodal_replay.py` — K-means diverse selection
- Read `src/continual/distillation.py` — Knowledge distillation

### 1.2 Results Collection
- Read `results/cmkl_DistMult.json` — Main CMKL results
- Read `results/ablation_*.json` — All 8 ablation results
- Read `results/*.json` — All baseline results for comparison
- From worklog: smoke test AP=0.1240, AF=0.0005, REM=0.9995

### 1.3 Experimental Setup
- Framework: PyTorch + PyG + PyKEEN
- Hardware: IBEX V100 32GB
- 5 seeds: [42, 123, 456, 789, 1024]
- Hyperparameters from `configs/cmkl.yaml`

### 1.4 Figures Needed
- `fig:cmkl_architecture` — CMKL framework diagram (encoders → fusion → decoder + EWC + replay)
- `fig:modality_fisher` — Per-modality Fisher information across tasks
- `fig:ablation_results` — Ablation bar charts
- `fig:forgetting_comparison` — CMKL vs baselines forgetting curves
- `fig:attention_weights` — Cross-modal attention visualization (if available)

---

## PHASE 2: Paper Planning — Section Mapping

### Core contribution
CMKL: first method addressing modality-specific forgetting in continual KG learning via modality-aware EWC and multimodal memory replay.

### Problem statement
Different modalities (structure, text, molecular) experience different distribution shifts across tasks in evolving biomedical KGs. Existing CKGE methods apply uniform regularization, ignoring modality-specific forgetting patterns.

### Prior work contrast
- Standard EWC: uniform lambda across all parameters
- LKGE: structure-only, no multimodal
- BER/Replay: random selection, no diversity
- MSCGL: multimodal but not continual

### Key findings
- CMKL achieves highest AP with lowest AF across all baselines
- Modality-aware EWC > global EWC (ablation proves per-modality lambdas help)
- Cross-attention fusion > concatenation
- K-means diverse replay > random replay
- Distillation provides additional forgetting reduction

### Section mapping
| Section | Source files |
|---------|-------------|
| Abstract | Aggregate from all sections |
| Introduction | Problem motivation, CMKL contribution |
| Related Work | CKGE methods, multimodal graph learning, biomedical KG reasoning |
| Problem Formulation | Formal CL-MKGE definition |
| Method | `src/models/`, `src/continual/` |
| Experiments | `results/*.json`, baselines, ablations |
| Analysis | Per-modality analysis, case studies |
| Conclusion | Summary + limitations |

---

## PHASE 3: Output Directory Structure

```
papers/paper_b_method/
├── main.tex                # Main document — \input{} all sections
├── neurips_2025.sty        # Copy from references/ReefNet_NeurIPS_2025/
├── sec/
│   ├── 0_abstract.tex      # 150-250 words, 6-part structure
│   ├── 1_intro.tex         # 5-paragraph structure with wrapfigure teaser
│   ├── 2_related.tex       # \paragraph{} topics, 20+ citations
│   ├── 3_method.tex        # CMKL methodology (4 subsections + algorithm)
│   ├── 4_experiments.tex   # Main results + ablations + analysis
│   ├── 5_conclusion.tex    # ~200 words + data availability
│   └── supplementary.tex   # S-numbered sections
├── tables/
│   ├── comparison.tex      # Table 1: CMKL vs prior CKGE methods (\cmark/\xmark)
│   ├── main_results.tex    # CMKL vs all baselines on LP
│   ├── ablation.tex        # All 8 ablation results
│   ├── lambda_analysis.tex # Per-modality lambda sensitivity
│   └── buffer_sweep.tex    # Buffer size sensitivity
├── figs/                   # Architecture diagrams, result plots
└── refs.bib                # BibTeX, \bibliographystyle{unsrt}
```

---

## PHASE 4: Section-by-Section Writing (following SKILL.md spec exactly)

### 4.1 Abstract (`sec/0_abstract.tex`)
Follow the **mandatory 6-part structure**:
1. Problem context (evolving biomedical KGs, modality-specific forgetting)
2. Introduce contribution ("We introduce **CMKL**,...")
3. Technical details (R-GCN + BiomedBERT + Morgan FP, cross-modal attention, per-modality Fisher)
4. Unique properties (first to address modality-specific forgetting)
5. Key results (AP improvement over baselines, AF reduction)
6. Release statement

### 4.2 Introduction (`sec/1_intro.tex`)
Follow the **5-paragraph structure**:
1. Broad motivation: biomedical drug repurposing, evolving KGs, catastrophic forgetting
2. Existing approaches and gaps: LKGE is structure-only, EWC uses uniform regularization
3. Our contribution with wrapfigure (CMKL architecture overview) + bold numbered pattern:
   - **(1) Modality-Aware EWC:** per-modality Fisher with different lambdas
   - **(2) Multimodal Memory Replay:** K-means diverse selection across modalities
   - **(3) Cross-Modal Attention Fusion:** bidirectional attention between modalities
4. Key findings from experiments
5. Contribution list (inline numbered, 4 items)

Include **Comparison Table 1** (CMKL vs prior CKGE methods features).

### 4.3 Related Work (`sec/2_related.tex`)
Use `\paragraph{}` topics:
- `\paragraph{Continual Knowledge Graph Embedding.}` — EWC for KG, LKGE, BER, STCKGE, IncDE
- `\paragraph{Multimodal Graph Learning.}` — MSCGL, OMG-NAS, MoDE, multimodal KGE
- `\paragraph{Biomedical KG Reasoning.}` — TxGNN, ProCyon, AMG-RAG
- `\paragraph{Elastic Weight Consolidation.}` — Original EWC, online EWC, progress & compress
- `\paragraph{Experience Replay for CL.}` — GEM, A-GEM, ER-GNN, BER
- Final positioning paragraph

### 4.4 Problem Formulation (in `sec/3_method.tex`, opening)
```
\section{Methodology}
\label{sec:method}
```

**Opening:** Formal definition of Continual Multimodal KG Learning:
- G_t = (V_t, E_t, X_t) where X_t = {X_struct, X_text, X_mol}
- Task sequence T = {T_1, T_2, ..., T_n}
- Objective: minimize ∑_i L(f_θ, T_i) + λ·Ω(θ, θ*)

### 4.5 Method Subsections (`sec/3_method.tex`)
- `\subsection{Architecture Overview}` — Full-width architecture figure
- `\subsection{Modality-Specific Encoders}`
  - `\noindent\textbf{Structural Encoder.}` — R-GCN with basis decomposition
  - `\noindent\textbf{Textual Encoder.}` — Frozen BiomedBERT + linear projection
  - `\noindent\textbf{Molecular Encoder.}` — Morgan fingerprint MLP
- `\subsection{Cross-Modal Attention Fusion}` — Bidirectional cross-attention + fusion MLP
- `\subsection{Modality-Aware EWC}` — Per-modality Fisher, per-modality lambdas, EWC penalty formula
- `\subsection{Multimodal Memory Replay}` — K-means diverse selection, multimodal embedding storage
- `\subsection{Training Procedure}` — **Algorithm box** with full pseudocode

**Algorithm pseudocode:**
```latex
\begin{algorithm}[t]
\caption{CMKL Training Procedure}
\begin{algorithmic}[1]
\REQUIRE Task sequence T = {T_1, ..., T_n}, model f_θ
\FOR{each task T_i}
  \STATE Encode: h_s = R-GCN(V, E), h_t = BiomedBERT(X_text), h_m = MLP(X_mol)
  \STATE Fuse: h = CrossAttn(h_s, h_t, h_m)
  \FOR{each epoch}
    \STATE L = L_task + λ·L_EWC + α·L_replay
    \STATE Update θ via gradient descent
  \ENDFOR
  \STATE Compute per-modality Fisher: F_s, F_t, F_m
  \STATE Add diverse exemplars to buffer via K-means
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

### 4.6 Experiments (`sec/4_experiments.tex`)
- `\subsection{Experimental Setup}` — Benchmark, baselines, metrics, hardware
- `\subsection{Main Results}` — CMKL vs 6 baselines, main results table
  - Wraptable with MRR, Hits@1/3/10, AP, AF, BWT, REM
  - Discussion: best model, improvement margins, what fails
- `\subsection{Ablation Studies}` — 8 ablations table
  - struct_only, text_only, concat_fusion, global_ewc, random_replay, distillation
  - Analysis of each ablation's impact
- `\subsection{Sensitivity Analysis}`
  - Buffer size sweep figure
  - Lambda sweep figure (per-modality)
- `\subsection{Analysis}`
  - Per-modality forgetting visualization
  - Which modality benefits most from protection?
  - When does multimodal help vs hurt?

### 4.7 Conclusion (`sec/5_conclusion.tex`)
Follow spec: Restate → Findings → Limitations → Release/impact + Data Availability Statement

### 4.8 Supplementary (`sec/supplementary.tex`)
- S1: Experimental Details (all hyperparameters, hardware)
- S2: Additional Results (per-seed results, per-relation breakdown)
- S3: Per-Modality Fisher Visualization
- S4: Limitations (scalability, PrimeKG-specific)

---

## PHASE 5: Post-Generation

### 5.1 LaTeX Compilation (MANDATORY)
```bash
cd papers/paper_b_method && \
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

## Custom Commands for Paper B

```latex
\newcommand{\ours}{CMKL\xspace}
\newcommand{\fullname}{Continual Multimodal Knowledge Graph Learner\xspace}
\newcommand{\nummodalities}{3\xspace}
\newcommand{\numentities}{129,312\xspace}
\newcommand{\numedges}{8,100,498\xspace}
\newcommand{\numablations}{8\xspace}
```

---

## Key References to Include (minimum 20+)

### Continual Learning (foundations)
- Kirkpatrick et al. 2017 — EWC
- Rolnick et al. 2019 — Experience Replay
- Zenke et al. 2017 — Synaptic Intelligence
- Li & Hoiem 2017 — LwF (Learning without Forgetting)
- Hinton et al. 2015 — Knowledge Distillation

### Continual Graph/KG Learning
- Cui et al. 2023 — LKGE
- Daruna et al. 2021 — Continual Graph Learning
- Wu et al. 2024 — PS-CKGE
- Song et al. 2023 — IncDE

### KG Embedding
- Bordes et al. 2013 — TransE
- Yang et al. 2015 — DistMult
- Sun et al. 2019 — RotatE
- Schlichtkrull et al. 2018 — R-GCN
- Ali et al. 2021 — PyKEEN

### Multimodal Graph Learning
- Liu et al. 2023 — MSCGL
- Zhang et al. 2024 — Multimodal KGE survey

### Biomedical
- Chandak et al. 2023 — PrimeKG
- Huang et al. 2024 — TxGNN
- Gu et al. 2021 — BiomedBERT (PubMedBERT)
- Rogers et al. 2016 — Morgan Fingerprints (extended connectivity)

### NLP/LLM for KG
- Devlin et al. 2019 — BERT
- Lewis et al. 2020 — RAG

---

## Completion Criteria
- [ ] Complete LaTeX draft following SKILL.md spec
- [ ] Architecture diagram (CMKL framework)
- [ ] Algorithm pseudocode box
- [ ] Comparison Table 1 with \cmark/\xmark
- [ ] Main results table with bold best
- [ ] Ablation table
- [ ] Sensitivity analysis figures
- [ ] 20+ verified BibTeX entries
- [ ] NeurIPS checklist completed
- [ ] Paper compiles to clean PDF
- [ ] All activities logged in worklog.md
