# Phase 7: Paper B - Method Paper
**Estimated time:** Weeks 26-30 | **PDF Section:** 9.2
**Depends on:** Phase 5 (experiments), Phase 6 (can run in parallel) | **Blocks:** None

---

## Objectives
- Write Paper B: A novel continual learning technique (CMKL) for multimodal biomedical KGs
- Target venues: ICML, NeurIPS, KDD, WWW

---

## Paper Structure

### 1. Introduction (1 page)
- Core challenge: continual multimodal biomedical KG reasoning under evolving knowledge
- Problem: existing CKGE methods ignore modality-specific forgetting patterns
- Key insight: different modalities (structure, text, molecular) experience different distribution shifts across tasks
- Contribution: CMKL with modality-aware EWC and multimodal memory replay

### 2. Related Work (1 page)
- CKGE methods: EWC for KG, LKGE, BER, STCKGE, IncDE
- Multimodal graph learning: MSCGL, OMG-NAS, MoDE
- Biomedical KG reasoning: TxGNN, ProCyon, AMG-RAG

### 3. Problem Formulation (0.5 page)
- Formal definition of continual multimodal KG learning
- Notation: G_t = (V_t, E_t, X_t) where X_t = {X_struct, X_text, X_mol}
- Objective: minimize forgetting while maximizing performance on evolving G_t

### 4. Method (2.5 pages)
- **4.1 Architecture Overview**: CMKL framework diagram
- **4.2 Modality-Specific Encoders**: R-GCN, BiomedBERT, Morgan fingerprint encoder
- **4.3 Cross-Modal Attention Fusion**: Why cross-attention > concatenation
- **4.4 Modality-Aware EWC**: Per-modality Fisher matrices, per-modality lambdas
- **4.5 Multimodal Memory Replay**: Diverse selection via K-means, storing multimodal embeddings
- **4.6 Training Procedure**: Full algorithm pseudocode

### 5. Experiments (2 pages)
- **5.1 Setup**: Benchmark description, baselines, metrics
- **5.2 Main Results**: CMKL vs. all baselines on LP and NC tasks
- **5.3 Ablation Studies**: All 7 ablations with analysis
- **5.4 Analysis**: Inter-modal vs. intra-modal forgetting, modality complementarity

### 6. Analysis (1 page)
- Which modality benefits most from protection?
- When does multimodal information help vs. hurt?
- Visualization of per-modality Fisher information across tasks
- Case studies: specific drug-disease predictions preserved/forgotten

### 7. Conclusion (0.5 page)
- Summary of contributions and findings
- Limitations: scalability, limited to PrimeKG entity types
- Future work: extension to more modalities, application to other biomedical tasks

---

## Key Selling Points
- **First method** to address modality-specific forgetting in continual KG learning
- Novel modality-aware regularization and multimodal replay strategies
- Practical application to biomedical drug repurposing under evolving knowledge
- Comprehensive evaluation on real temporal biomedical KG benchmark

---

## Writing Process

### Step 7.1: Set up LaTeX template
```
papers/paper_b_method/
├── main.tex          # Venue-appropriate template
├── references.bib
├── figures/
│   ├── cmkl_architecture.pdf     # CMKL framework diagram
│   ├── modality_fisher.pdf       # Per-modality Fisher visualization
│   ├── ablation_results.pdf      # Ablation bar charts
│   └── case_studies.pdf          # Specific prediction examples
└── tables/
    ├── main_results.tex          # CMKL vs. baselines
    ├── ablation_table.tex        # All ablation results
    └── hyperparameter_table.tex  # Per-modality lambda analysis
```

### Step 7.2: Write draft sections
Focus on clarity of the method section - reviewers need to understand the modality-aware EWC and multimodal replay clearly.

### Step 7.3: Algorithm pseudocode
Include a clear Algorithm box showing the full CMKL training procedure.

### Step 7.4: Internal review
- Verify all claims are supported by experimental evidence
- Ensure ablations clearly demonstrate the value of each component
- Check that related work properly differentiates from MSCGL and standard CKGE

**Log to worklog:** Draft completion dates per section, figure/table generation, review feedback.

---

## Completion Criteria
- [ ] Complete LaTeX draft in `papers/paper_b_method/`
- [ ] All figures and tables generated from actual results
- [ ] Algorithm pseudocode is clear and complete
- [ ] Ablation studies clearly support design decisions
- [ ] BibTeX references complete (30+ papers)
- [ ] Draft reviewed for consistency and clarity
- [ ] All activities logged in `worklog.md`
