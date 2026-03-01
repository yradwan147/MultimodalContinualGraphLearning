# Phase 5: Experiments and Ablation Studies
**Estimated time:** Weeks 18-22 | **PDF Sections:** 7, 8
**Depends on:** Phase 3 (baselines), Phase 4 (CMKL) | **Blocks:** Phase 6, 7

---

## Objectives
- Run the full experiment matrix (8 experiments)
- Run all ablation studies (7 ablations)
- Compute statistical significance across 5 random seeds
- Generate all results tables and figures

---

## Step 5.1: Full Experiment Matrix

| Exp | Method | Task 1 (LP) | Task 2 (KGQA) | Task 3 (NC) |
|-----|--------|-------------|---------------|-------------|
| E1 | Naive Sequential | Y | | Y |
| E2 | Joint Training (upper bound) | Y | | Y |
| E3 | EWC | Y | | Y |
| E4 | Experience Replay (BER) | Y | | Y |
| E5 | LKGE | Y | | |
| E6 | RAG Agent | | Y | |
| E7 | CMKL (proposed) | Y | | Y |
| E8 | CMKL + RAG | Y | Y | Y |

### For each experiment:
1. Run with 5 random seeds: [42, 123, 456, 789, 1024]
2. Compute results matrix R[i][j] per seed
3. Compute CL metrics: AP, AF, BWT, FWT, REM per seed
4. Report mean +/- std

### Script: `scripts/run_baselines.py` and `scripts/run_cmkl.py`
- Use config files from `configs/` for hyperparameters
- Log all results to wandb and to `results/` directory

**Log to worklog:** For each experiment - per-seed results, mean +/- std for all metrics, training time, GPU memory.

---

## Step 5.2: Ablation Studies

**Script:** `scripts/run_ablations.py`

### Ablation 1: Single modality (struct only)
- Remove text and molecular encoders from CMKL
- Tests: **value of multimodal features** for continual learning
- Implementation: Set `use_text=False, use_mol=False` in config

### Ablation 2: Single modality (text only)
- Remove GNN; use text embeddings only
- Tests: **structural vs. textual contribution**
- Implementation: Set `use_struct=False, use_mol=False`

### Ablation 3: Concatenation fusion
- Replace CrossModalAttentionFusion with `torch.cat() + MLP`
- Tests: **cross-attention vs. simple fusion**
- Implementation: Swap fusion module

### Ablation 4: Global EWC
- Use single lambda for all parameters (standard EWC, not modality-aware)
- Tests: **value of modality-specific regularization**
- Implementation: Set all lambdas equal

### Ablation 5: Random replay
- Replace K-means diverse selection with uniform random buffer
- Tests: **diverse vs. random buffer selection**
- Implementation: Set `selection_strategy='random'`

### Ablation 6: Buffer size sweep
- Sweep buffer sizes: [100, 250, 500, 1000, 2000, 5000]
- Tests: **memory budget sensitivity**

### Ablation 7: Lambda sweep
- Sweep per-modality lambdas independently
- Tests: **regularization strength per modality**
- Grid: lambda_struct x lambda_text x lambda_mol

All ablations run with 5 seeds, report mean +/- std.

**Log to worklog:** For each ablation - configuration, results, key finding.

---

## Step 5.3: Statistical Significance Tests

Use paired t-tests or Wilcoxon signed-rank tests at p < 0.05.

Key comparisons:
- CMKL vs. Naive Sequential (should be significant)
- CMKL vs. EWC (tests value of multimodal approach)
- CMKL vs. Experience Replay (tests value of modality-aware EWC)
- CMKL full vs. each ablation variant
- RAG vs. CMKL on KGQA task

**Log to worklog:** All pairwise p-values, which comparisons are significant.

---

## Step 5.4: Expected Results Analysis

### Expected: CMKL outperforms naive sequential and standard EWC

If results match expectations:
- Report the performance gains and analyze which component contributes most
- Use ablation results to decompose the contribution

### If CMKL performs worse than EWC:
1. Check cross-modal attention for overfitting - add dropout, reduce complexity
2. Verify multimodal features are correctly aligned - check embedding distributions
3. Try simpler late-fusion (concatenation + MLP) as intermediate step

### If forgetting is minimal even for naive baseline:
1. Temporal splits may not have enough distribution shift - verify emerged entities introduce new patterns
2. KGE model may be undertrained, masking forgetting - increase epochs
3. Ensure test sets cover entities specific to each task

### If RAG outperforms all parametric methods on KGQA:
1. This validates the RAG direction suggested by Prof. Zhang
2. Analyze WHERE RAG fails: likely multi-hop reasoning requiring structural understanding
3. Design hybrid: parametric GNN for structural reasoning + RAG for knowledge retrieval

### If Forward Transfer (FWT) is negative:
1. Check task ordering effects - try different orderings
2. Implement curriculum strategies: order tasks easy to hard

**Log to worklog:** Detailed analysis of results, unexpected findings, follow-up actions taken.

---

## Step 5.5: Generate Results Tables and Figures

**Script:** `scripts/generate_tables.py`

### Tables to generate:
1. **Main results table**: All methods x all metrics (AP, AF, BWT, FWT, MRR, Hits@1/3/10)
2. **Ablation table**: CMKL variants x all metrics
3. **Dataset statistics table**: Benchmark characteristics

### Figures to generate:
1. Results matrix heatmaps (one per method)
2. Forgetting curves over task sequence
3. Per-modality forgetting comparison
4. Buffer size sensitivity plot
5. Lambda sweep heatmap
6. Temporal KG evolution visualization (nodes/edges over snapshots)

Save all to `results/figures/` and `results/tables/`.

**Log to worklog:** List of generated figures and tables with file paths.

---

## Completion Criteria
- [ ] All 8 experiments completed with 5 seeds each
- [ ] All 7 ablation studies completed with 5 seeds each
- [ ] Statistical significance computed for all key comparisons
- [ ] Results tables generated in LaTeX format
- [ ] Results figures generated as PDF/PNG
- [ ] Analysis document written in `docs/experiment-results.md`
- [ ] Any unexpected results investigated and documented
- [ ] All activities logged in `worklog.md`
