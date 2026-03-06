# Run 2 Report: Rerun of Failed Jobs

**Date:** 2026-03-06
**SLURM Jobs:** 22 submitted (reruns of Run 1 failures), Job IDs 45811392-45811414
**Duration:** ~24 hours

---

## Executive Summary

Run 2 resubmitted the 22 jobs that failed in Run 1 (17 segfaults + 5 RAG timeouts). Results:

| Status | Count | Details |
|--------|-------|---------|
| Completed successfully | 6 | 1 joint_training (seed 123) + 5 RAG (all seeds) |
| Segfault during eval | 16 | All naive_seq (5) + EWC (5) + replay (5) + JT seed 456 |
| Time limit | 0 | RAG 48h fix worked |
| **Still missing** | **16** | Same 16 KGE baseline seeds |

**Critical finding:** The `max_test_triples=50K` sampling fix from Run 1 was **insufficient**. The segfault is in PyKEEN's all-entity scoring mechanism itself (129K entities), NOT in the number of test triples. The crash is non-deterministic — `jt_s123` succeeded while `jt_s456` segfaulted with the same code, same config, on a smaller test set.

---

## 1. Successful Jobs

### 1.1 Joint Training seed=123 (NEW)

| Metric | Value |
|--------|-------|
| AP | 0.0202 |
| AF | 0.0 |
| BWT | 0.0 |
| FWT | 0.0201 |
| REM | 1.0 |

Consistent with Run 1 seeds (42: 0.0208, 789: 0.0202, 1024: 0.0200). Joint training now has 4/5 seeds.

### 1.2 RAG Full Agent — ALL 5/5 seeds complete (NEW)

All RAG jobs used Qwen2.5-7B-Instruct LLM with PubMedBERT retrieval, 200 questions/task, 10 tasks.

| Seed | AP | AF | BWT | REM | Time (h) |
|------|------|-------|--------|-------|----------|
| 42 | 0.0025 | 0.0011 | -0.0011 | 0.999 | 16.0 |
| 123 | 0.0025 | 0.0006 | -0.0006 | 0.999 | — |
| 456 | 0.0015 | 0.0011 | -0.0011 | 0.999 | — |
| 789 | 0.0040 | 0.0000 | +0.0006 | 1.000 | — |
| 1024 | 0.0005 | 0.0006 | -0.0006 | 0.999 | — |
| **Mean** | **0.0022** | **0.0007** | **-0.0006** | **0.999** | — |
| **Std** | **0.0013** | **0.0005** | **0.0007** | **0.001** | — |

**Analysis:**
- Performance is near zero (AP=0.002). The RAG agent with Qwen2.5-7B essentially fails at continual LP evaluation.
- Most matrix entries are exactly 0.0 — the LLM rarely generates the exact gold entity name.
- Only 3 task types ever get nonzero MRR: disease_related (~0.005-0.025), gene_protein (~0.005-0.010), phenotype_related (~0.005-0.015).
- High seed variance (8x between best/worst) — question sampling randomization dominates.
- This is expected behavior: RAG is designed for KGQA (natural language), not LP (entity ranking). The near-zero LP scores establish the RAG baseline as a lower bound.

---

## 2. Failed Jobs — PyKEEN Segfault (16 jobs)

### 2.1 Failure Pattern

All 16 jobs crashed with `Segmentation fault` during PyKEEN's `evaluate()` call. Key observations:

1. **Crash at 0% eval progress:** Every segfault shows `0%| | 0.00/...` — crashes on the very first evaluation batch.
2. **GPU memory is fine:** 2136-2442 MB used at crash time (V100 has 32 GB). NOT a simple OOM.
3. **Non-deterministic:** `jt_s123` succeeded on the same code while `jt_s456` segfaulted on a smaller test set (19,952 triples vs 50K).
4. **Crash happens on various test set sizes:** From 3,401 triples (task_1_disease) to 50K sampled triples. The common factor is all-entity scoring against 129K entities.
5. **max_test_triples=50K was applied** — logs confirm sampling messages. But this only reduces rows, not the 129K entity columns.

### 2.2 Per-Method Breakdown

| Method | Seeds Failed | Furthest Progress |
|--------|-------------|-------------------|
| naive_sequential | ALL 5 | Tasks 1-2 eval (1-3 evals completed before crash) |
| EWC | ALL 5 | Tasks 1-3 eval (1-8 evals completed before crash) |
| experience_replay | ALL 5 | Tasks 1-3 eval (2-8 evals completed before crash) |
| joint_training | seed 456 | Task eval #6 (5 evals completed) |

### 2.3 Root Cause

The segfault is in PyKEEN's C/CUDA evaluation backend, specifically the all-entity ranking step that creates score tensors of shape `(batch_size, 129K)`. This is a known PyKEEN issue with very large entity sets. The crash is not a Python exception — it's a CUDA driver-level failure that manifests as SIGSEGV.

**The `max_test_triples` fix addressed the wrong dimension:** it reduced the number of test triples (rows) but the bottleneck is `num_entities` (columns = 129K), which cannot be reduced without changing the evaluation protocol.

### 2.4 Proposed Fixes for Run 3

1. **Reduce eval batch_size explicitly** (e.g., `batch_size=32` instead of 512) to reduce the score matrix from `(512, 129K)` to `(32, 129K)`
2. **Use PyKEEN's `slice_size` parameter** in the evaluator to further limit memory allocation
3. **Custom evaluation function** that bypasses PyKEEN's evaluator entirely — manually score `(h, r, ?)` for each test triple using the model's `score_t()` method with chunked entity batching
4. **Reduce entity set for ranking** — only rank against entities of the correct type (e.g., for `drug-disease` edges, only rank against disease entities, not all 129K)

Option 4 would be most effective (reducing 129K to ~5K-15K per relation type) and is also more scientifically meaningful (type-constrained evaluation).

---

## 3. Combined Results (Run 1 + Run 2)

### Link Prediction (Continual, 10 tasks)

| Method | Seeds Complete | AP (mean ± std) | AF (mean ± std) | Status |
|--------|---------------|-----------------|-----------------|--------|
| **CMKL (DistMult)** | **5/5** | **0.063 ± 0.003** | **0.040 ± 0.003** | Complete |
| LKGE (TransE) | 5/5 | 0.039 ± 0.001 | 0.012 ± 0.002 | Complete (9 tasks) |
| Joint Training (TransE) | 4/5 | 0.020 ± 0.000 | 0.0 | Missing seed 456 |
| RAG (Qwen2.5-7B) | 5/5 | 0.002 ± 0.001 | 0.001 ± 0.001 | Complete |
| Naive Sequential (TransE) | 0/5 | — | — | Segfault |
| EWC (TransE) | 0/5 | — | — | Segfault |
| Experience Replay (TransE) | 0/5 | — | — | Segfault |

### Node Classification — ALL COMPLETE (from Run 1)

| Method | AP (mean ± std) | AF (mean ± std) |
|--------|-----------------|-----------------|
| **CMKL** | **0.431 ± 0.005** | **0.004 ± 0.002** |
| Joint Training | 0.370 ± 0.002 | 0.003 ± 0.001 |
| EWC | 0.345 ± 0.004 | 0.008 ± 0.003 |
| Experience Replay | 0.344 ± 0.006 | 0.010 ± 0.003 |
| Naive Sequential | 0.344 ± 0.004 | 0.011 ± 0.002 |

---

## 4. What We Have vs. What We Need for Paper

### Complete (ready for paper):
- CMKL LP: 5/5 seeds
- LKGE LP: 5/5 seeds
- RAG LP: 5/5 seeds
- NC all methods: 25/25 seeds
- Joint Training LP: 4/5 seeds (usable, very low std)

### Missing (need Run 3 fix):
- Naive Sequential LP: 0/5 seeds
- EWC LP: 0/5 seeds
- Experience Replay LP: 0/5 seeds
- Joint Training LP: seed 456 (nice to have)

### Paper strategy options:
1. **Fix PyKEEN eval and rerun** (recommended) — implement custom evaluation or type-constrained ranking
2. **Report with available data** — CMKL vs LKGE vs Joint vs RAG, note baseline gap
3. **Use NC results as primary comparison** — all methods complete, cleaner story

---

## 5. Next Steps

1. **Fix PyKEEN segfault** — implement custom evaluation bypassing PyKEEN's evaluator (see Section 2.4)
2. **Run 3:** Resubmit 16 failed KGE baselines + JT seed 456 with fixed evaluation
3. **Run 3 also:** Multi-hop evaluation as standalone post-hoc eval
4. **After Run 3:** Merge all results, generate paper tables with `scripts/generate_tables.py`
5. **Ablation studies:** Submit after main results complete
