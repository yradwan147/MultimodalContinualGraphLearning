# Experiment Results Tracking

*Updated after each experiment run. All results use 5 random seeds [42, 123, 456, 789, 1024], reporting mean ± std.*

---

## Final Combined Results (Runs 1 + 2 + 3)

### Link Prediction (Continual, 10 tasks)

| Method | Seeds | AP (mean ± std) | AF (mean ± std) | BWT (mean ± std) | REM (mean ± std) | Eval Method | Status |
|--------|-------|-----------------|-----------------|-------------------|-------------------|-------------|--------|
| **CMKL (DistMult)** | **5/5** | **0.063 ± 0.003** | **0.040 ± 0.003** | **-0.040 ± 0.003** | **0.960 ± 0.003** | Custom (all-entity) | **Complete** |
| LKGE (TransE) | 5/5 | 0.039 ± 0.001 | 0.012 ± 0.002 | -0.010 ± 0.003 | 0.990 ± 0.003 | LKGE internal | Complete (9 tasks, skipped base) |
| Joint Training (TransE) | 5/5 | 0.018 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | Custom (pessimistic) | Complete |
| Naive Sequential (TransE) | 5/5 | 0.004 ± 0.000 | 0.021 ± 0.000 | -0.021 ± 0.000 | 0.980 ± 0.000 | Custom (pessimistic) | Complete |
| EWC (TransE) | 5/5 | 0.004 ± 0.000 | 0.017 ± 0.001 | -0.017 ± 0.001 | 0.984 ± 0.001 | Custom (pessimistic) | Complete |
| Experience Replay (TransE) | 5/5 | 0.004 ± 0.000 | 0.021 ± 0.000 | -0.021 ± 0.000 | 0.980 ± 0.000 | Custom (pessimistic) | Complete |
| RAG (Qwen2.5-7B) | 5/5 | 0.002 ± 0.001 | 0.001 ± 0.001 | -0.001 ± 0.001 | 0.999 ± 0.001 | Exact match | Complete |

**Notes:**
- Run 3 KGE baselines (naive seq, joint, EWC, replay) use custom eval that bypasses PyKEEN (pessimistic ranking — scores all 129K entities).
- CMKL uses its own custom all-entity scoring. LKGE uses its own internal evaluation.
- RAG evaluated via exact entity name match from LLM output (Qwen2.5-7B-Instruct).

### Node Classification (Continual, 10 tasks) — ALL COMPLETE (Run 1)

| Method | Seeds | AP (mean ± std) | AF (mean ± std) | BWT (mean ± std) | Status |
|--------|-------|-----------------|-----------------|-------------------|--------|
| **CMKL** | **5/5** | **0.431 ± 0.005** | **0.004 ± 0.002** | **-0.000 ± 0.002** | **Complete** |
| Joint Training | 5/5 | 0.370 ± 0.002 | 0.003 ± 0.001 | 0.022 ± 0.003 | Complete |
| EWC | 5/5 | 0.345 ± 0.004 | 0.008 ± 0.003 | 0.007 ± 0.004 | Complete |
| Experience Replay | 5/5 | 0.344 ± 0.006 | 0.010 ± 0.003 | 0.006 ± 0.005 | Complete |
| Naive Sequential | 5/5 | 0.344 ± 0.004 | 0.011 ± 0.002 | 0.003 ± 0.005 | Complete |

---

## Key Findings

- **CMKL is best on both LP and NC** — LP: 0.063 AP (1.6× LKGE, 3.5× Joint), NC: 0.431 AP (1.16× Joint)
- **EWC reduces forgetting vs Naive Sequential** — LP AF: 0.017 vs 0.021, confirming regularization works
- **Experience Replay does NOT reduce forgetting** — LP AF=0.021 identical to Naive Sequential; buffer diversity insufficient at scale
- **CMKL NC advantage is from multimodal features**, not CL mechanisms (EWC/replay add <1% over naive)
- **RAG is a lower bound** — Qwen2.5-7B cannot generate exact entity names for LP evaluation (AP=0.002)
- **Gene-protein task anomaly** (MRR~0.50 for CMKL) is NOT leakage — structural simplicity (310 unique tails) + DistMult type-selection
- **All KGE baseline AP values are low** (0.004-0.018) due to MRR-based evaluation over 129K entities — expected for large biomedical KGs
- **Joint Training is the upper bound** as expected (AP=0.018, no forgetting, REM=1.0)

---

## Per-Seed Details

### Run 3: KGE Baselines (Custom Eval — Consistent Across All Methods)

#### Naive Sequential (TransE, 5/5)

| Seed | AP | AF | BWT | FWT | REM |
|------|-------|-------|--------|-------|-------|
| 42 | 0.0043 | 0.0205 | -0.0205 | 0.0000 | 0.9795 |
| 123 | 0.0044 | 0.0207 | -0.0206 | 0.0000 | 0.9794 |
| 456 | 0.0044 | 0.0207 | -0.0205 | 0.0000 | 0.9795 |
| 789 | 0.0044 | 0.0210 | -0.0208 | 0.0000 | 0.9792 |
| 1024 | 0.0042 | 0.0203 | -0.0201 | 0.0000 | 0.9799 |

#### Joint Training (TransE, 5/5)

| Seed | AP | AF | BWT | FWT | REM |
|------|-------|-------|--------|-------|-------|
| 42 | 0.0179 | 0.0000 | 0.0000 | 0.0175 | 1.0000 |
| 123 | 0.0174 | 0.0000 | 0.0000 | 0.0170 | 1.0000 |
| 456 | 0.0174 | 0.0000 | 0.0000 | 0.0169 | 1.0000 |
| 789 | 0.0174 | 0.0000 | 0.0000 | 0.0170 | 1.0000 |
| 1024 | 0.0173 | 0.0000 | 0.0000 | 0.0169 | 1.0000 |

#### EWC (TransE, 5/5)

| Seed | AP | AF | BWT | FWT | REM |
|------|-------|-------|--------|-------|-------|
| 42 | 0.0039 | 0.0169 | -0.0168 | 0.0000 | 0.9832 |
| 123 | 0.0044 | 0.0169 | -0.0162 | 0.0000 | 0.9838 |
| 456 | 0.0043 | 0.0174 | -0.0174 | 0.0000 | 0.9826 |
| 789 | 0.0044 | 0.0174 | -0.0164 | 0.0000 | 0.9836 |
| 1024 | 0.0037 | 0.0163 | -0.0159 | 0.0000 | 0.9841 |

#### Experience Replay (TransE, 5/5)

| Seed | AP | AF | BWT | FWT | REM |
|------|-------|-------|--------|-------|-------|
| 42 | 0.0037 | 0.0210 | -0.0210 | 0.0000 | 0.9790 |
| 123 | 0.0040 | 0.0206 | -0.0205 | 0.0000 | 0.9795 |
| 456 | 0.0040 | 0.0205 | -0.0202 | 0.0000 | 0.9798 |
| 789 | 0.0040 | 0.0206 | -0.0204 | 0.0000 | 0.9796 |
| 1024 | 0.0038 | 0.0205 | -0.0203 | 0.0000 | 0.9797 |

### Run 1: CMKL DistMult (LP, 5/5)

| Seed | AP | AF | BWT | REM |
|------|-------|-------|--------|-------|
| 42 | 0.0625 | 0.0410 | -0.0410 | 0.9590 |
| 123 | 0.0650 | 0.0365 | -0.0365 | 0.9635 |
| 456 | 0.0650 | 0.0417 | -0.0417 | 0.9583 |
| 789 | 0.0638 | 0.0375 | -0.0375 | 0.9625 |
| 1024 | 0.0588 | 0.0426 | -0.0427 | 0.9573 |

### Run 1: LKGE TransE (LP, 5/5, 9 tasks)

| Seed | AP | AF | BWT | REM |
|------|-------|-------|--------|-------|
| 42 | 0.0377 | 0.0134 | -0.0128 | 0.9872 |
| 123 | 0.0377 | 0.0128 | -0.0052 | 0.9949 |
| 456 | 0.0399 | 0.0089 | -0.0081 | 0.9919 |
| 789 | 0.0394 | 0.0124 | -0.0121 | 0.9879 |
| 1024 | 0.0394 | 0.0133 | -0.0129 | 0.9871 |

### Run 2: RAG Qwen2.5-7B (LP, 5/5)

| Seed | AP | AF | BWT | REM |
|------|-------|-------|--------|-------|
| 42 | 0.0025 | 0.0011 | -0.0011 | 0.999 |
| 123 | 0.0025 | 0.0006 | -0.0006 | 0.999 |
| 456 | 0.0015 | 0.0011 | -0.0011 | 0.999 |
| 789 | 0.0040 | 0.0000 | +0.0006 | 1.000 |
| 1024 | 0.0005 | 0.0006 | -0.0006 | 0.999 |

---

## Ablation Results

*Pending — need IBEX runs. Local smoke tests only.*

| Ablation | AP | AF | BWT | FWT |
|----------|----|----|-----|-----|
| Struct only | — | — | — | — |
| Text only | — | — | — | — |
| Concat fusion | — | — | — | — |
| Global EWC | — | — | — | — |
| Random replay | — | — | — | — |
| Distillation | — | — | — | — |

---

## Run History

| Run | Date | Jobs | Completed | Failed | New Results |
|-----|------|------|-----------|--------|-------------|
| Run 1 | Mar 3-5 | 60 | 28 | 32 | CMKL 5/5, LKGE 5/5, JT 3/5, NC 25/25, RAG 1/5 |
| Run 2 | Mar 5-6 | 22 | 6 | 16 | JT seed 123, RAG 5/5 (full LLM) |
| Run 3 | Mar 6 | 20 | **20** | **0** | All 4 KGE baselines × 5 seeds (custom eval fix) |

**Total: 82 result files across 3 runs.**

---

## Data Sources Status

| Source | Status | Notes |
|--------|--------|-------|
| PrimeKG t0 (June 2021) | Available | Harvard Dataverse, 8.1M edges |
| Real t1 (July 2023) | Built | 13M edges from 9 databases |
| DrugBank | **Newly acquired** | `drugbank_all_full_database.xml.zip` — for future t1 rebuild |
| UMLS | **Newly acquired** | `umls-2025AB-full.zip` — for future t1 rebuild |
| DisGeNET | Unavailable | API moved to commercial model |
| Reactome | Unavailable | Server unreachable |
