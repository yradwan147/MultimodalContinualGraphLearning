# Experiment Results Tracking

*Updated after each experiment run. All results use 5 random seeds [42, 123, 456, 789, 1024], reporting mean ± std.*

---

## Combined Results (Run 1 + Run 2)

### Link Prediction (Continual, 10 tasks)

| Method | Seeds | AP (mean ± std) | AF (mean ± std) | BWT (mean ± std) | REM (mean ± std) | Status |
|--------|-------|-----------------|-----------------|-------------------|-------------------|--------|
| **CMKL (DistMult)** | **5/5** | **0.063 ± 0.003** | **0.040 ± 0.003** | **-0.040 ± 0.003** | **0.960 ± 0.003** | **Complete** |
| LKGE (TransE) | 5/5 | 0.039 ± 0.001 | 0.012 ± 0.002 | -0.010 ± 0.003 | 0.990 ± 0.003 | Complete (9 tasks, skipped base) |
| Joint Training (TransE) | 4/5 | 0.020 ± 0.000 | 0.0 | 0.0 | 1.0 | Missing seed 456 (segfault) |
| RAG (Qwen2.5-7B) | 5/5 | 0.002 ± 0.001 | 0.001 ± 0.001 | -0.001 ± 0.001 | 0.999 ± 0.001 | Complete |
| Naive Sequential (TransE) | 0/5 | — | — | — | — | **Segfault (PyKEEN eval)** |
| EWC (TransE) | 0/5 | — | — | — | — | **Segfault (PyKEEN eval)** |
| Experience Replay (TransE) | 0/5 | — | — | — | — | **Segfault (PyKEEN eval)** |

### Node Classification (Continual, 10 tasks) — ALL COMPLETE

| Method | Seeds | AP (mean ± std) | AF (mean ± std) | BWT (mean ± std) | Status |
|--------|-------|-----------------|-----------------|-------------------|--------|
| **CMKL** | **5/5** | **0.431 ± 0.005** | **0.004 ± 0.002** | **-0.000 ± 0.002** | **Complete** |
| Joint Training | 5/5 | 0.370 ± 0.002 | 0.003 ± 0.001 | 0.022 ± 0.003 | Complete |
| EWC | 5/5 | 0.345 ± 0.004 | 0.008 ± 0.003 | 0.007 ± 0.004 | Complete |
| Experience Replay | 5/5 | 0.344 ± 0.006 | 0.010 ± 0.003 | 0.006 ± 0.005 | Complete |
| Naive Sequential | 5/5 | 0.344 ± 0.004 | 0.011 ± 0.002 | 0.003 ± 0.005 | Complete |

---

## Key Findings

- **CMKL is best on both LP and NC** — LP: 0.063 AP (1.6x LKGE, 3.2x Joint), NC: 0.431 AP (1.16x Joint)
- **CMKL NC advantage is from multimodal features**, not CL mechanisms (EWC/replay add <1% over naive)
- **RAG is a lower bound** — Qwen2.5-7B cannot generate exact entity names for LP evaluation (AP=0.002)
- **Gene-protein task anomaly** (MRR~0.50) is NOT leakage — structural simplicity (310 unique tails) + DistMult type-selection
- **16 KGE baseline LP jobs still segfault** — PyKEEN all-entity eval with 129K entities crashes non-deterministically; needs custom eval fix

---

## Per-Seed Details

### CMKL DistMult (LP, 5/5)

| Seed | AP | AF | BWT | REM |
|------|-------|-------|--------|-------|
| 42 | 0.0625 | 0.0410 | -0.0410 | 0.9590 |
| 123 | 0.0650 | 0.0365 | -0.0365 | 0.9635 |
| 456 | 0.0650 | 0.0417 | -0.0417 | 0.9583 |
| 789 | 0.0638 | 0.0375 | -0.0375 | 0.9625 |
| 1024 | 0.0588 | 0.0426 | -0.0427 | 0.9573 |

### LKGE TransE (LP, 5/5, 9 tasks)

| Seed | AP | AF | BWT | REM |
|------|-------|-------|--------|-------|
| 42 | 0.0377 | 0.0134 | -0.0128 | 0.9872 |
| 123 | 0.0377 | 0.0128 | -0.0052 | 0.9949 |
| 456 | 0.0399 | 0.0089 | -0.0081 | 0.9919 |
| 789 | 0.0394 | 0.0124 | -0.0121 | 0.9879 |
| 1024 | 0.0394 | 0.0133 | -0.0129 | 0.9871 |

### Joint Training TransE (LP, 4/5)

| Seed | AP | AF | BWT | FWT | REM |
|------|-------|-----|-----|-------|-----|
| 42 | 0.0208 | 0.0 | 0.0 | 0.0207 | 1.0 |
| 123 | 0.0202 | 0.0 | 0.0 | 0.0201 | 1.0 |
| 789 | 0.0202 | 0.0 | 0.0 | 0.0201 | 1.0 |
| 1024 | 0.0200 | 0.0 | 0.0 | 0.0198 | 1.0 |

### RAG Qwen2.5-7B (LP, 5/5)

| Seed | AP | AF | BWT | REM |
|------|-------|-------|--------|-------|
| 42 | 0.0025 | 0.0011 | -0.0011 | 0.999 |
| 123 | 0.0025 | 0.0006 | -0.0006 | 0.999 |
| 456 | 0.0015 | 0.0011 | -0.0011 | 0.999 |
| 789 | 0.0040 | 0.0000 | +0.0006 | 1.000 |
| 1024 | 0.0005 | 0.0006 | -0.0006 | 0.999 |

---

## Ablation Results

*Pending — need IBEX runs after baseline LP fix.*

| Ablation | AP | AF | BWT | FWT |
|----------|----|----|-----|-----|
| Struct only | — | — | — | — |
| Text only | — | — | — | — |
| Concat fusion | — | — | — | — |
| Global EWC | — | — | — | — |
| Random replay | — | — | — | — |

---

## Run History

| Run | Date | Jobs | Completed | Failed | New Results |
|-----|------|------|-----------|--------|-------------|
| Run 1 | Mar 3-5 | 60 | 28 | 32 | CMKL 5/5, LKGE 5/5, JT 3/5, NC 25/25, RAG 1/5 |
| Run 2 | Mar 5-6 | 22 | 6 | 16 | JT seed 123, RAG 5/5 (full LLM) |

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
