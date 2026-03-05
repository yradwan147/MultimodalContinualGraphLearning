# Experiment Results Tracking

*Updated after each experiment run. All results use 5 random seeds [42, 123, 456, 789, 1024], reporting mean +/- std.*

---

## Run 1 (IBEX, 2026-03-03 to 2026-03-05)

**Status:** 28/60 jobs completed. 22 jobs resubmitted (Run 2). See `docs/run1_report.md` for full analysis.

### Link Prediction (Continual, 10 tasks)

| Method | Seeds | AP (mean +/- std) | AF (mean +/- std) | BWT (mean +/- std) | REM (mean +/- std) | Status |
|--------|-------|--------------------|--------------------|---------------------|---------------------|--------|
| CMKL (DistMult) | 5/5 | 0.0630 +/- 0.0025 | 0.0399 +/- 0.0027 | -0.0399 +/- 0.0027 | 0.9601 +/- 0.0027 | Complete |
| LKGE (TransE) | 5/5 | 0.0388 +/- 0.0010 | 0.0122 +/- 0.0018 | -0.0122 +/- 0.0018 | — | Complete (9 tasks, skipped base) |
| Joint Training (TransE) | 3/5 | 0.0203 +/- 0.0004 | 0.0 | 0.0 | 1.0 | Partial — seeds 123, 456 segfaulted |
| Naive Sequential (TransE) | 0/5 | — | — | — | — | Rerun (segfault) |
| EWC (TransE) | 0/5 | — | — | — | — | Rerun (segfault) |
| Experience Replay (TransE) | 0/5 | — | — | — | — | Rerun (segfault) |
| RAG (Qwen2.5-7B) | 1/5 | 0.0025 | 0.0 | — | — | Rerun (24h timeout, now with LLM) |

### Node Classification (Continual, 10 tasks)

| Method | Seeds | AP (mean +/- std) | AF (mean +/- std) | BWT (mean +/- std) | Status |
|--------|-------|--------------------|--------------------|---------------------|--------|
| **CMKL** | **5/5** | **0.431 +/- 0.005** | **0.004 +/- 0.002** | **-0.000 +/- 0.002** | **Complete** |
| Joint Training | 5/5 | 0.370 +/- 0.002 | 0.003 +/- 0.001 | 0.022 +/- 0.003 | Complete |
| EWC | 5/5 | 0.345 +/- 0.004 | 0.008 +/- 0.003 | 0.007 +/- 0.004 | Complete |
| Experience Replay | 5/5 | 0.344 +/- 0.006 | 0.010 +/- 0.003 | 0.006 +/- 0.005 | Complete |
| Naive Sequential | 5/5 | 0.344 +/- 0.004 | 0.011 +/- 0.002 | 0.003 +/- 0.005 | Complete |

**Key findings:**
- CMKL significantly outperforms all baselines on NC (0.431 vs 0.370 joint training) due to multimodal features
- NC forgetting is naturally minimal — EWC/replay add little value over naive sequential
- Gene-protein task MRR anomaly (0.50) is NOT leakage — structural simplicity (310 unique tails) + DistMult type-selection

---

## Ablation Results

| Ablation | AP | AF | BWT | FWT |
|----------|----|----|-----|-----|
| Struct only | — | — | — | — |
| Text only | — | — | — | — |
| Concat fusion | — | — | — | — |
| Global EWC | — | — | — | — |
| Random replay | — | — | — | — |

*Ablation runs pending — need IBEX after Run 2 completes.*

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
