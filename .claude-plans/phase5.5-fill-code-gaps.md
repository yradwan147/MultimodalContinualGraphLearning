# Phase 5.5: Fill All MCGL Code Gaps + Update SLURM for Comprehensive Testing

**Completed:** 2026-03-01 | **Status:** DONE
**Depends on:** Phase 4 (CMKL implementation) | **Blocks:** Phase 5 (IBEX experiments)

---

## Context

All 12 initial IBEX jobs OOM'd — this was GPU VRAM (not system RAM). The generic GPUs on IBEX have ~16GB VRAM, which isn't enough for PyKEEN's all-entity scoring with 129K+ entities at dim=256. Fix: switch to V100 (32GB VRAM) and add `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`.

Additionally, a project audit identified 5 code gaps that needed filling to complete the full experiment matrix from the PDF (8 experiments x 3 tasks = 20 SLURM jobs).

---

## SLURM OOM Fix

**All SLURM scripts**: Changed to `--gpus-per-node=1` + `--constraint=v100` and added `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` to every script.

---

## Step 1: Knowledge Distillation (DONE)

**Why:** Completes the CMKL continual learning toolkit and adds the `distillation` ablation study.

### Changes Made

**`src/continual/distillation.py`** (rewritten from stub):
- `KnowledgeDistillation` class with configurable temperature (T) and interpolation weight (alpha)
- `compute_distillation_loss()`: KL divergence between teacher/student score distributions
  - Formula: `L_soft = T^2 * KL(log_softmax(student/T) || softmax(teacher/T))`
- `compute_combined_loss()`: `L_total = alpha * L_hard + (1-alpha) * L_soft`
- `create_teacher_copy()`: Deep-copies model and freezes all parameters

**`src/models/cmkl.py`** (modified):
- Added config keys: `use_distillation` (False), `distillation_temperature` (2.0), `distillation_alpha` (0.5)
- Added `self.distillation` and `self._teacher_model` fields
- In `train_continually()`: creates frozen teacher copy after each task (for next task)
- In `_train_epoch()`: computes teacher scores (no_grad), gets student scores, adds distillation loss

**`scripts/run_ablations.py`** (modified):
- Added `"distillation"` to ABLATIONS list and dispatch (sets `use_distillation=True`)

**`scripts/run_cmkl.py`** (modified):
- Added `--use-distillation`, `--distillation-temperature`, `--distillation-alpha` CLI args

**Smoke test:** PASS

---

## Step 2: LKGE Result Parsing + Integration (DONE)

**Why:** Completes Baseline 5 (LKGE) in the experiment matrix.

### Changes Made

**`src/baselines/lkge.py`** (rewritten):
- `parse_results()`: 3 regex patterns for different LKGE output formats
  - Pattern 1: `Snapshot X - MRR: Y, Hits@1: Z, ...`
  - Pattern 2: `test on snapshot X` followed by metric lines
  - Pattern 3: Pipe-separated table rows
- `run_and_parse()`: Runs LKGE as subprocess, captures stdout, parses results

**`scripts/run_lkge.py`** (created, ~149 lines):
- CLI: `--tasks-dir`, `--model`, `--lkge-dir`, `--output-dir`, `--seeds`, `--quick`
- Quick mode: tests format conversion only (no actual LKGE run)
- Full mode: convert data -> run LKGE -> parse output -> compute CL metrics -> save JSON

**`slurm/run_lkge.sh`** (created):
- Standard SBATCH header with V100
- Auto-clones LKGE repo if `external/LKGE` doesn't exist

**Smoke test:** PASS (format conversion + log parsing)

---

## Step 3: RAG Agent + KGQA Pipeline (DONE)

**Why:** Completes Baseline 6 (RAG) and KGQA evaluation task. Largest gap (~400 lines).

### Changes Made

**`src/data/kgqa.py`** (created, ~165 lines):
- `QUESTION_TEMPLATES`: 24 relation-type to question template mappings
  - e.g., `"indication"` -> `"What drug treats {disease}?"`
- `generate_kgqa_questions()`: Converts test triples to QA pairs
- `generate_continual_kgqa_dataset()`: Per-task QA sets aligned with CL task sequence
- `_clean_entity_name()`: Strips prefixes (MONDO:, HP:, GO:, etc.)

**`src/baselines/rag_agent.py`** (rewritten from stub, ~343 lines):
- `BiomedicalRAGAgent` class with ChromaDB + optional HuggingFace LLM
- `_init_vectorstore()`: ChromaDB client + SentenceTransformerEmbeddingFunction (PubMedBERT)
- `_init_llm()`: HuggingFace text-generation pipeline with fallback to retrieval-only
- `index_kg_snapshot()`: Triple-to-NL conversion, batch indexing (40K per batch)
- `update_with_new_knowledge()`: Append new triples to existing collection
- `answer_question()`: Retrieve top-K -> LLM generation or majority-vote extraction
- `evaluate_kgqa()`: Batch evaluation returning EM, token_f1, accuracy

**`src/evaluation/metrics.py`** (modified):
- Added `compute_exact_match(prediction, ground_truth)`: Normalized string comparison
- Added `compute_token_f1(prediction, ground_truth)`: Token-level F1 with Counter intersection

**`scripts/run_rag.py`** (created, ~201 lines):
- CLI: `--no-llm`, `--questions-per-task`, `--quick`
- Pipeline: load tasks -> generate KGQA questions -> for each task: index KG -> evaluate all tasks seen so far -> build results matrix -> compute CL metrics -> save JSON

**`slurm/run_rag.sh`** (created):
- V100, `--mem=64G` (LLM needs more RAM), `--cpus-per-gpu=4`

**Smoke test:** PASS (question generation + EM/F1 metrics)

---

## Step 4: Node Classification Support (DONE)

**Why:** Completes Task 3 from the experiment matrix (4 baselines + CMKL on NC).

### Changes Made

**`src/data/node_classification.py`** (created, ~185 lines):
- `NODE_TYPE_LABELS`: PrimeKG 10-type mapping (anatomy=0, ..., pathway=9)
- `get_label_map()`: Returns the mapping dict
- `load_node_types()`: From node_index_map.csv or KG CSV
- `build_nc_dataset()`: Per-task NC datasets with train/val/test masks, 10-node minimum threshold

**`src/baselines/nc_baseline.py`** (created, ~204 lines):
- `NCClassifier(nn.Module)`: 2-layer MLP (Linear-ReLU-Dropout-Linear)
- `NCBaseline`: Trains classifier on frozen embeddings, early stopping on val accuracy
- `extract_pykeen_embeddings()`: Gets entity embeddings from PyKEEN models

**`src/models/cmkl.py`** (modified):
- Added optional `nc_classifier` head gated by `use_nc` config flag
- Head: `nn.Sequential(Linear(D,D), ReLU, Dropout(0.3), Linear(D, num_classes))`
- Added `classify_nodes(node_embeddings, node_ids)` method

**`src/evaluation/metrics.py`** (modified):
- Added `compute_nc_metrics(y_true, y_pred)`: accuracy, macro_f1, weighted_f1 via sklearn

**`scripts/run_nc.py`** (created, ~355 lines):
- `run_nc_kge_baseline()`: Train KGE continually -> extract embeddings -> train MLP classifier
- `run_nc_cmkl()`: Uses CMKL fused embeddings + integrated NC head
- CLI: `--method` (5 methods + "all"), `--seeds`, `--quick`

**`slurm/run_nc.sh`** (created):
- Standard SBATCH header, takes `$METHOD` as argument

**Smoke test:** PASS (label map + classifier + eval with 20 nodes)

---

## Step 5: Experiment Logger Verification (DONE)

**`src/utils/logging.py`** was already implemented in a prior session. Verified imports work.

---

## Step 6: SLURM Final Update (DONE)

### Changes Made

**All 7 SLURM scripts** updated with:
- `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`
- `mkdir -p slurm/slurm_logs`
- `--gres=gpu:v100:1` (32GB VRAM)

**`slurm/submit_all.sh`** rewritten for 20 jobs:
```
Link Prediction:      14 jobs (4 baselines + 1 LKGE + 1 CMKL + 8 ablations)
KGQA:                  1 job  (RAG agent)
Node Classification:   5 jobs (4 baselines + CMKL)
Total:                20 jobs
```

---

## Files Summary

| Action | File | Lines (approx.) |
|--------|------|-----------------|
| Rewrite | `src/continual/distillation.py` | ~80 |
| Modify | `src/models/cmkl.py` | +80 |
| Rewrite | `src/baselines/lkge.py` | +60 |
| Rewrite | `src/baselines/rag_agent.py` | ~343 |
| Modify | `src/evaluation/metrics.py` | +60 |
| Modify | `scripts/run_ablations.py` | +15 |
| Modify | `scripts/run_cmkl.py` | +10 |
| Create | `src/data/kgqa.py` | ~165 |
| Create | `src/data/node_classification.py` | ~185 |
| Create | `src/baselines/nc_baseline.py` | ~204 |
| Create | `scripts/run_lkge.py` | ~149 |
| Create | `scripts/run_rag.py` | ~201 |
| Create | `scripts/run_nc.py` | ~355 |
| Create | `slurm/run_lkge.sh` | ~25 |
| Create | `slurm/run_rag.sh` | ~25 |
| Create | `slurm/run_nc.sh` | ~25 |
| Rewrite | `slurm/submit_all.sh` | ~85 |
| Modify | 4 existing SLURM scripts | +2 lines each |

**Total new/modified code: ~1,300 lines across 18 files**

---

## Verification Results

All smoke tests passed:
1. `python scripts/run_ablations.py --ablation distillation --quick` -> AP, AF metrics produced
2. `python scripts/run_lkge.py --quick` -> LKGE format files generated, log parsing works
3. `python scripts/run_rag.py --quick` -> QA questions generated, EM/F1 metrics computed
4. `python scripts/run_nc.py --method naive_sequential --quick` -> NC accuracy/F1 produced
5. `slurm/submit_all.sh` lists all 20 sbatch commands
6. All 7 SLURM scripts have `--gres=gpu:v100:1` and `PYTORCH_CUDA_ALLOC_CONF`
