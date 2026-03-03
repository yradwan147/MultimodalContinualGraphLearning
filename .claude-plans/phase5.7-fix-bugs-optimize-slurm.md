# Plan: Fix All Bugs + Optimize SLURM + Add Monitoring

## Context
Deep analysis of all 12 result JSONs, 30 SLURM logs, and full source code audit revealed **5 critical bugs** producing wrong results, **infrastructure failures** causing all LKGE/RAG jobs to crash, and **SLURM timeout issues** where 18/30 jobs hit the 24h limit. This plan fixes everything before the next IBEX run so there are zero failed jobs.

### Key Findings from Latest Logs
- **LKGE**: `external/LKGE/src/config.py:8` HARD-CODES `args.emb_dim = 200`, overriding our `-emb_dim 100` CLI flag. Every LKGE run used emb_dim=200 despite our fix. This is the root cause.
- **RAG**: OOM killed at task 5 (task_2_gene_protein) after ~11.6h. ChromaDB indexing 5.67M triples + LLM uses >32G RAM. F1=0.0000 on ALL evaluations (potential eval bug).
- **CMKL**: Seed 42 completed in ~18.2h with MRR=1.0 everywhere (confirms _map_triples bug). Loss=0.0000 from epoch 1. Seed 2 hit 24h time limit.
- **Baselines**: naive_seq=17.9h/seed, joint=6.8h/seed, ewc=11.5h/seed, replay=18.1h/seed. Only 1-3 seeds complete per job.
- **NC**: ewc/replay/naive produce IDENTICAL macro_f1 values. joint_training and cmkl hit 24h limit.

---

## Priority 1: Fix CMKL `_map_triples` Bug (MRR=1.0)

**File:** `src/models/cmkl.py:590-609`

**Bug:** `_map_triples()` receives int64 arrays (already ID-mapped by `load_task_sequence()`), but calls `entity_to_id.get(int64_value, 0)` — keys are strings, so EVERY lookup returns default 0. All triples collapse to `(0, 0, 0)`, making training and evaluation trivially broken (MRR=1.0, loss=0.0000).

**Evidence:** `results/cmkl_DistMult.json` shows AP=1.0, AF=0.0. CMKL IBEX log shows loss=0.0000 from epoch 1 and MRR=1.0000 on ALL evaluations.

**Fix:** Since triples from `load_task_sequence()` are already int64 IDs, `_map_triples()` should be a no-op identity. Replace:
```python
def _map_triples(self, triples, entity_to_id, relation_to_id):
    # Triples are already int64 IDs from load_task_sequence()
    return np.asarray(triples, dtype=np.int64)
```

Also fix `_evaluate_mrr()` (line 801) which calls `_map_triples` — same fix applies transitively.

---

## Priority 2: Fix KGE Filtered Evaluation

**File:** `src/baselines/_base.py:249-254`

**Bug:** `additional_filter_triples=None` means **raw** ranking (not filtered). Standard KGE practice is filtered ranking — removing all known true triples from the ranking. Without filtering, any entity that is a valid answer for (h,r,?) but isn't the test triple gets counted as a false negative, pessimistically deflating scores.

**Evidence:** Baseline MRR values are very low (0.0001-0.046 across all baselines). While the task is hard, filtered ranking is the standard protocol needed for paper comparisons.

**Fix:** Change `evaluate_link_prediction()` to accept `all_known_triples` parameter:
```python
def evaluate_link_prediction(
    model, test_factory, device="cpu", batch_size=256,
    all_known_mapped_triples=None,  # NEW
):
    results = evaluator.evaluate(
        model=model,
        mapped_triples=test_factory.mapped_triples.to(device),
        additional_filter_triples=all_known_mapped_triples,  # Pass through
        batch_size=batch_size,
    )
```

Then update all callers (naive_sequential.py, ewc.py, experience_replay.py, joint_training.py) to pass concatenated train+val+test triples from all tasks seen so far.

Similarly, add filtered ranking to CMKL's `_evaluate_mrr()` by masking known (h,r,t) triples when computing ranks.

---

## Priority 3: Fix NC Identical Results

**File:** `scripts/run_nc.py:119-134`

**Bug:** Lines 132-133 use plain `train_epoch()` for ALL methods (naive_sequential, ewc, experience_replay). No EWC penalty or replay buffer is applied. All methods produce identical embeddings → identical NC results.

**Evidence:** `nc_naive_sequential.json` and `nc_experience_replay.json` have identical `results_matrix[0][0] = 0.5265560251438429`.

**Fix:** Add method-specific training inside the NC loop:
- `ewc`: Import and use `EWCTrainer` training logic (Fisher computation + EWC penalty)
- `experience_replay`: Import and use `ReplayTrainer` training logic (buffer sampling + mixed batches)
- `naive_sequential`: Keep using plain `train_epoch()`
- `joint_training`: Already handled correctly (accumulates all triples)

---

## Priority 4: Fix LKGE — Config Override + Parsing + Results Matrix

### 4A: LKGE config.py Hard-Codes emb_dim=200 (ROOT CAUSE OF OOM)

**File:** `external/LKGE/src/config.py:8`

**Bug:** `config.py` line 8 runs `args.emb_dim = 200` AFTER argparse, unconditionally overriding any CLI flag. Our `-emb_dim 100` was silently ignored. Every LKGE run used emb_dim=200.

**Evidence:** LKGE Namespace in log shows `emb_dim=200` despite CLI passing `-emb_dim 100`. OOM allocation is identical (17.03 GiB) across all attempts.

**Fix:** Patch `external/LKGE/src/config.py` to only set defaults when not specified:
```python
def config(args):
    args.learning_rate = 0.0001
    # DON'T override emb_dim if explicitly set via CLI
    if not hasattr(args, '_emb_dim_set'):
        args.emb_dim = 200
    args.batch_size = 2048
```
Or simpler: just comment out line 8 and rely on our wrapper to always pass emb_dim.

Also: `parse_args.py` line 22 lacks `type=int`, so CLI values are stored as strings. Fix by ensuring our wrapper passes integers correctly, or patch parse_args.py to add `type=int`.

### 4B: LKGE Parsing + Results Matrix

**File:** `src/baselines/lkge.py:300-366`

**Bug 1:** `_parse_log_content()` regex patterns don't match LKGE's actual PrettyTable output. All seeds produce `per_snapshot: {}` → empty results.

**Bug 2:** Results matrix construction at line 355: `results_matrix[n - 1, snap_id]` always fills only the LAST row.

**Fix:**
1. Patch config.py and parse_args.py to respect CLI args
2. Run LKGE with emb_dim=50 on a tiny test to capture actual output format
3. Rewrite regex patterns to match real PrettyTable output
4. Fix matrix construction to fill R[i][j] correctly
5. Add fallback: log full raw output if parsing fails

### 4C: LKGE Memory Optimization

Even with emb_dim=100 working correctly, task_0_base (5.67M training edges) may still be too large for LKGE's full-graph GCN on V100 32GB. Options:
- **emb_dim=50**: Reduces GCN memory by 4x from emb_dim=200
- **Exclude task_0_base**: Run LKGE on 9 incremental tasks only (note scalability limitation in paper)
- Both: use emb_dim=50 with all tasks, fall back to excluding task_0 if still OOM

SLURM config: `--mem=128G`, `--time=48:00:00`, `--constraint=v100`

---

## Priority 5: Fix RAG OOM + F1=0.0

### 5A: RAG RAM OOM (Killed at Task 5)

**Evidence:** Latest log (mcgl_rag_45765538.out) — SLURM OOM killer after ~11.6h. ChromaDB had indexed task_0_base (5.67M triples) → task_4 (~150K more), then OOM when adding task_2_gene_protein (2M triples). Total: ~8M+ embedded documents in RAM.

**Fix:**
- Bump SLURM `--mem=128G` (from 32G)
- Use `--no-llm` for RAG evaluation (retrieval-only is the fair baseline comparison since we're testing retrieval, not LLM generation quality)
- Remove LLM from SLURM config (saves ~16G VRAM + significant RAM for model weights)
- If still tight: add ChromaDB persistence (`persist_directory=`) to allow incremental indexing with disk-backed storage

### 5B: RAG F1=0.0000 on All Evaluations

**Evidence:** Even tasks where ChromaDB was fully indexed show F1=0.0000. This suggests the KGQA evaluation pipeline is broken.

**Fix:** Investigate `src/data/kgqa.py` and `src/baselines/rag_agent.py` evaluation:
- Check if questions are generated with entity names or int IDs
- Check if token_f1 comparison handles the answer format correctly
- Run a manual test: index 100 triples, generate 5 questions, print question+answer+prediction

---

## Priority 6: SLURM Separate Jobs (1 Seed Per Job)

**Problem:** Measured timing per seed:
| Method | Time/Seed | 5 Seeds Total | Status |
|--------|-----------|---------------|--------|
| naive_sequential | 17.9h | 89.5h | Only 1/5 seeds done |
| joint_training | 6.8h | 34h | 3/5 seeds done |
| ewc | 11.5h | 57.5h | 2/5 seeds done |
| experience_replay | 18.1h | 90.5h | Only 1/5 seeds done |
| cmkl | 18.2h | 91h | Only 1/5 seeds done |
| NC (each method) | ~17h | ~85h | 1/5 seeds each |

**Fix:** Create individual SLURM scripts that accept seed as a parameter. Each job runs 1 seed.

### Template: `slurm/run_baseline.sh` accepts baseline name + model + seed
```bash
#!/bin/bash --login
#SBATCH --time=30:00:00
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100
#SBATCH -J bl_${1}_s${3}   # e.g., bl_naive_s42

BASELINE=${1:?Usage: sbatch run_baseline.sh <baseline> <model> <seed>}
MODEL=${2:-TransE}
SEED=${3:?Must provide seed}

python scripts/run_baselines.py \
    --baseline $BASELINE --model $MODEL \
    --seeds $SEED \
    --output-dir results
```

### Per-method time limits (in the SLURM scripts):
- `naive_sequential`, `experience_replay`: `--time=30:00:00`
- `ewc`: `--time=20:00:00`
- `joint_training`: `--time=12:00:00`
- `cmkl`: `--time=30:00:00`
- `lkge`: `--time=48:00:00` + `--mem=128G`
- `rag`: `--time=24:00:00` + `--mem=128G` + no GPU if `--no-llm`
- `nc` (all methods): `--time=30:00:00`

### Each run script already saves per-seed results
The incremental save logic already writes after each seed. Since each job runs 1 seed, the output file is named by method+model (e.g., `naive_sequential_TransE.json`). To avoid race conditions, add `--output-suffix` arg so each seed writes to a unique file (e.g., `naive_sequential_TransE_seed42.json`).

### New: `scripts/merge_seed_results.py` (~50 lines)
After all jobs complete, merge per-seed JSON files:
```bash
python scripts/merge_seed_results.py --input-dir results --method naive_sequential
```

### New: `slurm/submit_all.sh`
Master script that submits all individual jobs with meaningful names:
```bash
#!/bin/bash
# Submit all experiments — 1 seed per job
SEEDS="42 123 456 789 1024"

for SEED in $SEEDS; do
    sbatch -J ns_s${SEED} slurm/run_baseline.sh naive_sequential TransE $SEED
    sbatch -J jt_s${SEED} slurm/run_baseline.sh joint_training TransE $SEED
    sbatch -J ewc_s${SEED} slurm/run_baseline.sh ewc TransE $SEED
    sbatch -J er_s${SEED} slurm/run_baseline.sh experience_replay TransE $SEED
    sbatch -J cmkl_s${SEED} slurm/run_cmkl.sh DistMult $SEED
    sbatch -J lkge_s${SEED} slurm/run_lkge.sh TransE $SEED
    sbatch -J nc_ns_s${SEED} slurm/run_nc.sh naive_sequential $SEED
    sbatch -J nc_ewc_s${SEED} slurm/run_nc.sh ewc $SEED
    sbatch -J nc_er_s${SEED} slurm/run_nc.sh experience_replay $SEED
    sbatch -J nc_jt_s${SEED} slurm/run_nc.sh joint_training $SEED
done

# RAG: single job with --no-llm, 1 seed at a time
for SEED in $SEEDS; do
    sbatch -J rag_s${SEED} slurm/run_rag.sh $SEED
done

echo "Submitted $(echo $SEEDS | wc -w) seeds x 11 methods = 55 jobs"
```

### Job naming convention:
`{method_abbrev}_s{seed}` — e.g., `ns_s42`, `ewc_s123`, `cmkl_s456`
Makes `squeue` output immediately readable.

---

## Priority 7: Structured Progress Reporting

**Goal:** User can monitor all jobs from terminal with a single command.

### Add to all Python scripts:
```python
# At start:
print("[STARTED] method=naive_sequential seed=42 tasks=10 epochs=100")

# After each task:
print(f"[PROGRESS] method=naive_sequential seed=42 task={task_idx+1}/{num_tasks} "
      f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

# On success:
print(f"[SUCCESS] method=naive_sequential seed=42 AP={ap:.4f} AF={af:.4f} time={total:.0f}s")

# On failure:
print(f"[FAILED] method=naive_sequential seed=42 error={str(e)[:200]}")
```

### Monitoring command (user runs from terminal):
```bash
# See status of all running/completed jobs:
grep -h '\[STARTED\]\|\[PROGRESS\]\|\[SUCCESS\]\|\[FAILED\]' slurm/slurm_logs/*.out | tail -50

# Or watch live:
watch -n 30 'grep -h "\[PROGRESS\]\|\[SUCCESS\]\|\[FAILED\]" slurm/slurm_logs/*.out | sort | tail -30'
```

### Add wrapper try/except to all script `main()` functions:
```python
def main():
    try:
        # ... existing code ...
        print(f"[SUCCESS] ...")
    except Exception as e:
        print(f"[FAILED] ... error={e}")
        raise  # Still fail the SLURM job
```

---

## Priority 8: Local Smoke Tests

**File (new):** `tests/test_smoke.py`

Quick tests that run locally before IBEX submission to catch bugs early:

1. **test_cmkl_map_triples**: Verify `_map_triples` is identity on int64 arrays
2. **test_filtered_evaluation**: Verify `evaluate_link_prediction` accepts and uses filter triples
3. **test_nc_methods_differ**: Verify EWC and replay NC training produce different results from naive
4. **test_baseline_quick**: Run `run_baselines.py --quick --baseline naive_sequential` end-to-end
5. **test_cmkl_quick**: Run `run_cmkl.py --quick` end-to-end, verify MRR < 1.0
6. **test_lkge_format_conversion**: Verify LKGE format conversion produces valid files
7. **test_progress_markers**: Verify `[STARTED]`/`[PROGRESS]`/`[SUCCESS]` appear in stdout

Run with: `python -m pytest tests/test_smoke.py -v`

---

## Priority 9: Multi-Hop Integration (Already Partly Done)

The `--eval-multihop` flag and `src/evaluation/multihop.py` are already implemented. Current status:
- `extract_all_path_types()` works (smoke tested)
- Baseline/CMKL scripts only count paths, don't run actual model scoring yet
- RAG multi-hop evaluation is fully implemented

**Remaining work:**
- In `run_baselines.py`: after getting the trained model, call `evaluate_multihop()` with `make_pykeen_score_fn(model)`
- In `run_cmkl.py`: call `evaluate_multihop()` with `make_cmkl_score_fn(model, h_fused)`
- Save multihop results in the per-seed JSON alongside standard metrics

---

## Files Summary

| Priority | Action | File | Change |
|----------|--------|------|--------|
| P1 | Fix | `src/models/cmkl.py` | `_map_triples` → identity for int64 |
| P2 | Fix | `src/baselines/_base.py` | Add `all_known_mapped_triples` param to `evaluate_link_prediction` |
| P2 | Fix | `src/baselines/naive_sequential.py` | Pass filter triples |
| P2 | Fix | `src/baselines/ewc.py` | Pass filter triples |
| P2 | Fix | `src/baselines/experience_replay.py` | Pass filter triples |
| P2 | Fix | `src/baselines/joint_training.py` | Pass filter triples |
| P2 | Fix | `src/models/cmkl.py` | Add filtered ranking to `_evaluate_mrr` |
| P3 | Fix | `scripts/run_nc.py` | Method-specific CL training |
| P4A | Patch | `external/LKGE/src/config.py` | Remove emb_dim hard-code override |
| P4A | Patch | `external/LKGE/src/parse_args.py` | Add `type=int` to emb_dim, batch_size |
| P4B | Fix | `src/baselines/lkge.py` | Rewrite parsing + matrix construction |
| P5A | Fix | `slurm/run_rag.sh` | `--mem=128G` + `--no-llm` |
| P5B | Fix | `src/baselines/rag_agent.py` or `src/data/kgqa.py` | Investigate F1=0.0 |
| P6 | Rewrite | `slurm/run_baseline.sh` | Accept seed param, per-method time limits |
| P6 | Rewrite | `slurm/run_cmkl.sh` | Accept seed param, 30h, 48G |
| P6 | Rewrite | `slurm/run_lkge.sh` | Accept seed param, 48h, 128G |
| P6 | Rewrite | `slurm/run_rag.sh` | Accept seed param, 128G, --no-llm |
| P6 | Rewrite | `slurm/run_nc.sh` | Accept method+seed params, 30h |
| P6 | Modify | All `scripts/run_*.py` | Add `--output-suffix` for per-seed files |
| P6 | Create | `scripts/merge_seed_results.py` | Merge per-seed JSONs |
| P6 | Create | `slurm/submit_all.sh` | Master submission script |
| P7 | Modify | All `scripts/run_*.py` | Add [STARTED]/[PROGRESS]/[SUCCESS]/[FAILED] |
| P8 | Create | `tests/test_smoke.py` | Local smoke tests |
| P9 | Modify | `scripts/run_baselines.py` | Actual multihop model scoring |
| P9 | Modify | `scripts/run_cmkl.py` | Actual multihop model scoring |

**~25 files modified/created**

---

## Verification Plan

### Local (before IBEX):
```bash
# 1. Smoke tests pass
python -m pytest tests/test_smoke.py -v

# 2. CMKL no longer returns MRR=1.0
python scripts/run_cmkl.py --quick --task-names task_1_disease_related task_3_phenotype_related
# Check: results/cmkl_DistMult.json AP < 1.0, loss decreasing

# 3. Baselines produce reasonable MRR (filtered)
python scripts/run_baselines.py --baseline naive_sequential --quick \
    --task-names task_1_disease_related task_3_phenotype_related
# Check: results file has MRR > 0 and < 1

# 4. LKGE config.py patched correctly
python -c "import sys; sys.path.insert(0,'external/LKGE/src'); from config import config; import argparse; a=argparse.Namespace(emb_dim=50, dataset='test', lifelong_name='LKGE'); config(a); assert a.emb_dim==50, f'config override: {a.emb_dim}'"

# 5. LKGE format conversion + CLI args work
python scripts/run_lkge.py --quick
# Check: lkge_TransE_quick.json has correct emb_dim in command

# 6. RAG quick mode (retrieval-only, 10 questions)
python scripts/run_rag.py --quick --task-names task_1_disease_related task_3_phenotype_related
# Check: F1 > 0 (if still 0, investigate kgqa.py)

# 7. Progress markers appear
python scripts/run_baselines.py --baseline naive_sequential --quick 2>&1 | grep '\[PROGRESS\]'

# 8. NC methods produce different results
python scripts/run_nc.py --quick --method ewc
python scripts/run_nc.py --quick --method naive_sequential
# Check: results differ
```

### On IBEX:
```bash
# Submit all jobs
bash slurm/submit_all.sh

# Monitor (single command for all jobs)
watch -n 30 'grep -h "\[PROGRESS\]\|\[SUCCESS\]\|\[FAILED\]" slurm/slurm_logs/*.out | sort | tail -30'

# Check for any OOM or failures
grep -h "OOM\|Killed\|FAILED\|Error" slurm/slurm_logs/*.out
```
