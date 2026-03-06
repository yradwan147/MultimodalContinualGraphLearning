# Run 3 Plan: Fix PyKEEN Segfault + Rerun All KGE Baselines

## Context

Runs 1 and 2 showed that PyKEEN's `RankBasedEvaluator` segfaults non-deterministically when scoring against 129K entities. The `max_test_triples=50K` fix was insufficient — it only reduced test triple count (rows), not the entity count (columns). 16/22 KGE baseline jobs still crashed in Run 2.

**Proof the fix works:** CMKL's `_evaluate_mrr()` (`src/models/cmkl.py:775-881`) does the exact same all-entity scoring (129K entities) but works perfectly — because it bypasses PyKEEN's evaluator entirely and uses direct PyTorch tensor operations.

**Consistency requirement (user-requested):** Since `evaluate_link_prediction()` is being rewritten, ALL 4 KGE baselines must be rerun (including Joint Training's 4 successful seeds) so every result comes from the same evaluation code.

---

## Step 1: Rewrite `evaluate_link_prediction()` — `src/baselines/_base.py` (lines 223-289)

Replace PyKEEN's `RankBasedEvaluator` with custom manual evaluation (same approach as CMKL):

1. Extract embeddings directly from PyKEEN model:
   ```python
   entity_emb = model.entity_representations[0](indices=None)  # [N, D]
   relation_emb = model.relation_representations[0](indices=None)  # [R, D]
   ```
2. Score all entities as tails per batch (batch_size=64):
   - TransE: `scores = -torch.cdist(h + r, entity_emb, p=1)`  → [B, N]
   - DistMult: `scores = (h * r) @ entity_emb.T`  → [B, N]
3. Build `hr_to_tails` filter dict from `all_known_mapped_triples` (same pattern as CMKL lines 819-825)
4. Mask known tails to `-inf`, compute rank of true tail
5. Return MRR, Hits@1, Hits@3, Hits@10
6. Call `torch.cuda.empty_cache()` before eval loop
7. Remove `from pykeen.evaluation import RankBasedEvaluator` entirely

Keep the existing `max_test_triples=50_000` sampling as additional safety.

## Step 2: Create `slurm/submit_run3.sh`

20 jobs total (4 KGE methods × 5 seeds):

| Method | Seeds | Reason |
|--------|-------|--------|
| naive_sequential | 42, 123, 456, 789, 1024 | Never completed (segfault) |
| ewc | 42, 123, 456, 789, 1024 | Never completed (segfault) |
| experience_replay | 42, 123, 456, 789, 1024 | Never completed (segfault) |
| joint_training | 42, 123, 456, 789, 1024 | Rerun all for eval consistency |

Uses existing `slurm/run_baseline.sh` (no changes needed).

## NOT modified (unaffected — different eval path)

- `src/models/cmkl.py` — CMKL uses its own `_evaluate_mrr()`, unchanged
- `src/baselines/lkge.py` — LKGE runs externally via subprocess
- `src/baselines/rag_agent.py` — RAG uses KGQA text evaluation
- `scripts/run_nc.py` — NC uses classification evaluation
- CMKL results (5/5), LKGE results (5/5), RAG results (5/5), NC results (25/25) — all valid, no rerun needed

---

## Verification

1. Local smoke test: `python scripts/run_baselines.py --baseline naive_sequential --model TransE --quick`
   - Should complete without segfault
   - Output JSON should have MRR, Hits@1, Hits@3, Hits@10
2. Compare quick-mode metrics with previous smoke test results (should be similar magnitude)
