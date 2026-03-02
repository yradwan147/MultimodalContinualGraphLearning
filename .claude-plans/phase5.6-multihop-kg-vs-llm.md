# Plan: Multi-Hop Evaluation + KG vs LLM Discussion Points

## Context
Prof. Zhang raised a critical question: since KG triples are individual pieces of knowledge, why not just use LLMs over them? The advantage of graph structure must be explicitly demonstrated. This plan adds:
1. **Multi-hop evaluation code** — empirically proves graph structure enables reasoning LLMs can't do
2. **Paper plan updates** — adds discussion sections and tables to both papers addressing KG vs LLM

---

## Step 1: Create `src/evaluation/multihop.py` (~250 lines)

**Core module for multi-hop path extraction and evaluation.**

### Constants: Biomedically meaningful 2-hop path types
```python
BIOMEDICAL_PATH_TYPES = [
    ("drug_protein", "disease_protein",    "drug → protein → disease"),   # drug repurposing
    ("drug_protein", "pathway_protein",    "drug → protein → pathway"),   # drug mechanism
    ("drug_protein", "bioprocess_protein", "drug → protein → bioprocess"),
    ("disease_protein", "pathway_protein", "disease → protein → pathway"),# disease mechanism
    ("disease_protein", "bioprocess_protein", "disease → protein → bioprocess"),
    ("drug_protein", "protein_protein",    "drug → protein → protein"),   # interaction chain
    ("disease_protein", "protein_protein", "disease → protein → protein"),
]
```

### Functions

**`build_adjacency_by_relation(triples, relation_to_id)`**
- Input: int64 triples array, relation mapping
- Output: `dict[int, dict[int, set[int]]]` — per-relation adjacency lists
- O(E) single pass over all training triples, ~200MB for 10M edges

**`build_direct_pair_set(triples)`**
- Input: int64 triples array
- Output: `set[tuple[int, int]]` of all directly-connected (head, tail) pairs
- Used to filter out multi-hop paths where source/target are also directly linked

**`extract_multihop_paths(adj_by_rel, rel1_id, rel2_id, direct_pairs, max_paths=10000, seed=42)`**
- Finds source→mid→target paths via adjacency intersection
- Filters out (source, target) pairs that are directly connected
- Samples up to `max_paths` if more found
- Returns list of `(source, rel1, mid, rel2, target)` tuples
- Efficient: O(E1 × avg_degree_rel2), ~seconds per path type

**`evaluate_multihop(score_fn, paths, num_entities, batch_size=256)`**
- For each path (src, r1, mid, r2, tgt): scores `(src, r2, ?)` and ranks `tgt`
- **Key insight:** R-GCN with 2 layers propagates 2-hop neighborhood info into src's embedding, so scoring `(src, r2, ?)` should rank true 2-hop targets highly
- RAG/flat embeddings don't encode this structural info
- Returns: `{multihop_MRR, multihop_Hits@1, multihop_Hits@10, num_paths}`

**`evaluate_multihop_rag(agent, paths, id_to_entity, id_to_relation, max_questions=500)`**
- Generates multi-hop questions from path templates
- E.g., "Drug X targets protein Y. What disease might X treat?"
- RAG must retrieve BOTH (drug,protein) AND (protein,disease) triples to answer
- Returns: `{multihop_EM, multihop_F1, num_questions}`

**`MULTIHOP_QUESTION_TEMPLATES`** — dict mapping (rel1, rel2) tuples to NL question templates for RAG evaluation

**Adapter functions:**
- `make_pykeen_score_fn(model, device)` — wraps PyKEEN model for multihop eval
- `make_cmkl_score_fn(cmkl_model, h_fused, device)` — wraps CMKL model

---

## Step 2: Add `compute_multihop_metrics()` to `src/evaluation/metrics.py` (~15 lines)

```python
def compute_multihop_metrics(ranks, ks=(1, 3, 10)):
    """Same as standard LP metrics but with 'multihop_' prefix."""
```

Reuses existing `compute_mrr()` and `compute_hits_at_k()`.

---

## Step 3: Integrate `--eval-multihop` into experiment scripts (~40 lines each)

### `scripts/run_baselines.py`
- Add `--eval-multihop` argparse flag
- After standard results matrix is built, if flag set:
  1. Collect all training triples seen so far
  2. Build adjacency + direct pair set
  3. Extract paths for each BIOMEDICAL_PATH_TYPES entry
  4. Create PyKEEN score_fn adapter
  5. Evaluate and append `multihop_results` to output JSON

### `scripts/run_cmkl.py`
- Same pattern with CMKL score_fn adapter
- Uses fused embeddings from `model.forward()` for scoring

### `scripts/run_rag.py`
- Same pattern but uses `evaluate_multihop_rag()` for NL question evaluation
- Generates multi-hop questions and evaluates RAG retrieval + reasoning

---

## Step 4: Create `scripts/run_multihop_eval.py` (~120 lines)

Standalone script for post-hoc multi-hop evaluation on saved checkpoints. Useful for running multi-hop analysis without retraining.

```
python scripts/run_multihop_eval.py \
    --tasks-dir data/benchmark/tasks \
    --results-dir results/ \
    --output results/multihop_comparison.json
```

---

## Step 5: Add `plot_multihop_comparison()` to `src/evaluation/visualization.py` (~40 lines)

Grouped bar chart: x-axis = path types, grouped bars = methods. Shows multi-hop MRR comparison.

---

## Step 6: SLURM Updates

### `slurm/run_baseline.sh`, `slurm/run_cmkl.sh`
Add `--eval-multihop` flag to the python command.

### New: `slurm/run_multihop.sh` (~20 lines)
Runs post-hoc `run_multihop_eval.py` after all main experiments complete.

### `slurm/submit_all.sh`
Add multihop eval job (depends on baseline + CMKL jobs completing).

---

## Step 7: Paper Plan Updates

### Phase 6 — Paper A (Benchmark): `.claude-plans/phase6-paper-a-benchmark.md`

**Add to `sec/2_related.tex`:**
```
\paragraph{Knowledge Graphs vs.\ Language Models.}
```
- Discuss Pan et al. 2024 survey (unifying LLMs and KGs)
- Position: KGs and LLMs are complementary, not competing
- KGs: structured multi-hop reasoning, verifiable, efficient to update
- LLMs: flexible NL understanding, zero-shot, but opaque and expensive to update
- PrimeKG specifically comes from structured databases (not text mining), so information is inherently relational

**Add `\subsection{Multi-Hop Evaluation}` in `sec/4_experiments.tex`:**
- Define the multi-hop prediction task
- Present 7 biomedical path types with biological motivation
- Results table: all baselines + CMKL multi-hop MRR
- Key finding: methods with graph structure (R-GCN) outperform flat methods

**Add new table: `tables/multihop_results.tex`**

**Add to Analysis subsection:**
- 1-2 paragraphs on why graph structure matters based on multi-hop evidence
- RAG struggles with multi-hop (must retrieve both triples independently)

**Add new references:**
- Pan et al. 2024 — Unifying LLMs and KGs (IEEE TKDE)
- Yao et al. 2023 — LLMs for KG completion
- Zhang et al. 2023 — Making LLMs better on KG tasks

### Phase 7 — Paper B (Method): `.claude-plans/phase7-paper-b-method.md`

**Add to `sec/2_related.tex`:**
```
\paragraph{LLMs for Knowledge Graphs.}
```
- LLM-based KG completion approaches and their limitations
- Graph structure as inductive bias that LLMs lack

**Add `\subsection{Multi-Hop Reasoning Analysis}` in `sec/4_experiments.tex`:**
- CMKL's R-GCN (2 layers) captures multi-hop better than flat KGE baselines AND RAG
- Cross-modal attention further improves multi-hop (text/molecular features on intermediate entities)

**Add `\subsection{Why Graph Structure Matters}` in Analysis/Discussion:**
Three-pronged argument:
1. **PrimeKG is from structured databases, not text.** Drug-protein targets from DrugBank assays, PPIs from BioGRID experiments, GO annotations — this has no textual equivalent in PubMed literature. LLMs trained on text miss this curated experimental knowledge.
2. **Multi-hop results prove structural encoding helps.** R-GCN propagates neighborhood info; RAG treats triples independently. Empirical multi-hop MRR gap demonstrates this.
3. **Efficiency.** Parameters, update cost, memory, forgetting control comparison table.

**Add new table: `tables/efficiency_comparison.tex`:**

| Aspect | KGE | CMKL | LLM Fine-tune | RAG |
|--------|-----|------|---------------|-----|
| Parameters | ~6M | ~2M | 7-70B | 7B + index |
| Update cost | Min (1 GPU) | Min (1 GPU) | Hours (8+ GPUs) | Re-index |
| Memory (infer.) | <1GB | <2GB | 14-140GB | 14GB+ |
| Forgetting ctrl | EWC/Replay | Modal-EWC | Catastrophic | None |
| Multi-hop | Embedding | R-GCN 2-hop | Implicit | Manual chain |
| Explainability | Similarity | Path-traceable | Black-box | Retrieved docs |

---

## Files Summary

| Action | File | Est. Lines |
|--------|------|------------|
| Create | `src/evaluation/multihop.py` | ~250 |
| Modify | `src/evaluation/metrics.py` | +15 |
| Modify | `src/evaluation/visualization.py` | +40 |
| Modify | `scripts/run_baselines.py` | +40 |
| Modify | `scripts/run_cmkl.py` | +40 |
| Modify | `scripts/run_rag.py` | +30 |
| Create | `scripts/run_multihop_eval.py` | ~120 |
| Create | `slurm/run_multihop.sh` | ~20 |
| Modify | `slurm/submit_all.sh` | +5 |
| Update | `.claude-plans/phase6-paper-a-benchmark.md` | rewrite sections |
| Update | `.claude-plans/phase7-paper-b-method.md` | rewrite sections |

**Total new code: ~555 lines across 9 files + 2 plan updates**

---

## Verification Plan

1. **Smoke test multi-hop extraction:**
   ```bash
   python -c "from src.evaluation.multihop import *; ..."
   ```
   Verify path extraction produces >100 paths per type on local task data.

2. **Smoke test multi-hop evaluation:**
   ```bash
   python scripts/run_baselines.py --baseline naive_sequential --quick --eval-multihop
   ```
   Verify `multihop_MRR` appears in output JSON.

3. **Smoke test CMKL multi-hop:**
   ```bash
   python scripts/run_cmkl.py --quick --eval-multihop
   ```

4. **Verify paper plan updates** have the new sections and tables specified.

5. **On IBEX:** Re-run experiments with `--eval-multihop` flag to collect multi-hop results alongside standard LP/CL metrics.
