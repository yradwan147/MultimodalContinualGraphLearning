# Phase 3: Baseline Implementation
**Estimated time:** Weeks 8-12 | **PDF Sections:** 5, 7.2
**Depends on:** Phase 2 (benchmark ready) | **Blocks:** Phase 5

---

## Objectives
- Implement 6 baseline methods for continual KG learning
- Implement the evaluation protocol (metrics computation)
- Validate each baseline produces expected behavior on the benchmark

---

## Step 3.1: Implement Evaluation Metrics

**File:** `src/evaluation/metrics.py`

### Link Prediction Metrics
- **MRR** (Mean Reciprocal Rank): Average of reciprocal rank of correct entity
- **Hits@K** (K=1, 3, 10): Proportion where correct entity is in top K
- **AUPRC**: Area Under Precision-Recall Curve (for drug repurposing)

### Continual Learning Metrics
Implement `evaluate_continual_learning(results_matrix, task_names)`:
- **Average Performance (AP):** `mean(R[-1, :])`
- **Average Forgetting (AF):** `mean(max_perf_j - final_perf_j)` for j < n-1
- **Backward Transfer (BWT):** `mean(R[-1, j] - R[j, j])` for j < n-1
- **Forward Transfer (FWT):** `mean(R[j-1, j])` for j > 0
- **Remembering (REM):** `1 - |min(BWT, 0)|`

Where R[i][j] = performance on task j's test set after training on task i.

**Log to worklog:** Metrics implementation verified with synthetic test matrices.

---

## Step 3.2: Baseline 1 - Naive Sequential Training (Lower Bound)

**File:** `src/baselines/naive_sequential.py`

Train KGE model (TransE/ComplEx/DistMult/RotatE) sequentially on each task without any continual learning mechanism. Uses PyKEEN.

### Key implementation details
- Collect all entity and relation IDs upfront for consistent indexing across tasks
- Use `pykeen.triples.TriplesFactory.from_labeled_triples()` with shared entity/relation mappings
- Initialize each task from previous model parameters (warm-start)
- After each task, evaluate on ALL tasks seen so far

### Expected outcome
High performance on most recent task, severe degradation on older tasks. This establishes the maximum-forgetting reference point.

### Troubleshooting
- Low performance even on current task: increase `num_epochs`, try different model
- PyKEEN pipeline doesn't support warm-starting: manually save/load `model.state_dict()` and use custom training loops with `pykeen.models.TransE(...)` directly
- Entity IDs don't align: ensure `entity_to_id` mapping covers all entities across all snapshots

**Log to worklog:** Model used, epochs, per-task MRR values, results matrix, CL metrics (AP, AF, BWT, FWT).

---

## Step 3.3: Baseline 2 - Joint Training (Upper Bound)

**File:** `src/baselines/joint_training.py`

Train on all tasks simultaneously - the upper bound for performance.

### Key implementation details
- Concatenate all training data, concatenate all validation data
- Single training run with combined TriplesFactory
- Evaluate on each task's test set separately

### Expected outcome
Best possible per-task performance (no forgetting by definition). Gap between joint and naive sequential quantifies how much forgetting matters.

**Log to worklog:** Per-task MRR values, overall AP.

---

## Step 3.4: Baseline 3 - EWC for Knowledge Graph Embeddings

**File:** `src/baselines/ewc.py`

Implement `EWC_KGE` class:
1. `compute_fisher(dataloader)`: Compute diagonal of Fisher Information Matrix after each task
   - `F_k ≈ (1/N) * sum_i (∂L_i/∂θ_k)^2`
2. `ewc_loss()`: `L_ewc = (λ/2) * sum_k F_k * (θ_k - θ*_k)^2`
3. `train_task(dataloader, optimizer, num_epochs)`: Standard training + EWC penalty

### Hyperparameters
- `lambda_ewc`: Start with 10.0, sweep over [0.1, 1.0, 10.0, 100.0, 1000.0]
- Reference: EWC with λ=10 reduced forgetting from 12.62% to 6.85% on FB15k-237

### Troubleshooting
- Fisher diagonal all near zero: model may not have converged - train longer
- EWC penalty too large (loss explodes): reduce `lambda_ewc`
- Forgetting still severe: Fisher approximation may be too crude - increase samples

**Log to worklog:** Lambda sweep results, best lambda, per-task MRR, CL metrics for each lambda.

---

## Step 3.5: Baseline 4 - Experience Replay (BER-style)

**File:** `src/baselines/experience_replay.py`

Implement `ExperienceReplayKGE` class:
1. `select_exemplars(triples, task_id)`: Three selection strategies:
   - `random`: Uniform random
   - `relation_balanced`: Equal samples per relation type
   - `attention`: Model attention scores (BER-style, advanced)
2. `train_task(task_triples, task_id, optimizer, num_epochs, replay_ratio=0.3)`:
   - Combine current task triples with replay buffer samples
   - After training, add exemplars to buffer

### Hyperparameters
- `buffer_size_per_task`: 500 (sweep: [100, 250, 500, 1000, 2000])
- `replay_ratio`: 0.3 (fraction of each batch from replay buffer)
- `selection_strategy`: 'relation_balanced' recommended for biomedical KGs

### Troubleshooting
- Replay doesn't help: increase `buffer_size_per_task` or `replay_ratio`
- Overfit on old tasks: decrease `replay_ratio`
- For heterogeneous relation types: `relation_balanced` may outperform `random`

**Log to worklog:** Strategy comparison, buffer size sweep results, best config, per-task MRR, CL metrics.

---

## Step 3.6: Baseline 5 - LKGE (Lifelong KGE)

**File:** `src/baselines/lkge.py`

LKGE is an existing open-source framework.

```bash
git clone https://github.com/nju-websoft/LKGE.git
cd LKGE
pip install -r requirements.txt
```

### Data format conversion
Implement `convert_to_lkge_format(task_sequence, output_dir)`:
- Each snapshot as a separate folder with `train.txt`, `valid.txt`, `test.txt`
- Each line: `head_entity\trelation\ttail_entity`

### Run command
```bash
python main.py -dataset PRIMEKG_TEMPORAL -gpu 0 -lifelong_name LKGE
```

**Log to worklog:** LKGE configuration used, any modifications needed, per-task results, CL metrics.

---

## Step 3.7: Baseline 6 - RAG-Based Biomedical Agent (for KGQA Task)

**File:** `src/baselines/rag_agent.py`

Implement `BiomedicalRAGAgent` class for Task 2 (KGQA):
1. `index_kg_snapshot(kg_triples, node_features)`: Convert triples to natural language, index in ChromaDB
2. `update_with_new_knowledge(new_triples, new_features)`: Add new documents to vector store
3. `answer_question(question, k=10)`: Retrieve relevant context, feed to LLM

### Key design decisions
- Embedding model: `pritamdeka/S-PubMedBert-MS-MARCO` or `BAAI/bge-base-en-v1.5`
- LLM: `meta-llama/Meta-Llama-3-8B-Instruct` (local) or OpenAI API
- RAG naturally avoids catastrophic forgetting (just add to index)
- But may suffer from **retrieval drift** as index grows

### Troubleshooting
- Poor retrieval: try domain-specific embedding models
- LLM hallucinations: add verification step checking against KG triples
- Slow vector store: use approximate nearest neighbor search

**Log to worklog:** Embedding model, LLM used, indexing time per snapshot, QA accuracy, comparison with parametric methods.

---

## Step 3.8: Implement Statistical Testing

**File:** `src/evaluation/statistical.py`

```python
from scipy import stats

def compute_significance(results_a, results_b, alpha=0.05):
    """Paired t-test between two methods across random seeds."""
    t_stat, p_value = stats.ttest_rel(results_a, results_b)
    return {'t_statistic': t_stat, 'p_value': p_value, 'significant': p_value < alpha}
```

Run all experiments with **5 random seeds** (42, 123, 456, 789, 1024) and report mean +/- std.

**Log to worklog:** Significance test results for all pairwise method comparisons.

---

## Step 3.9: Results Visualization

**File:** `src/evaluation/visualization.py`

Implement plotting utilities for:
- Results matrix heatmaps (R[i][j] per method)
- Bar charts comparing AP, AF, BWT, FWT across methods
- Forgetting curves over task sequence
- Per-relation-type performance breakdown

Save all plots to `results/figures/`.

**Log to worklog:** Figures generated, key visual takeaways.

---

## Completion Criteria
- [ ] All 6 baselines implemented and producing results
- [ ] Evaluation metrics verified with known-correct test cases
- [ ] Each baseline run with 5 random seeds
- [ ] Results logged in `results/` directory
- [ ] Statistical significance computed
- [ ] Visualization plots generated
- [ ] `docs/experiment-results.md` updated with baseline comparison table
- [ ] All activities logged in `worklog.md`
