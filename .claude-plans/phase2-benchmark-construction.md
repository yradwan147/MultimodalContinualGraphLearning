# Phase 2: Temporal Biomedical KG Benchmark Construction
**Estimated time:** Weeks 3-8 | **PDF Section:** 4
**Depends on:** Phase 1 | **Blocks:** Phase 3, 4, 5

---

## Objectives
- Build temporal snapshots t0, t1 (and optionally t2) of PrimeKG
- Compute temporal diffs between snapshots
- Define continual learning task sequences
- Extract multimodal features
- Create train/val/test splits per task

---

## Step 2.1: Download PrimeKG t0 (June 2021)

**File:** `src/data/download.py`

```python
# Two options - implement both with fallback
# Option A: TDC (preferred - simpler)
# Option B: Harvard Dataverse direct download
```

### Expected output
DataFrame with columns: `[relation, display_relation, x_index, x_id, x_type, x_name, x_source, y_index, y_id, y_type, y_name, y_source]`

### Troubleshooting
- If Harvard Dataverse is down, use TDC as fallback
- If kg.csv is very large (~2GB), use chunked reading: `pd.read_csv('kg.csv', chunksize=100000)`
- Verify shape: ~8.1M rows, 12 columns

**Log to worklog:** Download method used, file size, row/column counts.

---

## Step 2.2: Rebuild PrimeKG t1 (July 2023 Sources)

**Requires manual steps:**
```bash
git clone https://github.com/mims-harvard/PrimeKG.git
cd PrimeKG
pip install -r requirements.txt
bash datasets/primary_data_resources.sh
cd datasets/processing_scripts/
python bgee.py             # -> anatomy_gene.csv
python ctd.py              # -> exposure_data.csv
python drugbank_drug_drug.py    # -> drug_drug.csv
python drugbank_drug_protein.py # -> drug_protein.csv
# ... run all 17 processing scripts listed in README
cd ../..
jupyter nbconvert --to notebook --execute build_graph.ipynb
# Produces: kg_raw.csv, kg_giant.csv, kg.csv
```

### Critical notes
- **DrugBank** requires free academic license at https://go.drugbank.com/
- **DisGeNET** requires registration at https://www.disgenet.org/
- Some URLs in scripts may have changed; check `datasets/primary_data_resources.sh`
- The July 2023 update notes that `datasets/feature_construction/` scripts may remain out-of-date

### Troubleshooting
- If a database URL is broken, search for current URL and update script
- If build_graph.ipynb fails, ensure all 17 processing scripts ran successfully
- If node/edge counts differ significantly, verify database versions match July 2023 releases

**Log to worklog:** Each processing script result, any URL fixes needed, final t1 node/edge counts.

---

## Step 2.3: Obtain PrimeKG-U (t2, Feb 2026) - Optional

Monitor CellAwareGNN paper's supplementary materials. If not publicly available, construct t2 by running PrimeKG build scripts with February 2026 database downloads.

**Log to worklog:** Source of t2, construction method, node/edge counts.

---

## Step 2.4: Compute Temporal Diffs

**File:** `src/data/temporal_diff.py`

Implement `compute_kg_diff(kg_old_path, kg_new_path)` that:
1. Creates triple identifiers: `x_id|relation|y_id`
2. Computes set differences for added/removed/persistent triples
3. Identifies emerged/disappeared entities
4. Identifies emerged relation types
5. Returns stats dict + the actual triple sets

### Expected output
Dictionary with: added_triples, removed_triples, persistent_triples, emerged_entities, disappeared_entities, emerged_relations counts.

Numbers should reflect realistic biomedical evolution (new drugs approved 2021-2023, new disease characterizations).

### Troubleshooting
- If added_triples is 0 or very small: same database versions were used for both builds
- If removed_triples is very large: database may have changed ID scheme - normalize IDs
- If entity IDs don't match: create mapping table using entity names as secondary keys

**Log to worklog:** Diff statistics for t0->t1 (and t1->t2 if applicable).

---

## Step 2.5: Define Task Sequences for Continual Learning

**File:** `src/data/task_sequence.py`

Implement `create_task_sequence(kg_t0, kg_t1, strategy='entity_type')` with three strategies:
- `entity_type`: Group new triples by entity type (drug, disease, gene/protein). **Recommended.**
- `relation_type`: Group by relation type
- `temporal`: Use publication dates for finer-grained splits

### Design decision
The `entity_type` strategy best mirrors real-world knowledge evolution - new drugs getting approved is a distinct "event" from new diseases being characterized.

**Log to worklog:** Strategy chosen, number of tasks created, triples per task.

---

## Step 2.6: Extract Multimodal Features

**File:** `src/data/features.py`

Implement `extract_multimodal_features(kg_path, drug_features_path, disease_features_path)`:
- Drug features: clinical descriptions from DrugBank (description, indication, pharmacodynamics, mechanism of action, toxicity, protein binding, metabolism, half-life, route of elimination)
- Disease features: MONDO definition, UMLS description, Mayo Clinic clinical features
- Molecular features (optional): RDKit Morgan fingerprints from SMILES strings
- Text embeddings (optional): BiomedBERT (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`)

### Notes
- `engineer_features.ipynb` and `mapping_mayo.ipynb` in PrimeKG repo handle feature extraction
- IBM Research built a Multimodal KG from PrimeKG with 13 modalities

**Log to worklog:** Feature types extracted, dimensionalities, coverage (% of nodes with each feature type).

---

## Step 2.7: Create Train/Val/Test Splits per Task

**File:** `src/data/splits.py`

Implement `create_splits_per_task(tasks, val_ratio=0.1, test_ratio=0.2, seed=42)`:
- Training set: for learning the current task
- Validation set: for hyperparameter tuning and early stopping
- Test set: for evaluating on current AND all previous tasks

### Critical constraints
- Tasks with < 100 triples: merge with related task or discard
- Test sets for old tasks must be preserved across all future tasks
- **No data leakage:** test triples from task i should NOT appear in training sets of tasks i+1, i+2, ...

**Log to worklog:** Split sizes per task, any tasks merged/discarded, leakage verification results.

---

## Step 2.8: Save Benchmark to Disk

Save the complete benchmark:
```
data/benchmark/
├── snapshots/
│   ├── kg_t0.csv
│   ├── kg_t1.csv
│   └── kg_t2.csv (optional)
├── diffs/
│   ├── diff_t0_t1.json
│   └── diff_t1_t2.json (optional)
├── tasks/
│   ├── task_0/
│   │   ├── train.txt
│   │   ├── valid.txt
│   │   └── test.txt
│   ├── task_1/
│   │   └── ...
│   └── task_N/
│       └── ...
├── features/
│   ├── drug_features.csv
│   ├── disease_features.csv
│   └── text_embeddings.pt (optional)
├── statistics.json
└── README.md
```

Also create `notebooks/02_benchmark_stats.ipynb` with visualizations of the benchmark.

**Log to worklog:** Final benchmark statistics, directory contents, total disk size.

---

## Completion Criteria
- [ ] PrimeKG t0 and t1 downloaded/built
- [ ] Temporal diffs computed with meaningful added/removed counts
- [ ] Task sequences defined (at least 3-5 tasks)
- [ ] Multimodal features extracted for drug and disease nodes
- [ ] Train/val/test splits created with no leakage
- [ ] Benchmark saved to `data/benchmark/`
- [ ] `notebooks/02_benchmark_stats.ipynb` shows benchmark visualizations
- [ ] All activities logged in `worklog.md`
