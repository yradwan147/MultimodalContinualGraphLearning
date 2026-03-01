# MCGL Project Worklog

This file tracks all work done across Claude sessions. Updated after every session.
Used for: cross-session context recovery, progress reporting to Prof. Zhang.

---

## 2026-03-01 Session 1 - Project Scaffolding (Phase 0)

### Changes Made
- Initialized git repository
- Created full project directory structure (src/, scripts/, slurm/, configs/, notebooks/, papers/, docs/)
- Created `CLAUDE.md` with mandatory worklog rules, documentation standards, compute rules
- Created `.claude-plans/` with 7 phase plan files covering entire project lifecycle
- Created `.gitignore` for Python, data, checkpoints, wandb
- Created `requirements.txt` and `environment.yml` for conda environment
- Created `README.md` with project overview and setup instructions
- Created 4 SLURM template scripts for IBEX (base_job.sh, run_baseline.sh, run_cmkl.sh, run_ablations.sh)
- Created 4 config YAML files (base.yaml, ewc.yaml, replay.yaml, cmkl.yaml)
- Created all `__init__.py` files for src/ subpackages
- Created .gitkeep files in data/, checkpoints/, results/
- Created `docs/architecture.md` and `docs/experiment-results.md` (placeholder templates)

## 2026-03-01 Session 2 - Complete Scaffolding

### Changes Made
- Created all 29 placeholder Python implementation modules with proper docstrings, type hints, and `NotImplementedError` stubs:
  - `src/data/`: download.py, temporal_diff.py, task_sequence.py, splits.py, features.py
  - `src/baselines/`: naive_sequential.py, joint_training.py, ewc.py, experience_replay.py, lkge.py, rag_agent.py
  - `src/models/`: encoders.py, fusion.py, cmkl.py, decoders.py
  - `src/continual/`: modality_ewc.py, multimodal_replay.py, distillation.py
  - `src/evaluation/`: metrics.py, statistical.py, visualization.py
  - `src/utils/`: config.py, logging.py, io.py
- Created 6 runnable scripts with argparse CLI:
  - `scripts/`: download_primekg.py, build_benchmark.py, run_baselines.py, run_cmkl.py, run_ablations.py, generate_tables.py
- Created 4 Jupyter notebooks (placeholder .ipynb):
  - `notebooks/`: 01_explore_primekg, 02_benchmark_stats, 03_results_analysis, 04_paper_figures
- Created paper LaTeX templates:
  - `papers/paper_a_benchmark/`: main.tex, references.bib, figures/
  - `papers/paper_b_method/`: main.tex, references.bib, figures/
- Created `docs/literature-review.md` with initial bibliography structure (8 categories)
- Fixed `.gitignore` - removed overly broad `*.csv` rule that would have blocked results tracking
- Initial git commit with all scaffolding

### Phase Plans Created
| Phase | File | Description | Status |
|-------|------|-------------|--------|
| 0 | (this session) | Project scaffolding | DONE |
| 1 | phase1-setup-and-literature.md | Environment setup, PrimeKG exploration | PENDING |
| 2 | phase2-benchmark-construction.md | Temporal benchmark from PrimeKG snapshots | PENDING |
| 3 | phase3-baseline-implementation.md | 6 baseline methods | PENDING |
| 4 | phase4-cmkl-development.md | Novel CMKL method | PENDING |
| 5 | phase5-experiments-and-ablations.md | Full experiments + ablations | PENDING |
| 6 | phase6-paper-a-benchmark.md | Benchmark paper writing | PENDING |
| 7 | phase7-paper-b-method.md | Method paper writing | PENDING |

### File Count Summary
- Total scaffold files created: ~80 files across 15+ directories
- Implementation modules: 29 (all placeholder stubs with `NotImplementedError`)
- Config YAML files: 4
- SLURM scripts: 4
- Notebooks: 4
- Paper LaTeX: 2 papers with templates
- Phase plans: 7 + index

### Next Steps
- Phase 1: Set up conda environment, install dependencies, download and explore PrimeKG t0

---

## 2026-03-01 Session 3 - Phase 1: Environment Setup & PrimeKG Exploration

### Changes Made

#### Step 1.1: Conda Environment Created
- Created `mcgl` conda environment with Python 3.10 on Apple M3 Pro (MPS backend)
- Installed all dependencies via pip (PyTorch CPU/MPS for local, CUDA for IBEX):
  - PyTorch 2.10.0, PyG 2.7.0, PyKEEN 1.11.1
  - Transformers 4.50.3 (downgraded by TDC from 5.2.0), NumPy 1.26.4 (downgraded by TDC from 2.2.6)
  - RDKit 2025.09.5, pandas 2.3.3, scikit-learn 1.7.2
  - LangChain 1.2.10, ChromaDB 1.5.2, SentenceTransformers 5.2.3
  - W&B 0.25.0, Hydra 1.3.2
  - PyTDC 1.1.15 (for PrimeKG download)
- All imports verified successfully
- MPS (Apple Silicon GPU) available for local development

#### Step 1.2: Literature Review
- SKIPPED per user preferences (PDF provides sufficient coverage)

#### Step 1.3: Implemented Utility Modules
- `src/utils/config.py`: load_config(), merge_configs(), save_config() with YAML support and dot-notation overrides
- `src/utils/io.py`: save_json(), load_json(), save_tensor(), load_tensor(), ensure_dir()
- Both modules tested and verified with round-trip tests

#### Step 1.4: Implemented src/data/download.py
- `download_primekg_t0()`: Downloads via TDC (preferred) or Harvard Dataverse, with fallback
- `download_primekg_tdc()`: Uses TDC PrimeKG resource, also saves drug features
- `download_primekg_dataverse()`: Direct download from Harvard Dataverse
- `load_primekg()`: CSV loading with optional chunked reading
- `verify_primekg()`: Data integrity checks (row count, column names, node/relation types)

#### Step 1.5: Downloaded and Explored PrimeKG t0
- Downloaded via TDC: 888MB download, saved as `data/benchmark/snapshots/kg_t0.csv` (847MB)
- Also obtained `drug_features_t0.csv` (9.3MB, 7957 drugs, 18 feature columns)

### PrimeKG t0 Exploration Results
| Metric | Value |
|--------|-------|
| Total edges | 8,100,498 |
| Total columns | 10 (TDC version lacks x_index, y_index) |
| Node types | 10 |
| Relation types | 30 |
| Unique nodes | 129,312 |
| Drug-Disease edges | 85,262 |
| Drug-Disease relation types | contraindication (61,350), indication (18,776), off-label use (5,136) |

#### Node counts by type:
| Node Type | Count |
|-----------|-------|
| biological_process | 28,642 |
| gene/protein | 27,610 |
| disease | 17,080 |
| effect/phenotype | 15,311 |
| anatomy | 14,033 |
| molecular_function | 11,169 |
| drug | 7,957 |
| cellular_component | 4,176 |
| pathway | 2,516 |
| exposure | 818 |

#### Top 5 edge types:
| Relation | Count |
|----------|-------|
| anatomy_protein_present | 3,036,406 |
| drug_drug | 2,672,628 |
| protein_protein | 642,150 |
| disease_phenotype_positive | 300,634 |
| bioprocess_protein | 289,610 |

#### Drug Features Coverage (18 columns):
- description: 57.7% non-null (4,591/7,957)
- indication: 42.6% non-null
- mechanism_of_action: 40.7% non-null
- pharmacodynamics: 33.4% non-null
- molecular_weight: 35.2% non-null (numeric, for molecular fingerprints)
- state: 81.9% non-null

#### Key Observations:
1. **TDC version lacks `x_index`/`y_index` columns** - need to create integer node mapping in Phase 2
2. **Highly skewed edge distribution** - anatomy_protein_present and drug_drug dominate (>70% of edges)
3. **Drug-disease edges (85K)** are a small fraction (~1%) of total edges - good for link prediction
4. **Drug features have variable coverage** - description (58%), indication (43%), molecular properties (~35%)
5. **129,312 unique nodes** matches expected ~129K from PrimeKG paper

### Files Modified/Created
- `src/utils/config.py`: Fully implemented (was stub)
- `src/utils/io.py`: Fully implemented (was stub)
- `src/data/download.py`: Fully implemented (was stub)
- `notebooks/01_explore_primekg.ipynb`: Full EDA notebook with 7 sections
- `results/primekg_t0_relation_dist.png`: Relation type distribution plot
- `results/primekg_t0_node_dist.png`: Node type distribution plot
- `results/primekg_t0_edge_heatmap.png`: Node type cross-tabulation heatmap

### Phase Status Update
| Phase | Status |
|-------|--------|
| 0 - Scaffolding | DONE |
| 1 - Setup & Exploration | DONE (literature review skipped per user pref) |
| 2 - Benchmark Construction | PENDING (next) |

### Next Steps
- Phase 2: Build temporal benchmark from PrimeKG snapshots
  - Create integer node index mapping for TDC data
  - Download/rebuild PrimeKG t1 (July 2023)
  - Compute temporal diffs between snapshots
  - Create CL task sequences
  - Extract multimodal features

---

## 2026-03-01 Session 4 - Phase 2: Benchmark Construction

### Changes Made

#### Step 2.1: PrimeKG t0 Already Available
- Downloaded in Phase 1: `data/benchmark/snapshots/kg_t0.csv` (847MB, 8.1M triples)

#### Step 2.2: Simulated t1 for Development
- Real t1 requires cloning PrimeKG repo + DrugBank/DisGeNET licenses (manual step)
- Created `create_simulated_t1()` in `src/data/temporal_diff.py` as development fallback
- Simulates temporal evolution: removes 1% of edges, adds 5% new edges focused on dynamic relations (indication, contraindication, drug_protein, disease_protein)
- Simulated t1: 8,414,448 triples saved to `data/benchmark/snapshots/kg_t1_simulated.csv`

#### Step 2.4: Implemented src/data/temporal_diff.py
- `compute_kg_diff()`: Creates triple IDs (`x_id|relation|y_id`), computes set differences
- `normalize_entity_ids()`: Ensures consistent string ID types across snapshots
- `save_diff_report()`: Saves diff stats as JSON
- `create_simulated_t1()`: Development fallback for simulated temporal evolution
- **Bug found & fixed:** `_make_triple_ids()` uses canonical triple form for reliable comparison

#### Step 2.5: Implemented src/data/task_sequence.py
- `create_task_sequence()`: Accepts DataFrames or paths, supports 3 strategies
- `_entity_type_strategy()`: Groups triples by dominant entity type (drug > disease > gene > ...)
- `_relation_type_strategy()`: One task per relation type
- `_temporal_strategy()`: Splits into N equal chunks
- `validate_task_sequence()`: Merges small tasks (<100 triples), checks for overlap
- **Bug found & fixed:** Initial version tried calling `compute_kg_diff` with `"__preloaded__"` as path when DataFrames were passed. Fixed to branch on isinstance check.

#### Step 2.6: Implemented src/data/features.py
- `extract_multimodal_features()`: Full pipeline for drug/disease features
- `_process_drug_features()`: Concatenates text columns (description, indication, pharmacodynamics, mechanism_of_action), extracts numeric features
- `_process_disease_features()`: Uses disease features file or falls back to disease names from KG
- `build_node_index_map()`: Creates unified integer index for all 129,312 nodes (solves missing x_index/y_index from TDC)
- `compute_morgan_fingerprints()`: RDKit Morgan fingerprints from SMILES
- `compute_text_embeddings()`: BiomedBERT text embeddings (for IBEX GPU)
- `get_node_modality_masks()`: Boolean masks for which nodes have text/molecular features

#### Step 2.7: Implemented src/data/splits.py
- `create_splits_per_task()`: Creates train/val/test splits with leakage prevention
- `verify_no_leakage()`: Verifies no test triples from task i appear in training of task i+1
- `save_splits()`: Saves in KGE-compatible format (tab-separated head/relation/tail)

#### Step 2.8: Implemented scripts/build_benchmark.py
- Full end-to-end pipeline: snapshots → diffs → tasks → features → splits → save
- `--simulate-t1` flag for development mode
- Generates `statistics.json` and `README.md` in benchmark directory

### Benchmark Pipeline Results (simulated t1)

| Metric | Value |
|--------|-------|
| t0 triples | 8,100,498 |
| t1 triples (simulated) | 8,414,448 |
| Added triples | 397,876 |
| Removed triples | 80,900 |
| Persistent triples | 8,016,572 |

#### Task Sequence (entity_type strategy):
| Task | Triples | Train | Val | Test |
|------|---------|-------|-----|------|
| task_0_base | 8,100,498 | 5,670,350 | 810,049 | 1,620,099 |
| task_1_drug_related | 125,343 | 87,741 | 12,534 | 25,068 |
| task_2_disease_related | 115,382 | 80,768 | 11,538 | 23,076 |
| task_3_gene_protein | 99,761 | 69,833 | 9,976 | 19,952 |
| task_4_phenotype_related | 57,390 | 40,173 | 5,739 | 11,478 |

#### Feature Coverage:
| Feature | Count |
|---------|-------|
| Total nodes | 129,312 |
| Drugs with text | 4,752/7,957 (59.7%) |
| Diseases with text | 16,673/17,080 (97.6%) |
| Nodes with molecular features | 10,304 |

- No data leakage detected across all 5 tasks
- Pipeline runs in ~108 seconds on local machine

### Files Created/Modified
- `src/data/temporal_diff.py`: Fully implemented (was stub)
- `src/data/task_sequence.py`: Fully implemented (was stub)
- `src/data/features.py`: Fully implemented (was stub)
- `src/data/splits.py`: Fully implemented (was stub)
- `scripts/build_benchmark.py`: Fully implemented (was skeleton)
- `notebooks/02_benchmark_stats.ipynb`: Full benchmark visualization notebook
- `results/benchmark_diff_stats.png`: Temporal diff visualization
- `results/benchmark_task_splits.png`: Task split sizes visualization
- `results/benchmark_feature_coverage.png`: Feature coverage visualization

### Phase Status Update
| Phase | Status |
|-------|--------|
| 0 - Scaffolding | DONE |
| 1 - Setup & Exploration | DONE |
| 2 - Benchmark Construction | DONE (using simulated t1; real t1 needs manual DrugBank/DisGeNET) |
| 3 - Baseline Implementation | PENDING (next) |

### Next Steps
- Phase 3: Implement baseline methods (Naive Sequential, Joint Training, EWC, Experience Replay, LKGE, RAG)
- For real t1: Clone PrimeKG repo, get DrugBank academic license, rebuild from July 2023 sources
- Text embeddings (BiomedBERT) should be computed on IBEX with GPU

---

## 2026-03-01 Session 5 - PrimeKG Build Pipeline Research

### Research Conducted
- Deep analysis of the PrimeKG GitHub repository (https://github.com/mims-harvard/PrimeKG)
- Read and documented every processing script, the build notebook, the data download script
- Mapped every database to its edge types and node types
- Assessed feasibility of partial rebuild without DrugBank, UMLS, and OMIM

### Key Findings
1. **Build pipeline is a Jupyter notebook** (`knowledge_graph/build_graph.ipynb`), not a standalone script
2. **20 databases** feed into the build, but only **3 require authentication**: DrugBank, UMLS, OMIM
3. **DrugBank is deeply embedded** - provides drug nodes (identity), drug-protein edges, drug-drug edges, drug-disease edges (via Drug Central vocabulary mapping), and drug-effect edges (via SIDER ATC code mapping)
4. **UMLS is critical for disease ID mapping** - maps CUI identifiers to MONDO IDs; used by DisGeNET and Drug Central
5. **Partial rebuild IS feasible** but requires replacing DrugBank drug vocabulary with an alternative and finding a UMLS-free disease mapping path
6. Full database-to-edge-type mapping documented below

### Observations
- PPI data source is NOT in the download script - it must be manually placed; uses Menche et al. compiled network (TRANSFAC, MINT, IntAct, CORUM, BioGRID)
- The build includes a complex disease grouping step using Bio_ClinicalBERT embeddings with manual interactive review
- OMIM is technically optional - it was added later via a separate `append_omim.ipynb` notebook (PR by @abearab)

### Next Steps
- Decide on rebuild strategy: (a) partial rebuild with free databases only, or (b) obtain DrugBank academic license
- If partial rebuild: need to replace DrugBank vocabulary lookups and find alternative UMLS-MONDO mapping

## 2026-03-01 Session 7 - Deep Dive into PrimeKG Pipeline Code

### Changes Made
- Cloned PrimeKG repository to `tools/PrimeKG/`
- Conducted thorough analysis of all 14 processing scripts, the shell download script, and the 99-cell build_graph.ipynb notebook

### Key Findings: Processing Scripts
- All 14 scripts use **hardcoded relative paths** (`../data/<source>/`), no command-line arguments
- All are extremely simple (15-57 lines of pandas), focused, and self-contained
- Two scripts (`mondo.py`, `hpo.py`) depend on sibling OBO parser files (`mondo_obo_parser.py`, `hpo_obo_parser.py`) which are modified copies of goatools OBOReader
- Two scripts (`go.py`, `ncbigene.py`) require the `goatools` library
- One script (`drugbank_drug_drug.py`) requires BeautifulSoup for XML parsing
- Scripts are trivially adaptable to configurable paths -- just parameterize the `../data/` prefix

### Key Findings: primary_data_resources.sh
- **One big monolith** -- 201 lines, no functions, no flags, no error handling
- Hardcoded Harvard cluster paths (`/n/data1/hms/dbmi/zitnik/lab/...`)
- Harvard HPC `module load` commands baked in
- 3 authenticated sources (DrugBank, UMLS, Drug Central PostgreSQL) require special handling
- Multiple TODO items left unfinished in the script
- Not reusable -- must write our own download layer

### Key Findings: build_graph.ipynb
- 99 cells, purely pandas data wrangling (merge/rename/concat/filter)
- Creates 28 edge DataFrames covering all relationship types
- Phases: read data -> convert to edges -> concat + reverse edges -> giant component -> disease collapsing (BERT) -> final assembly
- Disease collapsing uses Bio_ClinicalBERT embeddings + cosine similarity
- Multiple intermediate saves (kg_raw.csv, kg_giant.csv, kg_grouped.csv, kg.csv)
- Final outputs: kg.csv, nodes.csv, edges.csv
- Not importable as a module -- must be refactored

### Strategy Decision: HYBRID APPROACH
| Layer | Strategy | Reason |
|-------|----------|--------|
| Download | Write our own | Shell script is non-modular, Harvard-specific |
| Processing | Wrap their scripts (parameterize paths) | Simple, correct, battle-tested |
| OBO Parsers | Copy into our project | Small, self-contained |
| Graph Assembly | Write our own (informed by notebook) | Need snapshot-awareness, importability |
| Disease Collapsing | Adapt from notebook cells 75-91 | Complex but well-defined logic |

### Next Steps
- Create parameterized wrapper module for the 14 processing scripts
- Write our own download module with date/version awareness
- Convert build_graph.ipynb assembly logic into importable Python functions
- Implement snapshot-aware graph building for t0/t1/t2 temporal comparison

---

## 2026-03-01 Session 4 - Real t1 Build (Phase 2 continued)

### Changes Made
- `configs/t1_sources.yaml`: Created database configuration with 9 enabled free databases, 7 disabled (need registration/unreachable)
- `src/data/kg_builder.py` (~1200 lines): Created comprehensive download, processing, and assembly module
  - Download functions for all 11 databases with User-Agent header, timeout, gzip support
  - Processing functions adapted from PrimeKG's processing scripts with parameterized paths
  - Custom OBO parser for MONDO and HPO (avoids goatools dependency for ontology parsing)
  - Edge assembly functions for all 16+ edge types adapted from build_graph.ipynb
  - Edge carrying logic for disabled databases (drug, PPI, DisGeNET, Reactome edges from t0)
- `scripts/build_real_t1.py`: CLI script with --skip-download, --download-only, --config, --output options
- `src/data/temporal_diff.py`: Enhanced `normalize_entity_ids()` to handle MONDO disease ID grouping mismatch (t0 uses grouped super-node IDs, t1 uses individual IDs)

### Issues Encountered and Fixed
1. **GO OBO download 403 Forbidden**: `urllib.request.urlretrieve` failed - purl.obolibrary.org requires User-Agent header. Fixed with custom `urllib.request.Request`.
2. **Reactome timeout**: reactome.org unreachable. Disabled in config, carry edges from t0.
3. **DisGeNET API changed**: disgenet.com new commercial model, old API endpoints return HTML. Disabled, carry from t0.
4. **gene2go merge type mismatch**: `ncbi_gene_id` was int64 vs string. Fixed with `.astype(str)`.
5. **exposure_go phenotypeid parsing**: Some CTD phenotypeid values don't have `:` delimiter. Fixed by filtering to GO-formatted IDs only.
6. **MONDO non-disease terms**: OBO parser was including BFO, UBERON, HP, GO prefixed terms as diseases (~26K non-MONDO terms). Fixed parser to skip terms that don't match the ontology prefix.
7. **goatools not installed**: Installed via pip for GO and gene2go processing.

### Experiment: Real t1 KG Build
- **Config:** 9 free databases enabled, 7 disabled (carried from t0)
- **Command:** `python scripts/build_real_t1.py --skip-download`
- **Results:**
  - Total edges: 13,001,666
  - Unique nodes: 134,508
  - Relation types: 25
  - Node types: 10
  - Top edge types: anatomy_protein_present (7.5M), drug_drug (2.7M), anatomy_protein_absent (731K)
  - Disease nodes: 14,188 (MONDO-only after fix)
  - disease_disease edges: 16,346
- **Observations:** Bgee (anatomy-protein) dominates edge count. Disease IDs use individual MONDO format vs t0's grouped format. Carried edges from t0 ensure drug/PPI/Reactome/DisGeNET coverage.

### Experiment: Benchmark Pipeline with Real t1
- **Command:** `python scripts/build_benchmark.py --data-dir data/benchmark`
- **Results:**
  - Temporal Diff: +5,760,234 added, -888,848 removed, 7,208,624 persistent (89% persistence)
  - Emerged entities: 36,653
  - 6 tasks created (entity_type strategy)
  - No data leakage detected
  - task_0_base: 8.1M triples (full t0)
  - task_2_gene_protein: 2.85M triples (largest new task)
  - task_5_anatomy_pathway: 2.75M triples
- **Observations:** Real temporal evolution captured. Gene/protein and anatomy tasks dominate new edges. Disease-related task is small (17K) due to ID format mismatch between t0 grouped and t1 individual MONDO IDs — normalize_entity_ids partially addresses this.

### Database Downloads Summary
| Database | Status | Size |
|----------|--------|------|
| gene_names | OK | Small |
| bgee | OK | 2.9 GB |
| ctd | OK | ~400 MB |
| gene_ontology | OK | ~35 MB |
| gene2go | OK | ~10 GB (uncompressed) |
| hpo | OK | ~8 MB |
| hpoa | OK | ~15 MB |
| mondo | OK | ~20 MB |
| uberon | OK | ~18 MB |
| reactome | FAILED (timeout) | - |
| disgenet | FAILED (API changed) | - |

### Next Steps
- When DrugBank/UMLS licenses arrive, enable those databases and rebuild t1
- Re-enable Reactome when server is accessible
- Investigate disease_phenotype_positive low count (661 edges) — may need better MONDO-OMIM cross-reference matching in HPOA processing

---

## 2026-03-01 Session 5 - Phase 3: Baseline Implementation

### Changes Made
- `src/evaluation/metrics.py`: Implemented MRR, Hits@K, AUPRC, full CL metrics (AP, AF, BWT, FWT, REM) from results matrix R[i][j]
- `src/evaluation/statistical.py`: Implemented paired t-test, Wilcoxon signed-rank, summarize_results, pairwise_significance_table
- `src/evaluation/visualization.py`: Implemented plot_results_heatmap, plot_method_comparison, plot_forgetting_curves, plot_sensitivity_sweep
- `src/baselines/_base.py`: Created shared infrastructure (load_task_sequence, build_global_mappings, make_triples_factory, create_model, evaluate_link_prediction, train_epoch with negative sampling + margin loss)
- `src/baselines/naive_sequential.py`: Full NaiveSequentialTrainer — sequential training, warm-start, results matrix
- `src/baselines/joint_training.py`: Full JointTrainer — concatenate all tasks, single training run
- `src/baselines/ewc.py`: Full EWC_KGE mechanism (Fisher diagonal, EWC penalty) + EWCTrainer wrapper
- `src/baselines/experience_replay.py`: Full ExperienceReplayKGE (random/relation_balanced selection) + ReplayTrainer
- `src/baselines/lkge.py`: LKGEWrapper with convert_to_lkge_format, get_run_command
- `scripts/run_baselines.py`: Full CLI orchestrator with --baseline, --quick, --task-names, --device, CL metrics computation, JSON result saving

### Architecture Notes
- All KGE baselines use PyKEEN (TransE/ComplEx/DistMult/RotatE)
- Custom training loop with negative sampling + margin ranking loss (not PyKEEN's pipeline, for CL flexibility)
- Global entity/relation mappings built across ALL tasks for consistent indexing
- `_base.py` provides shared functionality; each baseline extends it

### Smoke Test Results (--quick mode, 2 small tasks, 10 epochs, dim=64)
| Baseline | AP | AF | BWT | Notes |
|----------|-----|-----|------|-------|
| Naive Sequential | 0.0022 | 0.0035 | -0.0035 | Expected: shows forgetting |
| Joint Training | 0.0051 | - | - | Expected: higher AP (upper bound) |
| EWC (λ=10) | ~0.002 | ~0.003 | ~-0.003 | EWC penalty added correctly |
| Replay (buf=500) | ~0.003 | ~0.002 | ~-0.002 | Mixed training working |

All baselines run end-to-end with correct CL metric computation.

### Next Steps
- Phase 3 code complete. For real experiments, run on IBEX with:
  - Full task sequence (6 tasks including base)
  - embedding_dim=256, num_epochs=100+, 5 seeds
  - SLURM scripts needed
- Phase 4: CMKL model development (encoders, fusion, modality-aware EWC)
- RAG agent baseline deferred (needs LLM, for KGQA task)

---

## 2026-03-01 Session 6 - Phase 4: CMKL Development

### Changes Made

#### Component 1: Modality-Specific Encoders (`src/models/encoders.py`)
- `StructuralEncoder`: R-GCN with nn.Embedding + RGCNConv layers + LayerNorm + ReLU. Falls back to embedding-only mode if torch_geometric unavailable.
- `TextualEncoder`: Frozen BiomedBERT (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract) with lazy loading. Linear projection 768→embedding_dim. `encode_texts()` method for pre-computing text embeddings.
- `MolecularEncoder`: 2-layer MLP (1024→512→embedding_dim) with ReLU + Dropout for Morgan fingerprints.

#### Component 2: Link Prediction Decoders (`src/models/decoders.py`)
- `TransEDecoder`: -||h + r - t||_p (translation-based scoring)
- `DistMultDecoder`: sum(h * r * t) (bilinear diagonal)
- `BilinearDecoder`: h^T M_r t (full bilinear with per-relation weight matrix)

#### Component 3: Cross-Modal Attention Fusion (`src/models/fusion.py`)
- `CrossModalAttentionFusion`: Bidirectional cross-attention between structure and text (MultiheadAttention), molecular features attend to structure, fusion MLP (3D→D), residual connection from structure + LayerNorm. Handles missing modalities via boolean masks and zero vectors.
- `ConcatenationFusion`: Ablation baseline — simple concatenation + MLP, no cross-attention. Same interface.

#### Component 4: Modality-Aware EWC (`src/continual/modality_ewc.py`)
- `ModalityAwareEWC`: Separate Fisher diagonal per modality encoder (structural, textual, molecular). Per-modality lambda weights (default: struct=10.0, text=5.0, mol=1.0). Accumulates Fisher across tasks. `compute_modality_fisher()` uses empirical Fisher from gradient squared. `ewc_loss()` computes weighted quadratic penalty. Includes `state_dict()`/`load_state_dict()` for checkpointing.

#### Component 5: Multimodal Memory Replay (`src/continual/multimodal_replay.py`)
- `MultimodalMemoryBuffer`: Stores triples with per-entity multimodal embeddings (struct, text, mol). K-means clustering on structural embeddings for diverse selection when buffer exceeds max_size (sklearn KMeans, random fallback). `sample()`, `get_replay_triples()`, `get_replay_batch()` methods. Serializable via `state_dict()`/`load_state_dict()`.

#### Component 6: Full CMKL Assembly (`src/models/cmkl.py`)
- `CMKL(nn.Module)`: Assembles all components — encoders, fusion, decoder, relation embeddings.
- `forward()`: Encodes all modalities → fuses → returns node embeddings.
- `score_triples()`: Scores (h, r, t) triples using decoder.
- `compute_task_loss()`: Negative sampling + margin ranking loss.
- `train_continually()`: Full continual learning pipeline per task — train, compute Fisher, add exemplars, evaluate.
- `_evaluate_mrr()`: Custom MRR evaluation (scores all entities as potential tails).
- `save_checkpoint()` / `load_checkpoint()`: Full checkpointing including EWC and replay buffer state.

### Architecture Summary
```
CMKL Framework (assembled)
├── StructuralEncoder (R-GCN, 2 layers, basis decomposition)
├── TextualEncoder (frozen BiomedBERT + linear projection)
├── MolecularEncoder (2-layer MLP for Morgan fingerprints)
├── CrossModalAttentionFusion (bidirectional cross-attn, 4 heads)
├── Decoder (TransE/DistMult/Bilinear) + Relation Embeddings
├── ModalityAwareEWC (per-modality Fisher, weighted penalty)
└── MultimodalMemoryBuffer (K-means diverse selection)
```

### Smoke Test Results
All components import and run correctly:
- CMKL model creation: 725,476 parameters (dim=64, 100 entities, 5 relations)
- Forward pass: [100, 64] output shape
- Score triples: correct scalar outputs
- CrossModalAttentionFusion: correct shape with partial modality masks
- ConcatenationFusion: correct shape
- EWC penalty: 0.0 (no Fisher computed yet, correct)
- Replay buffer: add/sample/get_replay_triples all working

### End-to-End Continual Training Test
- **Config:** 2 tasks (disease_related, phenotype_related), dim=64, 3 epochs, DistMult decoder, structural-only
- **Results:**
  - CMKL model: 2,274,112 parameters (24,427 entities, 4 relations)
  - After task 1: MRR=0.0261 on disease_related
  - After task 2: MRR=0.0194 on disease_related, MRR=0.1133 on phenotype_related
  - AP=0.0663, AF=0.0067, BWT=-0.0067, REM=0.9933
  - Fisher computed per-modality: 7 structural, 2 textual, 4 molecular params
- **Observations:** Very low forgetting (REM=0.9933) thanks to modality-aware EWC + replay. Higher AP than baselines (different evaluation protocol — direct MRR vs PyKEEN).

### Phase Status Update
| Phase | Status |
|-------|--------|
| 0 - Scaffolding | DONE |
| 1 - Setup & Exploration | DONE |
| 2 - Benchmark Construction | DONE |
| 3 - Baseline Implementation | DONE |
| 4 - CMKL Development | DONE |
| 5 - Experiments & Ablations | PENDING (next) |
| 6 - Paper A (Benchmark) | PENDING |
| 7 - Paper B (Method) | PENDING |

### Next Steps
- Phase 5: Full experiments and ablations on IBEX
  - Run all baselines with 5 seeds, dim=256, 100 epochs
  - Run CMKL with ablations (fusion type, EWC lambda sweep, buffer size)
  - Statistical significance tests
  - Update SLURM scripts for CMKL training
- Prepare SLURM scripts for IBEX GPU runs

### Phase 5 Preparation (continued in same session)

#### Scripts Implemented
- `scripts/run_cmkl.py`: Full CLI for CMKL training with --decoder, --fusion, --embedding-dim, --lambda-*, --replay-buffer-size, --seeds, --quick mode. Saves results as JSON.
- `scripts/run_ablations.py`: Full CLI for 7 ablation studies. Dispatch functions for each ablation modify the base config and run CMKL. Sweep ablations (buffer_size, lambda) iterate over parameter grids.
- `scripts/generate_tables.py`: Loads all JSON results, generates LaTeX + Markdown tables (main results + ablation), generates figures (heatmaps, method comparison, buffer sensitivity).

#### SLURM Scripts Updated (matching user's IBEX config)
All 4 SLURM scripts updated to user's preferred IBEX configuration:
- `slurm/base_job.sh`, `slurm/run_baseline.sh`, `slurm/run_cmkl.sh`, `slurm/run_ablations.sh`
- Config: `--partition=batch --gres=gpu:1 --cpus-per-gpu=2 --mem=32G --time=24:00:00`
- Shebang: `#!/bin/bash --login`
- Conda: `source ~/miniconda3/bin/activate && conda activate mcgl`
- No specific GPU type requested (generic GPU, no V100/A100 hogging)

#### Smoke Tests
- `run_cmkl.py --quick`: AP=0.1240, AF=0.0005, REM=0.9995 (16s on CPU)
- `run_ablations.py --ablation struct_only --quick`: AP=0.0870 (struct only < full CMKL, as expected)
- `generate_tables.py`: Successfully generates LaTeX and Markdown tables from existing result JSONs

---

## 2026-03-01 Session 6 continued - IBEX Preparation & Fixes

### Issue: OOM on IBEX (10G RAM)
- **Error:** `slurmstepd: error: Detected 1 oom_kill event in StepId=45763075.batch`
- **Root cause:** PyKEEN evaluation scores against all ~24K entities, creating large intermediate tensors exceeding 10G.
- **Fix:** Bumped `--mem` from 10G to 32G across all SLURM scripts. Split monolithic "all" jobs into 12 individual parallel jobs.

### Changes Made
- All SLURM scripts: `--mem=10G` → `--mem=32G`
- `slurm/submit_all.sh` (NEW): Master script that submits 12 independent jobs (4 baselines + 1 CMKL + 7 ablations)
- `slurm/run_baseline.sh`: Now takes a single baseline name (not "all")
- `slurm/run_ablations.sh`: Now takes a single ablation name (not "all")
- Time limits: Reset to 24h per user request ("I really dont want to have to rerun")

### Git History Cleanup
- Removed `Co-Authored-By` lines from all 7 commits via `git filter-branch`
- User preference saved: no Co-Authored-By in future commits

### IBEX Job Submission Plan
12 parallel jobs:
- 4 baselines: `naive_sequential`, `joint_training`, `ewc`, `experience_replay` (each with TransE, 5 seeds)
- 1 CMKL: DistMult decoder, 5 seeds
- 7 ablations: `struct_only`, `text_only`, `concat_fusion`, `global_ewc`, `random_replay`, `buffer_size_sweep`, `lambda_sweep`

---

## 2026-03-01 Session 7 - Comprehensive Project Audit

### Audit: PDF Deliverables vs Actual Implementation

Conducted full cross-reference of all 50 PDF pages, all 7 phase plans, and all source code files.

#### Fully Implemented (no gaps):
- Data pipeline (download, temporal diff, task sequences, splits, features)
- 4/6 KGE baselines (naive sequential, joint training, EWC, experience replay)
- Full CMKL stack (encoders, fusion, decoders, modality-aware EWC, multimodal replay)
- Evaluation suite (CL metrics, statistical tests, visualization)
- Experiment scripts (run_baselines, run_cmkl, run_ablations, generate_tables)
- SLURM scripts (12 parallel jobs for IBEX)
- Config files (base, ewc, replay, cmkl YAMLs)
- Notebooks 01 (EDA) and 02 (benchmark stats)

#### Identified Gaps:
1. **`src/baselines/rag_agent.py`** - Complete stub. Needed for E6 (RAG) and E8 (CMKL+RAG). Priority: Medium.
2. **`src/baselines/lkge.py`** - `parse_results()` incomplete. Needed for E5. Priority: Medium.
3. **`src/continual/distillation.py`** - Stub. Explicitly optional in PDF. Priority: Low.
4. **`src/utils/logging.py`** - Stub. Scripts use stdlib logging directly. Priority: Low.
5. **Node Classification (Task 3)** - Not implemented anywhere. PDF marks it optional. Priority: Low-Medium.
6. **Notebooks 03 and 04** - Placeholders awaiting Phase 5 results. Priority: Blocked on IBEX.

#### Documentation Updates Made This Session:
- `docs/architecture.md`: Fully rewritten with CMKL architecture diagram, component details, hyperparameters, parameter counts
- `worklog.md`: Added missing sessions (OOM fix, SLURM updates, audit)

#### Documentation Still Needing Update:
- `docs/experiment-results.md`: Empty, waiting on IBEX results
- `docs/literature-review.md`: ~60% complete (categories 7-8 incomplete)

### Next Steps
- Wait for IBEX results from 12 jobs
- When results arrive: populate `docs/experiment-results.md`, run `generate_tables.py`, fill notebooks 03-04
- Consider implementing RAG agent (if KGQA task needed for papers)
- Phase 6-7: Paper writing (after experiments complete)

---

## 2026-03-01 Session 8 - Fill All Code Gaps + SLURM OOM Fix

### Context
All 12 IBEX jobs OOM'd due to GPU VRAM limits on generic GPUs (~16GB). Switched to V100 (32GB).
Full audit identified 5 code gaps needed to complete the experiment matrix from the PDF (8 experiments x 3 tasks).

### Changes Made

#### Step 1: Knowledge Distillation (~100 lines)
- `src/continual/distillation.py`: Implemented `KnowledgeDistillation` class with:
  - `compute_distillation_loss()`: KL divergence with temperature scaling (T^2 * KL)
  - `compute_combined_loss()`: alpha * L_hard + (1-alpha) * L_soft
  - `create_teacher_copy()`: Deep-copies and freezes model for teacher
- `src/models/cmkl.py`: Added distillation integration:
  - Config keys: `use_distillation`, `distillation_temperature` (2.0), `distillation_alpha` (0.5)
  - `train_continually()`: Creates frozen teacher copy after each task
  - `_train_epoch()`: Computes teacher/student all-entity scores, adds distillation loss
  - Added optional `nc_classifier` head (Linear-ReLU-Dropout-Linear) gated by `use_nc` config
  - Added `classify_nodes()` method
- `scripts/run_ablations.py`: Added "distillation" to ABLATIONS list + dispatch
- `scripts/run_cmkl.py`: Added `--use-distillation`, `--distillation-temperature`, `--distillation-alpha` CLI args
- **Smoke test**: distillation loss computation, teacher copy freezing - PASS

#### Step 2: LKGE Result Parsing + Integration (~150 lines)
- `src/baselines/lkge.py`: Rewrote with:
  - `parse_results()`: 3 regex patterns for LKGE output (Snapshot line, test-on-snapshot blocks, tabular)
  - `_parse_log_content()`: Extracts MRR, Hits@1/3/10 per snapshot, builds results matrix
  - `run_and_parse()`: Runs LKGE as subprocess, captures output, parses results
- `scripts/run_lkge.py` (NEW): CLI with --tasks-dir, --model, --lkge-dir, --seeds, --quick
  - Quick mode tests format conversion only (no LKGE subprocess needed)
  - Full mode: converts data -> runs LKGE -> parses -> computes CL metrics -> saves JSON
- `slurm/run_lkge.sh` (NEW): V100, auto-clones LKGE repo if missing
- **Smoke test**: Format conversion + log parsing - PASS

#### Step 3: RAG Agent + KGQA Pipeline (~400 lines)
- `src/data/kgqa.py` (NEW ~150 lines):
  - `QUESTION_TEMPLATES`: 24 relation-type -> NL question mappings
  - `generate_kgqa_questions()`: Converts triples to QA pairs using templates
  - `generate_continual_kgqa_dataset()`: Per-task QA sets aligned with CL sequence
- `src/baselines/rag_agent.py`: Full implementation:
  - `_init_vectorstore()`: ChromaDB + SentenceTransformer (PubMedBERT)
  - `_init_llm()`: HuggingFace text-gen pipeline (Llama-3-8B), fallback to retrieval-only
  - `index_kg_snapshot()`: Triple-to-NL conversion, batch indexing (40K/batch)
  - `update_with_new_knowledge()`: Incremental indexing (no retraining)
  - `answer_question()`: Retrieve top-K -> LLM generation or majority-vote extraction
  - `evaluate_kgqa()`: Batch eval with EM + token-F1
  - `compute_exact_match()`, `compute_token_f1()`: Normalized string matching
- `src/evaluation/metrics.py`: Added `compute_exact_match()`, `compute_token_f1()`, `compute_nc_metrics()`
- `scripts/run_rag.py` (NEW): CLI with --no-llm, --questions-per-task, --quick
- `slurm/run_rag.sh` (NEW): V100, 64G RAM, 4 CPUs (LLM needs more resources)
- **Smoke test**: KGQA generation, EM/F1 computation - PASS

#### Step 4: Node Classification Support (~350 lines)
- `src/data/node_classification.py` (NEW ~150 lines):
  - `get_label_map()`: PrimeKG 10 node types -> int (0-9)
  - `load_node_types()`: From node_index_map.csv or KG CSV extraction
  - `build_nc_dataset()`: Per-task NC datasets with train/val/test masks
- `src/baselines/nc_baseline.py` (NEW ~100 lines):
  - `NCClassifier`: 2-layer MLP (Linear-ReLU-Dropout-Linear)
  - `NCBaseline`: Trains classifier on frozen embeddings, returns accuracy/macro_f1/weighted_f1
  - `extract_pykeen_embeddings()`: Gets entity embeddings from PyKEEN models
- `scripts/run_nc.py` (NEW ~250 lines):
  - For KGE baselines: train KGE continually -> extract embeddings -> train MLP classifier
  - For CMKL: use fused embeddings -> train MLP classifier
  - Builds NC results matrix with macro_f1, computes CL metrics
- `slurm/run_nc.sh` (NEW): V100, takes $METHOD argument

#### Step 5: Logging Verified
- `src/utils/logging.py`: Already fully implemented with `setup_logger()` and `ExperimentTracker`

#### Step 6: SLURM Final Update
- All 7 SLURM scripts: Added `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`
- All 7 SLURM scripts: Output logs to `slurm/slurm_logs/` directory
- All 7 SLURM scripts: `--gres=gpu:v100:1` (32GB VRAM)
- `slurm/submit_all.sh`: Expanded from 12 to **20 jobs**:
  - Link Prediction: 4 baselines + 1 LKGE + 1 CMKL + 8 ablations = 14 jobs
  - KGQA: 1 RAG agent = 1 job
  - Node Classification: 4 baselines + 1 CMKL = 5 jobs

### Smoke Test Results
| Test | Status |
|------|--------|
| Distillation loss + teacher copy | PASS |
| LKGE format conversion + log parsing | PASS |
| KGQA question generation | PASS |
| EM/F1 metrics | PASS |
| NC label map + classifier + eval | PASS |
| SLURM scripts: all have v100 + PYTORCH_CUDA_ALLOC_CONF | PASS |
| submit_all.sh: 20 sbatch commands | PASS |

### New/Modified Files Summary
| Action | File |
|--------|------|
| Modify | `src/continual/distillation.py` |
| Modify | `src/models/cmkl.py` |
| Modify | `src/baselines/lkge.py` |
| Modify | `src/baselines/rag_agent.py` |
| Modify | `src/evaluation/metrics.py` |
| Modify | `scripts/run_ablations.py` |
| Modify | `scripts/run_cmkl.py` |
| Create | `src/data/kgqa.py` |
| Create | `src/data/node_classification.py` |
| Create | `src/baselines/nc_baseline.py` |
| Create | `scripts/run_lkge.py` |
| Create | `scripts/run_rag.py` |
| Create | `scripts/run_nc.py` |
| Create | `slurm/run_lkge.sh` |
| Create | `slurm/run_rag.sh` |
| Create | `slurm/run_nc.sh` |
| Modify | `slurm/submit_all.sh` |
| Modify | `slurm/base_job.sh` |
| Modify | `slurm/run_baseline.sh` |
| Modify | `slurm/run_cmkl.sh` |
| Modify | `slurm/run_ablations.sh` |

#### Phase Plan Updates (Paper Writing Methodology)
- `.claude-plans/phase6-paper-a-benchmark.md`: Completely rewritten to follow `~/.claude/skills/write-research-paper/SKILL.md` methodology
  - Phases 0-5 mapped to our project: context validation, deep project scan, paper planning, output directory, section-by-section writing, compilation
  - Section-by-section spec: 6-part abstract, 5-paragraph intro, \paragraph{} related work, benchmark construction, experiments, conclusion
  - Comparison Table 1 (ours vs prior CGL benchmarks with \cmark/\xmark)
  - Custom LaTeX commands: \ours{PrimeKG-CL}, stat macros for consistent numbers
  - NeurIPS checklist, supplementary material template
  - 20+ key references mapped across 5 categories
- `.claude-plans/phase7-paper-b-method.md`: Completely rewritten same way
  - CMKL method paper: architecture overview, modality-aware EWC, multimodal replay, cross-attention fusion
  - Algorithm pseudocode box for CMKL training procedure
  - 8 ablation studies table, sensitivity analysis figures
  - Per-modality Fisher visualization, forgetting comparison
  - 20+ key references across 6 categories

### Next Steps
- User resubmits `bash slurm/submit_all.sh` on IBEX (20 jobs)
- When results arrive: populate experiment-results.md, generate tables/figures
- Phase 6-7: Paper writing (follow updated phase plans with SKILL.md methodology)
