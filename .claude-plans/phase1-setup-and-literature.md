# Phase 1: Environment Setup & Literature Review
**Estimated time:** Weeks 1-3 | **PDF Sections:** 1, 2, 5.1, 11
**Depends on:** Phase 0 (project scaffolding) | **Blocks:** Phase 2, 3

---

## Objectives
- Set up the Python development environment with all dependencies
- Conduct and document the literature review
- Explore PrimeKG t0 to understand the data

---

## Step 1.1: Create Conda Environment

```bash
conda create -n mcgl python=3.10
conda activate mcgl
```

### Core dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install pykeen
pip install TxGNN
pip install transformers
pip install pandas numpy scikit-learn matplotlib seaborn
pip install rdkit        # molecular fingerprints
pip install networkx     # graph analysis
```

### RAG experiment dependencies
```bash
pip install langchain chromadb sentence-transformers
pip install openai       # if using OpenAI API
```

### Experiment tracking
```bash
pip install wandb
```

### Verification
- `python -c "import torch; print(torch.cuda.is_available())"` -> True (if GPU available)
- `python -c "import torch_geometric; print(torch_geometric.__version__)"` -> version string
- `python -c "import pykeen; print(pykeen.get_version())"` -> version string

**Log to worklog:** Environment creation, all installed package versions, CUDA availability.

---

## Step 1.2: Literature Review Deep-Dive

Create `docs/literature-review.md` with annotated bibliography organized by topic.

### Categories to cover (from PDF Section 2):

#### 2.1 Biomedical Knowledge Graphs
- **PrimeKG** (Chandak et al.): 129,375 nodes, 8.1M edges, 10 types, 30 edge types. Primary KG.
- **PrimeKG-U** (CellAwareGNN, Feb 2026): Expanded to ~140K nodes, ~14M edges
- **PrimeKG++/BioMedKG**: Extended with biological sequences, multimodal contrastive learning
- **TarKG**: 15 entity types, 43 relation types, chemical structures + protein sequences

#### 2.2 Models Built on PrimeKG
- **TxGNN** (Huang et al., Nature Medicine 2024): GNN for zero-shot drug repurposing, bipartite reasoning graph, ~5.7M edges, 30 relation types
- **ProCyon** (Queen et al., 2025): Multimodal foundation model for protein phenotypes
- **GraphAge** (Ahmed et al., PNAS Nexus 2025): DNA methylation as graph for biological age prediction

#### 2.3 Continual Knowledge Graph Embedding (CKGE)
Key methods to review:
| Method | Year | Approach | Key Idea |
|--------|------|----------|----------|
| EWC for KG | 2024 | Regularization | Fisher Information Matrix penalty on TransE |
| LKGE | 2023 | Autoencoder + Transfer | Masked KG autoencoder, embedding transfer |
| CLKGE | 2024 | Transfer + Retention | Knowledge transfer and retention across snapshots |
| IncDE | 2024 | Incremental Distillation | Hierarchical ordering, two-stage training |
| STCKGE | 2025 | Spatial Transformation | Dual-component entity representations |
| BER | 2024 | Experience Replay + Distillation | Attention-based fact selection for replay |
| DiCGRL | 2020 | Disentangled Representation | Continual graph learning with disentangled reps |

Benchmarks: PS-CKGE (SIGIR 2025), FB15k-237, WN18RR, ICEWS05-15

#### 2.4 Multimodal Continual Graph Learning
- **MSCGL** (Cai et al., WWW 2022): Seminal work, AdaMGNN + NAS + Group Sparse Regularization. No official code.
- **OMG-NAS** (AAAI 2024): Multimodal graph NAS for distribution shifts
- **MoDE** (NeurIPS 2025): Intra-modal and inter-modal forgetting

#### 2.5 RAG-Based Biomedical Systems
- **AMG-RAG** (EMNLP Findings 2025): Agentic Medical Graph-RAG, 74.1% F1 on MEDQA
- **KGT**: KG-based Thought for pan-cancer QA
- **BioGraphRAG**: 34M PubMed abstracts, 11.4M entities, 42.5M relations

#### 2.6 Evaluation Metrics
- Link Prediction: MRR, Hits@K (K=1,3,10), AUPRC
- Continual Learning: AP, AF, BWT, FWT, REM

**Log to worklog:** Each paper reviewed with 2-3 sentence summary and relevance to project.

---

## Step 1.3: Download and Explore PrimeKG t0

```python
# Option A: Via TDC
from tdc.resource import PrimeKG
data = PrimeKG(path='./data')
drug_features = data.get_features(feature_type='drug')

# Option B: Direct from Harvard Dataverse
import pandas as pd
kg_t0 = pd.read_csv('data/kg.csv', low_memory=False)
```

### Exploration checklist:
- [ ] Verify shape: ~8.1M rows, 12 columns
- [ ] Column names: relation, display_relation, x_index, x_id, x_type, x_name, x_source, y_index, y_id, y_type, y_name, y_source
- [ ] Count nodes by type (should match table in PDF Section 2.1)
- [ ] Count edge/relation types (should be 30)
- [ ] Examine drug features (description, indication, pharmacodynamics, etc.)
- [ ] Examine disease features (MONDO definition, UMLS description, Mayo Clinic)
- [ ] Save exploration results to `notebooks/01_explore_primekg.ipynb`

**Log to worklog:** PrimeKG download method, actual shape, node/edge counts, any discrepancies from expected values.

---

## Completion Criteria
- [ ] Conda environment `mcgl` is functional with all dependencies
- [ ] `docs/literature-review.md` has 30+ annotated papers
- [ ] PrimeKG t0 downloaded and explored
- [ ] `notebooks/01_explore_primekg.ipynb` shows data statistics
- [ ] All activities logged in `worklog.md`
