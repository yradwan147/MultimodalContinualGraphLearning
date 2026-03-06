# Multimodal Continual Graph Learning (MCGL)

Continual learning on evolving, multimodal biomedical knowledge graphs -- incorporating new drugs, diseases, proteins, and relationships without catastrophically forgetting previously learned knowledge.

## Core Research Question

How can graph-based models perform continual multimodal biomedical KG reasoning (drug-disease link prediction, biomedical KGQA) under evolving knowledge graphs, while minimizing catastrophic forgetting and avoiding full retraining?

## Project Status

| Phase | Status |
|-------|--------|
| 0 - Scaffolding | DONE |
| 1 - Setup & Exploration | DONE |
| 2 - Benchmark Construction | DONE (real t1 from 9 databases) |
| 3 - Baseline Implementation | DONE (6 baselines) |
| 4 - CMKL Development | DONE (all 6 components) |
| 5 - Experiments | DONE (Runs 1-3, 82 result files, all methods complete) |
| 6 - Paper A (Benchmark) | IN PROGRESS |
| 7 - Paper B (Method) | IN PROGRESS |

## Results Summary

### Link Prediction (Continual, 10 tasks, 5 seeds)

| Method | AP | AF | BWT | REM |
|--------|-----|-----|------|------|
| **CMKL (DistMult)** | **0.063 ± 0.003** | 0.040 ± 0.003 | -0.040 ± 0.003 | 0.960 ± 0.003 |
| LKGE (TransE) | 0.039 ± 0.001 | **0.012 ± 0.002** | -0.010 ± 0.003 | 0.990 ± 0.003 |
| Joint Training | 0.018 ± 0.000 | 0.000 | 0.000 | 1.000 |
| EWC | 0.004 ± 0.000 | 0.017 ± 0.001 | -0.017 ± 0.001 | 0.984 ± 0.001 |
| Naive Sequential | 0.004 ± 0.000 | 0.021 ± 0.000 | -0.021 ± 0.000 | 0.980 ± 0.000 |
| Experience Replay | 0.004 ± 0.000 | 0.021 ± 0.000 | -0.021 ± 0.000 | 0.980 ± 0.000 |
| RAG (Qwen2.5-7B) | 0.002 ± 0.001 | 0.001 ± 0.001 | -0.001 ± 0.001 | 0.999 ± 0.001 |

### Node Classification (Continual, 10 tasks, 5 seeds)

| Method | AP | AF | BWT |
|--------|-----|-----|------|
| **CMKL** | **0.431 ± 0.005** | **0.004 ± 0.002** | -0.000 ± 0.002 |
| Joint Training | 0.370 ± 0.002 | 0.003 ± 0.001 | 0.022 ± 0.003 |
| EWC | 0.345 ± 0.004 | 0.008 ± 0.003 | 0.007 ± 0.004 |

Full results: [`docs/experiment-results.md`](docs/experiment-results.md)

## Project Outputs

- **Paper A (Benchmark):** PrimeKG-CL — a temporal, multimodal biomedical KG benchmark for continual learning
- **Paper B (Method):** CMKL — Continual Multimodal Knowledge Graph Learner with modality-aware EWC and multimodal replay

## Quick Start

```bash
# Create environment
conda env create -f environment.yml
conda activate mcgl

# Or with pip
pip install -r requirements.txt

# Download PrimeKG
python scripts/download_primekg.py

# Build benchmark
python scripts/build_benchmark.py

# Run baselines
python scripts/run_baselines.py --config configs/base.yaml

# Run CMKL
python scripts/run_cmkl.py --config configs/cmkl.yaml
```

## Project Structure

```
src/
├── data/          # Data loading, benchmark construction
├── baselines/     # 6 baseline implementations
├── models/        # CMKL model (encoders, fusion, decoders)
├── continual/     # CL mechanisms (modality-aware EWC, multimodal replay)
├── evaluation/    # Metrics, statistical tests, visualization
└── utils/         # Config, logging, I/O

scripts/           # Top-level runnable scripts
slurm/             # SLURM job scripts for IBEX cluster
configs/           # Experiment YAML configs
notebooks/         # Jupyter notebooks for exploration
papers/            # LaTeX paper drafts
docs/              # Documentation and experiment reports
```

## Key Technologies

- **PrimeKG** - Primary biomedical knowledge graph (129K+ nodes, 8.1M+ edges)
- **PyKEEN** - KG embedding training
- **PyTorch Geometric** - GNN implementation (R-GCN)
- **HuggingFace Transformers** - BiomedBERT for text encoding
- **LangChain + ChromaDB** - RAG pipeline for KGQA
- **Weights & Biases** - Experiment tracking

## Compute

- Local: data exploration, development, small tests
- IBEX (KAUST): model training, experiments, sweeps (V100 GPUs)
