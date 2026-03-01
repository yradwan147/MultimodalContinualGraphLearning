# Multimodal Continual Graph Learning (MCGL)

Continual learning on evolving, multimodal biomedical knowledge graphs -- incorporating new drugs, diseases, proteins, and relationships without catastrophically forgetting previously learned knowledge.

## Core Research Question

How can graph-based models perform continual multimodal biomedical KG reasoning (drug-disease link prediction, biomedical KGQA) under evolving knowledge graphs, while minimizing catastrophic forgetting and avoiding full retraining?

## Project Outputs

- **Paper A (Benchmark):** A temporal, multimodal biomedical KG benchmark for continual learning evaluation
- **Paper B (Method):** CMKL -- a novel continual learning technique with modality-aware regularization and multimodal memory replay

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
```

## Key Technologies

- **PrimeKG** - Primary biomedical knowledge graph (129K+ nodes, 8.1M+ edges)
- **PyKEEN** - KG embedding training and evaluation
- **PyTorch Geometric** - GNN implementation (R-GCN, GAT)
- **HuggingFace Transformers** - BiomedBERT for text encoding
- **LangChain + ChromaDB** - RAG pipeline for KGQA
- **Weights & Biases** - Experiment tracking

## Compute

- Local: data exploration, development, small tests
- IBEX (KAUST): model training, experiments, sweeps (V100 GPUs)
