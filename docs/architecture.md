# CMKL Architecture Notes

## Overview

CMKL (Continual Multimodal Knowledge Graph Learner) is a framework for continual learning on evolving, multimodal biomedical knowledge graphs. It addresses **modality-specific forgetting** by leveraging multimodal complementarity: different modalities (structure, text, molecular) experience different distribution shifts across tasks, and CMKL treats them accordingly.

**Core insight:** Per-modality regularization and diverse multimodal replay reduce catastrophic forgetting more effectively than modality-agnostic approaches.

## Architecture Diagram

```
                    ┌─────────────────────────────────────┐
                    │          CMKL Framework              │
                    └─────────────┬───────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
┌───────┴────────┐  ┌────────────┴──────────┐  ┌──────────┴──────────┐
│   Structural   │  │      Textual          │  │     Molecular       │
│   Encoder      │  │      Encoder          │  │     Encoder         │
│   (R-GCN)      │  │  (frozen BiomedBERT)  │  │  (2-layer MLP)      │
│   2 layers     │  │   768 → D projection  │  │  1024 → 512 → D    │
└───────┬────────┘  └────────────┬──────────┘  └──────────┬──────────┘
        │                        │                         │
        └─────────────┬──────────┴─────────────────────────┘
                      │
        ┌─────────────┴──────────────────┐
        │  Cross-Modal Attention Fusion  │
        │  - struct ↔ text cross-attn    │
        │  - mol → struct cross-attn     │
        │  - Fusion MLP (3D → D)         │
        │  - Residual + LayerNorm        │
        └─────────────┬──────────────────┘
                      │
              ┌───────┴────────┐
              │  Fused Node    │
              │  Embeddings    │
              │  [N, D]        │
              └───────┬────────┘
                      │
    ┌─────────────────┼──────────────────────┐
    │                 │                      │
┌───┴───┐    ┌───────┴────────┐    ┌────────┴─────────┐
│Decoder│    │ Modality-Aware │    │   Multimodal     │
│TransE/│    │     EWC        │    │  Memory Replay   │
│DistMul│    │ (per-modality  │    │ (K-means diverse │
│Bilinea│    │  Fisher + λ)   │    │   selection)     │
└───────┘    └────────────────┘    └──────────────────┘
```

## Component Details

### 1. Modality-Specific Encoders (`src/models/encoders.py`)

| Encoder | Input | Architecture | Output | Trainable? |
|---------|-------|-------------|--------|------------|
| StructuralEncoder | Graph (edge_index, edge_type) | nn.Embedding → 2x RGCNConv + LayerNorm + ReLU | [N, D] | Yes |
| TextualEncoder | Pre-computed 768-dim BiomedBERT embeddings | Linear(768, D) | [N_text, D] | Projection only (BERT frozen) |
| MolecularEncoder | 1024-dim Morgan fingerprints | Linear(1024,512) → ReLU → Dropout → Linear(512,D) | [N_mol, D] | Yes |

- **Not all nodes have all modalities**: Only drugs have molecular features (~8K nodes), only drugs+diseases have text descriptions (~22K nodes). Missing modalities handled via boolean masks + zero vectors.
- **BiomedBERT model**: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` (frozen to save compute; only projection layer trains)
- **Structural encoder fallback**: If `torch_geometric` unavailable, falls back to embedding-only mode (no message passing)

### 2. Cross-Modal Attention Fusion (`src/models/fusion.py`)

**CrossModalAttentionFusion** (default):
- 3 `nn.MultiheadAttention` modules (4 heads each):
  - `struct_to_text`: Structure attends to text (Query=struct, Key/Value=text)
  - `text_to_struct`: Text attends to structure (bidirectional)
  - `mol`: Molecular attends to structure
- Fusion MLP: `Linear(3D, D) → ReLU → Linear(D, D)`
- Residual connection from structural embeddings + LayerNorm
- Missing modalities: zero vectors for nodes without text/molecular features

**ConcatenationFusion** (ablation baseline):
- Simple `torch.cat([struct, text, mol]) → MLP` (no cross-attention)
- Same interface as CrossModalAttentionFusion for easy swapping

### 3. Link Prediction Decoders (`src/models/decoders.py`)

| Decoder | Score Function | Parameters |
|---------|---------------|------------|
| TransE | `-\|\|h + r - t\|\|_p` | p_norm (default 2) |
| DistMult | `sum(h * r * t)` | None extra |
| Bilinear | `h^T M_r t` | Per-relation D×D matrix |

- Relation embeddings: `nn.Embedding(num_relations, D)` (shared, except Bilinear uses own matrices)

### 4. Modality-Aware EWC (`src/continual/modality_ewc.py`)

**Key innovation**: Separate Fisher Information Matrix diagonal per modality encoder.

```
L_ewc = (1/2) * Σ_m [ λ_m * Σ_k F_m,k * (θ_m,k - θ*_m,k)^2 ]
```

| Modality | Default λ | Rationale |
|----------|-----------|-----------|
| Structural | 10.0 | Graph structure changes most with new entities/edges |
| Textual | 5.0 | Text descriptions may evolve but less dramatically |
| Molecular | 1.0 | Molecular structures are chemically stable |

- Fisher computed via empirical approximation: `F_k ≈ (1/N) Σ_i (∂L_i/∂θ_k)^2`
- Accumulates across tasks (summing Fisher diagonals)
- Stores old parameter snapshots for penalty computation
- Serializable via `state_dict()` / `load_state_dict()`

### 5. Multimodal Memory Replay (`src/continual/multimodal_replay.py`)

- **Buffer stores**: Triples + per-entity embeddings (structural, textual, molecular)
- **Diverse selection**: When buffer exceeds `max_size`, uses K-means clustering on structural embeddings to keep most diverse samples (one closest to each cluster center)
- **Default max_size**: 1000 (sweep: [100, 250, 500, 1000, 2000, 5000])
- **Replay during training**: Sample batch from buffer, compute loss on replay triples, add to total loss with `replay_weight`

### 6. Full CMKL Assembly (`src/models/cmkl.py`, ~786 lines)

**Training pipeline per task:**
1. **Encode**: Structural (R-GCN) + Textual (projection) + Molecular (MLP)
2. **Fuse**: Cross-modal attention → fused embeddings [N, D]
3. **Train**: `L = L_task + L_ewc + replay_weight * L_replay`
   - Task loss: margin ranking with negative sampling (corrupt head or tail)
   - EWC loss: modality-weighted Fisher penalty
   - Replay loss: same margin ranking on buffered triples
4. **Post-task**: Compute per-modality Fisher, add exemplars to replay buffer
5. **Evaluate**: MRR on all tasks seen so far → populate results matrix R[i][j]

**Key methods:**
- `forward()` → encode all modalities → fuse → return node embeddings
- `score_triples()` → score (h, r, t) using selected decoder
- `compute_task_loss()` → negative sampling + margin ranking loss
- `train_continually()` → full CL pipeline across task sequence
- `_evaluate_mrr()` → rank-based evaluation (all entities as candidate tails)
- `save_checkpoint()` / `load_checkpoint()` → full state including EWC + replay

## Default Hyperparameters (from `configs/cmkl.yaml`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| embedding_dim | 256 | All encoders project to this dim |
| num_gnn_layers | 2 | R-GCN layers |
| num_heads | 4 | Cross-attention heads |
| dropout | 0.1 | Encoder/fusion dropout |
| lr | 0.001 | Adam optimizer |
| batch_size | 512 | Training batch size |
| num_epochs | 100 | Per task |
| margin | 1.0 | Margin ranking loss |
| lambda_struct | 10.0 | EWC weight for structural |
| lambda_text | 5.0 | EWC weight for textual |
| lambda_mol | 1.0 | EWC weight for molecular |
| replay_buffer_size | 1000 | Max exemplars in buffer |
| replay_weight | 0.5 | Weight of replay loss |
| decoder_type | DistMult | Score function |
| fusion_type | cross_attention | Fusion mechanism |

## Parameter Count (dim=256, ~24K entities, ~30 relations)

Approximate breakdown:
- Node embeddings: ~24K × 256 = ~6.3M
- R-GCN layers: 2 × (256 × 256 × 30 bases) ≈ varies with num_bases
- Textual projection: 768 × 256 = ~197K
- Molecular MLP: 1024×512 + 512×256 = ~655K
- Cross-attention: 3 × (3 × 256 × 256) = ~590K
- Fusion MLP: 768×256 + 256×256 = ~262K
- Relation embeddings: 30 × 256 = ~8K
- **Total**: ~8-10M parameters (varies with entity count and R-GCN bases)
