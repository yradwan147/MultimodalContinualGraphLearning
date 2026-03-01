# Phase 4: CMKL (Continual Multimodal KG Learner) Development
**Estimated time:** Weeks 12-18 | **PDF Section:** 6
**Depends on:** Phase 2 (benchmark), Phase 3 (baselines for comparison) | **Blocks:** Phase 5

---

## Objectives
- Implement the novel CMKL framework with 4 key components
- The core contribution: **modality-aware continual learning** that leverages multimodal complementarity to reduce forgetting while handling heterogeneous distribution shifts across modalities

---

## Architecture Overview

```
CMKL Framework
├── Modality-Specific Encoders
│   ├── Structural GNN Encoder (R-GCN/GAT)
│   ├── Textual LM Encoder (BiomedBERT)
│   └── Molecular Fingerprint Encoder
├── Cross-Modal Attention Fusion
│   └── (modality-specific + cross-modal attention)
├── Continual Learning Module
│   ├── Modality-Aware EWC
│   ├── Multimodal Memory Replay
│   └── Knowledge Distillation (old -> new)
└── Output: Link prediction scores / Node embeddings
```

---

## Step 4.1: Component 1 - Modality-Specific Encoders

**File:** `src/models/encoders.py`

### StructuralEncoder (R-GCN)
```python
class StructuralEncoder(nn.Module):
    """R-GCN encoder for graph structure."""
    # Uses RGCNConv from torch_geometric.nn
    # Parameters: num_nodes, num_relations, embedding_dim=256, num_layers=2
    # Forward: node_emb -> RGCNConv layers with LayerNorm + ReLU
```

### TextualEncoder (Frozen BiomedBERT)
```python
class TextualEncoder(nn.Module):
    """Frozen BiomedBERT encoder for textual node features."""
    # Uses microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
    # Freeze BERT weights to save compute
    # Linear projection from 768 -> projection_dim (256)
    # Forward: tokenize -> BERT -> CLS token -> projection
```

### MolecularEncoder (for drug nodes)
```python
class MolecularEncoder(nn.Module):
    """Encoder for molecular fingerprints (drug nodes)."""
    # Input: 1024-dim Morgan fingerprints
    # Two-layer MLP: 1024 -> 512 -> 256
```

### Key considerations
- Not all nodes have all modalities (only drugs have molecular features, only drugs/diseases have text)
- Must handle missing modalities gracefully with zero vectors or learned default embeddings

**Log to worklog:** Encoder architectures, parameter counts, handling of missing modalities.

---

## Step 4.2: Component 2 - Cross-Modal Attention Fusion

**File:** `src/models/fusion.py`

```python
class CrossModalAttentionFusion(nn.Module):
    """Fuse representations from multiple modalities using cross-attention."""
    # Unlike concatenation (MSCGL), cross-attention captures complementary information
    #
    # cross_attn_struct_to_text: MultiheadAttention(embed_dim, num_heads=4)
    # cross_attn_text_to_struct: MultiheadAttention(embed_dim, num_heads=4)
    # cross_attn_mol: MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
    # fusion_mlp: Linear(embed_dim*3, embed_dim) -> ReLU -> Linear(embed_dim, embed_dim)
    # LayerNorm + residual connection
    #
    # Forward:
    #   1. Cross-attention: structure attends to text
    #   2. For nodes without modalities, use zero vectors
    #   3. Concatenate [h_struct_enhanced, h_text_full, h_mol_full]
    #   4. MLP projection + residual + LayerNorm
```

### Design decisions
- Cross-attention is bidirectional between structure and text
- Molecular features attend to structural embeddings
- Residual connection from structural encoder (always available) ensures graceful degradation
- `node_has_text` and `node_has_mol` boolean masks handle missing modalities

**Log to worklog:** Fusion architecture, attention head count, dimension choices, ablation ideas noted.

---

## Step 4.3: Component 3 - Modality-Aware Continual Learning (EWC variant)

**File:** `src/continual/modality_ewc.py`

```python
class ModalityAwareEWC:
    """EWC with separate Fisher matrices per modality.

    Key insight: Different modalities experience different distribution shifts.
    Molecular structures are stable; text descriptions evolve;
    graph structure changes with new entities.
    Per-modality Fisher matrices allow modality-specific regularization strength.
    """
    # __init__: lambda_struct=10.0, lambda_text=5.0, lambda_mol=1.0
    #
    # compute_modality_fisher(dataloader):
    #   For each modality in ['structural', 'textual', 'molecular']:
    #     Compute Fisher diagonal for that encoder's parameters
    #     Accumulate across tasks
    #     Store old parameter values
    #
    # ewc_loss():
    #   total = 0
    #   For each modality:
    #     total += lambda_modality * sum(F_k * (theta_k - theta*_k)^2)
    #   return total / 2
```

### Hyperparameters to tune
- `lambda_struct`: 10.0 (graph structure changes most with new entities)
- `lambda_text`: 5.0 (text descriptions may evolve but less dramatically)
- `lambda_mol`: 1.0 (molecular structures are relatively stable)
- Sweep each independently: [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]

**Log to worklog:** Per-modality lambda sweep results, optimal values, comparison with global EWC.

---

## Step 4.4: Component 4 - Multimodal Memory Replay

**File:** `src/continual/multimodal_replay.py`

```python
class MultimodalMemoryBuffer:
    """Memory buffer that stores triples with their multimodal features.

    Three strategies:
    1. Full multimodal: Store complete multimodal features for each triple
    2. Partial modality: Store different modality subsets for different triples
    3. Cross-modal reconstruction: Store one modality, reconstruct others
    """
    # add_exemplars(triples, struct_embs, text_embs, mol_embs, masks):
    #   Store exemplars with all available modality embeddings
    #   When buffer exceeds max_size, use K-means clustering on structural
    #   embeddings to keep most diverse samples
    #
    # sample(batch_size):
    #   Return random batch from buffer
```

### Key design decisions
- K-means based diverse selection ensures replay covers different graph regions
- Storing embeddings (not raw features) saves memory and avoids re-encoding
- `max_size` default: 1000, sweep: [100, 250, 500, 1000, 2000, 5000]

**Log to worklog:** Buffer strategy used, max size sweep results, memory footprint, replay effectiveness.

---

## Step 4.5: Assemble Full CMKL Model

**File:** `src/models/cmkl.py`

Assemble all 4 components into the full CMKL model:
1. Initialize encoders, fusion module, and link prediction decoder
2. Training loop with modality-aware EWC + multimodal replay
3. After each task: compute per-modality Fisher, add exemplars to buffer

### Link Prediction Decoder
**File:** `src/models/decoders.py`

Implement score functions:
- TransE-style: `||h + r - t||`
- DistMult-style: `h * r * t`
- Bilinear: `h^T M_r t`

### Training Pipeline
```python
for task_id, task_data in enumerate(task_sequence):
    # 1. Encode: structural, textual, molecular
    # 2. Fuse: cross-modal attention
    # 3. Train: task loss + EWC penalty + replay loss
    # 4. After training: compute Fisher per modality, add to replay buffer
    # 5. Evaluate: on all tasks seen so far
```

**Log to worklog:** Full model parameter count, training time per task, memory usage, convergence behavior.

---

## Step 4.6: Knowledge Distillation (Optional Enhancement)

**File:** `src/continual/distillation.py`

If EWC + replay isn't sufficient, add knowledge distillation:
- Old model (teacher) generates soft targets on new task's data
- New model (student) learns from both hard targets and soft targets
- This preserves the old model's "dark knowledge" about entity relationships

**Log to worklog:** Whether distillation was needed, improvement over base CMKL.

---

## Completion Criteria
- [ ] All 4 CMKL components implemented and unit tested
- [ ] Full CMKL model assembles and trains on the benchmark
- [ ] CMKL produces results matrix comparable to or better than baselines
- [ ] Modality-aware EWC demonstrably outperforms global EWC
- [ ] Multimodal replay outperforms random replay
- [ ] All code documented with docstrings and inline comments
- [ ] Architecture documented in `docs/architecture.md`
- [ ] All activities logged in `worklog.md`
