"""Modality-specific encoders for the CMKL framework.

Three encoder types for different modalities in biomedical KGs:
1. StructuralEncoder: R-GCN for graph structure (all nodes)
2. TextualEncoder: Frozen BiomedBERT for text descriptions (drug/disease nodes)
3. MolecularEncoder: MLP for Morgan fingerprints (drug nodes only)

Not all nodes have all modalities - encoders must handle missing modalities
gracefully via zero vectors or learned default embeddings.

Usage:
    from src.models.encoders import StructuralEncoder, TextualEncoder, MolecularEncoder
    struct_enc = StructuralEncoder(num_nodes=129375, num_relations=30)
    text_enc = TextualEncoder()
    mol_enc = MolecularEncoder()
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class StructuralEncoder:
    """R-GCN encoder for graph structure.

    Uses RGCNConv layers from torch_geometric.nn with LayerNorm and ReLU.

    Args:
        num_nodes: Total number of nodes in the KG.
        num_relations: Total number of relation types.
        embedding_dim: Output embedding dimension.
        num_layers: Number of R-GCN layers.
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        embedding_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        raise NotImplementedError("Phase 4: Implement StructuralEncoder")

    def forward(self, edge_index: "Tensor", edge_type: "Tensor") -> "Tensor":
        """Forward pass through R-GCN layers.

        Args:
            edge_index: Edge indices [2, num_edges].
            edge_type: Edge type labels [num_edges].

        Returns:
            Node embeddings [num_nodes, embedding_dim].
        """
        raise NotImplementedError("Phase 4: Implement forward pass")


class TextualEncoder:
    """Frozen BiomedBERT encoder for textual node features.

    Uses microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract with frozen
    weights. A linear projection maps from 768 to projection_dim.

    Args:
        model_name: HuggingFace model identifier.
        projection_dim: Output embedding dimension.
    """

    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        projection_dim: int = 256,
    ) -> None:
        raise NotImplementedError("Phase 4: Implement TextualEncoder")

    def forward(self, texts: list[str]) -> "Tensor":
        """Encode text descriptions.

        Tokenizes input, runs through frozen BERT, extracts [CLS] token
        embedding, and projects to target dimension.

        Args:
            texts: List of text descriptions.

        Returns:
            Text embeddings [batch_size, projection_dim].
        """
        raise NotImplementedError("Phase 4: Implement forward pass")


class MolecularEncoder:
    """Encoder for molecular fingerprints (drug nodes).

    Two-layer MLP: input_dim (1024) -> 512 -> projection_dim (256).

    Args:
        input_dim: Morgan fingerprint dimensionality.
        projection_dim: Output embedding dimension.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        projection_dim: int = 256,
    ) -> None:
        raise NotImplementedError("Phase 4: Implement MolecularEncoder")

    def forward(self, fingerprints: "Tensor") -> "Tensor":
        """Encode molecular fingerprints.

        Args:
            fingerprints: Morgan fingerprint vectors [batch_size, input_dim].

        Returns:
            Molecular embeddings [batch_size, projection_dim].
        """
        raise NotImplementedError("Phase 4: Implement forward pass")
