"""Multimodal memory replay buffer for CMKL.

Stores triples with their multimodal features (structural, textual, molecular
embeddings). Uses K-means clustering on structural embeddings for diverse
exemplar selection, ensuring replay covers different graph regions.

Three replay strategies:
1. Full multimodal: Store complete multimodal features for each triple
2. Partial modality: Store different modality subsets for different triples
3. Cross-modal reconstruction: Store one modality, reconstruct others

Usage:
    from src.continual.multimodal_replay import MultimodalMemoryBuffer
    buffer = MultimodalMemoryBuffer(max_size=1000)
    buffer.add_exemplars(triples, struct_embs, text_embs, mol_embs, masks)
    replay_batch = buffer.sample(batch_size=64)
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MultimodalMemoryBuffer:
    """Memory buffer that stores triples with their multimodal features.

    Args:
        max_size: Maximum number of exemplars in the buffer.
            Sweep: [100, 250, 500, 1000, 2000, 5000].
        strategy: Replay strategy - 'full_multimodal', 'partial_modality',
            or 'cross_modal_reconstruction'.
    """

    def __init__(
        self,
        max_size: int = 1000,
        strategy: str = "full_multimodal",
    ) -> None:
        self.max_size = max_size
        self.strategy = strategy
        self.buffer: list[dict] = []

    def add_exemplars(
        self,
        triples: "np.ndarray",
        struct_embs: "Tensor",
        text_embs: "Tensor",
        mol_embs: "Tensor",
        node_has_text: "Tensor",
        node_has_mol: "Tensor",
    ) -> None:
        """Add representative samples to the buffer after training on a task.

        Stores each triple with all available modality embeddings.
        When buffer exceeds max_size, uses K-means clustering on structural
        embeddings to keep the most diverse samples.

        Args:
            triples: Array of (head, relation, tail) triples.
            struct_embs: Structural embeddings for head/tail entities.
            text_embs: Text embeddings (for nodes with text).
            mol_embs: Molecular embeddings (for drug nodes).
            node_has_text: Boolean mask for text availability.
            node_has_mol: Boolean mask for molecular availability.
        """
        raise NotImplementedError("Phase 4: Implement exemplar addition")

    def _diverse_selection(self) -> None:
        """Select diverse exemplars using K-means clustering on structural embeddings.

        Clusters all buffer items, then keeps the exemplar closest to each
        cluster center to maximize diversity.
        """
        raise NotImplementedError("Phase 4: Implement K-means diverse selection")

    def sample(self, batch_size: int) -> list[dict]:
        """Sample a batch from the replay buffer.

        Args:
            batch_size: Number of exemplars to sample.

        Returns:
            List of exemplar dicts with triple and embedding data.
        """
        raise NotImplementedError("Phase 4: Implement buffer sampling")

    def __len__(self) -> int:
        return len(self.buffer)
