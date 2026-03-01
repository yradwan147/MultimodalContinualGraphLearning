"""Link prediction decoders for CMKL and baselines.

Score functions for predicting links in knowledge graphs:
- TransE: ||h + r - t|| (translation-based)
- DistMult: <h, r, t> (bilinear diagonal)
- Bilinear: h^T M_r t (full bilinear)

Usage:
    from src.models.decoders import TransEDecoder, DistMultDecoder
    decoder = TransEDecoder(embedding_dim=256)
    scores = decoder(head_embs, rel_embs, tail_embs)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TransEDecoder:
    """TransE-style link prediction decoder.

    Score: -||h + r - t||_p

    Args:
        embedding_dim: Dimension of entity/relation embeddings.
        p_norm: Norm order (1 or 2).
    """

    def __init__(self, embedding_dim: int = 256, p_norm: int = 2) -> None:
        raise NotImplementedError("Phase 4: Implement TransE decoder")

    def forward(
        self,
        head: "Tensor",
        relation: "Tensor",
        tail: "Tensor",
    ) -> "Tensor":
        """Compute TransE scores.

        Args:
            head: Head entity embeddings [batch, dim].
            relation: Relation embeddings [batch, dim].
            tail: Tail entity embeddings [batch, dim].

        Returns:
            Score tensor [batch].
        """
        raise NotImplementedError("Phase 4: Implement TransE scoring")


class DistMultDecoder:
    """DistMult-style link prediction decoder.

    Score: sum(h * r * t)

    Args:
        embedding_dim: Dimension of entity/relation embeddings.
    """

    def __init__(self, embedding_dim: int = 256) -> None:
        raise NotImplementedError("Phase 4: Implement DistMult decoder")

    def forward(
        self,
        head: "Tensor",
        relation: "Tensor",
        tail: "Tensor",
    ) -> "Tensor":
        """Compute DistMult scores.

        Args:
            head: Head entity embeddings [batch, dim].
            relation: Relation embeddings [batch, dim].
            tail: Tail entity embeddings [batch, dim].

        Returns:
            Score tensor [batch].
        """
        raise NotImplementedError("Phase 4: Implement DistMult scoring")


class BilinearDecoder:
    """Full bilinear link prediction decoder.

    Score: h^T M_r t (with per-relation weight matrix).

    Args:
        embedding_dim: Dimension of entity/relation embeddings.
        num_relations: Number of relation types.
    """

    def __init__(self, embedding_dim: int = 256, num_relations: int = 30) -> None:
        raise NotImplementedError("Phase 4: Implement Bilinear decoder")

    def forward(
        self,
        head: "Tensor",
        relation_ids: "Tensor",
        tail: "Tensor",
    ) -> "Tensor":
        """Compute bilinear scores.

        Args:
            head: Head entity embeddings [batch, dim].
            relation_ids: Relation type indices [batch].
            tail: Tail entity embeddings [batch, dim].

        Returns:
            Score tensor [batch].
        """
        raise NotImplementedError("Phase 4: Implement Bilinear scoring")
