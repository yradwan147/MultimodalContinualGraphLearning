"""Link prediction decoders for CMKL and baselines.

Score functions for predicting links in knowledge graphs:
- TransE: -||h + r - t|| (translation-based)
- DistMult: <h, r, t> (bilinear diagonal)
- Bilinear: h^T M_r t (full bilinear)

Usage:
    from src.models.decoders import TransEDecoder, DistMultDecoder
    decoder = TransEDecoder(embedding_dim=256)
    scores = decoder(head_embs, rel_embs, tail_embs)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransEDecoder(nn.Module):
    """TransE-style link prediction decoder.

    Score: -||h + r - t||_p (higher is more plausible).

    Args:
        embedding_dim: Dimension of entity/relation embeddings.
        p_norm: Norm order (1 or 2).
    """

    def __init__(self, embedding_dim: int = 256, p_norm: int = 2) -> None:
        super().__init__()
        self.p_norm = p_norm

    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """Compute TransE scores: -||h + r - t||_p."""
        return -torch.norm(head + relation - tail, p=self.p_norm, dim=-1)


class DistMultDecoder(nn.Module):
    """DistMult-style link prediction decoder.

    Score: sum(h * r * t).

    Args:
        embedding_dim: Dimension of entity/relation embeddings.
    """

    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()

    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """Compute DistMult scores: sum(h * r * t)."""
        return (head * relation * tail).sum(dim=-1)


class BilinearDecoder(nn.Module):
    """Full bilinear link prediction decoder.

    Score: h^T M_r t (with per-relation weight matrix).

    Args:
        embedding_dim: Dimension of entity/relation embeddings.
        num_relations: Number of relation types.
    """

    def __init__(self, embedding_dim: int = 256, num_relations: int = 30) -> None:
        super().__init__()
        self.relation_matrices = nn.Parameter(
            torch.randn(num_relations, embedding_dim, embedding_dim) * 0.01
        )

    def forward(self, head: torch.Tensor, relation_ids: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """Compute bilinear scores: h^T M_r t.

        Args:
            head: [batch, dim].
            relation_ids: [batch] integer relation type indices.
            tail: [batch, dim].

        Returns:
            Score tensor [batch].
        """
        M = self.relation_matrices[relation_ids]  # [batch, dim, dim]
        # h^T M_r t = sum_ij h_i M_ij t_j
        return torch.einsum("bi,bij,bj->b", head, M, tail)
