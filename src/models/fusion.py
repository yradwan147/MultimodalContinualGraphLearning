"""Cross-modal attention fusion for the CMKL framework.

Fuses representations from multiple modalities using cross-attention.
Unlike simple concatenation (used in MSCGL), cross-attention captures
complementary information between modalities while allowing each modality
to attend to relevant signals from others.

Architecture:
- Bidirectional cross-attention between structure and text
- Molecular features attend to structural embeddings
- Residual connection from structural encoder (always available)
- LayerNorm for stable training
- Handles missing modalities via boolean masks

Usage:
    from src.models.fusion import CrossModalAttentionFusion
    fusion = CrossModalAttentionFusion(embed_dim=256, num_heads=4)
    h_fused = fusion(h_struct, h_text, h_mol, node_has_text, node_has_mol)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CrossModalAttentionFusion(nn.Module):
    """Fuse representations from multiple modalities using cross-attention.

    Cross-attention allows each modality to attend to relevant signals from
    other modalities. Structure is always available as the backbone; text and
    molecular features are optional per-node.

    For nodes missing a modality, zero vectors are used in the fusion MLP,
    with the residual connection from structure ensuring graceful degradation.

    Args:
        embed_dim: Embedding dimension (must be same for all modalities).
        num_heads: Number of attention heads.
        dropout: Dropout probability in attention and MLP.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Bidirectional cross-attention: structure <-> text
        self.cross_attn_struct_to_text = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.cross_attn_text_to_struct = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )

        # Molecular features attend to structural embeddings
        self.cross_attn_mol_to_struct = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )

        # Fusion MLP: combine enhanced struct + text + mol into embed_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        # LayerNorm for residual connection
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        h_struct: torch.Tensor,
        h_text: torch.Tensor,
        h_mol: torch.Tensor,
        node_has_text: torch.Tensor,
        node_has_mol: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse multimodal embeddings via cross-attention.

        Args:
            h_struct: [N, D] structural embeddings for all nodes.
            h_text: [N_text, D] textual embeddings (only for nodes with text).
            h_mol: [N_mol, D] molecular embeddings (only for drug nodes).
            node_has_text: [N] boolean mask indicating which nodes have text.
            node_has_mol: [N] boolean mask indicating which nodes have mol.

        Returns:
            Fused node embeddings [N, D].
        """
        N, D = h_struct.shape
        device = h_struct.device

        # --- 1. Structure-Text cross-attention (bidirectional) ---
        # For nodes with text: structure attends to text, text attends to structure
        h_struct_enhanced = h_struct.clone()
        h_text_full = torch.zeros(N, D, device=device)

        text_idx = node_has_text.nonzero(as_tuple=True)[0]
        if text_idx.numel() > 0 and h_text.shape[0] > 0:
            # Subset of structural embeddings for text nodes
            h_struct_subset = h_struct[text_idx]  # [N_text, D]

            # Structure attends to text (query=struct, key/value=text)
            # Unsqueeze to add sequence dim for MultiheadAttention: [N_text, 1, D]
            s2t_out, _ = self.cross_attn_struct_to_text(
                h_struct_subset.unsqueeze(1),
                h_text.unsqueeze(1),
                h_text.unsqueeze(1),
            )
            h_struct_enhanced[text_idx] = h_struct_subset + s2t_out.squeeze(1)

            # Text attends to structure (query=text, key/value=struct)
            t2s_out, _ = self.cross_attn_text_to_struct(
                h_text.unsqueeze(1),
                h_struct_subset.unsqueeze(1),
                h_struct_subset.unsqueeze(1),
            )
            h_text_full[text_idx] = h_text + t2s_out.squeeze(1)

        # --- 2. Molecular-Structure cross-attention ---
        h_mol_full = torch.zeros(N, D, device=device)

        mol_idx = node_has_mol.nonzero(as_tuple=True)[0]
        if mol_idx.numel() > 0 and h_mol.shape[0] > 0:
            h_struct_mol_subset = h_struct[mol_idx]  # [N_mol, D]

            # Molecular attends to structure
            m2s_out, _ = self.cross_attn_mol_to_struct(
                h_mol.unsqueeze(1),
                h_struct_mol_subset.unsqueeze(1),
                h_struct_mol_subset.unsqueeze(1),
            )
            h_mol_full[mol_idx] = h_mol + m2s_out.squeeze(1)

        # --- 3. Concatenate and fuse via MLP ---
        # [N, 3D] -> MLP -> [N, D]
        h_concat = torch.cat([h_struct_enhanced, h_text_full, h_mol_full], dim=-1)
        h_fused = self.fusion_mlp(h_concat)

        # --- 4. Residual connection from structure + LayerNorm ---
        h_fused = self.layer_norm(h_fused + h_struct)

        return h_fused


class ConcatenationFusion(nn.Module):
    """Simple concatenation + MLP fusion (ablation baseline).

    Used as an ablation to compare against cross-modal attention.
    Concatenates available modality embeddings and projects via MLP.
    No cross-attention — just direct concatenation.

    Args:
        embed_dim: Per-modality embedding dimension.
        num_modalities: Number of modalities to fuse.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_modalities: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        h_struct: torch.Tensor,
        h_text: torch.Tensor,
        h_mol: torch.Tensor,
        node_has_text: torch.Tensor,
        node_has_mol: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse via concatenation + MLP.

        Missing modalities are filled with zeros before concatenation.

        Args:
            h_struct: [N, D] structural embeddings.
            h_text: [N_text, D] textual embeddings.
            h_mol: [N_mol, D] molecular embeddings.
            node_has_text: [N] boolean mask for text availability.
            node_has_mol: [N] boolean mask for molecular availability.

        Returns:
            Fused node embeddings [N, D].
        """
        N, D = h_struct.shape
        device = h_struct.device

        # Scatter text/mol embeddings into full-size tensors
        h_text_full = torch.zeros(N, D, device=device)
        text_idx = node_has_text.nonzero(as_tuple=True)[0]
        if text_idx.numel() > 0 and h_text.shape[0] > 0:
            h_text_full[text_idx] = h_text

        h_mol_full = torch.zeros(N, D, device=device)
        mol_idx = node_has_mol.nonzero(as_tuple=True)[0]
        if mol_idx.numel() > 0 and h_mol.shape[0] > 0:
            h_mol_full[mol_idx] = h_mol

        h_concat = torch.cat([h_struct, h_text_full, h_mol_full], dim=-1)
        h_fused = self.fusion_mlp(h_concat)
        h_fused = self.layer_norm(h_fused + h_struct)

        return h_fused
