"""Modality-Aware Elastic Weight Consolidation for CMKL.

EWC with separate Fisher matrices per modality. Key insight: different
modalities experience different distribution shifts across tasks. Molecular
structures are stable; text descriptions evolve; graph structure changes
with new entities. Per-modality Fisher matrices allow modality-specific
regularization strength.

Default lambdas: lambda_struct=10.0, lambda_text=5.0, lambda_mol=1.0

Usage:
    from src.continual.modality_ewc import ModalityAwareEWC
    ewc = ModalityAwareEWC(model, lambda_struct=10.0, lambda_text=5.0, lambda_mol=1.0)
    ewc.compute_modality_fisher(dataloader)
    penalty = ewc.ewc_loss()
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ModalityAwareEWC:
    """EWC with separate Fisher matrices per modality.

    Args:
        model: The CMKL model with named encoders (structural_encoder,
            textual_encoder, molecular_encoder).
        lambda_struct: Regularization strength for structural encoder.
            Graph structure changes most with new entities.
        lambda_text: Regularization strength for textual encoder.
            Text descriptions may evolve but less dramatically.
        lambda_mol: Regularization strength for molecular encoder.
            Molecular structures are relatively stable.
    """

    def __init__(
        self,
        model: "nn.Module",
        lambda_struct: float = 10.0,
        lambda_text: float = 5.0,
        lambda_mol: float = 1.0,
    ) -> None:
        self.model = model
        self.lambdas = {
            "structural": lambda_struct,
            "textual": lambda_text,
            "molecular": lambda_mol,
        }
        self.fisher_per_modality: dict = {}
        self.old_params_per_modality: dict = {}

    def compute_modality_fisher(
        self,
        dataloader: "DataLoader",
        device: str = "cuda",
    ) -> None:
        """Compute separate Fisher diagonals for each encoder's parameters.

        For each modality in ['structural', 'textual', 'molecular']:
        - Compute Fisher diagonal for that encoder's parameters
        - Accumulate across tasks (sum)
        - Store old parameter values as reference

        Args:
            dataloader: DataLoader for the current task's training data.
            device: Device for computation.
        """
        raise NotImplementedError("Phase 4: Implement per-modality Fisher computation")

    def ewc_loss(self) -> "torch.Tensor":
        """Compute modality-weighted EWC penalty.

        total = sum over modalities of:
            lambda_m * sum_k(F_k * (theta_k - theta*_k)^2) / 2

        Returns:
            Scalar tensor with total modality-aware EWC penalty.
        """
        raise NotImplementedError("Phase 4: Implement modality-aware EWC loss")
