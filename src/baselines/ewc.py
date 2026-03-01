"""Baseline 3: Elastic Weight Consolidation (EWC) for Knowledge Graph Embeddings.

Protects important parameters learned in previous tasks by adding a quadratic
penalty based on the Fisher Information Matrix. After each task, computes the
Fisher diagonal to identify important parameters, then adds a penalty term
to the loss when training on subsequent tasks.

L_total = L_task + (lambda/2) * sum_k F_k * (theta_k - theta*_k)^2

Reference: EWC with lambda=10 reduced forgetting from 12.62% to 6.85% on FB15k-237.

Usage:
    from src.baselines.ewc import EWC_KGE
    ewc = EWC_KGE(model, lambda_ewc=10.0)
    ewc.train_task(dataloader, optimizer, num_epochs)
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Optional

logger = logging.getLogger(__name__)


class EWC_KGE:
    """Elastic Weight Consolidation for Knowledge Graph Embeddings.

    Args:
        model: The KGE model (PyTorch nn.Module).
        lambda_ewc: Regularization strength. Start with 10.0,
            sweep over [0.1, 1.0, 10.0, 100.0, 1000.0].
    """

    def __init__(self, model: "nn.Module", lambda_ewc: float = 10.0) -> None:
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_diags: dict = {}  # param_name -> Fisher diagonal
        self.old_params: dict = {}  # param_name -> parameter values after previous task

    def compute_fisher(
        self,
        dataloader: "DataLoader",
        device: str = "cuda",
    ) -> None:
        """Compute diagonal of Fisher Information Matrix after training on a task.

        Uses the empirical Fisher approximation:
        F_k ~ (1/N) * sum_i (dL_i/d_theta_k)^2

        Accumulates Fisher across tasks (sum) and stores current parameters
        as reference for the EWC penalty.

        Args:
            dataloader: DataLoader for the task's training data.
            device: Device for computation.
        """
        raise NotImplementedError("Phase 3: Implement Fisher computation")

    def ewc_loss(self) -> "torch.Tensor":
        """Compute the EWC penalty term.

        L_ewc = (lambda/2) * sum_k F_k * (theta_k - theta*_k)^2

        Returns:
            Scalar tensor with EWC penalty.
        """
        raise NotImplementedError("Phase 3: Implement EWC loss")

    def train_task(
        self,
        dataloader: "DataLoader",
        optimizer: "Optimizer",
        num_epochs: int,
        device: str = "cuda",
    ) -> list[float]:
        """Train on a single task with EWC regularization.

        After training, automatically computes Fisher for this task.

        Args:
            dataloader: Training data for current task.
            optimizer: PyTorch optimizer.
            num_epochs: Number of training epochs.
            device: Device for computation.

        Returns:
            List of per-epoch total loss values.
        """
        raise NotImplementedError("Phase 3: Implement EWC training loop")
