"""Baseline 2: Joint Training (Upper Bound).

Trains on all tasks simultaneously by concatenating all training data.
This is the upper bound for performance since there is no forgetting
by definition. The gap between joint and naive sequential quantifies
how much forgetting matters on this benchmark.

Usage:
    from src.baselines.joint_training import JointTrainer
    trainer = JointTrainer(model_name='TransE')
    per_task_results = trainer.train(task_sequence)
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class JointTrainer:
    """Train KGE model on all tasks simultaneously (upper bound reference).

    Args:
        model_name: KGE model type - 'TransE', 'ComplEx', 'DistMult', or 'RotatE'.
        embedding_dim: Dimension of entity/relation embeddings.
        num_epochs: Training epochs.
        lr: Learning rate.
        device: 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        model_name: str = "TransE",
        embedding_dim: int = 256,
        num_epochs: int = 200,
        lr: float = 0.001,
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device

    def train(
        self,
        task_sequence: OrderedDict[str, dict],
    ) -> dict[str, dict]:
        """Train on concatenated data from all tasks.

        Args:
            task_sequence: OrderedDict of {task_name: {'train': array,
                'val': array, 'test': array}}.

        Returns:
            Dict mapping task_name -> metrics dict for each task's test set.
        """
        raise NotImplementedError("Phase 3: Implement joint training")
