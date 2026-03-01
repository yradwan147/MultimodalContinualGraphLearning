"""Baseline 1: Naive Sequential Training (Lower Bound).

Trains a KGE model (TransE/ComplEx/DistMult/RotatE) on each task sequentially
without any continual learning mechanism. Establishes the maximum-forgetting
reference point. Uses PyKEEN for KGE training and evaluation.

Expected outcome: High performance on the most recent task but severe
degradation on older tasks.

Usage:
    from src.baselines.naive_sequential import NaiveSequentialTrainer
    trainer = NaiveSequentialTrainer(model_name='TransE')
    results_matrix = trainer.train(task_sequence)
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class NaiveSequentialTrainer:
    """Train KGE model sequentially without any CL mechanism.

    Collects all entity and relation IDs upfront for consistent indexing,
    then trains on each task using the previous model as initialization.

    Args:
        model_name: KGE model type - 'TransE', 'ComplEx', 'DistMult', or 'RotatE'.
        embedding_dim: Dimension of entity/relation embeddings.
        num_epochs: Training epochs per task.
        lr: Learning rate.
        device: 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        model_name: str = "TransE",
        embedding_dim: int = 256,
        num_epochs: int = 100,
        lr: float = 0.001,
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device
        self.model = None

    def train(
        self,
        task_sequence: OrderedDict[str, dict],
    ) -> np.ndarray:
        """Train sequentially on all tasks and build results matrix.

        Args:
            task_sequence: OrderedDict of {task_name: {'train': array,
                'val': array, 'test': array}}.

        Returns:
            results_matrix: R[i][j] = performance on task j's test set
                after training on task i. Shape (n_tasks, n_tasks).
        """
        raise NotImplementedError("Phase 3: Implement naive sequential training")

    def _build_global_mappings(
        self,
        task_sequence: OrderedDict[str, dict],
    ) -> tuple[dict, dict]:
        """Build entity_to_id and relation_to_id covering all tasks.

        Args:
            task_sequence: Full task sequence.

        Returns:
            Tuple of (entity_to_id, relation_to_id) dicts.
        """
        raise NotImplementedError("Phase 3: Implement global ID mappings")

    def _evaluate_task(self, task_data: dict) -> float:
        """Evaluate current model on a single task's test set.

        Args:
            task_data: Dict with 'test' key containing test triples.

        Returns:
            MRR score on the test set.
        """
        raise NotImplementedError("Phase 3: Implement task evaluation")
