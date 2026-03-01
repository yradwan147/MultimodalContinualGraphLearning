"""Baseline 4: Experience Replay for continual KGE (BER-style).

Maintains a memory buffer of representative triples from previous tasks.
During training on new tasks, replays old triples to prevent forgetting.
Supports multiple exemplar selection strategies.

Usage:
    from src.baselines.experience_replay import ExperienceReplayKGE
    replay = ExperienceReplayKGE(model, buffer_size_per_task=500)
    replay.train_task(task_triples, task_id, optimizer, num_epochs)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ExperienceReplayKGE:
    """Experience replay for continual KGE, inspired by BER.

    Args:
        model: The KGE model (PyTorch nn.Module).
        buffer_size_per_task: Max exemplars to store per task.
            Sweep: [100, 250, 500, 1000, 2000].
        selection_strategy: How to select exemplars - 'random',
            'relation_balanced', or 'attention'.
    """

    def __init__(
        self,
        model: "nn.Module",
        buffer_size_per_task: int = 500,
        selection_strategy: str = "relation_balanced",
    ) -> None:
        self.model = model
        self.buffer_size_per_task = buffer_size_per_task
        self.selection_strategy = selection_strategy
        self.memory_buffer: list = []  # List of (triple, task_id) tuples

    def select_exemplars(
        self,
        triples: np.ndarray,
        task_id: int,
    ) -> list[tuple]:
        """Select representative triples for the memory buffer.

        Strategies:
        - 'random': Uniform random selection.
        - 'relation_balanced': Equal samples per relation type.
            Recommended for biomedical KGs with heterogeneous relation types.
        - 'attention': Use model attention scores (BER-style, advanced).

        Args:
            triples: Array of (head, relation, tail) triples.
            task_id: Identifier for the current task.

        Returns:
            List of (triple, task_id) tuples for the buffer.
        """
        raise NotImplementedError("Phase 3: Implement exemplar selection")

    def train_task(
        self,
        task_triples: np.ndarray,
        task_id: int,
        optimizer: "Optimizer",
        num_epochs: int,
        replay_ratio: float = 0.3,
        device: str = "cuda",
    ) -> list[float]:
        """Train on current task with experience replay.

        Combines current task triples with samples from the replay buffer.
        After training, adds new exemplars to the buffer.

        Args:
            task_triples: Training triples for current task.
            task_id: Current task identifier.
            optimizer: PyTorch optimizer.
            num_epochs: Number of training epochs.
            replay_ratio: Fraction of each batch from replay buffer.
            device: Device for computation.

        Returns:
            List of per-epoch loss values.
        """
        raise NotImplementedError("Phase 3: Implement replay training loop")
