"""Train/validation/test split creation for continual learning tasks.

Creates non-leaking splits where test triples from task i never appear
in training sets of tasks i+1, i+2, etc. This is critical for valid
continual learning evaluation.

Usage:
    from src.data.splits import create_splits_per_task
    splits = create_splits_per_task(tasks, val_ratio=0.1, test_ratio=0.2)
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def create_splits_per_task(
    tasks: OrderedDict[str, "pd.DataFrame"],
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, dict]:
    """Create train/val/test splits for each task in the sequence.

    For continual learning evaluation:
    - Training set: used for learning the current task
    - Validation set: used for hyperparameter tuning and early stopping
    - Test set: used for evaluating on current AND all previous tasks

    Args:
        tasks: OrderedDict mapping task names to DataFrames of triples.
        val_ratio: Fraction of triples for validation.
        test_ratio: Fraction of triples for testing.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping task_name -> {'train': array, 'val': array, 'test': array,
            'n_train': int, 'n_val': int, 'n_test': int}.
    """
    raise NotImplementedError("Phase 2: Implement split creation")


def verify_no_leakage(
    splits: dict[str, dict],
) -> bool:
    """Verify no data leakage across task splits.

    Ensures test triples from task i do not appear in training
    sets of tasks i+1, i+2, etc.

    Args:
        splits: Split dictionary from create_splits_per_task.

    Returns:
        True if no leakage detected.

    Raises:
        ValueError: If leakage is detected, with details.
    """
    raise NotImplementedError("Phase 2: Implement leakage verification")


def save_splits(
    splits: dict[str, dict],
    output_dir: str,
) -> None:
    """Save splits to disk in LKGE-compatible format.

    Each task gets a directory with train.txt, valid.txt, test.txt.
    Each line: head_entity<tab>relation<tab>tail_entity.

    Args:
        splits: Split dictionary from create_splits_per_task.
        output_dir: Base directory for saving (e.g., data/benchmark/tasks/).
    """
    raise NotImplementedError("Phase 2: Implement split saving")
