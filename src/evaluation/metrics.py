"""Evaluation metrics for continual knowledge graph learning.

Link prediction metrics: MRR, Hits@K, AUPRC
Continual learning metrics: AP, AF, BWT, FWT, REM

The results matrix R[i][j] represents performance on task j's test set
after training on task i. This matrix is the core data structure for
computing all continual learning metrics.

Usage:
    from src.evaluation.metrics import evaluate_continual_learning, compute_mrr
    cl_metrics = evaluate_continual_learning(results_matrix, task_names)
    mrr = compute_mrr(ranks)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def evaluate_continual_learning(
    results_matrix: np.ndarray,
    task_names: list[str],
) -> dict[str, float]:
    """Compute all continual learning metrics from results matrix R.

    R[i][j] = performance on task j's test set after training on task i.

    Args:
        results_matrix: NumPy array of shape (n_tasks, n_tasks).
        task_names: List of task names for reporting.

    Returns:
        Dict with 'Average Performance (AP)', 'Average Forgetting (AF)',
        'Backward Transfer (BWT)', 'Forward Transfer (FWT)',
        'Remembering (REM)'.
    """
    raise NotImplementedError("Phase 3: Implement CL metrics computation")


def compute_mrr(ranks: np.ndarray) -> float:
    """Compute Mean Reciprocal Rank.

    Args:
        ranks: Array of ranks for correct entities.

    Returns:
        MRR score.
    """
    raise NotImplementedError("Phase 3: Implement MRR")


def compute_hits_at_k(ranks: np.ndarray, k: int = 10) -> float:
    """Compute Hits@K metric.

    Args:
        ranks: Array of ranks for correct entities.
        k: Cutoff value (1, 3, or 10).

    Returns:
        Hits@K score (proportion of ranks <= k).
    """
    raise NotImplementedError("Phase 3: Implement Hits@K")


def compute_auprc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> float:
    """Compute Area Under Precision-Recall Curve.

    Used for drug repurposing evaluation following TxGNN protocol.

    Args:
        y_true: Binary ground truth labels.
        y_scores: Predicted scores.

    Returns:
        AUPRC score.
    """
    raise NotImplementedError("Phase 3: Implement AUPRC")
