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

import numpy as np
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Link prediction metrics
# ---------------------------------------------------------------------------

def compute_mrr(ranks: np.ndarray) -> float:
    """Compute Mean Reciprocal Rank.

    Args:
        ranks: Array of integer ranks for correct entities (1-based).

    Returns:
        MRR score in [0, 1].
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    return float(np.mean(1.0 / ranks))


def compute_hits_at_k(ranks: np.ndarray, k: int = 10) -> float:
    """Compute Hits@K metric.

    Args:
        ranks: Array of integer ranks for correct entities (1-based).
        k: Cutoff value (1, 3, or 10).

    Returns:
        Hits@K score (proportion of ranks <= k).
    """
    ranks = np.asarray(ranks)
    return float(np.mean(ranks <= k))


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
    return float(average_precision_score(y_true, y_scores))


def compute_link_prediction_metrics(
    ranks: np.ndarray,
    ks: tuple[int, ...] = (1, 3, 10),
) -> dict[str, float]:
    """Compute all standard link prediction metrics from ranks.

    Args:
        ranks: Array of integer ranks (1-based) for correct entities.
        ks: Tuple of K values for Hits@K.

    Returns:
        Dict with 'MRR' and 'Hits@K' for each K.
    """
    ranks = np.asarray(ranks)
    metrics = {"MRR": compute_mrr(ranks)}
    for k in ks:
        metrics[f"Hits@{k}"] = compute_hits_at_k(ranks, k)
    return metrics


# ---------------------------------------------------------------------------
# Continual learning metrics
# ---------------------------------------------------------------------------

def evaluate_continual_learning(
    results_matrix: np.ndarray,
    task_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute all continual learning metrics from results matrix R.

    R[i][j] = performance on task j's test set after training on task i.
    The matrix is lower-triangular: R[i][j] is only valid when i >= j
    (can only evaluate on tasks seen so far).

    Args:
        results_matrix: NumPy array of shape (n_tasks, n_tasks).
        task_names: Optional list of task names for logging.

    Returns:
        Dict with AP, AF, BWT, FWT, REM metrics.
    """
    R = np.asarray(results_matrix, dtype=np.float64)
    n = R.shape[0]

    if n < 2:
        return {
            "Average Performance (AP)": float(R[0, 0]),
            "Average Forgetting (AF)": 0.0,
            "Backward Transfer (BWT)": 0.0,
            "Forward Transfer (FWT)": 0.0,
            "Remembering (REM)": 1.0,
        }

    # AP: Average Performance after training on the final task
    # AP = (1/n) * sum_{j=0}^{n-1} R[n-1, j]
    ap = float(np.mean(R[-1, :]))

    # AF: Average Forgetting
    # For each task j < n-1: forgetting_j = max_{i in [j..n-2]} R[i,j] - R[n-1,j]
    forgetting = []
    for j in range(n - 1):
        max_perf = np.max(R[j:n - 1, j])  # best perf on task j before final task
        forgetting.append(max_perf - R[-1, j])
    af = float(np.mean(forgetting))

    # BWT: Backward Transfer
    # BWT = (1/(n-1)) * sum_{j=0}^{n-2} (R[n-1, j] - R[j, j])
    bwt_vals = [R[-1, j] - R[j, j] for j in range(n - 1)]
    bwt = float(np.mean(bwt_vals))

    # FWT: Forward Transfer
    # FWT = (1/(n-1)) * sum_{j=1}^{n-1} R[j-1, j]
    # R[j-1, j] = performance on task j BEFORE training on it
    # (using model trained up to task j-1)
    fwt_vals = [R[j - 1, j] for j in range(1, n)]
    fwt = float(np.mean(fwt_vals))

    # REM: Remembering = 1 - |min(BWT, 0)|
    rem = 1.0 - abs(min(bwt, 0.0))

    metrics = {
        "Average Performance (AP)": ap,
        "Average Forgetting (AF)": af,
        "Backward Transfer (BWT)": bwt,
        "Forward Transfer (FWT)": fwt,
        "Remembering (REM)": rem,
    }

    if task_names:
        logger.info(f"CL Metrics ({len(task_names)} tasks):")
        for name, val in metrics.items():
            logger.info(f"  {name}: {val:.4f}")

    return metrics
