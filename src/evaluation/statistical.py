"""Statistical significance testing for experiment comparisons.

Uses paired t-tests or Wilcoxon signed-rank tests to compare methods
across random seeds. All experiments use 5 seeds: [42, 123, 456, 789, 1024].
Report mean +/- std with significance at p < 0.05.

Usage:
    from src.evaluation.statistical import compute_significance, summarize_results
    sig = compute_significance(results_method_a, results_method_b)
    summary = summarize_results(all_seed_results)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 1024]


def compute_significance(
    results_method_a: np.ndarray,
    results_method_b: np.ndarray,
    alpha: float = 0.05,
    test: str = "paired_t",
) -> dict:
    """Test statistical significance between two methods across random seeds.

    Args:
        results_method_a: Array of metric values across seeds for method A.
        results_method_b: Array of metric values across seeds for method B.
        alpha: Significance threshold.
        test: Test type - 'paired_t' or 'wilcoxon'.

    Returns:
        Dict with 't_statistic' (or 'w_statistic'), 'p_value', 'significant'.
    """
    raise NotImplementedError("Phase 3: Implement significance testing")


def summarize_results(
    all_seed_results: list[dict],
) -> dict[str, str]:
    """Aggregate results across seeds into mean +/- std format.

    Args:
        all_seed_results: List of metric dicts, one per seed.

    Returns:
        Dict mapping metric_name -> "mean +/- std" string.
    """
    raise NotImplementedError("Phase 3: Implement result summarization")


def pairwise_significance_table(
    method_results: dict[str, list[dict]],
    metric: str = "Average Performance (AP)",
    alpha: float = 0.05,
) -> "pd.DataFrame":
    """Compute pairwise significance between all methods.

    Args:
        method_results: Dict mapping method_name -> list of per-seed results.
        metric: Which metric to compare.
        alpha: Significance threshold.

    Returns:
        DataFrame with p-values for all method pairs.
    """
    raise NotImplementedError("Phase 3: Implement pairwise significance table")
