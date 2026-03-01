"""Visualization utilities for experiment results.

Generates publication-quality plots for:
- Results matrix heatmaps (R[i][j] per method)
- Bar charts comparing AP, AF, BWT, FWT across methods
- Forgetting curves over task sequence
- Per-relation-type performance breakdown
- Buffer size and lambda sweep sensitivity plots

Usage:
    from src.evaluation.visualization import plot_results_heatmap, plot_forgetting_curves
    plot_results_heatmap(results_matrix, task_names, save_path='results/figures/heatmap.pdf')
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def plot_results_heatmap(
    results_matrix: "np.ndarray",
    task_names: list[str],
    method_name: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Plot results matrix as a heatmap.

    R[i][j] = performance on task j after training on task i.

    Args:
        results_matrix: Shape (n_tasks, n_tasks).
        task_names: Task labels for axes.
        method_name: Title for the plot.
        save_path: If provided, save figure to this path.
    """
    raise NotImplementedError("Phase 3: Implement results heatmap")


def plot_method_comparison(
    method_metrics: dict[str, dict],
    metrics_to_plot: list[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Bar chart comparing methods across CL metrics.

    Args:
        method_metrics: Dict of method_name -> {metric: value}.
        metrics_to_plot: Which metrics to include (default: AP, AF, BWT, FWT).
        save_path: If provided, save figure to this path.
    """
    raise NotImplementedError("Phase 3: Implement method comparison plot")


def plot_forgetting_curves(
    method_results: dict[str, "np.ndarray"],
    task_names: list[str],
    save_path: Optional[str] = None,
) -> None:
    """Plot forgetting curves over task sequence for multiple methods.

    Shows how performance on task 0 degrades as more tasks are learned.

    Args:
        method_results: Dict of method_name -> results_matrix.
        task_names: Task labels.
        save_path: If provided, save figure to this path.
    """
    raise NotImplementedError("Phase 3: Implement forgetting curves")


def plot_sensitivity_sweep(
    sweep_values: list,
    sweep_results: list[dict],
    param_name: str,
    save_path: Optional[str] = None,
) -> None:
    """Plot hyperparameter sensitivity (buffer size, lambda sweeps).

    Args:
        sweep_values: List of parameter values tested.
        sweep_results: List of metric dicts, one per value.
        param_name: Name of the swept parameter.
        save_path: If provided, save figure to this path.
    """
    raise NotImplementedError("Phase 5: Implement sensitivity plot")
