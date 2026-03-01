"""Experiment logging and tracking utilities.

Provides unified logging to console, file, and optionally Weights & Biases.
Tracks training metrics, evaluation results, and experiment metadata.

Usage:
    from src.utils.logging import setup_logger, ExperimentTracker
    logger = setup_logger('experiment_name', log_dir='results/logs')
    tracker = ExperimentTracker(use_wandb=False)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def setup_logger(
    name: str,
    log_dir: str = "results/logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up a logger with console and file handlers.

    Args:
        name: Logger name (used for log filename).
        log_dir: Directory for log files.
        level: Logging level.

    Returns:
        Configured Logger instance.
    """
    raise NotImplementedError("Phase 1: Implement logger setup")


class ExperimentTracker:
    """Track experiment metrics to file and optionally W&B.

    Args:
        experiment_name: Name for the experiment run.
        results_dir: Directory for saving results.
        use_wandb: Whether to log to Weights & Biases.
        wandb_project: W&B project name.
    """

    def __init__(
        self,
        experiment_name: str = "mcgl",
        results_dir: str = "results",
        use_wandb: bool = False,
        wandb_project: str = "mcgl",
    ) -> None:
        self.experiment_name = experiment_name
        self.results_dir = Path(results_dir)
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics dict.

        Args:
            metrics: Dict of metric_name -> value.
            step: Optional step number (epoch, task, etc.).
        """
        raise NotImplementedError("Phase 1: Implement metric logging")

    def log_config(self, config: dict) -> None:
        """Log experiment configuration.

        Args:
            config: Configuration dict.
        """
        raise NotImplementedError("Phase 1: Implement config logging")

    def save_results(self, results: dict, filename: str) -> str:
        """Save results dict to JSON file.

        Args:
            results: Results dictionary.
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        raise NotImplementedError("Phase 1: Implement results saving")
