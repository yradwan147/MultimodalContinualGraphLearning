"""Experiment configuration management.

Loads YAML config files and provides a unified configuration interface.
Supports config merging (base + experiment-specific) and CLI overrides.

Usage:
    from src.utils.config import load_config
    config = load_config('configs/base.yaml', overrides={'training.lr': 0.0005})
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def load_config(
    config_path: str,
    overrides: Optional[dict[str, Any]] = None,
) -> dict:
    """Load experiment configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.
        overrides: Dict of dot-separated keys to override values.
            e.g., {'training.lr': 0.0005, 'model.embedding_dim': 128}.

    Returns:
        Configuration dictionary.
    """
    raise NotImplementedError("Phase 1: Implement config loading")


def merge_configs(base: dict, override: dict) -> dict:
    """Deep merge two configuration dicts (override takes precedence).

    Args:
        base: Base configuration.
        override: Override configuration.

    Returns:
        Merged configuration dict.
    """
    raise NotImplementedError("Phase 1: Implement config merging")


def save_config(config: dict, output_path: str) -> None:
    """Save configuration to YAML file (for experiment reproducibility).

    Args:
        config: Configuration dictionary.
        output_path: Path to save YAML file.
    """
    raise NotImplementedError("Phase 1: Implement config saving")
