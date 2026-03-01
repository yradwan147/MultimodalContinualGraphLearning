"""Data I/O utilities for loading and saving experiment artifacts.

Handles reading/writing of various formats: CSV, JSON, PyTorch tensors,
NumPy arrays, and pickle files.

Usage:
    from src.utils.io import save_json, load_json, save_tensor, load_tensor
    save_json(data, 'results/metrics.json')
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Save data to JSON file.

    Args:
        data: JSON-serializable data.
        path: Output file path.
        indent: JSON indentation level.
    """
    raise NotImplementedError("Phase 1: Implement JSON saving")


def load_json(path: str) -> Any:
    """Load data from JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Deserialized data.
    """
    raise NotImplementedError("Phase 1: Implement JSON loading")


def save_tensor(tensor: "torch.Tensor", path: str) -> None:
    """Save PyTorch tensor to disk.

    Args:
        tensor: Tensor to save.
        path: Output file path.
    """
    raise NotImplementedError("Phase 2: Implement tensor saving")


def load_tensor(path: str, device: str = "cpu") -> "torch.Tensor":
    """Load PyTorch tensor from disk.

    Args:
        path: Path to saved tensor.
        device: Device to load tensor onto.

    Returns:
        Loaded tensor.
    """
    raise NotImplementedError("Phase 2: Implement tensor loading")


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist.

    Args:
        path: Directory path.

    Returns:
        Path object for the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
