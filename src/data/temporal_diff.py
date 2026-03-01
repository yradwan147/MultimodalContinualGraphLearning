"""Compute temporal diffs between PrimeKG snapshots.

Identifies added/removed/persistent triples, emerged/disappeared entities,
and new relation types between two KG snapshots. This is the core mechanism
for constructing the continual learning benchmark from real temporal evolution.

Triple identifiers are constructed as: x_id|relation|y_id

Usage:
    from src.data.temporal_diff import compute_kg_diff
    stats, added, removed, emerged = compute_kg_diff('data/kg_t0.csv', 'data/kg_t1.csv')
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def compute_kg_diff(
    kg_old_path: str,
    kg_new_path: str,
) -> tuple[dict, set, set, set]:
    """Compute differences between two KG snapshots.

    Creates triple identifiers (x_id|relation|y_id), then computes set
    differences to find added, removed, and persistent triples. Also
    identifies emerged/disappeared entities and new relation types.

    Args:
        kg_old_path: Path to older snapshot CSV.
        kg_new_path: Path to newer snapshot CSV.

    Returns:
        Tuple of (stats_dict, added_triples, removed_triples, emerged_entities).
        stats_dict contains counts for all categories.
    """
    raise NotImplementedError("Phase 2: Implement temporal diff computation")


def normalize_entity_ids(
    kg_old: pd.DataFrame,
    kg_new: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize entity IDs across snapshots.

    Some databases may change their ID schemes between releases.
    This creates a mapping using entity names as secondary keys.

    Args:
        kg_old: Older snapshot DataFrame.
        kg_new: Newer snapshot DataFrame.

    Returns:
        Tuple of normalized DataFrames with consistent IDs.
    """
    raise NotImplementedError("Phase 2: Implement ID normalization")


def save_diff_report(
    stats: dict,
    output_path: str,
) -> None:
    """Save temporal diff statistics as JSON.

    Args:
        stats: Dictionary from compute_kg_diff.
        output_path: Path to save JSON report.
    """
    raise NotImplementedError("Phase 2: Implement diff report saving")
