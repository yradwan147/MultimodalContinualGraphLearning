"""Define continual learning task sequences from temporal KG diffs.

Groups new triples into discrete learning tasks using various strategies.
The entity_type strategy is recommended as it mirrors real-world knowledge
evolution (new drugs approved vs. new diseases characterized).

Strategies:
- entity_type: Group by entity type (drug, disease, gene/protein)
- relation_type: Group by relation type
- temporal: Use publication dates for finer-grained splits

Usage:
    from src.data.task_sequence import create_task_sequence
    tasks = create_task_sequence(kg_t0, kg_t1, strategy='entity_type')
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def create_task_sequence(
    kg_t0: pd.DataFrame,
    kg_t1: pd.DataFrame,
    strategy: str = "entity_type",
) -> OrderedDict[str, pd.DataFrame]:
    """Create a sequence of tasks from temporal KG diffs.

    Each task contains triples grouped by the chosen strategy,
    representing a distinct learning episode in the continual
    learning setup.

    Args:
        kg_t0: Older KG snapshot DataFrame.
        kg_t1: Newer KG snapshot DataFrame.
        strategy: Grouping strategy - 'entity_type', 'relation_type',
            or 'temporal'.

    Returns:
        OrderedDict mapping task names to DataFrames of new triples.
    """
    raise NotImplementedError("Phase 2: Implement task sequence creation")


def _entity_type_strategy(
    added_triples: pd.DataFrame,
) -> OrderedDict[str, pd.DataFrame]:
    """Group new triples by the entity type they introduce.

    New drug-related triples become one task, new disease-related
    triples another, etc. This mirrors real-world knowledge evolution.

    Args:
        added_triples: DataFrame of triples added between snapshots.

    Returns:
        OrderedDict of task_name -> triples_dataframe.
    """
    raise NotImplementedError("Phase 2: Implement entity_type strategy")


def _relation_type_strategy(
    added_triples: pd.DataFrame,
) -> OrderedDict[str, pd.DataFrame]:
    """Group new triples by relation type.

    Args:
        added_triples: DataFrame of triples added between snapshots.

    Returns:
        OrderedDict of task_name -> triples_dataframe.
    """
    raise NotImplementedError("Phase 2: Implement relation_type strategy")


def validate_task_sequence(
    tasks: OrderedDict[str, pd.DataFrame],
    min_triples: int = 100,
) -> OrderedDict[str, pd.DataFrame]:
    """Validate and clean task sequence.

    Merges tasks with fewer than min_triples into related tasks
    and verifies no overlap between tasks.

    Args:
        tasks: Task sequence from create_task_sequence.
        min_triples: Minimum triples per task (smaller tasks get merged).

    Returns:
        Validated and potentially merged task sequence.
    """
    raise NotImplementedError("Phase 2: Implement task validation")
