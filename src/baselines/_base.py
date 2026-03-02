"""Shared infrastructure for continual KGE baselines.

Provides:
- Memory-efficient task loading (int IDs, not string arrays)
- Global entity/relation mapping across all tasks
- PyKEEN TriplesFactory creation from pre-mapped triples
- Model creation (TransE, ComplEx, DistMult, RotatE)
- Evaluation (MRR, Hits@K) on arbitrary test sets
- Results matrix tracking

All concrete baselines extend this module's functions.
"""

from __future__ import annotations

import logging
import resource
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from pykeen.models import TransE, ComplEx, DistMult, RotatE
from pykeen.triples import TriplesFactory

logger = logging.getLogger(__name__)


def _log_mem(label: str) -> None:
    """Log current RSS memory usage."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    logger.info(f"[MEM] {label}: {rss_mb:.0f} MB")


MODEL_REGISTRY = {
    "TransE": TransE,
    "ComplEx": ComplEx,
    "DistMult": DistMult,
    "RotatE": RotatE,
}


# ---------------------------------------------------------------------------
# Memory-efficient data loading: stream files → int ID arrays directly
# Never creates numpy string arrays (which use fixed-width dtype and
# consume ~50 GB for 8M triples).
# ---------------------------------------------------------------------------

def _scan_vocab(
    tasks_dir: Path,
    task_names: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """Stream all task files to build entity/relation vocabularies.

    Reads files line by line without storing strings in memory.

    Returns:
        (entity_to_id, relation_to_id) dicts.
    """
    entities: set[str] = set()
    relations: set[str] = set()

    for name in task_names:
        for split_file in ("train.txt", "valid.txt", "test.txt"):
            path = tasks_dir / name / split_file
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 3:
                        entities.add(parts[0])
                        entities.add(parts[2])
                        relations.add(parts[1])

    entity_to_id = {e: i for i, e in enumerate(sorted(entities))}
    relation_to_id = {r: i for i, r in enumerate(sorted(relations))}
    logger.info(f"Global vocab: {len(entity_to_id):,} entities, "
                f"{len(relation_to_id)} relations")
    return entity_to_id, relation_to_id


def _load_mapped_triples(
    path: Path,
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
) -> np.ndarray:
    """Load a triples file directly as int64 array using pre-built mappings.

    Args:
        path: Path to tab-separated triples file.
        entity_to_id: Entity string → int mapping.
        relation_to_id: Relation string → int mapping.

    Returns:
        int64 numpy array of shape (n, 3) with columns [head_id, relation_id, tail_id].
    """
    ids: list[list[int]] = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                h, r, t = parts
                if h in entity_to_id and r in relation_to_id and t in entity_to_id:
                    ids.append([entity_to_id[h], relation_to_id[r], entity_to_id[t]])
    if not ids:
        return np.empty((0, 3), dtype=np.int64)
    return np.array(ids, dtype=np.int64)


def load_task_sequence(
    tasks_dir: str | Path,
    task_names: list[str] | None = None,
) -> tuple[OrderedDict[str, dict[str, np.ndarray]], dict[str, int], dict[str, int]]:
    """Load task sequence as memory-efficient int arrays.

    Two-pass approach:
    1. Stream all files to build entity/relation vocabularies
    2. Stream again to convert triples directly to int64 arrays

    Memory usage: ~200 MB for 8M triples (vs ~50 GB with numpy string arrays).

    Args:
        tasks_dir: Path to benchmark/tasks directory.
        task_names: Specific task names to load. If None, loads all sorted.

    Returns:
        Tuple of (tasks, entity_to_id, relation_to_id) where tasks is
        OrderedDict mapping task_name → {'train': int64_array, 'val': ..., 'test': ...}.
    """
    tasks_dir = Path(tasks_dir)

    if task_names is None:
        task_names = sorted([
            d.name for d in tasks_dir.iterdir()
            if d.is_dir() and (d / "train.txt").exists()
        ])

    _log_mem("before loading tasks")

    # Pass 1: build vocab by streaming files
    entity_to_id, relation_to_id = _scan_vocab(tasks_dir, task_names)
    _log_mem("after vocab scan")

    # Pass 2: load triples as int arrays
    split_map = {"train": "train.txt", "val": "valid.txt", "test": "test.txt"}
    tasks: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    for name in task_names:
        task_dir = tasks_dir / name
        task_data: dict[str, np.ndarray] = {}
        for split_key, filename in split_map.items():
            task_data[split_key] = _load_mapped_triples(
                task_dir / filename, entity_to_id, relation_to_id
            )
        tasks[name] = task_data

        total = sum(len(v) for v in task_data.values())
        logger.info(f"Loaded {name}: {total:,} triples "
                    f"(train={len(task_data['train']):,}, "
                    f"val={len(task_data['val']):,}, "
                    f"test={len(task_data['test']):,})")
        _log_mem(f"after loading {name}")

    return tasks, entity_to_id, relation_to_id


def make_triples_factory(
    mapped_triples: np.ndarray,
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
) -> TriplesFactory:
    """Create a PyKEEN TriplesFactory from pre-mapped int triples.

    Args:
        mapped_triples: int64 array of shape (n, 3) with [head_id, rel_id, tail_id].
        entity_to_id: Entity string → int mapping.
        relation_to_id: Relation string → int mapping.

    Returns:
        TriplesFactory ready for PyKEEN training/evaluation.
    """
    tensor = torch.as_tensor(mapped_triples, dtype=torch.long)
    return TriplesFactory(
        mapped_triples=tensor,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )


def create_model(
    model_name: str,
    triples_factory: TriplesFactory,
    embedding_dim: int = 256,
    random_seed: int = 42,
) -> torch.nn.Module:
    """Create a PyKEEN KGE model.

    Args:
        model_name: One of 'TransE', 'ComplEx', 'DistMult', 'RotatE'.
        triples_factory: TriplesFactory defining the entity/relation vocab.
        embedding_dim: Embedding dimension.
        random_seed: Random seed for reproducibility.

    Returns:
        Initialized PyKEEN model.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY)}")

    torch.manual_seed(random_seed)
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(
        triples_factory=triples_factory,
        embedding_dim=embedding_dim,
        random_seed=random_seed,
    )
    return model


def evaluate_link_prediction(
    model: torch.nn.Module,
    test_factory: TriplesFactory,
    device: str = "cpu",
    batch_size: int = 256,
) -> dict[str, float]:
    """Evaluate model on a test set using rank-based metrics.

    Computes MRR, Hits@1, Hits@3, Hits@10 using PyKEEN's evaluator.

    Args:
        model: Trained PyKEEN model.
        test_factory: TriplesFactory for test data.
        device: Device for evaluation.
        batch_size: Evaluation batch size.

    Returns:
        Dict with MRR, Hits@1, Hits@3, Hits@10.
    """
    from pykeen.evaluation import RankBasedEvaluator

    _log_mem(f"before eval ({test_factory.num_triples:,} test triples)")
    model = model.to(device)
    model.eval()
    evaluator = RankBasedEvaluator()

    results = evaluator.evaluate(
        model=model,
        mapped_triples=test_factory.mapped_triples.to(device),
        additional_filter_triples=None,
        batch_size=batch_size,
    )
    _log_mem("after eval")

    # Extract realistic (pessimistic) metrics
    metrics = {
        "MRR": results.get_metric("both.realistic.inverse_harmonic_mean_rank"),
        "Hits@1": results.get_metric("both.realistic.hits_at_1"),
        "Hits@3": results.get_metric("both.realistic.hits_at_3"),
        "Hits@10": results.get_metric("both.realistic.hits_at_10"),
    }
    return metrics


def train_epoch(
    model: torch.nn.Module,
    train_factory: TriplesFactory,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    batch_size: int = 256,
    extra_loss_fn: callable | None = None,
) -> float:
    """Train model for one epoch using sLCWA (stochastic local closed-world assumption).

    Uses negative sampling with the model's built-in loss function.

    Args:
        model: PyKEEN KGE model.
        train_factory: Training TriplesFactory.
        optimizer: PyTorch optimizer.
        device: Device.
        batch_size: Training batch size.
        extra_loss_fn: Optional callable returning additional loss term
            (e.g., EWC penalty). Added to the base KGE loss.

    Returns:
        Average loss over the epoch.
    """
    model = model.to(device)
    model.train()

    mapped = train_factory.mapped_triples.to(device)
    n = mapped.shape[0]

    # Shuffle
    perm = torch.randperm(n, device=device)
    mapped = mapped[perm]

    total_loss = 0.0
    n_batches = 0

    for start in range(0, n, batch_size):
        batch = mapped[start:start + batch_size]

        # Generate negative samples
        neg_batch = _generate_negatives(
            batch, model.num_entities, device=device
        )

        # Compute score-based loss
        pos_scores = model.score_hrt(batch)
        neg_scores = model.score_hrt(neg_batch)

        # Margin ranking loss (for TransE-like models) or
        # use model's built-in loss
        loss = _margin_loss(pos_scores, neg_scores, margin=1.0)

        if extra_loss_fn is not None:
            loss = loss + extra_loss_fn()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def _generate_negatives(
    pos_batch: torch.Tensor,
    num_entities: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate negative samples by corrupting head or tail.

    Randomly replaces head or tail entity with a random entity.

    Args:
        pos_batch: Positive triples of shape (batch, 3).
        num_entities: Total number of entities.
        device: Device.

    Returns:
        Negative triples of shape (batch, 3).
    """
    neg = pos_batch.clone()
    n = neg.shape[0]
    # 50% corrupt head, 50% corrupt tail
    mask = torch.rand(n, device=device) < 0.5
    random_entities = torch.randint(0, num_entities, (n,), device=device)
    neg[mask, 0] = random_entities[mask]
    neg[~mask, 2] = random_entities[~mask]
    return neg


def _margin_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Margin ranking loss for KGE training.

    L = max(0, margin - pos_score + neg_score)

    Higher score = more plausible triple.
    """
    return torch.nn.functional.relu(margin - pos_scores + neg_scores).mean()


def get_device(requested: str = "auto") -> str:
    """Get the best available device.

    Args:
        requested: 'auto', 'cuda', 'mps', or 'cpu'.

    Returns:
        Device string.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested
