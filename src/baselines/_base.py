"""Shared infrastructure for continual KGE baselines.

Provides ContinualKGETrainer base class with:
- Task data loading from benchmark directory
- Global entity/relation mapping across all tasks
- PyKEEN TriplesFactory creation with shared vocab
- Model creation (TransE, ComplEx, DistMult, RotatE)
- Evaluation (MRR, Hits@K) on arbitrary test sets
- Results matrix tracking

All concrete baselines extend this class.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from pykeen.models import TransE, ComplEx, DistMult, RotatE
from pykeen.triples import TriplesFactory

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "TransE": TransE,
    "ComplEx": ComplEx,
    "DistMult": DistMult,
    "RotatE": RotatE,
}


def load_triples_file(path: str | Path) -> np.ndarray:
    """Load a tab-separated triples file (head, relation, tail).

    Args:
        path: Path to .txt file with tab-separated triples.

    Returns:
        NumPy array of shape (n, 3) with string entries.
    """
    triples = []
    with open(str(path)) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                triples.append(parts)
    return np.array(triples, dtype=str)


def load_task_sequence(
    tasks_dir: str | Path,
    task_names: list[str] | None = None,
) -> OrderedDict[str, dict[str, np.ndarray]]:
    """Load task sequence from benchmark directory.

    Each task directory should contain train.txt, valid.txt, test.txt
    with tab-separated triples (head, relation, tail).

    Args:
        tasks_dir: Path to benchmark/tasks directory.
        task_names: Specific task names to load. If None, loads all
            sorted alphabetically.

    Returns:
        OrderedDict mapping task_name -> {'train': array, 'val': array, 'test': array}.
    """
    tasks_dir = Path(tasks_dir)

    if task_names is None:
        task_names = sorted([
            d.name for d in tasks_dir.iterdir()
            if d.is_dir() and (d / "train.txt").exists()
        ])

    tasks = OrderedDict()
    for name in task_names:
        task_dir = tasks_dir / name
        tasks[name] = {
            "train": load_triples_file(task_dir / "train.txt"),
            "val": load_triples_file(task_dir / "valid.txt"),
            "test": load_triples_file(task_dir / "test.txt"),
        }
        total = sum(len(v) for v in tasks[name].values())
        logger.info(f"Loaded {name}: {total:,} triples "
                    f"(train={len(tasks[name]['train']):,}, "
                    f"val={len(tasks[name]['val']):,}, "
                    f"test={len(tasks[name]['test']):,})")

    return tasks


def build_global_mappings(
    task_sequence: OrderedDict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Build entity_to_id and relation_to_id covering all tasks.

    Args:
        task_sequence: Full task sequence from load_task_sequence.

    Returns:
        Tuple of (entity_to_id, relation_to_id) dicts.
    """
    entities = set()
    relations = set()

    for task_data in task_sequence.values():
        for split in task_data.values():
            if len(split) > 0:
                entities.update(split[:, 0])  # heads
                entities.update(split[:, 2])  # tails
                relations.update(split[:, 1])  # relations

    entity_to_id = {e: i for i, e in enumerate(sorted(entities))}
    relation_to_id = {r: i for i, r in enumerate(sorted(relations))}

    logger.info(f"Global vocab: {len(entity_to_id):,} entities, "
                f"{len(relation_to_id)} relations")
    return entity_to_id, relation_to_id


def make_triples_factory(
    triples: np.ndarray,
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
) -> TriplesFactory:
    """Create a PyKEEN TriplesFactory with global mappings.

    Args:
        triples: Array of shape (n, 3) with string (head, relation, tail).
        entity_to_id: Global entity mapping.
        relation_to_id: Global relation mapping.

    Returns:
        TriplesFactory with mapped triples.
    """
    return TriplesFactory.from_labeled_triples(
        triples,
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

    model = model.to(device)
    model.eval()
    evaluator = RankBasedEvaluator()

    results = evaluator.evaluate(
        model=model,
        mapped_triples=test_factory.mapped_triples.to(device),
        additional_filter_triples=None,
        batch_size=batch_size,
    )

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
