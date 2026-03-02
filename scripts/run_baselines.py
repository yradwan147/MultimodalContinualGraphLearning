"""Run baseline experiments on the temporal benchmark.

Supports 4 KGE baselines: naive_sequential, joint_training, ewc,
experience_replay. Runs with configurable seeds and logs results.

Usage:
    # Run a single baseline
    python scripts/run_baselines.py --baseline naive_sequential --tasks-dir data/benchmark/tasks

    # Run with specific tasks (skip the huge base task for local testing)
    python scripts/run_baselines.py --baseline naive_sequential \
        --task-names task_1_disease_related task_3_phenotype_related

    # Run all baselines
    python scripts/run_baselines.py --baseline all

    # Quick local test (small embedding, few epochs)
    python scripts/run_baselines.py --baseline naive_sequential --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def log_memory(label: str) -> None:
    """Log current RSS memory usage."""
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS returns bytes, Linux returns KB
    if sys.platform == "darwin":
        rss_mb = rss_mb / (1024 * 1024)
    else:
        rss_mb = rss_mb / 1024
    logger.info(f"[MEMORY] {label}: {rss_mb:.0f} MB RSS")


SEEDS = [42, 123, 456, 789, 1024]
KGE_BASELINES = ["naive_sequential", "joint_training", "ewc", "experience_replay"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument(
        "--baseline",
        choices=KGE_BASELINES + ["all"],
        required=True,
        help="Which baseline to run",
    )
    parser.add_argument(
        "--tasks-dir",
        default="data/benchmark/tasks",
        help="Path to benchmark tasks directory",
    )
    parser.add_argument(
        "--task-names",
        nargs="+",
        default=None,
        help="Specific task names to use (default: all tasks in directory)",
    )
    parser.add_argument("--model", default="TransE", help="KGE model type")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42],
        help="Random seeds (default: [42]; use --seeds 42 123 456 789 1024 for full)",
    )
    parser.add_argument("--output-dir", default="results", help="Output directory")

    # EWC-specific
    parser.add_argument("--lambda-ewc", type=float, default=10.0)
    parser.add_argument("--fisher-samples", type=int, default=1000)

    # Replay-specific
    parser.add_argument("--buffer-size", type=int, default=500)
    parser.add_argument("--selection-strategy", default="relation_balanced")
    parser.add_argument("--replay-ratio", type=float, default=0.3)

    # Quick mode for local testing
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: small embedding (64), few epochs (10)",
    )

    args = parser.parse_args()

    if args.quick:
        args.embedding_dim = 64
        args.num_epochs = 10
        logger.info("Quick mode: embedding_dim=64, num_epochs=10")

    baselines = KGE_BASELINES if args.baseline == "all" else [args.baseline]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.baselines._base import load_task_sequence
    from src.evaluation.metrics import evaluate_continual_learning

    log_memory("before loading tasks")

    # Load tasks once (returns int arrays + mappings)
    task_seq, entity_to_id, relation_to_id = load_task_sequence(
        args.tasks_dir, args.task_names
    )
    task_names = list(task_seq.keys())
    logger.info(f"Loaded {len(task_names)} tasks: {task_names}")

    # Log per-task sizes
    for name, data in task_seq.items():
        total = sum(len(v) for v in data.values())
        logger.info(f"  {name}: {total:,} triples "
                    f"(train={len(data['train']):,})")
    log_memory("after loading tasks")

    for baseline_name in baselines:
        print(f"\n{'=' * 60}")
        print(f"Baseline: {baseline_name}")
        print(f"Model: {args.model}, dim={args.embedding_dim}, "
              f"epochs={args.num_epochs}, lr={args.lr}")
        print(f"Seeds: {args.seeds}")
        print(f"{'=' * 60}")

        all_seed_results = []

        for seed in args.seeds:
            logger.info(f"\n--- Seed {seed} ---")
            start = time.time()

            log_memory(f"before {baseline_name} seed={seed}")
            R = _run_baseline(
                baseline_name, task_seq,
                entity_to_id, relation_to_id,
                args, seed,
            )
            log_memory(f"after {baseline_name} seed={seed}")

            elapsed = time.time() - start
            logger.info(f"Seed {seed} completed in {elapsed:.1f}s")

            # Compute CL metrics
            cl_metrics = evaluate_continual_learning(R, task_names)
            cl_metrics["seed"] = seed
            cl_metrics["results_matrix"] = R.tolist()
            all_seed_results.append(cl_metrics)

            # Print summary
            for name, val in cl_metrics.items():
                if isinstance(val, float):
                    logger.info(f"  {name}: {val:.4f}")

        # Save results
        result_path = output_dir / f"{baseline_name}_{args.model}.json"
        with open(result_path, "w") as f:
            json.dump({
                "baseline": baseline_name,
                "model": args.model,
                "embedding_dim": args.embedding_dim,
                "num_epochs": args.num_epochs,
                "lr": args.lr,
                "task_names": task_names,
                "seeds": args.seeds,
                "results": all_seed_results,
            }, f, indent=2)
        logger.info(f"Results saved to {result_path}")

        # Print aggregate summary
        if len(all_seed_results) > 1:
            from src.evaluation.statistical import summarize_results
            summary = summarize_results(all_seed_results)
            print(f"\n--- {baseline_name} Summary ({len(args.seeds)} seeds) ---")
            for name, val in summary.items():
                if name not in ("seed", "results_matrix"):
                    print(f"  {name}: {val}")


def _run_baseline(
    name: str,
    task_seq: dict,
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
    args: argparse.Namespace,
    seed: int,
) -> "np.ndarray":
    """Run a single baseline with a single seed. Returns results matrix."""
    import numpy as np

    if name == "naive_sequential":
        from src.baselines.naive_sequential import NaiveSequentialTrainer
        trainer = NaiveSequentialTrainer(
            model_name=args.model,
            embedding_dim=args.embedding_dim,
            num_epochs=args.num_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            seed=seed,
        )
        return trainer.train(task_seq, entity_to_id, relation_to_id)

    elif name == "joint_training":
        from src.baselines.joint_training import JointTrainer
        trainer = JointTrainer(
            model_name=args.model,
            embedding_dim=args.embedding_dim,
            num_epochs=args.num_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            seed=seed,
        )
        result = trainer.train(task_seq, entity_to_id, relation_to_id)
        return result["results_matrix"]

    elif name == "ewc":
        from src.baselines.ewc import EWCTrainer
        trainer = EWCTrainer(
            model_name=args.model,
            embedding_dim=args.embedding_dim,
            num_epochs=args.num_epochs,
            lr=args.lr,
            lambda_ewc=args.lambda_ewc,
            batch_size=args.batch_size,
            fisher_samples=args.fisher_samples,
            device=args.device,
            seed=seed,
        )
        return trainer.train(task_seq, entity_to_id, relation_to_id)

    elif name == "experience_replay":
        from src.baselines.experience_replay import ReplayTrainer
        trainer = ReplayTrainer(
            model_name=args.model,
            embedding_dim=args.embedding_dim,
            num_epochs=args.num_epochs,
            lr=args.lr,
            buffer_size_per_task=args.buffer_size,
            selection_strategy=args.selection_strategy,
            replay_ratio=args.replay_ratio,
            batch_size=args.batch_size,
            device=args.device,
            seed=seed,
        )
        return trainer.train(task_seq, entity_to_id, relation_to_id)

    else:
        raise ValueError(f"Unknown baseline: {name}")


if __name__ == "__main__":
    main()
