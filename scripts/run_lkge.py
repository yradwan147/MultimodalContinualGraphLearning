"""Run LKGE baseline on the temporal benchmark.

Converts our benchmark to LKGE format, runs LKGE as a subprocess,
parses results, and computes CL metrics.

Usage:
    # Quick mode: only test format conversion (no LKGE run)
    python scripts/run_lkge.py --quick

    # Full run (requires LKGE repo cloned to external/LKGE)
    python scripts/run_lkge.py --seeds 42 123 456 789 1024

    # Custom LKGE directory
    python scripts/run_lkge.py --lkge-dir /path/to/LKGE --model TransE
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 1024]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LKGE baseline")
    parser.add_argument("--tasks-dir", default="data/benchmark/tasks")
    parser.add_argument("--task-names", nargs="+", default=None)
    parser.add_argument("--model", default="TransE",
                        choices=["TransE", "DistMult", "ComplEx", "RotatE"])
    parser.add_argument("--lkge-dir", default="external/LKGE",
                        help="Path to cloned LKGE repository")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: only test format conversion, no LKGE run")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.baselines._base import load_task_sequence
    from src.baselines.lkge import LKGEWrapper
    from src.evaluation.metrics import evaluate_continual_learning

    # Load tasks
    task_seq, entity_to_id, relation_to_id = load_task_sequence(
        args.tasks_dir, args.task_names
    )
    task_names = list(task_seq.keys())

    logger.info(f"Tasks: {task_names}")
    logger.info(f"Entities: {len(entity_to_id):,}, Relations: {len(relation_to_id)}")

    # Convert to LKGE format
    wrapper = LKGEWrapper(lkge_dir=args.lkge_dir)
    lkge_data_dir = str(output_dir / "lkge_format")
    wrapper.convert_to_lkge_format(task_seq, lkge_data_dir)

    # Verify format conversion
    lkge_path = Path(lkge_data_dir)
    assert (lkge_path / "entity2id.txt").exists(), "entity2id.txt not created"
    assert (lkge_path / "relation2id.txt").exists(), "relation2id.txt not created"
    n_snapshots = len(list(lkge_path.glob("snapshot_*")))
    logger.info(f"LKGE format: {n_snapshots} snapshots in {lkge_data_dir}")

    if args.quick:
        logger.info("Quick mode: format conversion successful, skipping LKGE run")
        # Create a dummy result for testing
        result = {
            "method": "lkge",
            "model": args.model,
            "task_names": task_names,
            "num_snapshots": n_snapshots,
            "lkge_data_dir": lkge_data_dir,
            "status": "format_conversion_only",
            "lkge_command": wrapper.get_run_command(
                lkge_data_dir, model=args.model, num_epochs=args.num_epochs),
        }
        result_path = output_dir / f"lkge_{args.model}_quick.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Quick result saved to {result_path}")
        return

    # Full run: execute LKGE for each seed
    all_seed_results = []

    for seed in args.seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"LKGE Seed {seed}")
        logger.info(f"{'='*60}")
        start = time.time()

        seed_output = str(output_dir / f"lkge_output_seed{seed}")
        results = wrapper.run_and_parse(
            dataset_dir=lkge_data_dir,
            output_dir=seed_output,
            model=args.model,
            num_epochs=args.num_epochs,
            seed=seed,
        )
        elapsed = time.time() - start

        if "error" in results:
            logger.error(f"LKGE failed for seed {seed}: {results['error']}")
            continue

        results["seed"] = seed
        results["training_time_s"] = elapsed

        # Compute CL metrics if we have a results matrix
        if results.get("results_matrix"):
            R = np.array(results["results_matrix"])
            cl_metrics = evaluate_continual_learning(R, task_names)
            results.update(cl_metrics)
            logger.info(f"  AP={cl_metrics['Average Performance (AP)']:.4f}, "
                        f"AF={cl_metrics['Average Forgetting (AF)']:.4f}")

        all_seed_results.append(results)

    # Save results
    result_path = output_dir / f"lkge_{args.model}.json"
    with open(result_path, "w") as f:
        json.dump({
            "method": "lkge",
            "model": args.model,
            "task_names": task_names,
            "seeds": args.seeds,
            "results": all_seed_results,
        }, f, indent=2, default=str)
    logger.info(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
