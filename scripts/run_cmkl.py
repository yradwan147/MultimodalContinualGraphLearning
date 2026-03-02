"""Run CMKL experiments on the temporal benchmark.

Trains and evaluates the full CMKL model with modality-aware EWC
and multimodal memory replay. Runs with configurable random seeds.

Usage:
    # Quick local test
    python scripts/run_cmkl.py --quick --task-names task_1_disease_related task_3_phenotype_related

    # Full run (for IBEX)
    python scripts/run_cmkl.py --seeds 42 123 456 789 1024

    # With specific decoder
    python scripts/run_cmkl.py --decoder DistMult --embedding-dim 256
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
    parser = argparse.ArgumentParser(description="Run CMKL experiments")
    parser.add_argument(
        "--tasks-dir", default="data/benchmark/tasks",
        help="Path to benchmark tasks directory",
    )
    parser.add_argument(
        "--task-names", nargs="+", default=None,
        help="Specific task names (default: all tasks in directory)",
    )
    parser.add_argument("--decoder", default="DistMult", choices=["TransE", "DistMult", "Bilinear"])
    parser.add_argument("--fusion", default="cross_attention", choices=["cross_attention", "concatenation"])
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-gnn-layers", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--device", default="auto")

    # EWC hyperparameters
    parser.add_argument("--lambda-struct", type=float, default=10.0)
    parser.add_argument("--lambda-text", type=float, default=5.0)
    parser.add_argument("--lambda-mol", type=float, default=1.0)

    # Replay hyperparameters
    parser.add_argument("--replay-buffer-size", type=int, default=1000)
    parser.add_argument("--replay-weight", type=float, default=0.5)

    # Distillation hyperparameters
    parser.add_argument("--use-distillation", action="store_true",
                        help="Enable knowledge distillation")
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--distillation-alpha", type=float, default=0.5)

    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42],
        help="Random seeds (default: [42])",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: dim=64, epochs=5, 1 seed",
    )
    args = parser.parse_args()

    if args.quick:
        args.embedding_dim = 64
        args.num_epochs = 5
        args.num_gnn_layers = 1
        args.num_attention_heads = 2
        args.replay_buffer_size = 100
        logger.info("Quick mode: dim=64, epochs=5, 1 GNN layer")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.baselines._base import load_task_sequence
    from src.evaluation.metrics import evaluate_continual_learning
    from src.models.cmkl import CMKL

    # Load tasks
    task_seq, entity_to_id, relation_to_id = load_task_sequence(
        args.tasks_dir, args.task_names
    )
    task_names = list(task_seq.keys())

    logger.info(f"Tasks: {task_names}")
    logger.info(f"Entities: {len(entity_to_id):,}, Relations: {len(relation_to_id)}")

    config = {
        "num_entities": len(entity_to_id),
        "num_relations": len(relation_to_id),
        "embedding_dim": args.embedding_dim,
        "num_gnn_layers": args.num_gnn_layers,
        "num_attention_heads": args.num_attention_heads,
        "fusion_type": args.fusion,
        "decoder_type": args.decoder,
        "lambda_struct": args.lambda_struct,
        "lambda_text": args.lambda_text,
        "lambda_mol": args.lambda_mol,
        "replay_buffer_size": args.replay_buffer_size,
        "replay_weight": args.replay_weight,
        "use_distillation": args.use_distillation,
        "distillation_temperature": args.distillation_temperature,
        "distillation_alpha": args.distillation_alpha,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
    }

    logger.info(f"Config: dim={config['embedding_dim']}, epochs={config['num_epochs']}, "
                f"decoder={config['decoder_type']}, fusion={config['fusion_type']}")
    logger.info(f"EWC lambdas: struct={config['lambda_struct']}, "
                f"text={config['lambda_text']}, mol={config['lambda_mol']}")
    logger.info(f"Replay: buffer={config['replay_buffer_size']}, weight={config['replay_weight']}")

    all_seed_results = []

    for seed in args.seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Seed {seed}")
        logger.info(f"{'='*60}")
        start = time.time()

        model = CMKL(config)
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"CMKL parameters: {param_count:,}")

        results = model.train_continually(
            task_sequence=task_seq,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            device=args.device,
            seed=seed,
        )

        elapsed = time.time() - start
        logger.info(f"Seed {seed} completed in {elapsed:.1f}s")

        # Compute CL metrics
        R = np.array(results["results_matrix"])
        cl_metrics = evaluate_continual_learning(R, task_names)
        cl_metrics["seed"] = seed
        cl_metrics["results_matrix"] = results["results_matrix"]
        cl_metrics["training_time_s"] = elapsed
        all_seed_results.append(cl_metrics)

        for name, val in cl_metrics.items():
            if isinstance(val, float):
                logger.info(f"  {name}: {val:.4f}")

    # Save results
    result_path = output_dir / f"cmkl_{args.decoder}.json"
    with open(result_path, "w") as f:
        json.dump({
            "method": "cmkl",
            "decoder": args.decoder,
            "fusion": args.fusion,
            "config": config,
            "task_names": task_names,
            "seeds": args.seeds,
            "results": all_seed_results,
        }, f, indent=2)
    logger.info(f"Results saved to {result_path}")

    # Aggregate summary
    if len(all_seed_results) > 1:
        from src.evaluation.statistical import summarize_results
        summary = summarize_results(all_seed_results)
        print(f"\n--- CMKL Summary ({len(args.seeds)} seeds) ---")
        for name, val in summary.items():
            if name not in ("seed", "results_matrix", "training_time_s"):
                print(f"  {name}: {val}")


if __name__ == "__main__":
    main()
