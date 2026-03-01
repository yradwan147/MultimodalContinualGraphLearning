"""Run baseline experiments on the temporal benchmark.

Supports all 6 baselines: naive_sequential, joint_training, ewc,
experience_replay, lkge, rag_agent. Runs with 5 random seeds and
logs results to results/ directory.

Usage:
    python scripts/run_baselines.py --baseline ewc --config configs/ewc.yaml
    python scripts/run_baselines.py --baseline all --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEEDS = [42, 123, 456, 789, 1024]
BASELINES = [
    "naive_sequential",
    "joint_training",
    "ewc",
    "experience_replay",
    "lkge",
    "rag_agent",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument(
        "--baseline",
        choices=BASELINES + ["all"],
        required=True,
        help="Which baseline to run",
    )
    parser.add_argument("--config", default="configs/base.yaml", help="Config file")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Random seeds",
    )
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    from src.utils.config import load_config

    config = load_config(args.config)
    baselines = BASELINES if args.baseline == "all" else [args.baseline]

    for baseline_name in baselines:
        print(f"\n{'='*60}")
        print(f"Running baseline: {baseline_name}")
        print(f"Seeds: {args.seeds}")
        print(f"{'='*60}")
        # Phase 3: Implement baseline dispatch and execution


if __name__ == "__main__":
    main()
