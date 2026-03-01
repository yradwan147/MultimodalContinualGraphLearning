"""Run CMKL experiments on the temporal benchmark.

Trains and evaluates the full CMKL model with modality-aware EWC
and multimodal memory replay. Runs with 5 random seeds.

Usage:
    python scripts/run_cmkl.py --config configs/cmkl.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEEDS = [42, 123, 456, 789, 1024]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CMKL experiments")
    parser.add_argument("--config", default="configs/cmkl.yaml", help="Config file")
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

    print(f"Running CMKL with config: {args.config}")
    print(f"Seeds: {args.seeds}")
    # Phase 4: Implement CMKL training and evaluation


if __name__ == "__main__":
    main()
