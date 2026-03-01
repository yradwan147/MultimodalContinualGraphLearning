"""Run ablation studies for CMKL.

7 ablation studies:
1. Struct only (no text, no mol)
2. Text only (no struct, no mol)
3. Concatenation fusion (vs. cross-attention)
4. Global EWC (vs. modality-aware)
5. Random replay (vs. K-means diverse)
6. Buffer size sweep (100-5000)
7. Lambda sweep (per-modality)

Usage:
    python scripts/run_ablations.py --ablation struct_only --config configs/cmkl.yaml
    python scripts/run_ablations.py --ablation all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEEDS = [42, 123, 456, 789, 1024]
ABLATIONS = [
    "struct_only",
    "text_only",
    "concat_fusion",
    "global_ewc",
    "random_replay",
    "buffer_size_sweep",
    "lambda_sweep",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument(
        "--ablation",
        choices=ABLATIONS + ["all"],
        required=True,
        help="Which ablation to run",
    )
    parser.add_argument("--config", default="configs/cmkl.yaml", help="Base config")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Random seeds",
    )
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    ablations = ABLATIONS if args.ablation == "all" else [args.ablation]

    for ablation_name in ablations:
        print(f"\n{'='*60}")
        print(f"Running ablation: {ablation_name}")
        print(f"{'='*60}")
        # Phase 5: Implement ablation dispatch and execution


if __name__ == "__main__":
    main()
