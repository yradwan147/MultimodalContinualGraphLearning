"""Baseline 5: LKGE (Lifelong Knowledge Graph Embedding) wrapper.

Wraps the external LKGE framework (https://github.com/nju-websoft/LKGE)
for use with our PrimeKG temporal benchmark. Handles data format conversion
between our benchmark format and LKGE's expected directory structure.

Usage:
    from src.baselines.lkge import LKGEWrapper
    wrapper = LKGEWrapper(lkge_dir='external/LKGE')
    wrapper.convert_and_run(task_sequence, output_dir='results/lkge')
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LKGEWrapper:
    """Wrapper for the LKGE external framework.

    Args:
        lkge_dir: Path to cloned LKGE repository.
        gpu_id: GPU device ID for LKGE.
    """

    def __init__(
        self,
        lkge_dir: str = "external/LKGE",
        gpu_id: int = 0,
    ) -> None:
        self.lkge_dir = Path(lkge_dir)
        self.gpu_id = gpu_id

    def convert_to_lkge_format(
        self,
        task_sequence: dict,
        output_dir: str,
    ) -> str:
        """Convert task sequence to LKGE's expected directory format.

        Creates snapshot_0/, snapshot_1/, etc. with train.txt, valid.txt,
        test.txt in each. Format: head_entity<tab>relation<tab>tail_entity.

        Args:
            task_sequence: Our benchmark task sequence.
            output_dir: Directory to write LKGE-formatted data.

        Returns:
            Path to the converted dataset directory.
        """
        raise NotImplementedError("Phase 3: Implement LKGE format conversion")

    def run(
        self,
        dataset_dir: str,
        lifelong_name: str = "LKGE",
    ) -> dict:
        """Run LKGE on the converted dataset.

        Executes: python main.py -dataset PRIMEKG_TEMPORAL -gpu <id> -lifelong_name LKGE

        Args:
            dataset_dir: Path to LKGE-formatted dataset.
            lifelong_name: LKGE method variant.

        Returns:
            Dict of results parsed from LKGE output.
        """
        raise NotImplementedError("Phase 3: Implement LKGE runner")

    def parse_results(self, output_dir: str) -> dict:
        """Parse LKGE output files into our metrics format.

        Args:
            output_dir: LKGE output directory.

        Returns:
            Dict with per-task metrics compatible with our evaluation pipeline.
        """
        raise NotImplementedError("Phase 3: Implement LKGE result parsing")
