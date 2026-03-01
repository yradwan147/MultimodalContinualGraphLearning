"""Baseline 5: LKGE (Lifelong Knowledge Graph Embedding) wrapper.

Wraps the external LKGE framework (https://github.com/nju-websoft/LKGE)
for use with our PrimeKG temporal benchmark. Handles data format conversion
between our benchmark format and LKGE's expected directory structure.

Usage:
    from src.baselines.lkge import LKGEWrapper
    wrapper = LKGEWrapper(lkge_dir='external/LKGE')
    wrapper.convert_to_lkge_format(task_sequence, 'data/lkge_format')
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np

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
        task_sequence: OrderedDict[str, dict[str, np.ndarray]],
        output_dir: str,
    ) -> str:
        """Convert task sequence to LKGE's expected directory format.

        Creates snapshot_0/, snapshot_1/, etc. with train.txt, valid.txt,
        test.txt in each. Format: head_entity<tab>relation<tab>tail_entity.

        Also creates entity2id.txt and relation2id.txt at the top level.

        Args:
            task_sequence: Our benchmark task sequence.
            output_dir: Directory to write LKGE-formatted data.

        Returns:
            Path to the converted dataset directory.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Collect all entities and relations
        entities = set()
        relations = set()
        for data in task_sequence.values():
            for split in data.values():
                if len(split) > 0:
                    entities.update(split[:, 0])
                    entities.update(split[:, 2])
                    relations.update(split[:, 1])

        entity_list = sorted(entities)
        relation_list = sorted(relations)

        # Write entity2id.txt
        with open(out / "entity2id.txt", "w") as f:
            f.write(f"{len(entity_list)}\n")
            for i, e in enumerate(entity_list):
                f.write(f"{e}\t{i}\n")

        # Write relation2id.txt
        with open(out / "relation2id.txt", "w") as f:
            f.write(f"{len(relation_list)}\n")
            for i, r in enumerate(relation_list):
                f.write(f"{r}\t{i}\n")

        # Write each snapshot
        for idx, (name, data) in enumerate(task_sequence.items()):
            snap_dir = out / f"snapshot_{idx}"
            snap_dir.mkdir(exist_ok=True)

            for split_name, split_data in data.items():
                fname = {"train": "train.txt", "val": "valid.txt", "test": "test.txt"}
                if split_name in fname and len(split_data) > 0:
                    with open(snap_dir / fname[split_name], "w") as f:
                        for h, r, t in split_data:
                            f.write(f"{h}\t{r}\t{t}\n")

            logger.info(f"  Wrote snapshot_{idx} ({name}): "
                       f"{len(data.get('train', [])):,} train triples")

        logger.info(f"LKGE dataset written to {out} "
                    f"({len(task_sequence)} snapshots, "
                    f"{len(entity_list)} entities, "
                    f"{len(relation_list)} relations)")
        return str(out)

    def get_run_command(
        self,
        dataset_dir: str,
        lifelong_name: str = "LKGE",
        model: str = "TransE",
        num_epochs: int = 100,
    ) -> str:
        """Generate the command to run LKGE.

        This returns the command string since LKGE runs as a subprocess
        (typically on IBEX via SLURM).

        Args:
            dataset_dir: Path to LKGE-formatted dataset.
            lifelong_name: LKGE method variant.
            model: KGE model to use.
            num_epochs: Training epochs.

        Returns:
            Command string to execute.
        """
        cmd = (
            f"cd {self.lkge_dir} && "
            f"python main.py "
            f"-dataset {dataset_dir} "
            f"-gpu {self.gpu_id} "
            f"-lifelong_name {lifelong_name} "
            f"-model {model} "
            f"-epoch {num_epochs}"
        )
        return cmd

    def parse_results(self, output_dir: str) -> dict:
        """Parse LKGE output files into our metrics format.

        Args:
            output_dir: LKGE output directory.

        Returns:
            Dict with per-task metrics.
        """
        results_path = Path(output_dir)
        if not results_path.exists():
            logger.warning(f"LKGE output not found: {output_dir}")
            return {}

        # LKGE outputs results in a log file
        # Parse the log to extract MRR, Hits@K per snapshot
        metrics = {}
        log_files = list(results_path.glob("*.log")) + list(results_path.glob("*.txt"))
        for lf in log_files:
            with open(lf) as f:
                content = f.read()
            logger.info(f"Found LKGE output: {lf.name} ({len(content)} chars)")
            metrics["raw_log"] = content

        return metrics
