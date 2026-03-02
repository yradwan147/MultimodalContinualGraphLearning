"""Baseline 5: LKGE (Lifelong Knowledge Graph Embedding) wrapper.

Wraps the external LKGE framework (https://github.com/nju-websoft/LKGE)
for use with our PrimeKG temporal benchmark. Handles data format conversion
between our benchmark format and LKGE's expected directory structure,
subprocess execution, and result parsing.

Usage:
    from src.baselines.lkge import LKGEWrapper
    wrapper = LKGEWrapper(lkge_dir='external/LKGE')
    wrapper.convert_to_lkge_format(task_sequence, 'data/lkge_format')
    results = wrapper.run_and_parse(dataset_dir, output_dir)
"""

from __future__ import annotations

import logging
import re
import subprocess
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

        # Write each snapshot — LKGE expects dirs named 0/, 1/, 2/, ...
        for idx, (name, data) in enumerate(task_sequence.items()):
            snap_dir = out / str(idx)
            snap_dir.mkdir(exist_ok=True)

            for split_name, split_data in data.items():
                fname = {"train": "train.txt", "val": "valid.txt", "test": "test.txt"}
                if split_name in fname and len(split_data) > 0:
                    with open(snap_dir / fname[split_name], "w") as f:
                        for h, r, t in split_data:
                            f.write(f"{h}\t{r}\t{t}\n")

            logger.info(f"  Wrote {idx}/ ({name}): "
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
        snapshot_num: int | None = None,
        seed: int = 42,
    ) -> str:
        """Generate the command to run LKGE.

        Args:
            dataset_dir: Path to LKGE-formatted dataset.
            lifelong_name: LKGE method variant.
            model: KGE model to use.
            num_epochs: Training epochs.
            snapshot_num: Number of snapshots. Auto-detected if None.
            seed: Random seed.

        Returns:
            Command string to execute.
        """
        # LKGE expects: -data_path <parent>/ -dataset <folder_name>
        # It constructs paths as: data_path + dataset + '/'
        abs_dataset_dir = Path(dataset_dir).resolve()
        data_path = str(abs_dataset_dir.parent) + "/"
        dataset_name = abs_dataset_dir.name
        # Checkpoint and log paths must also be absolute to avoid
        # writing into the LKGE repo directory
        save_path = str(abs_dataset_dir.parent / "lkge_checkpoints") + "/"
        log_path = str(abs_dataset_dir.parent / "lkge_logs") + "/"
        # Auto-detect snapshot count from dataset directory
        if snapshot_num is None:
            # Dirs are named 0/, 1/, 2/, ... — count numeric directories
            snapshot_num = len([
                d for d in abs_dataset_dir.iterdir()
                if d.is_dir() and d.name.isdigit()
            ])
        cmd = (
            f"python main.py "
            f"-data_path {data_path} "
            f"-dataset {dataset_name} "
            f"-save_path {save_path} "
            f"-log_path {log_path} "
            f"-gpu {self.gpu_id} "
            f"-snapshot_num {snapshot_num} "
            f"-lifelong_name {lifelong_name} "
            f"-embedding_model {model} "
            f"-epoch_num {num_epochs} "
            f"-seed {seed}"
        )
        return cmd

    def run_and_parse(
        self,
        dataset_dir: str,
        output_dir: str,
        lifelong_name: str = "LKGE",
        model: str = "TransE",
        num_epochs: int = 100,
        timeout: int = 86400,
        seed: int = 42,
    ) -> dict:
        """Run LKGE as a subprocess and parse the output.

        Args:
            dataset_dir: Path to LKGE-formatted dataset.
            output_dir: Directory for LKGE outputs.
            lifelong_name: LKGE method variant.
            model: KGE model to use.
            num_epochs: Training epochs.
            timeout: Max runtime in seconds (default 24h).
            seed: Random seed.

        Returns:
            Dict with parsed metrics per snapshot.
        """
        cmd = self.get_run_command(
            dataset_dir, lifelong_name, model, num_epochs, seed=seed
        )
        logger.info(f"Running LKGE: {cmd}")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        log_file = out_path / "lkge_run.log"

        # Resolve lkge_dir to absolute so cwd works correctly
        abs_lkge_dir = str(self.lkge_dir.resolve())
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=abs_lkge_dir,
            )
            output = result.stdout + "\n" + result.stderr

            with open(log_file, "w") as f:
                f.write(output)

            logger.info(f"LKGE completed (returncode={result.returncode})")

            if result.returncode != 0:
                logger.warning(f"LKGE non-zero exit: {result.returncode}")
                logger.warning(f"stderr: {result.stderr[:500]}")

        except subprocess.TimeoutExpired:
            logger.error(f"LKGE timed out after {timeout}s")
            return {"error": "timeout"}
        except FileNotFoundError:
            logger.error(f"LKGE not found at {self.lkge_dir}")
            return {"error": "lkge_not_found"}

        return self.parse_results(output_dir)

    def parse_results(self, output_dir: str) -> dict:
        """Parse LKGE output files into our metrics format.

        Extracts MRR, Hits@1, Hits@3, Hits@10 per snapshot from LKGE log
        output using regex patterns matching LKGE's standard output format.

        Args:
            output_dir: LKGE output directory.

        Returns:
            Dict with 'per_snapshot' metrics and 'results_matrix'.
        """
        results_path = Path(output_dir)
        if not results_path.exists():
            logger.warning(f"LKGE output not found: {output_dir}")
            return {}

        # Collect all text from log files
        content = ""
        log_files = (list(results_path.glob("*.log"))
                     + list(results_path.glob("*.txt"))
                     + list(results_path.glob("**/*.log")))
        for lf in log_files:
            with open(lf) as f:
                content += f.read() + "\n"

        if not content.strip():
            logger.warning("No LKGE log content found")
            return {"raw_log": "", "per_snapshot": {}}

        return self._parse_log_content(content)

    def _parse_log_content(self, content: str) -> dict:
        """Parse LKGE log text to extract per-snapshot metrics.

        LKGE outputs metrics in patterns like:
            Snapshot 0 - MRR: 0.1234 - Hits@1: 0.0567 - Hits@3: 0.1234 - Hits@10: 0.2345
        or:
            [Snapshot 0] MRR=0.1234, Hits@1=0.0567, Hits@3=0.1234, Hits@10=0.2345
        or table format:
            snapshot | MRR   | Hits@1 | Hits@3 | Hits@10
            0        | 0.123 | 0.056  | 0.123  | 0.234

        Args:
            content: Raw log text.

        Returns:
            Dict with per_snapshot metrics and results_matrix.
        """
        per_snapshot: dict[int, dict[str, float]] = {}

        # Pattern 1: "Snapshot X - MRR: Y" or "Snapshot X: MRR: Y"
        snap_pattern = re.compile(
            r"[Ss]napshot\s+(\d+)\s*[-:]\s*"
            r"MRR[:\s=]+([0-9.]+).*?"
            r"Hits@1[:\s=]+([0-9.]+).*?"
            r"Hits@3[:\s=]+([0-9.]+).*?"
            r"Hits@10[:\s=]+([0-9.]+)",
            re.IGNORECASE,
        )
        for m in snap_pattern.finditer(content):
            snap_id = int(m.group(1))
            per_snapshot[snap_id] = {
                "MRR": float(m.group(2)),
                "Hits@1": float(m.group(3)),
                "Hits@3": float(m.group(4)),
                "Hits@10": float(m.group(5)),
            }

        # Pattern 2: "test on snapshot X" followed by individual metric lines
        if not per_snapshot:
            test_snap = re.compile(
                r"[Tt]est(?:ing)?\s+(?:on\s+)?[Ss]napshot\s+(\d+)", re.IGNORECASE
            )
            mrr_pat = re.compile(r"MRR[:\s=]+([0-9.]+)", re.IGNORECASE)
            h1_pat = re.compile(r"Hits@1[:\s=]+([0-9.]+)", re.IGNORECASE)
            h3_pat = re.compile(r"Hits@3[:\s=]+([0-9.]+)", re.IGNORECASE)
            h10_pat = re.compile(r"Hits@10[:\s=]+([0-9.]+)", re.IGNORECASE)

            lines = content.split("\n")
            current_snap = None
            for line in lines:
                snap_m = test_snap.search(line)
                if snap_m:
                    current_snap = int(snap_m.group(1))
                    per_snapshot[current_snap] = {}
                    continue
                if current_snap is not None:
                    for pat, key in [(mrr_pat, "MRR"), (h1_pat, "Hits@1"),
                                     (h3_pat, "Hits@3"), (h10_pat, "Hits@10")]:
                        m = pat.search(line)
                        if m:
                            per_snapshot[current_snap][key] = float(m.group(1))

        # Pattern 3: Tabular format (pipe-separated)
        if not per_snapshot:
            table_row = re.compile(
                r"^\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*"
                r"([0-9.]+)\s*\|\s*([0-9.]+)",
                re.MULTILINE,
            )
            for m in table_row.finditer(content):
                snap_id = int(m.group(1))
                per_snapshot[snap_id] = {
                    "MRR": float(m.group(2)),
                    "Hits@1": float(m.group(3)),
                    "Hits@3": float(m.group(4)),
                    "Hits@10": float(m.group(5)),
                }

        # Build results matrix using MRR as the primary metric
        # R[i][j] = MRR on snapshot j after training through snapshot i
        # LKGE typically reports final performance on each snapshot
        if per_snapshot:
            n = max(per_snapshot.keys()) + 1
            results_matrix = np.zeros((n, n))
            for snap_id, metrics in per_snapshot.items():
                # LKGE evaluates on all snapshots after training on each
                # The parsed metrics represent final evaluation
                results_matrix[n - 1, snap_id] = metrics.get("MRR", 0.0)
            results_matrix = results_matrix.tolist()
        else:
            results_matrix = []

        return {
            "per_snapshot": {
                str(k): v for k, v in sorted(per_snapshot.items())
            },
            "results_matrix": results_matrix,
            "raw_log": content[:5000],
        }
