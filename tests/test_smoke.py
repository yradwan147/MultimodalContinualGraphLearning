"""Smoke tests to catch bugs before IBEX submission.

Run with: python -m pytest tests/test_smoke.py -v

These tests verify critical fixes:
1. CMKL _map_triples is identity on int64 arrays
2. Filtered evaluation accepts and uses filter triples
3. NC methods produce different training (EWC/replay vs naive)
4. LKGE config.py no longer overrides emb_dim
5. RAG entity name cleaning matches gold answers
6. LKGE log parsing handles PrettyTable format
7. Progress markers are present in scripts
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestCMKLMapTriples:
    """P1: Verify _map_triples is identity for int64 arrays."""

    def test_map_triples_identity(self):
        """_map_triples should return the same int64 array unchanged."""
        from src.models.cmkl import CMKL

        config = {
            "num_entities": 100,
            "num_relations": 10,
            "embedding_dim": 32,
        }
        model = CMKL(config)

        triples = np.array([[0, 1, 2], [50, 5, 99], [10, 0, 30]], dtype=np.int64)
        entity_to_id = {"entity_0": 0, "entity_1": 1}  # Doesn't matter
        relation_to_id = {"rel_0": 0}

        result = model._map_triples(triples, entity_to_id, relation_to_id)
        np.testing.assert_array_equal(result, triples)
        assert result.dtype == np.int64

    def test_map_triples_no_collapse_to_zero(self):
        """Ensure triples don't all collapse to (0, 0, 0)."""
        from src.models.cmkl import CMKL

        config = {"num_entities": 1000, "num_relations": 30, "embedding_dim": 32}
        model = CMKL(config)

        triples = np.array([[5, 3, 10], [100, 15, 999]], dtype=np.int64)
        result = model._map_triples(triples, {}, {})

        # No row should be all zeros unless input was all zeros
        assert not np.all(result == 0), "Triples collapsed to all zeros!"


class TestFilteredEvaluation:
    """P2: Verify evaluate_link_prediction accepts filter triples."""

    def test_accepts_filter_param(self):
        """evaluate_link_prediction should accept all_known_mapped_triples."""
        import inspect
        from src.baselines._base import evaluate_link_prediction

        sig = inspect.signature(evaluate_link_prediction)
        assert "all_known_mapped_triples" in sig.parameters, \
            "evaluate_link_prediction missing all_known_mapped_triples parameter"


class TestNCMethodsHaveCL:
    """P3: Verify NC uses method-specific CL training."""

    def test_ewc_imports_in_nc(self):
        """run_nc.py should import EWC_KGE for EWC training."""
        nc_path = Path(__file__).parent.parent / "scripts" / "run_nc.py"
        content = nc_path.read_text()
        assert "EWC_KGE" in content, "run_nc.py doesn't import EWC_KGE"
        assert "ewc.ewc_loss()" in content, "run_nc.py doesn't use ewc_loss()"

    def test_replay_imports_in_nc(self):
        """run_nc.py should import ExperienceReplayKGE for replay training."""
        nc_path = Path(__file__).parent.parent / "scripts" / "run_nc.py"
        content = nc_path.read_text()
        assert "ExperienceReplayKGE" in content, \
            "run_nc.py doesn't import ExperienceReplayKGE"
        assert "replay.add_task" in content, \
            "run_nc.py doesn't add tasks to replay buffer"


class TestLKGEConfig:
    """P4A: Verify LKGE config.py doesn't override emb_dim."""

    def test_config_no_emb_dim_override(self):
        """config.py should NOT have 'args.emb_dim = 200' line."""
        config_path = (Path(__file__).parent.parent
                       / "external" / "LKGE" / "src" / "config.py")
        if not config_path.exists():
            pytest.skip("LKGE not cloned")

        content = config_path.read_text()
        # Should NOT have a direct assignment like "args.emb_dim = 200"
        lines = content.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "args.emb_dim = 200" not in stripped, \
                f"config.py still overrides emb_dim: {stripped}"

    def test_parse_args_has_type_int(self):
        """parse_args.py should have type=int for emb_dim."""
        parse_path = (Path(__file__).parent.parent
                      / "external" / "LKGE" / "src" / "parse_args.py")
        if not parse_path.exists():
            pytest.skip("LKGE not cloned")

        content = parse_path.read_text()
        assert "type=int" in content, "parse_args.py missing type=int"


class TestLKGEParsing:
    """P4B: Verify LKGE log parsing handles PrettyTable format."""

    def test_parse_prettytable_format(self):
        """Parser should extract metrics from PrettyTable output."""
        from src.baselines.lkge import LKGEWrapper

        wrapper = LKGEWrapper()
        # Simulate LKGE PrettyTable output
        log_content = """
+------------+--------+--------+--------+--------+---------+
| Snapshot:0 |  MRR   | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
+------------+--------+--------+--------+--------+---------+
|     0      | 0.1234 | 0.0567 | 0.0890 | 0.1100 | 0.2345  |
+------------+--------+--------+--------+--------+---------+
+------------+--------+--------+--------+--------+---------+
| Snapshot:1 |  MRR   | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
+------------+--------+--------+--------+--------+---------+
|     0      | 0.1100 | 0.0500 | 0.0800 | 0.1000 | 0.2100  |
|     1      | 0.2345 | 0.1234 | 0.1890 | 0.2200 | 0.3456  |
+------------+--------+--------+--------+--------+---------+
Forward transfer: 0.0123  Backward transfer: -0.0134
"""
        result = wrapper._parse_log_content(log_content)

        assert "results_matrix" in result
        assert len(result["results_matrix"]) == 2  # 2 snapshots
        R = np.array(result["results_matrix"])
        # R[0][0] = MRR on snapshot 0 after training on snapshot 0
        assert abs(R[0, 0] - 0.1234) < 1e-4
        # R[1][0] = MRR on snapshot 0 after training on snapshot 1
        assert abs(R[1, 0] - 0.1100) < 1e-4
        # R[1][1] = MRR on snapshot 1 after training on snapshot 1
        assert abs(R[1, 1] - 0.2345) < 1e-4

        assert "transfer" in result
        assert abs(result["transfer"]["forward_transfer"] - 0.0123) < 1e-4
        assert abs(result["transfer"]["backward_transfer"] - (-0.0134)) < 1e-4


class TestRAGEntityCleaning:
    """P5B: Verify RAG answer cleaning matches gold answer format."""

    def test_clean_entity_name(self):
        """_clean_entity_name should strip prefixes and replace underscores."""
        from src.data.kgqa import _clean_entity_name

        assert _clean_entity_name("MONDO:0005148") == "0005148"
        assert _clean_entity_name("DrugBank:DB12345") == "DB12345"
        assert _clean_entity_name("some_entity_name") == "some entity name"
        assert _clean_entity_name("HP:0001234") == "0001234"

    def test_rag_uses_clean_entity(self):
        """rag_agent.py should use _clean_entity_name for answers."""
        rag_path = (Path(__file__).parent.parent
                    / "src" / "baselines" / "rag_agent.py")
        content = rag_path.read_text()
        assert "_clean_entity_name" in content, \
            "rag_agent.py doesn't use _clean_entity_name"


class TestProgressMarkers:
    """P7: Verify progress reporting markers exist in scripts."""

    @pytest.mark.parametrize("script", [
        "scripts/run_baselines.py",
        "scripts/run_cmkl.py",
        "scripts/run_lkge.py",
        "scripts/run_rag.py",
        "scripts/run_nc.py",
    ])
    def test_markers_present(self, script):
        """Each script should have [STARTED], [PROGRESS], [SUCCESS], [FAILED]."""
        script_path = Path(__file__).parent.parent / script
        content = script_path.read_text()

        for marker in ["[STARTED]", "[PROGRESS]", "[SUCCESS]", "[FAILED]"]:
            assert marker in content, f"{script} missing {marker} marker"


class TestOutputSuffix:
    """P6: Verify --output-suffix is supported."""

    @pytest.mark.parametrize("script", [
        "scripts/run_baselines.py",
        "scripts/run_cmkl.py",
        "scripts/run_lkge.py",
        "scripts/run_rag.py",
        "scripts/run_nc.py",
    ])
    def test_output_suffix_arg(self, script):
        """Each script should accept --output-suffix."""
        script_path = Path(__file__).parent.parent / script
        content = script_path.read_text()
        assert "output-suffix" in content or "output_suffix" in content, \
            f"{script} missing --output-suffix argument"
