"""Integration tests: TD-H02 pLM embedding precompute → cross-scale NPZ
consumption (2026-08-16).

Runs the real CLI (mock backbone, offline) end to end and then assembles
a complete ``ptm2cellnet.cross-scale.npz.v1`` archive from the embedding
cache plus synthetic graph/target arrays, proving that the precomputed
embeddings are directly consumable by ``CrossScaleNPZDataset`` (the same
dataset the cross-scale trainer and offline predictor use).
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cross_scale_dataset import (  # noqa: E402
    CROSS_SCALE_DATA_SCHEMA_VERSION,
    CrossScaleNPZDataset,
)


def _run_cli(input_path: Path, output_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_cross_scale_embeddings.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--backbones",
            "mock",
            *extra,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _assemble_full_npz(embedding_cache: Path, out_path: Path, sample_ids: list, L: int) -> None:
    """Build a schema-v1 archive from the embedding cache + synthetic graph."""
    archive = np.load(embedding_cache, allow_pickle=False)
    n = len(sample_ids)
    G = 8  # 细胞图基因节点数（与 cell_edge_index / delta_expression 对齐）
    arrays: dict = {
        "schema_version": np.array(CROSS_SCALE_DATA_SCHEMA_VERSION),
        "sample_id": np.array(sample_ids),
        "mock_embeddings": archive["mock_embeddings"],
        "signal_edge_index": np.array([[0, 1, 1], [1, 2, 0]], dtype=np.int64),
        "signal_gene_map": np.eye(L, G, dtype=np.float32),  # [L, G]
        "cell_edge_index": np.array([[0, 1], [1, 0]], dtype=np.int64),
        "cell_edge_weight": np.array([0.5, 0.5], dtype=np.float32),
        "delta_expression": np.zeros((n, G), dtype=np.float32),
        "cell_state": np.zeros((n,), dtype=np.int64),
    }
    np.savez_compressed(out_path, **arrays)


class TestEmbeddingCliToDataset:
    def test_sequence_requests_json_with_fasta_source(self, tmp_path) -> None:
        request = tmp_path / "sequence_requests.json"
        request.write_text(
            json.dumps({"dataset": "demo", "genes": ["AHR", "MISSING"]}),
            encoding="utf-8",
        )
        fasta = tmp_path / "sequences.fasta"
        fasta.write_text(">sp|P11111|AHR_HUMAN GN=AHR\nACDEFG\n", encoding="utf-8")
        out = tmp_path / "out"
        proc = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "prepare_cross_scale_embeddings.py"),
                "--genes-json",
                str(request),
                "--sequence-fasta",
                str(fasta),
                "--output",
                str(out),
                "--backbones",
                "mock",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 1, proc.stderr
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["n_processed"] == 1
        assert manifest["n_failed"] == 1
        assert manifest["failed_samples"][0]["sample_id"] == "MISSING"

    def test_end_to_end_mock_backbone(self, tmp_path) -> None:
        input_path = tmp_path / "in.tsv"
        input_path.write_text("s1\tACDEFGHIK\ns2\tACDE\n", encoding="utf-8")
        out = tmp_path / "out"

        proc = _run_cli(input_path, out)
        assert proc.returncode == 0, proc.stderr

        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["n_processed"] == 2
        assert manifest["n_failed"] == 0
        assert manifest["input"]["sha256"]

        # 组装完整 v1 NPZ 并验证 CrossScaleNPZDataset 可消费
        cache = out / "embeddings.npz"
        L = int(np.load(cache, allow_pickle=False)["mock_embeddings"].shape[1])
        full_npz = tmp_path / "train.npz"
        _assemble_full_npz(cache, full_npz, ["s1", "s2"], L)

        dataset = CrossScaleNPZDataset(full_npz, backbone_names=("mock",))
        assert dataset.sample_count == 2
        assert dataset.sequence_length == L
        assert dataset.gene_count == 8
        assert dataset.contract()["backbones"] == ["mock"]
        assert dataset.contract()["has_cell_edge_weight"] is True
        item = dataset[0]
        assert item["inputs"]["mock_embeddings"].shape == (L, 8)
        assert item["sample_id"] == "s1"
        assert item["targets"]["delta_expression"].shape == (8,)

    def test_failed_samples_do_not_break_consumption(self, tmp_path) -> None:
        input_path = tmp_path / "in.tsv"
        input_path.write_text("good\tACDEFG\nbad\t\n", encoding="utf-8")
        out = tmp_path / "out"

        proc = _run_cli(input_path, out)
        assert proc.returncode == 1  # 有失败样本 => exit 1（非致命）
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["n_processed"] == 1
        assert manifest["n_failed"] == 1
        assert manifest["failed_samples"][0]["sample_id"] == "bad"

    def test_resume_idempotent(self, tmp_path) -> None:
        input_path = tmp_path / "in.tsv"
        input_path.write_text("s1\tACDEFG\ns2\tACDE\n", encoding="utf-8")
        out = tmp_path / "out"
        assert _run_cli(input_path, out).returncode == 0
        first_manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        # resume 运行：全部命中，退出 0 且产出不变
        proc = _run_cli(input_path, out, "--resume")
        assert proc.returncode == 0, proc.stderr
        second_manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert second_manifest["n_processed"] == 2
        assert second_manifest["outputs"]["sha256"] == first_manifest["outputs"]["sha256"]
