"""Small real train→val→test→checkpoint CLI integration test."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.cross_scale_dataset import CROSS_SCALE_DATA_SCHEMA_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write(path, samples):
    rng = np.random.default_rng(3)
    np.savez(
        path,
        schema_version=np.array(CROSS_SCALE_DATA_SCHEMA_VERSION),
        ankh39_embeddings=rng.normal(size=(samples, 3, 2)).astype("float32"),
        esm2_embeddings=rng.normal(size=(samples, 3, 3)).astype("float32"),
        prott5_embeddings=rng.normal(size=(samples, 3, 4)).astype("float32"),
        signal_edge_index=np.array([[0, 1], [1, 2]], dtype="int64"),
        signal_gene_map=np.ones((3, 4), dtype="float32"),
        cell_edge_index=np.array([[0, 1, 2], [1, 2, 3]], dtype="int64"),
        delta_expression=rng.normal(size=(samples, 4)).astype("float32"),
        cell_state=np.arange(samples, dtype="int64") % 2,
    )


def test_train_cross_scale_cli_exports_self_describing_artifact(tmp_path):
    for name, samples in (("train", 4), ("val", 2), ("test", 2)):
        _write(tmp_path / f"{name}.npz", samples)
    config = {
        "cross_scale": {
            "protein_dim": 5,
            "signal_input_dim": 5,
            "signal_hidden_dim": 6,
            "signal_output_dim": 5,
            "cell_gene_feature_dim": 2,
            "cell_hidden_dim": 6,
            "num_cell_genes": 4,
            "num_cell_states": 2,
            "num_ptm_types": 2,
            "max_position": 8,
            "dropout": 0.0,
            "plm_backbone_dims": {"ankh39": 2, "esm2": 3, "prott5": 4},
            "required_backbones": ["ankh39", "esm2", "prott5"],
        },
        "training": {"batch_size": 2, "max_epochs": 1, "device": "cpu", "learning_rate": 0.001},
        "labels": {"cell_states": ["resting", "active"]},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_cross_scale.py",
            "--config",
            str(config_path),
            "--train-data",
            str(tmp_path / "train.npz"),
            "--val-data",
            str(tmp_path / "val.npz"),
            "--test-data",
            str(tmp_path / "test.npz"),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["ok"] is True
    assert {"best.pt", "last.pt", "config.yaml", "metrics.json", "artifact_manifest.json"}.issubset(
        {path.name for path in output.iterdir()}
    )
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    assert checkpoint["optimizer_state_dict"]
    assert checkpoint["rng_state"]["torch_cpu"].numel() > 0
    manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_contracts"]["train"]["sha256"]
    assert manifest["label_vocabulary"] == ["resting", "active"]
