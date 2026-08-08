"""Full training artifact → offline prediction CLI integration test."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from src.data.cross_scale_dataset import CROSS_SCALE_DATA_SCHEMA_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write(path, samples, *, targets=True):
    rng = np.random.default_rng(17)
    arrays = {
        "schema_version": np.array(CROSS_SCALE_DATA_SCHEMA_VERSION),
        "ankh39_embeddings": rng.normal(size=(samples, 3, 2)).astype("float32"),
        "esm2_embeddings": rng.normal(size=(samples, 3, 3)).astype("float32"),
        "prott5_embeddings": rng.normal(size=(samples, 3, 4)).astype("float32"),
        "signal_edge_index": np.array([[0, 1], [1, 2]], dtype="int64"),
        "signal_gene_map": np.ones((3, 4), dtype="float32"),
        "cell_edge_index": np.array([[0, 1, 2], [1, 2, 3]], dtype="int64"),
        "sample_id": np.array([f"sample-{index}" for index in range(samples)]),
    }
    if targets:
        arrays["delta_expression"] = rng.normal(size=(samples, 4)).astype("float32")
        arrays["cell_state"] = np.arange(samples, dtype="int64") % 2
    np.savez(path, **arrays)


def test_train_artifact_can_be_reloaded_for_batch_prediction(tmp_path):
    for name, samples in (("train", 4), ("val", 2), ("test", 2)):
        _write(tmp_path / f"{name}.npz", samples)
    inference = tmp_path / "inference.npz"
    _write(inference, 3, targets=False)
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
        },
        "training": {"batch_size": 2, "max_epochs": 1, "device": "cpu"},
        "labels": {"cell_states": ["resting", "active"]},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    artifact = tmp_path / "artifact"
    trained = subprocess.run(
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
            str(artifact),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert trained.returncode == 0, trained.stderr

    output = tmp_path / "predictions.npz"
    predicted = subprocess.run(
        [
            sys.executable,
            "scripts/predict_cross_scale.py",
            "--artifact",
            str(artifact),
            "--data",
            str(inference),
            "--output",
            str(output),
            "--batch-size",
            "2",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert predicted.returncode == 0, predicted.stderr
    assert json.loads(predicted.stdout)["sample_count"] == 3
    with np.load(output, allow_pickle=False) as payload:
        assert payload["delta_expression"].shape == (3, 4)
        assert payload["cell_state_probabilities"].shape == (3, 2)
        assert payload["sample_id"].tolist() == ["sample-0", "sample-1", "sample-2"]
    provenance = json.loads((tmp_path / "predictions.npz.provenance.json").read_text(encoding="utf-8"))
    assert provenance["artifact_schema_version"] == "ptm2cellnet.cross-scale.artifact.v1"
