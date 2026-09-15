"""F-01 跨尺度 API 集成测试：真实训练 artifact → 在线服务全链路。

链路：``train_cross_scale.py`` 训练产物 → ``POST /cross-scale/initialize``
（graph_ref 登记默认图）→ ``/cross-scale/predict``（embedding 与 sequence
双路径）→ ``/cross-scale/batch_predict``，并与离线 CLI
（``predict_cross_scale.py``）的概率输出做数值一致性对拍。
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.routes.cross_scale import reset_cross_scale_state
from src.data.cross_scale_dataset import CROSS_SCALE_DATA_SCHEMA_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELS = ["resting", "active"]


def _write_npz(path, samples, *, targets=True):
    rng = np.random.default_rng(17)
    arrays = {
        "schema_version": np.array(CROSS_SCALE_DATA_SCHEMA_VERSION),
        "ankh39_embeddings": rng.normal(size=(samples, 3, 2)).astype("float32"),
        "esm2_embeddings": rng.normal(size=(samples, 3, 3)).astype("float32"),
        "prott5_embeddings": rng.normal(size=(samples, 3, 4)).astype("float32"),
        "signal_edge_index": np.array([[0, 1], [1, 2]], dtype="int64"),
        "signal_gene_map": np.ones((3, 4), dtype="float32"),
        "cell_edge_index": np.array([[0, 1, 2], [1, 2, 3]], dtype="int64"),
        "cell_edge_weight": np.array([0.5, 1.0, 0.8], dtype="float32"),
        "sample_id": np.array([f"sample-{index}" for index in range(samples)]),
    }
    if targets:
        arrays["delta_expression"] = rng.normal(size=(samples, 4)).astype("float32")
        arrays["cell_state"] = np.arange(samples, dtype="int64") % 2
    np.savez(path, **arrays)


@pytest.fixture(scope="module")
def trained_artifact(tmp_path_factory):
    """训练一个最小跨尺度 artifact（真实 train_cross_scale.py 链路）。"""
    tmp_path = tmp_path_factory.mktemp("cross_scale_api")
    for name, samples in (("train", 4), ("val", 2), ("test", 2)):
        _write_npz(tmp_path / f"{name}.npz", samples)
    inference = tmp_path / "inference.npz"
    _write_npz(inference, 3, targets=False)
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
            "num_ptm_types": 5,
            "max_position": 32,
            "dropout": 0.0,
            "plm_backbone_dims": {"ankh39": 2, "esm2": 3, "prott5": 4},
            # 在线 sequence 路径的显式工程开关（API 层会通过 fallback_flags 报告）
            "allow_plm_fallback": True,
            "allow_graph_approximation": True,
        },
        "training": {"batch_size": 2, "max_epochs": 1, "device": "cpu"},
        "labels": {"cell_states": LABELS},
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
    return artifact, inference, tmp_path


@pytest.fixture()
def api_client(trained_artifact, tmp_path, monkeypatch):
    artifact, inference, _ = trained_artifact
    # artifact 在模块级 tmp_path 下；allowed roots 指向其父目录
    monkeypatch.setenv("PTM2CELLNET_ALLOWED_ROOTS", str(artifact.parent))
    monkeypatch.delenv("PTM2CELLNET_API_KEY", raising=False)
    monkeypatch.setenv("PTM2CELLNET_ENV", "test")
    reset_cross_scale_state()
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/cross-scale/initialize",
        json={
            "artifact_path": str(artifact),
            "device": "cpu",
            "graph_ref": str(inference),
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["graph_registered"] is True
    return client


def test_embedding_prediction_matches_offline_cli(api_client, trained_artifact):
    """API embedding 路径与离线 CLI（同 artifact + 同 NPZ）概率一致。"""
    _artifact, inference, tmp_path = trained_artifact
    output = tmp_path / "predictions.npz"
    predicted = subprocess.run(
        [
            sys.executable,
            "scripts/predict_cross_scale.py",
            "--artifact",
            str(_artifact),
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

    response = api_client.post(
        "/api/v1/cross-scale/predict",
        json={"embedding_ref": str(inference), "sample_index": 1},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["cell_state"] in LABELS
    # NPZ 自带 signal_gene_map，无需均匀映射构造
    assert body["fallback_flags"].get("signal_map_uniform") is not True

    with np.load(output, allow_pickle=False) as payload:
        cli_row = payload["cell_state_probabilities"][1]
    api_probs = [body["probabilities"][label] for label in LABELS]
    assert np.allclose(api_probs, cli_row, atol=1e-5), (api_probs, cli_row)
    assert body["cell_state"] == LABELS[int(np.argmax(cli_row))]


def test_sequence_and_batch_paths(api_client):
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    single = api_client.post(
        "/api/v1/cross-scale/predict",
        json={"sequence": sequence, "allow_uniform_signal_map": True},
    )
    assert single.status_code == 200, single.text
    assert single.json()["fallback_flags"]["plm_fallback_used"] is True
    assert single.json()["fallback_flags"]["signal_map_uniform"] is True

    batch = api_client.post(
        "/api/v1/cross-scale/batch_predict",
        json={
            "samples": [
                {"sequence": sequence, "sample_id": "s0", "allow_uniform_signal_map": True},
                {"sequence": sequence, "sample_id": "s1", "allow_uniform_signal_map": True},
            ],
            "batch_size": 2,
        },
    )
    assert batch.status_code == 200, batch.text
    body = batch.json()
    assert body["summary"] == {"total": 2, "succeeded": 2, "failed": 0}
    assert [p["sample_id"] for p in body["predictions"]] == ["s0", "s1"]
    # 组批与单样本路径结果一致（同序列、同图、确定性推理）
    assert body["predictions"][0]["probabilities"] == pytest.approx(single.json()["probabilities"], abs=1e-6)


def test_api_key_gate_guarded_subpath(monkeypatch, trained_artifact):
    """跨尺度 initialize 与标准 /initialize 共用 API key 门禁语义。"""
    artifact, _inference, _tmp = trained_artifact
    monkeypatch.setenv("PTM2CELLNET_ALLOWED_ROOTS", str(artifact.parent))
    monkeypatch.setenv("PTM2CELLNET_API_KEY", "secret")
    monkeypatch.delenv("PTM2CELLNET_ENV", raising=False)
    reset_cross_scale_state()
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/cross-scale/initialize",
        json={"artifact_path": str(artifact)},
        headers={"X-API-Key": "wrong"},
    )
    assert response.status_code == 403
