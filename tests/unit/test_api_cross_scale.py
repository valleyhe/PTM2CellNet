"""F-01 跨尺度 API 三端点单元测试。

覆盖 ``POST /cross-scale/initialize`` / ``/cross-scale/predict`` /
``/cross-scale/batch_predict``：合法 artifact 加载、API key 门禁、路径
穿越防护、sequence / embedding 双输入、PTM 位点映射、显式 fallback
标记、批量失败策略与状态隔离。
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.routes.cross_scale import (
    CROSS_SCALE_STATE,
    PTM_TYPE_TO_IDX,
    reset_cross_scale_state,
)
from src.api.routes.state import STATE
from src.inference.cross_scale_predictor import (
    ARTIFACT_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
)
from src.models.cross_scale import CrossScalePTM2CellNet

LABELS = ["resting", "active"]
SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"


def _artifact_config() -> dict:
    return {
        "cross_scale": {
            "protein_dim": 6,
            "signal_input_dim": 6,
            "signal_hidden_dim": 8,
            "signal_output_dim": 6,
            "cell_gene_feature_dim": 2,
            "cell_hidden_dim": 8,
            "num_cell_genes": 4,
            "num_cell_states": 2,
            "num_ptm_types": 5,
            "max_position": 32,
            "dropout": 0.0,
            "plm_backbone_dims": {"ankh39": 2, "esm2": 3, "prott5": 4},
            # 允许 fallback 是显式工程选择：无本地 pLM 权重的环境仍可服务
            # 序列请求，但响应必须通过 fallback_flags 报告（契约）。
            "allow_plm_fallback": True,
            "allow_graph_approximation": True,
        },
        "labels": {"cell_states": LABELS},
    }


def _build_artifact(
    artifact_dir: Path,
    *,
    drop_best: bool = False,
) -> Path:
    """构造一个可通过 load_cross_scale_artifact 全部校验的最小 artifact。"""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config = _artifact_config()
    torch.manual_seed(7)
    model = CrossScalePTM2CellNet.from_config(config)
    checkpoint = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": 1,
        "best_val_loss": 0.5,
        "model_state_dict": model.state_dict(),
        "artifact_context": {"config": config, "label_vocabulary": LABELS},
        "model_info": model.get_model_info(),
    }
    best_path = artifact_dir / "best.pt"
    last_path = artifact_dir / "last.pt"
    torch.save(checkpoint, best_path)
    torch.save(checkpoint, last_path)
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_type": "cross_scale",
        "best_checkpoint": "best.pt",
        "last_checkpoint": "last.pt",
        "label_vocabulary": LABELS,
        "config": config,
    }
    (artifact_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    if drop_best:
        best_path.unlink()
    return artifact_dir


def _write_embedding_npz(path: Path, *, samples: int = 2, dims=None, length: int = 3) -> Path:
    # length=3 与 _write_graph_npz 的 signal_gene_map N_signal=3 对齐，
    # 使 embedding 路径可直接消费服务端默认图的 signal→gene 映射。
    dims = dims or {"ankh39": 2, "esm2": 3, "prott5": 4}
    rng = np.random.default_rng(11)
    arrays = {
        "schema_version": np.array("ptm2cellnet.embedding-cache.v1"),
        "sample_id": np.array([f"s{i}" for i in range(samples)]),
    }
    for name, dim in dims.items():
        arrays[f"{name}_embeddings"] = rng.normal(size=(samples, length, dim)).astype("float32")
        arrays[f"{name}_attention_mask"] = np.ones((samples, length), dtype="float32")
    np.savez(path, **arrays)
    return path


def _write_graph_npz(path: Path, *, num_genes: int = 4) -> Path:
    """服务端默认图：cell/signal 边与 signal→gene 映射（与训练 NPZ 同构）。"""
    np.savez(
        path,
        cell_edge_index=np.array([[0, 1, 2], [1, 2, 3]], dtype="int64"),
        cell_edge_weight=np.array([0.5, 1.0, 0.8], dtype="float32"),
        signal_edge_index=np.array([[0, 1], [1, 2]], dtype="int64"),
        signal_gene_map=np.ones((3, num_genes), dtype="float32"),
    )
    return path


@pytest.fixture(autouse=True)
def _isolate_cross_scale_state():
    reset_cross_scale_state()
    yield
    reset_cross_scale_state()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PTM2CELLNET_ALLOWED_ROOTS", str(tmp_path))
    monkeypatch.delenv("PTM2CELLNET_API_KEY", raising=False)
    monkeypatch.setenv("PTM2CELLNET_ENV", "test")
    return TestClient(create_app())


def _initialize(client: TestClient, artifact_dir: Path, **overrides):
    payload = {"artifact_path": str(artifact_dir), "device": "cpu"}
    payload.update(overrides)
    return client.post("/api/v1/cross-scale/initialize", json=payload)


def _initialize_ready(client: TestClient, tmp_path: Path, **overrides):
    """构造 artifact + 默认图并完成 initialize（推理测试的标准前置）。"""
    artifact = _build_artifact(tmp_path / "artifact")
    graph = _write_graph_npz(tmp_path / "graph.npz")
    return _initialize(client, artifact, graph_ref=str(graph), **overrides)


class TestInitialize:
    def test_success_registers_state_and_digest(self, client, tmp_path):
        artifact = _build_artifact(tmp_path / "artifact")
        graph = _write_graph_npz(tmp_path / "graph.npz")
        response = _initialize(client, artifact, graph_ref=str(graph))
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "success"
        assert body["model_type"] == "cross_scale"
        assert body["readiness"] == "ready"
        assert body["graph_registered"] is True
        assert body["label_vocabulary"] == LABELS
        assert len(body["manifest_digest"]) == 64
        summary = body["model_summary"]
        assert summary["protein_backbones"] == ["ankh39", "esm2", "prott5"]
        assert summary["num_cell_genes"] == 4
        assert summary["num_cell_states"] == 2
        assert summary["allow_plm_fallback"] is True
        assert summary["allow_graph_approximation"] is True
        assert CROSS_SCALE_STATE.model is not None
        assert CROSS_SCALE_STATE.manifest_digest == body["manifest_digest"]

    def test_api_key_enforced_in_production(self, client, tmp_path, monkeypatch):
        monkeypatch.delenv("PTM2CELLNET_API_KEY", raising=False)
        monkeypatch.delenv("PTM2CELLNET_ENV", raising=False)
        artifact = _build_artifact(tmp_path / "artifact")
        assert _initialize(client, artifact).status_code == 503

    def test_wrong_api_key_rejected(self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("PTM2CELLNET_API_KEY", "secret")
        artifact = _build_artifact(tmp_path / "artifact")
        missing = client.post(
            "/api/v1/cross-scale/initialize",
            json={"artifact_path": str(artifact)},
        )
        assert missing.status_code == 401
        wrong = client.post(
            "/api/v1/cross-scale/initialize",
            json={"artifact_path": str(artifact)},
            headers={"X-API-Key": "nope"},
        )
        assert wrong.status_code == 403

    def test_path_traversal_rejected(self, client, tmp_path):
        assert _initialize(client, Path("/etc")).status_code == 403

    def test_missing_artifact_dir_is_404(self, client, tmp_path):
        assert _initialize(client, tmp_path / "nope").status_code == 404

    def test_invalid_artifact_is_400(self, client, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        response = _initialize(client, empty)
        assert response.status_code == 400
        assert "artifact" in response.json()["detail"]

    def test_last_checkpoint_fallback_only_without_strict(self, client, tmp_path):
        artifact = _build_artifact(tmp_path / "artifact", drop_best=True)
        strict = _initialize(client, artifact)  # strict_assets 默认 True
        assert strict.status_code == 400
        relaxed = _initialize(client, artifact, strict_assets=False)
        assert relaxed.status_code == 200, relaxed.text

    def test_does_not_touch_standard_state(self, client, tmp_path):
        assert _initialize_ready(client, tmp_path).status_code == 200
        # 独立状态槽：跨尺度加载不得影响标准模型槽位
        assert STATE.model is None

    def test_graph_ref_missing_cell_edge_rejected(self, client, tmp_path):
        artifact = _build_artifact(tmp_path / "artifact")
        bad_graph = tmp_path / "bad_graph.npz"
        np.savez(bad_graph, signal_edge_index=np.array([[0], [1]], dtype="int64"))
        response = _initialize(client, artifact, graph_ref=str(bad_graph))
        assert response.status_code == 400
        assert "cell_edge_index" in response.json()["detail"]

    def test_graph_ref_node_out_of_range_rejected(self, client, tmp_path):
        artifact = _build_artifact(tmp_path / "artifact")
        bad_graph = tmp_path / "bad_graph.npz"
        np.savez(bad_graph, cell_edge_index=np.array([[0, 99], [1, 98]], dtype="int64"))
        response = _initialize(client, artifact, graph_ref=str(bad_graph))
        assert response.status_code == 400
        assert "越界" in response.json()["detail"]


class TestPredict:
    def test_requires_initialization(self, client):
        response = client.post(
            "/api/v1/cross-scale/predict", json={"sequence": SEQUENCE}
        )
        assert response.status_code == 503

    def test_sequence_prediction_with_fallback_flags(self, client, tmp_path):
        _initialize_ready(client, tmp_path)
        response = client.post(
            "/api/v1/cross-scale/predict",
            json={"sequence": SEQUENCE, "sample_id": "s1", "allow_uniform_signal_map": True},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["sample_id"] == "s1"
        assert body["cell_state"] in LABELS
        assert pytest.approx(sum(body["probabilities"].values()), abs=1e-4) == 1.0
        assert body["confidence"] == pytest.approx(
            max(body["probabilities"].values()), abs=1e-6
        )
        # 契约：禁止隐式随机 fallback —— 无已加载 backbone 时必须显式标记
        assert body["fallback_flags"]["plm_fallback_used"] is True
        # 服务端默认图已提供 signal_edge_index，无需图近似
        assert body["fallback_flags"]["graph_approximation_used"] is False
        # 均匀 signal 映射是显式请求的工程选择，必须被标记
        assert body["fallback_flags"]["signal_map_uniform"] is True
        assert body["delta_expression"] is None
        assert body["provenance"]["manifest_digest"] == CROSS_SCALE_STATE.manifest_digest
        assert body["processing_time_ms"] is not None

    def test_sequence_with_delta_expression(self, client, tmp_path):
        _initialize_ready(client, tmp_path)
        response = client.post(
            "/api/v1/cross-scale/predict",
            json={"sequence": SEQUENCE, "include_delta_expression": True, "allow_uniform_signal_map": True},
        )
        assert response.status_code == 200
        assert len(response.json()["delta_expression"]) == 4

    def test_ptm_sites_are_applied(self, client, tmp_path):
        _initialize_ready(client, tmp_path)
        response = client.post(
            "/api/v1/cross-scale/predict",
            json={
                "sequence": SEQUENCE, "allow_uniform_signal_map": True,
                "ptm_sites": [{"position": 4, "type": "phosphorylation"}],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["ptm_sites_applied"] == 1

    def test_unknown_ptm_type_rejected(self, client, tmp_path):
        _initialize_ready(client, tmp_path)
        response = client.post(
            "/api/v1/cross-scale/predict",
            json={
                "sequence": SEQUENCE,
                "ptm_sites": [{"position": 4, "type": "glycosylation"}],
            },
        )
        assert response.status_code == 400
        assert "PTM" in response.json()["detail"]

    def test_ptm_position_beyond_sequence_rejected(self, client, tmp_path):
        _initialize_ready(client, tmp_path)
        response = client.post(
            "/api/v1/cross-scale/predict",
            json={
                "sequence": SEQUENCE,
                "ptm_sites": [{"position": 999, "type": "phosphorylation"}],
            },
        )
        assert response.status_code == 400

    def test_sequence_and_embedding_are_exclusive(self, client, tmp_path):
        embedding = _write_embedding_npz(tmp_path / "emb.npz")
        _initialize_ready(client, tmp_path)
        both = client.post(
            "/api/v1/cross-scale/predict",
            json={"sequence": SEQUENCE, "embedding_ref": str(embedding)},
        )
        assert both.status_code == 400
        neither = client.post("/api/v1/cross-scale/predict", json={})
        assert neither.status_code == 400

    def test_embedding_ref_prediction(self, client, tmp_path):
        embedding = _write_embedding_npz(tmp_path / "emb.npz", samples=2)
        _initialize_ready(client, tmp_path)
        response = client.post(
            "/api/v1/cross-scale/predict",
            json={"embedding_ref": str(embedding), "sample_index": 1},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["cell_state"] in LABELS
        # embedding 路径消费预计算真实表示，不触及 pLM fallback 向量
        assert body["fallback_flags"]["plm_fallback_used"] is False
        assert body["provenance"]["input_mode"] == "embedding_ref"

    def test_embedding_dim_mismatch_rejected(self, client, tmp_path):
        bad = _write_embedding_npz(tmp_path / "bad.npz", dims={"ankh39": 9, "esm2": 3, "prott5": 4})
        _initialize_ready(client, tmp_path)
        response = client.post(
            "/api/v1/cross-scale/predict", json={"embedding_ref": str(bad)}
        )
        assert response.status_code == 400
        assert "backbone dim" in response.json()["detail"]

    def test_embedding_missing_backbone_array_rejected(self, client, tmp_path):
        partial = tmp_path / "partial.npz"
        np.savez(
            partial,
            schema_version=np.array("ptm2cellnet.embedding-cache.v1"),
            ankh39_embeddings=np.zeros((1, 5, 2), dtype="float32"),
        )
        _initialize_ready(client, tmp_path)
        response = client.post(
            "/api/v1/cross-scale/predict", json={"embedding_ref": str(partial)}
        )
        assert response.status_code == 400
        assert "esm2_embeddings" in response.json()["detail"]

    def test_embedding_path_traversal_rejected(self, client, tmp_path):
        _initialize_ready(client, tmp_path)
        response = client.post(
            "/api/v1/cross-scale/predict",
            json={"embedding_ref": "/etc/passwd"},
        )
        assert response.status_code in (403, 404)

    def test_missing_graph_fails_explicitly(self, client, tmp_path):
        # 模型契约：cell graph 不可静默替换 —— 未登记 graph_ref 时显式 400
        _initialize(client, _build_artifact(tmp_path / "artifact"))
        response = client.post(
            "/api/v1/cross-scale/predict", json={"sequence": SEQUENCE}
        )
        assert response.status_code == 400
        assert "cell graph" in response.json()["detail"]

    def test_sequence_without_uniform_map_rejected(self, client, tmp_path):
        # 默认图 signal_gene_map N_signal=3 与序列长度不匹配，跳过后缺映射；
        # 未显式 allow_uniform_signal_map 时必须 400 而非静默构造
        _initialize_ready(client, tmp_path)
        response = client.post(
            "/api/v1/cross-scale/predict", json={"sequence": SEQUENCE}
        )
        assert response.status_code == 400
        assert "signal_gene_map" in response.json()["detail"]

    def test_cross_scale_npz_supplies_own_graph(self, client, tmp_path):
        # 跨尺度 NPZ（schema v1）自带图结构（cell_edge_index + 匹配 L 的
        # signal_gene_map）时优先生效，无需 uniform 映射
        rng = np.random.default_rng(3)
        npz = tmp_path / "sample.npz"
        np.savez(
            npz,
            schema_version=np.array("ptm2cellnet.cross-scale.npz.v1"),
            sample_id=np.array(["npz-sample"]),
            ankh39_embeddings=rng.normal(size=(1, 3, 2)).astype("float32"),
            esm2_embeddings=rng.normal(size=(1, 3, 3)).astype("float32"),
            prott5_embeddings=rng.normal(size=(1, 3, 4)).astype("float32"),
            cell_edge_index=np.array([[0, 1, 2], [1, 2, 3]], dtype="int64"),
            signal_edge_index=np.array([[0, 1], [1, 2]], dtype="int64"),
            signal_gene_map=np.ones((3, 4), dtype="float32"),
        )
        _initialize_ready(client, tmp_path)
        response = client.post(
            "/api/v1/cross-scale/predict", json={"embedding_ref": str(npz)}
        )
        assert response.status_code == 200, response.text
        assert response.json()["fallback_flags"].get("signal_map_uniform") is not True

    def test_reset_clears_state(self, client, tmp_path):
        _initialize_ready(client, tmp_path)
        reset_cross_scale_state()
        assert CROSS_SCALE_STATE.model is None
        assert CROSS_SCALE_STATE.default_graph is None
        response = client.post(
            "/api/v1/cross-scale/predict", json={"sequence": SEQUENCE}
        )
        assert response.status_code == 503


class TestBatchPredict:
    def test_plain_sequences_are_batched(self, client, tmp_path):
        _initialize_ready(client, tmp_path)
        samples = [
            {"sequence": SEQUENCE, "sample_id": f"b{i}", "allow_uniform_signal_map": True} for i in range(3)
        ]
        response = client.post(
            "/api/v1/cross-scale/batch_predict",
            json={"samples": samples, "batch_size": 2},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert [p["sample_id"] for p in body["predictions"]] == ["b0", "b1", "b2"]
        assert body["summary"] == {"total": 3, "succeeded": 3, "failed": 0}
        assert body["errors"] == []
        assert body["provenance"]["batch_size"] == 2

    def test_mixed_samples_and_error_collection(self, client, tmp_path):
        embedding = _write_embedding_npz(tmp_path / "emb.npz")
        _initialize_ready(client, tmp_path)
        samples = [
            {"sequence": SEQUENCE, "sample_id": "good", "allow_uniform_signal_map": True},
            {"sequence": SEQUENCE, "sample_id": "bad-type",
             "ptm_sites": [{"position": 2, "type": "not-a-ptm"}]},
            {"embedding_ref": str(embedding), "sample_id": "good-emb"},
            {"sequence": SEQUENCE, "sample_id": "both", "embedding_ref": str(embedding)},
        ]
        response = client.post(
            "/api/v1/cross-scale/batch_predict", json={"samples": samples}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["summary"]["total"] == 4
        assert body["summary"]["succeeded"] == 2
        assert body["summary"]["failed"] == 2
        assert [error["index"] for error in body["errors"]] == [1, 3]
        assert {p["sample_id"] for p in body["predictions"]} == {"good", "good-emb"}

    def test_fail_fast_aborts_on_first_error(self, client, tmp_path):
        _initialize_ready(client, tmp_path)
        samples = [
            {"sequence": SEQUENCE, "sample_id": "ok", "allow_uniform_signal_map": True},
            {"sequence": SEQUENCE, "ptm_sites": [{"position": 2, "type": "not-a-ptm"}]},
        ]
        response = client.post(
            "/api/v1/cross-scale/batch_predict",
            json={"samples": samples, "fail_fast": True},
        )
        assert response.status_code == 400

    def test_empty_samples_rejected_by_schema(self, client, tmp_path):
        _initialize_ready(client, tmp_path)
        response = client.post(
            "/api/v1/cross-scale/batch_predict", json={"samples": []}
        )
        assert response.status_code == 422

    def test_requires_initialization(self, client):
        response = client.post(
            "/api/v1/cross-scale/batch_predict",
            json={"samples": [{"sequence": SEQUENCE}]},
        )
        assert response.status_code == 503


class TestPtmVocabulary:
    def test_vocabulary_is_one_based_and_sourced_from_default_types(self):
        from src.data.features import DEFAULT_PTM_TYPES

        assert PTM_TYPE_TO_IDX["phosphorylation"] == 1
        assert set(PTM_TYPE_TO_IDX) == set(DEFAULT_PTM_TYPES)
        assert sorted(PTM_TYPE_TO_IDX.values()) == list(
            range(1, len(DEFAULT_PTM_TYPES) + 1)
        )
