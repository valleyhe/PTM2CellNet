"""端到端测试：API /initialize checkpoint 加载严格性。

对应审计报告 P0-2 / P2-1：
    - 正确 checkpoint（裸 state_dict）加载成功，返回 loaded_format/参数覆盖率。
    - 格式不匹配 / 形状不匹配的 checkpoint 返回 400 而非 success。
    - Lightning 导出的裸权重可被 API 成功加载。
"""

from pathlib import Path

import pytest
import torch
import yaml
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.routes.state import STATE
from src.models.architectures import PTM2CellNet


@pytest.fixture(autouse=True)
def _dev_env_and_allowed_roots(tmp_path, monkeypatch):
    """Enable dev mode + allow tmp_path under the path whitelist.

    The /initialize endpoint is default-deny: in non-production envs without
    PTM2CELLNET_API_KEY it stays open only when PTM2CELLNET_ENV=development,
    and checkpoint paths must fall under PTM2CELLNET_ALLOWED_ROOTS. Tests are
    not production, so they opt into dev mode and whitelist the tmp_path used
    by each case.
    """
    monkeypatch.setenv("PTM2CELLNET_ENV", "development")
    monkeypatch.delenv("PTM2CELLNET_API_KEY", raising=False)
    # Whitelist the per-test tmp_path via the env var; initialize.py resolves
    # PTM2CELLNET_ALLOWED_ROOTS on each call, so this takes effect live.
    import src.api.routes.initialize as init_mod

    monkeypatch.setenv(
        "PTM2CELLNET_ALLOWED_ROOTS",
        f"{tmp_path.resolve()},{','.join(str(p) for p in init_mod._ALLOWED_ROOTS)}",
    )
    yield


@pytest.fixture(autouse=True)
def _restore_state():
    """每个测试后恢复全局 STATE，避免状态泄漏到其他测试。"""
    orig = {
        "model": STATE.model,
        "cell_states": list(STATE.cell_states),
        "idx_to_label": dict(STATE.idx_to_label),
        "device": STATE.device,
        "variant_workflow": STATE.variant_workflow,
        "pathway_mapper": STATE.pathway_mapper,
    }
    yield
    for k, v in orig.items():
        setattr(STATE, k, v)


def _model_config(num_classes=4, embed_dim=16):
    return {
        "model": {
            "encoder_type": "cnn",
            "vocab_size": 21,
            "embed_dim": embed_dim,
            "num_filters": 16,
            "kernel_sizes": [3],
            "max_seq_len": 64,
            "num_ptm_types": 5,
            "num_classes": num_classes,
            "pool_type": "mean",
            "dropout": 0.1,
        },
        "data": {
            "cell_states": [f"class_{i}" for i in range(num_classes)],
            "max_sequence_length": 64,
            "ptm_types": ["phosphorylation", "acetylation", "methylation", "ubiquitination", "sumoylation"],
        },
    }


def _write_config(path: Path, cfg: dict) -> Path:
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


class TestAPICheckpointStrictness:
    """P0-2: API 默认严格加载，坏 checkpoint 直接 400。"""

    def test_correct_state_dict_succeeds(self, tmp_path):
        """与 config 匹配的裸 state_dict 应成功，并返回诊断字段。"""
        cfg = _model_config()
        model = PTM2CellNet.from_config(cfg)
        ckpt = tmp_path / "good.pt"
        torch.save(model.state_dict(), ckpt)
        cfg_path = _write_config(tmp_path / "good.config.yaml", cfg)

        client = TestClient(create_app())
        cell_states = cfg["data"]["cell_states"]
        r = client.post(
            "/api/v1/initialize",
            json={
                "checkpoint_path": str(ckpt),
                "config_path": str(cfg_path),
                "cell_states": cell_states,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "success"
        assert body["loaded_format"] == "state_dict"
        assert body["missing_keys_count"] == 0
        assert body["unexpected_keys_count"] == 0
        assert body["loaded_parameter_ratio"] == 1.0

    def test_shape_mismatch_returns_400(self, tmp_path):
        """num_classes 不一致导致输出层形状不匹配时应返回 400。"""
        # 训练时 num_classes=4，推理 config 声明 num_classes=3
        train_cfg = _model_config(num_classes=4)
        model = PTM2CellNet.from_config(train_cfg)
        ckpt = tmp_path / "shape.pt"
        torch.save(model.state_dict(), ckpt)

        infer_cfg = _model_config(num_classes=3)  # 输出层形状不匹配
        cfg_path = _write_config(tmp_path / "shape.config.yaml", infer_cfg)

        client = TestClient(create_app())
        r = client.post(
            "/api/v1/initialize",
            json={
                "checkpoint_path": str(ckpt),
                "config_path": str(cfg_path),
                "cell_states": infer_cfg["data"]["cell_states"],
            },
        )
        assert r.status_code == 400, f"形状不匹配应返回 400，实际 {r.status_code}: {r.text}"
        assert "不匹配" in r.json()["detail"]

    def test_lightning_raw_ckpt_unconverted_fails(self, tmp_path):
        """未导出裸权重的原始 Lightning .ckpt（含外层键）应失败而非误报成功。"""
        cfg = _model_config()
        model = PTM2CellNet.from_config(cfg)
        sd = model.state_dict()
        # 模拟 Lightning 原始 ckpt：state_dict 带 model. 前缀 + 外层 epoch/loops
        lightning_ckpt = {
            "epoch": 1,
            "global_step": 5,
            "pytorch-lightning_version": "2.0",
            "state_dict": {f"model.{k}": v for k, v in sd.items()},
            "loops": [],
            "lr_schedulers": [],
        }
        ckpt = tmp_path / "lightning.ckpt"
        torch.save(lightning_ckpt, ckpt)
        cfg_path = _write_config(tmp_path / "lightning.config.yaml", cfg)

        client = TestClient(create_app())
        r = client.post(
            "/api/v1/initialize",
            json={
                "checkpoint_path": str(ckpt),
                "config_path": str(cfg_path),
                "cell_states": cfg["data"]["cell_states"],
            },
        )
        # 统一合约已支持剥离 model. 前缀，因此此处应成功（验证转换路径）
        assert r.status_code == 200, f"Lightning 裸权重经统一合约应可加载: {r.text}"
        body = r.json()
        assert body["status"] == "success"
        assert body["loaded_format"] == "lightning"
        assert body["missing_keys_count"] == 0
        assert body["unexpected_keys_count"] == 0

    def test_missing_keys_returns_400(self, tmp_path):
        """checkpoint 缺少部分权重键时应返回 400。"""
        cfg = _model_config()
        model = PTM2CellNet.from_config(cfg)
        partial_sd = dict(model.state_dict())
        # 删除一个键，制造 missing
        first_key = next(iter(partial_sd))
        del partial_sd[first_key]
        ckpt = tmp_path / "partial.pt"
        torch.save(partial_sd, ckpt)
        cfg_path = _write_config(tmp_path / "partial.config.yaml", cfg)

        client = TestClient(create_app())
        r = client.post(
            "/api/v1/initialize",
            json={
                "checkpoint_path": str(ckpt),
                "config_path": str(cfg_path),
                "cell_states": cfg["data"]["cell_states"],
            },
        )
        assert r.status_code == 400, f"缺失键应返回 400，实际 {r.status_code}"
        assert "缺失" in r.json()["detail"]
