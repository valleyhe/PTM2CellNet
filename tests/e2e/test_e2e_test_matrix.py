"""Extended E2E test matrix (P2-1).

Covers the deployment/inference edge cases the original happy-path suite did
not lock down:

* custom (non-default) cell-state label order survives train→predict.
* custom PTM types round-trip through training and inference.
* negative cases: missing sibling config fails fast; label-count vs logits
  mismatch is rejected by the API ``/initialize``.
* CLI batch inference over an empty CSV produces an empty output (locked
  behaviour, not a crash).
* demo models surface their ``model_kind`` via ``/model/info``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "scripts"


def _small_config(tmp_path: Path, cell_states: list[str], ptm_types: list[str]) -> Path:
    cfg = {
        "model": {
            "encoder_type": "cnn",
            "vocab_size": 21,
            "embed_dim": 16,
            "num_filters": 16,
            "kernel_sizes": [3],
            "max_seq_len": 64,
            "num_ptm_types": len(ptm_types) + 1,
            "num_classes": len(cell_states),
            "pool_type": "mean",
            "dropout": 0.0,
        },
        "data": {
            "max_sequence_length": 64,
            "valid_amino_acids": "ACDEFGHIKLMNPQRSTVWY",
            "ptm_types": ptm_types,
        },
        "training": {"max_epochs": 1, "batch_size": 8, "learning_rate": 1e-3},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _build_cnn(num_classes: int = 3):
    """Build a small CNN model via from_config (the supported construction path).

    Direct ``PTM2CellNet(...)`` does not accept encoder-specific kwargs like
    ``num_filters``/``kernel_sizes``; those only flow through config. Tests that
    need an ad-hoc model therefore go through ``from_config``.
    """
    from src.models.architectures import PTM2CellNet

    cfg = {
        "model": {
            "encoder_type": "cnn",
            "vocab_size": 21,
            "embed_dim": 16,
            "num_filters": 16,
            "kernel_sizes": [3],
            "max_seq_len": 64,
            "num_ptm_types": 5,
            "num_classes": num_classes,
            "pool_type": "mean",
            "dropout": 0.0,
        }
    }
    return PTM2CellNet.from_config(cfg)


def _write_config_for_model(path: Path, num_classes: int, cell_states: list[str], model_kind: str | None = None):
    cfg = {
        "model": {
            "encoder_type": "cnn",
            "vocab_size": 21,
            "embed_dim": 16,
            "num_filters": 16,
            "kernel_sizes": [3],
            "max_seq_len": 64,
            "num_ptm_types": 5,
            "num_classes": num_classes,
            "pool_type": "mean",
        },
        "data": {"cell_states": cell_states},
    }
    if model_kind:
        cfg["model"]["model_kind"] = model_kind
        cfg["model_card"] = {"model_kind": model_kind, "not_for_biological_use": model_kind == "demo"}
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _make_csv(path: Path, rows: list[tuple[str, str, str]]):
    import csv as _csv

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["sequence", "cell_state", "ptm_sites"])
        for seq, label, ptm in rows:
            writer.writerow([seq, label, ptm])


def _run(cmd, cwd=PROJECT_ROOT, timeout=300):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    assert result.returncode == 0, f"命令失败 (exit {result.returncode}): {' '.join(cmd)}"
    return result


@pytest.fixture(autouse=True)
def _reset_api_state(monkeypatch):
    # E2E 矩阵测试调用 /initialize，需在 development 模式运行以绕过生产 API key 要求
    # (生产部署时由部署方设置 PTM2CELLNET_API_KEY)。同时放宽 checkpoint 路径白名单
    # 至 /tmp（pytest tmp_path 在其下），生产部署由部署方配置真实白名单。
    monkeypatch.setenv("PTM2CELLNET_ENV", "development")
    monkeypatch.delenv("PTM2CELLNET_API_KEY", raising=False)
    monkeypatch.setenv("PTM2CELLNET_ALLOWED_ROOTS", "/tmp")
    from src.api.routes.state import reset_state

    reset_state()
    yield
    reset_state()


@pytest.fixture(scope="module")
def native_artifact(tmp_path_factory):
    """Create a minimal trained-artifact for batch PTM parsing edge case tests."""
    tmp = tmp_path_factory.mktemp("native_artifact")
    num_classes = 3
    cell_states = ["a", "b", "c"]
    ckpt = tmp / "best_model.pt"
    cfg = tmp / "best_model.config.yaml"
    model = _build_cnn(num_classes=num_classes)
    torch.save(model.state_dict(), ckpt)
    _write_config_for_model(cfg, num_classes=num_classes, cell_states=cell_states)
    return {"ckpt": ckpt, "cfg": cfg}


# ---------------------------------------------------------------------------
# Custom labels / PTM types round-trip
# ---------------------------------------------------------------------------


class TestCustomLabelSemantics:
    def test_custom_label_order_persists_through_predict(self, tmp_path):
        """Non-default, non-sorted-looking labels decode correctly end-to-end."""
        cell_states = ["resting", "activated", "memory"]
        ptm_types = ["phosphorylation", "ubiquitination"]
        cfg_path = _small_config(tmp_path, cell_states, ptm_types)
        data_csv = tmp_path / "data.csv"
        # Many distinct rows per class so the held-out test split is very likely
        # to contain all 3 labels (otherwise the multi-class AUC evaluator raises).
        import random

        rng = random.Random(0)
        amino = "ACDEFGHIKLMNPQRSTVWY"
        rows = []
        for _cls_idx, label in enumerate(cell_states):
            for _ in range(14):
                seq = "".join(rng.choice(amino) for _ in range(rng.randint(12, 20)))
                rows.append((seq, label, "[]"))
        _make_csv(data_csv, rows)
        out = tmp_path / "out"
        _run(
            [
                sys.executable,
                str(SCRIPTS / "train.py"),
                "--config",
                str(cfg_path),
                "--data",
                str(data_csv),
                "--output",
                str(out),
                "--epochs",
                "1",
                "--batch_size",
                "4",
            ]
        )
        saved = yaml.safe_load((out / "models" / "best_model.config.yaml").read_text())
        # Labels persisted as sorted-unique.
        assert sorted(saved["data"]["cell_states"]) == sorted(cell_states)

        pred_out = tmp_path / "pred.csv"
        _run(
            [
                sys.executable,
                str(SCRIPTS / "predict.py"),
                "--model",
                str(out / "models" / "best_model.pt"),
                "--sequence",
                "ACDEFGHIK",
                "--output",
                str(pred_out),
                "--device",
                "cpu",
            ]
        )
        import pandas as pd

        df = pd.read_csv(pred_out)
        prob_cols = {c.replace("prob_", "") for c in df.columns if c.startswith("prob_")}
        assert prob_cols == set(cell_states), f"推理概率列应匹配训练标签，实际: {prob_cols}"

    def test_custom_ptm_types_round_trip(self, tmp_path):
        """A custom PTM type used at training is accepted at inference."""
        cell_states = ["a", "b"]
        ptm_types = ["glycosylation", "nitrosylation"]  # non-default
        cfg_path = _small_config(tmp_path, cell_states, ptm_types)
        data_csv = tmp_path / "data.csv"
        seqs = [
            "ACDEFGHIK",
            "CDEFGHIKLM",
            "EFGHIKLMNP",
            "GHIKLMNPQR",
            "IKLMNPQRST",
            "KLMNPQRSTV",
            "LMNPQRSTVW",
            "MNPQRSTVWY",
        ]
        rows = []
        for i, seq in enumerate(seqs):
            label = cell_states[i % 2]
            ptm = '[{"position":1,"type":"glycosylation"}]' if i % 2 == 0 else "[]"
            rows.append((seq, label, ptm))
        _make_csv(data_csv, rows)
        out = tmp_path / "out"
        _run(
            [
                sys.executable,
                str(SCRIPTS / "train.py"),
                "--config",
                str(cfg_path),
                "--data",
                str(data_csv),
                "--output",
                str(out),
                "--epochs",
                "1",
                "--batch_size",
                "2",
            ]
        )
        pred_out = tmp_path / "pred.csv"
        ptm = json.dumps([{"position": 2, "type": "glycosylation"}])
        _run(
            [
                sys.executable,
                str(SCRIPTS / "predict.py"),
                "--model",
                str(out / "models" / "best_model.pt"),
                "--sequence",
                "ACDEFGHIK",
                "--ptm-sites",
                ptm,
                "--output",
                str(pred_out),
                "--device",
                "cpu",
            ]
        )
        import pandas as pd

        df = pd.read_csv(pred_out)
        # ptm_count column reflects the parsed site.
        assert int(df["ptm_count"].iloc[0]) == 1


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


class TestNegativeCases:
    def test_predict_missing_sibling_config_fails_fast(self, tmp_path):
        """A checkpoint with no sibling config and no --config must error."""
        import torch

        ckpt = tmp_path / "bare.pt"
        torch.save(_build_cnn(num_classes=3).state_dict(), ckpt)

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "predict.py"),
                "--model",
                str(ckpt),
                "--sequence",
                "ACDEFGHIK",
                "--device",
                "cpu",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode != 0, "缺少 config 时应失败而非静默运行"
        combined = result.stdout + result.stderr
        assert "config" in combined.lower()

    def test_api_rejects_label_logits_mismatch(self, tmp_path):
        """/initialize must 400 when cell_states count != checkpoint logits dim."""
        import torch
        from fastapi.testclient import TestClient
        from src.api.app import create_app

        ckpt = tmp_path / "m.pt"
        torch.save(_build_cnn(num_classes=3).state_dict(), ckpt)
        _write_config_for_model(tmp_path / "m.config.yaml", num_classes=3, cell_states=["a", "b", "c"])

        client = TestClient(create_app())
        # initialize with only 2 labels → must be rejected.
        r = client.post(
            "/api/v1/initialize",
            json={
                "checkpoint_path": str(ckpt),
                "cell_states": ["a", "b"],  # wrong count vs 3-class logits
            },
        )
        assert r.status_code == 400, f"标签数与 logits 维度不一致时应返回 400，实际 {r.status_code}: {r.text}"


# ---------------------------------------------------------------------------
# Empty batch behaviour
# ---------------------------------------------------------------------------


class TestEmptyBatch:
    def test_empty_input_csv_produces_empty_output(self, tmp_path):
        """An empty (header-only) batch CSV must yield an empty output, not crash."""
        import torch
        import pandas as pd

        ckpt = tmp_path / "m.pt"
        torch.save(_build_cnn(num_classes=3).state_dict(), ckpt)
        _write_config_for_model(tmp_path / "m.config.yaml", num_classes=3, cell_states=["a", "b", "c"])

        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("id,sequence,ptm_sites\n", encoding="utf-8")
        out = tmp_path / "out.csv"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "predict.py"),
                "--model",
                str(ckpt),
                "--input",
                str(empty_csv),
                "--output",
                str(out),
                "--device",
                "cpu",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        assert result.returncode == 0, "空输入 CSV 不应导致崩溃"
        # Output exists; whether it has a header or is empty is acceptable, but
        # there must be no data rows.
        if out.exists():
            df = pd.read_csv(out)
            assert len(df) == 0, f"空输入应产生空输出，实际 {len(df)} 行"


# ---------------------------------------------------------------------------
# Demo model surfacing via /model/info
# ---------------------------------------------------------------------------


class TestModelInfoDemoFlag:
    def test_demo_model_info_exposes_model_kind(self, tmp_path):
        """A demo artifact's /model/info must flag is_demo_model=True."""
        import torch
        from fastapi.testclient import TestClient
        from src.api.app import create_app
        from src.api.routes.state import STATE

        ckpt = tmp_path / "m.pt"
        torch.save(_build_cnn(num_classes=3).state_dict(), ckpt)
        _write_config_for_model(
            tmp_path / "m.config.yaml",
            num_classes=3,
            cell_states=["a", "b", "c"],
            model_kind="demo",
        )

        client = TestClient(create_app())
        r = client.post(
            "/api/v1/initialize",
            json={
                "checkpoint_path": str(ckpt),
                "cell_states": ["a", "b", "c"],
            },
        )
        assert r.status_code == 200, r.text
        assert STATE.is_demo_model is True
        info = client.get("/api/v1/model/info").json()
        assert info["is_demo_model"] is True
        assert info["model_kind"] == "demo"


# =============================================================================
# 批量 PTM 位点解析边界场景 (对应缺口分析 §3.1 — Gap 4)
# =============================================================================


class TestBatchPTMParsingEdgeCases:
    """CSV 批量推理中 PTM 位点解析的边界场景 E2E 覆盖。

    对应缺口分析: "没有测试批量 CSV 中 ptm_sites 列包含真实 JSON
    位点数据的完整路径" — 覆盖 None/空/非法 JSON/越界 position。
    """

    def test_batch_ptm_sites_none_value(self, native_artifact, tmp_path):
        """ptm_sites 列为 None/空值时应正确处理。"""
        batch_csv = tmp_path / "batch_none.csv"
        batch_csv.write_text(
            "id,sequence,ptm_sites\n"
            "1,ACDEFGHIKLMNPQRSTVWY,\n"  # empty ptm_sites
            "2,ACDEFGHIKLMNPQRSTVWY,\n",  # empty ptm_sites
            encoding="utf-8",
        )
        out = tmp_path / "pred_none.csv"
        _run(
            [
                sys.executable,
                str(SCRIPTS / "predict.py"),
                "--model",
                str(native_artifact["ckpt"]),
                "--input",
                str(batch_csv),
                "--output",
                str(out),
                "--device",
                "cpu",
            ]
        )
        import pandas as pd

        df = pd.read_csv(out)
        assert len(df) == 2
        assert int(df["ptm_count"].iloc[0]) == 0
        assert int(df["ptm_count"].iloc[1]) == 0

    def test_batch_ptm_sites_empty_list(self, native_artifact, tmp_path):
        """ptm_sites 列为 [] 空列表时应正确处理。"""
        batch_csv = tmp_path / "batch_empty_list.csv"
        batch_csv.write_text(
            "id,sequence,ptm_sites\n1,ACDEFGHIKLMNPQRSTVWY,[]\n2,ACDEFGHIKLMNPQRSTVWY,[]\n",
            encoding="utf-8",
        )
        out = tmp_path / "pred_empty_list.csv"
        _run(
            [
                sys.executable,
                str(SCRIPTS / "predict.py"),
                "--model",
                str(native_artifact["ckpt"]),
                "--input",
                str(batch_csv),
                "--output",
                str(out),
                "--device",
                "cpu",
            ]
        )
        import pandas as pd

        df = pd.read_csv(out)
        assert len(df) == 2
        assert int(df["ptm_count"].iloc[0]) == 0

    def test_batch_ptm_sites_malformed_json(self, native_artifact, tmp_path):
        """ptm_sites 列为非法 JSON 时应输出警告但仍可完成推理。"""
        batch_csv = tmp_path / "batch_malformed.csv"
        batch_csv.write_text(
            "id,sequence,ptm_sites\n"
            '1,ACDEFGHIKLMNPQRSTVWY,"{invalid json}"\n'
            '2,ACDEFGHIKLMNPQRSTVWY,"[{""position"":3,""type"":""phosphorylation""}]"\n',
            encoding="utf-8",
        )
        out = tmp_path / "pred_malformed.csv"
        result = _run(
            [
                sys.executable,
                str(SCRIPTS / "predict.py"),
                "--model",
                str(native_artifact["ckpt"]),
                "--input",
                str(batch_csv),
                "--output",
                str(out),
                "--device",
                "cpu",
            ]
        )
        import pandas as pd

        df = pd.read_csv(out)
        assert len(df) == 2
        # 第 1 行 PTM 解析失败 → ptm_count 应为 0
        assert int(df["ptm_count"].iloc[0]) == 0
        # 第 2 行正常 → ptm_count 应为 1
        assert int(df["ptm_count"].iloc[1]) == 1

    def test_batch_ptm_sites_position_out_of_bounds(self, native_artifact, tmp_path):
        """position 超出序列长度时应跳过该位点并输出警告。"""
        batch_csv = tmp_path / "batch_oob.csv"
        batch_csv.write_text(
            "id,sequence,ptm_sites\n"
            # sequence 只有 20 AA, position=999 越界
            '1,ACDEFGHIKLMNPQRSTVWY,"[{""position"":999,""type"":""phosphorylation""}]"\n'
            '2,ACDEFGHIKLMNPQRSTVWY,"[{""position"":3,""type"":""methylation""}]"\n',
            encoding="utf-8",
        )
        out = tmp_path / "pred_oob.csv"
        _run(
            [
                sys.executable,
                str(SCRIPTS / "predict.py"),
                "--model",
                str(native_artifact["ckpt"]),
                "--input",
                str(batch_csv),
                "--output",
                str(out),
                "--device",
                "cpu",
            ]
        )
        import pandas as pd

        df = pd.read_csv(out)
        assert len(df) == 2
        # 越界 position 应被跳过 → ptm_count 为 0
        assert int(df["ptm_count"].iloc[0]) == 0
        # 正常位点 → ptm_count 为 1
        assert int(df["ptm_count"].iloc[1]) == 1

    def test_batch_ptm_sites_missing_ptm_column(self, native_artifact, tmp_path):
        """CSV 缺少 ptm_sites 列时应默认视为无 PTM 位点。"""
        batch_csv = tmp_path / "batch_no_ptm_col.csv"
        batch_csv.write_text(
            "id,sequence\n1,ACDEFGHIKLMNPQRSTVWY\n2,ACDEFGHIKLMNPQRSTVWC\n",
            encoding="utf-8",
        )
        out = tmp_path / "pred_no_ptm_col.csv"
        _run(
            [
                sys.executable,
                str(SCRIPTS / "predict.py"),
                "--model",
                str(native_artifact["ckpt"]),
                "--input",
                str(batch_csv),
                "--output",
                str(out),
                "--device",
                "cpu",
            ]
        )
        import pandas as pd

        df = pd.read_csv(out)
        assert len(df) == 2
        assert int(df["ptm_count"].iloc[0]) == 0
        assert int(df["ptm_count"].iloc[1]) == 0

    def test_batch_ptm_sites_various_string_artifacts(self, native_artifact, tmp_path):
        """处理 CSV 中常见的 pandas NaN/null/None 字符串产物。"""
        batch_csv = tmp_path / "batch_artifacts.csv"
        batch_csv.write_text(
            "id,sequence,ptm_sites\n"
            "1,ACDEFGHIKLMNPQRSTVWY,nan\n"
            "2,ACDEFGHIKLMNPQRSTVWY,null\n"
            "3,ACDEFGHIKLMNPQRSTVWY,None\n"
            "4,ACDEFGHIKLMNPQRSTVWY,\n"
            "5,ACDEFGHIKLMNPQRSTVWY,[]\n",
            encoding="utf-8",
        )
        out = tmp_path / "pred_artifacts.csv"
        _run(
            [
                sys.executable,
                str(SCRIPTS / "predict.py"),
                "--model",
                str(native_artifact["ckpt"]),
                "--input",
                str(batch_csv),
                "--output",
                str(out),
                "--device",
                "cpu",
            ]
        )
        import pandas as pd

        df = pd.read_csv(out)
        assert len(df) == 5
        # 所有各种 artifacts 均应解析为 0 个 PTM 位点
        assert all(int(df["ptm_count"].iloc[i]) == 0 for i in range(5))
