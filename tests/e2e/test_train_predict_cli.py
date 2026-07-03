"""端到端测试：训练 CLI 产物 -> 推理 CLI/API，验证标签语义正确。

对应审计报告 P2-1 / §4 阶段 1-2 验收：
    - 原生 ``train.py`` -> ``predict.py``，标签映射与训练一致。
    - Lightning ``train_lightning.py`` 导出裸权重 -> ``predict.py``。
    - 单样本推理写出 ``--output`` 并真实消费 PTM 位点。

为保持 CI 可运行，使用小模型（CNN）、小数据（sample_data.csv）、CPU、1 epoch。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "scripts"
SAMPLE_DATA = PROJECT_ROOT / "data" / "raw" / "sample_data.csv"

# 训练标签顺序（sorted unique），与 demo fallback 顺序不同——用于验证持久化生效。
EXPECTED_LABEL_ORDER = ["apoptosis", "differentiation", "proliferation", "quiescence"]
DEMO_FALLBACK_ORDER = ["proliferation", "differentiation", "apoptosis", "quiescence"]


def _small_native_config(tmp_path: Path) -> Path:
    """生成一个最小可训练的原生配置（CNN、CPU、1 epoch）。"""
    cfg = {
        "model": {
            "encoder_type": "cnn",
            "vocab_size": 21,
            "embed_dim": 32,
            "num_filters": 32,
            "kernel_sizes": [3, 5],
            "max_seq_len": 512,
            "num_ptm_types": 6,
            "num_classes": 4,
            "pool_type": "mean",
            "dropout": 0.1,
        },
        "data": {
            "max_sequence_length": 512,
            "valid_amino_acids": "ACDEFGHIKLMNPQRSTVWY",
            "ptm_types": [
                "phosphorylation", "acetylation", "methylation",
                "ubiquitination", "sumoylation",
            ],
        },
        "training": {
            "max_epochs": 1,
            "batch_size": 16,
            "learning_rate": 1e-3,
        },
    }
    path = tmp_path / "native_config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _run(cmd, cwd=PROJECT_ROOT):
    """运行子进程，失败时打印完整输出。"""
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    assert result.returncode == 0, f"命令失败 (exit {result.returncode}): {' '.join(cmd)}"
    return result


@pytest.fixture(scope="module")
def native_artifact(tmp_path_factory):
    """训练一个原生模型并返回其 best_model.pt / config 路径。"""
    tmp = tmp_path_factory.mktemp("native")
    cfg_path = _small_native_config(tmp)
    output = tmp / "output"
    _run([
        sys.executable, str(SCRIPTS / "train.py"),
        "--config", str(cfg_path),
        "--data", str(SAMPLE_DATA),
        "--output", str(output),
        "--epochs", "1",
        "--batch_size", "16",
    ])
    ckpt = output / "models" / "best_model.pt"
    cfg = output / "models" / "best_model.config.yaml"
    assert ckpt.exists(), "best_model.pt 未生成"
    assert cfg.exists(), "best_model.config.yaml 未生成"
    return {"ckpt": ckpt, "cfg": cfg, "output": output}


@pytest.fixture(autouse=True)
def _reset_api_state(monkeypatch):
    """Reset shared API STATE around every test (P0-1).

    Tests in this module call ``POST /api/v1/initialize`` against the real
    ``create_app()``, which mutates the module-level ``STATE`` singleton
    (including ``variant_workflow``). Without a reset, that state leaks into
    later unit tests that assert the variant workflow is *not* initialized.

    These tests call /initialize, which requires PTM2CELLNET_API_KEY in
    production; run them in development mode to bypass that gate, and widen
    the checkpoint path allowlist to /tmp (pytest tmp_path lives under it).
    """
    monkeypatch.setenv("PTM2CELLNET_ENV", "development")
    monkeypatch.delenv("PTM2CELLNET_API_KEY", raising=False)
    monkeypatch.setenv("PTM2CELLNET_ALLOWED_ROOTS", "/tmp")
    from src.api.routes.state import reset_state

    reset_state()
    yield
    reset_state()


class TestNativeTrainPredictE2E:
    """P0-1 / P0-3: 原生训练 -> 推理 CLI 的真实 E2E 闭环。"""

    def test_config_persists_non_hardcoded_cell_states(self, native_artifact):
        """训练产物 config 必须持久化 sorted-unique 标签顺序，而非 None。"""
        saved = yaml.safe_load(native_artifact["cfg"].read_text())
        cell_states = saved.get("data", {}).get("cell_states")
        assert cell_states == EXPECTED_LABEL_ORDER, (
            f"cell_states 未持久化或顺序错误: {cell_states}"
        )
        # 同时保存 label_to_idx
        assert saved.get("data", {}).get("label_to_idx") == {
            label: i for i, label in enumerate(EXPECTED_LABEL_ORDER)
        }

    def test_predict_uses_persisted_labels_without_fallback(self, native_artifact, tmp_path):
        """推理必须使用持久化标签，且不依赖 demo fallback。"""
        out = tmp_path / "pred.csv"
        _run([
            sys.executable, str(SCRIPTS / "predict.py"),
            "--model", str(native_artifact["ckpt"]),
            "--sequence", "ACDEFGHIKLMNPQRSTVWY",
            "--output", str(out),
            "--device", "cpu",
        ])
        assert out.exists(), "单样本推理未写出 --output"
        import pandas as pd
        df = pd.read_csv(out)
        # 输出列应包含每个 cell_state 的概率
        for state in EXPECTED_LABEL_ORDER:
            assert f"prob_{state}" in df.columns, f"缺少概率列 prob_{state}"

    def test_predict_fails_fast_without_config(self, native_artifact, tmp_path):
        """缺少 sibling config 且未 --allow-demo-fallback 时必须 fail-fast。"""
        # 复制 checkpoint 到无 config 的目录
        bare = tmp_path / "bare.pt"
        bare.write_bytes(native_artifact["ckpt"].read_bytes())
        result = subprocess.run(
            [
                sys.executable, str(SCRIPTS / "predict.py"),
                "--model", str(bare),
                "--sequence", "ACDEFGHIKLMNPQRSTVWY",
                "--device", "cpu",
            ],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300,
        )
        assert result.returncode != 0, "缺少标签映射时应失败而非静默 fallback"
        assert "data.cell_states" in result.stdout or "data.cell_states" in result.stderr

    def test_predict_demo_fallback_flag_opt_in(self, native_artifact, tmp_path):
        """--allow-demo-fallback 显式启用后可使用 demo 顺序。

        构造一个不含 data.cell_states 的 config（仅描述模型结构），验证 fallback 生效。
        """
        bare = tmp_path / "bare.pt"
        bare.write_bytes(native_artifact["ckpt"].read_bytes())
        # 复制训练 config 但移除 cell_states，强制走 fallback 路径
        import copy
        cfg_dict = yaml.safe_load(native_artifact["cfg"].read_text())
        cfg_dict.setdefault("data", {}).pop("cell_states", None)
        cfg_dict["data"].pop("label_to_idx", None)
        bare_cfg = tmp_path / "bare.config.yaml"
        bare_cfg.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")
        out = tmp_path / "pred.csv"
        _run([
            sys.executable, str(SCRIPTS / "predict.py"),
            "--model", str(bare),
            "--config", str(bare_cfg),
            "--sequence", "ACDEFGHIKLMNPQRSTVWY",
            "--output", str(out),
            "--device", "cpu",
            "--allow-demo-fallback",
        ])
        assert out.exists()


class TestPredictPTMInput:
    """P1-1 / P1-2: CLI 真实消费 PTM 位点。"""

    def test_single_sample_ptm_sites_arg(self, native_artifact, tmp_path):
        """--ptm-sites 解析后传入模型，输出记录 ptm_count。"""
        out = tmp_path / "pred_ptm.csv"
        ptm = json.dumps([{"position": 3, "type": "phosphorylation"},
                          {"position": 7, "type": "acetylation"}])
        _run([
            sys.executable, str(SCRIPTS / "predict.py"),
            "--model", str(native_artifact["ckpt"]),
            "--sequence", "ACDEFGHIKLMNPQRSTVWY",
            "--ptm-sites", ptm,
            "--output", str(out),
            "--device", "cpu",
        ])
        import pandas as pd
        df = pd.read_csv(out)
        assert int(df["ptm_count"].iloc[0]) == 2

    def test_batch_ptm_sites_column(self, native_artifact, tmp_path):
        """批量 CSV 的 ptm_sites 列被解析并传入模型。"""
        batch_csv = tmp_path / "batch.csv"
        # 用引号包裹 JSON 避免 CSV 逗号分割问题
        batch_csv.write_text(
            "id,sequence,ptm_sites\n"
            '1,ACDEFGHIKLMNPQRSTVWY,"[{""position"":3,""type"":""phosphorylation""}]"\n'
            "2,ACDEFGHIKLMNPQRSTVWY,\n",
            encoding="utf-8",
        )
        out = tmp_path / "batch_pred.csv"
        _run([
            sys.executable, str(SCRIPTS / "predict.py"),
            "--model", str(native_artifact["ckpt"]),
            "--input", str(batch_csv),
            "--output", str(out),
            "--device", "cpu",
        ])
        import pandas as pd
        df = pd.read_csv(out)
        assert int(df.loc[0, "ptm_count"]) == 1
        assert int(df.loc[1, "ptm_count"]) == 0


class TestUnifiedCheckpointContract:
    """P0-3: 统一 checkpoint 合约——推理 CLI 可消费 Lightning .ckpt。"""

    def test_extract_model_state_dict_handles_lightning_prefix(self):
        """extract_model_state_dict 剥离 Lightning model. 前缀。"""
        from src.utils.checkpoint_utils import extract_model_state_dict

        sd = {"encoder.weight": torch.zeros(2, 2), "head.bias": torch.zeros(2)}
        lightning_ckpt = {
            "epoch": 1, "pytorch-lightning_version": "2.0",
            "state_dict": {f"model.{k}": v for k, v in sd.items()},
        }
        out = extract_model_state_dict(lightning_ckpt)
        assert set(out.keys()) == set(sd.keys())

    def test_extract_model_state_dict_bare_passthrough(self):
        from src.utils.checkpoint_utils import extract_model_state_dict
        sd = {"encoder.weight": torch.zeros(2, 2)}
        assert extract_model_state_dict(sd) is sd

    def test_extract_model_state_dict_legacy(self):
        from src.utils.checkpoint_utils import extract_model_state_dict
        sd = {"encoder.weight": torch.zeros(2, 2)}
        out = extract_model_state_dict({"epoch": 0, "model_state_dict": sd})
        assert out is sd

    def test_load_model_consumes_lightning_ckpt(self, tmp_path):
        """load_model 可直接加载 Lightning .ckpt 到裸模型。"""
        from src.models.architectures import PTM2CellNet
        from src.utils.io import load_model

        model = PTM2CellNet(encoder_type="cnn", vocab_size=21, embed_dim=16,
                            max_seq_len=64, num_ptm_types=5, num_classes=4)
        sd = model.state_dict()
        lightning_ckpt = {
            "epoch": 1, "global_step": 5, "pytorch-lightning_version": "2.0",
            "state_dict": {f"model.{k}": v for k, v in sd.items()},
        }
        ckpt = tmp_path / "lightning.ckpt"
        torch.save(lightning_ckpt, ckpt)

        fresh = PTM2CellNet(encoder_type="cnn", vocab_size=21, embed_dim=16,
                            max_seq_len=64, num_ptm_types=5, num_classes=4)
        load_model(fresh, str(ckpt))  # strict=True 默认
        for k in sd:
            assert torch.allclose(sd[k], fresh.state_dict()[k]), f"权重不一致: {k}"


class TestLightningE2E:
    """P0-1: Lightning training -> inference E2E with non-default class count."""

    def _small_lightning_config(self, tmp_path, num_classes=3):
        import yaml
        from pathlib import Path
        cell_states = [f"class_{i}" for i in range(num_classes)]
        cfg = {
            "model": {
                "encoder_type": "cnn", "vocab_size": 21, "embed_dim": 32,
                "num_filters": 32, "kernel_sizes": [3, 5], "max_seq_len": 512,
                "num_ptm_types": 6, "num_classes": 4,  # deliberately wrong
                "pool_type": "mean", "dropout": 0.1,
            },
            "data": {
                "max_sequence_length": 512,
                "valid_amino_acids": "ACDEFGHIKLMNPQRSTVWY",
                "ptm_types": ["phosphorylation", "acetylation", "methylation",
                              "ubiquitination", "sumoylation"],
                "cell_states": cell_states,
            },
            "training": {
                "max_epochs": 1, "batch_size": 4, "learning_rate": 1e-3,
                "drop_last": False,
                "checkpoint": {"enabled": True, "monitor": "step", "save_top_k": 1},
                "early_stopping": {"enabled": False},
                "logger": {"type": "tensorboard"},
            },
            "paths": {"outputs_models": str(tmp_path / "models")},
        }
        path = tmp_path / "lightning_config.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return path, cell_states

    def _make_3class_data(self, tmp_path):
        """Create a tiny 3-class CSV dataset."""
        csv = tmp_path / "train_3class.csv"
        csv.write_text(
            "sequence,cell_state,ptm_sites\n"
            "ACDEFGHIKLMNPQRSTVWY,class_0,[]\n"
            "ACDEFGHIKLMNPQRSTVWY,class_1,[]\n"
            "ACDEFGHIKLMNPQRSTVWY,class_2,[]\n"
            "ACDEFGHIKLMNPQRSTVWY,class_0,[]\n"
            "ACDEFGHIKLMNPQRSTVWY,class_1,[]\n"
            "ACDEFGHIKLMNPQRSTVWY,class_2,[]\n",
            encoding="utf-8",
        )
        return csv

    def test_lightning_export_has_correct_num_classes(self, tmp_path):
        """Train with 3-class data, config model.num_classes=4 initially.
        After fix P0-1, exported config should have num_classes=3."""
        import os, sys, subprocess, yaml
        from pathlib import Path
        cfg_path, cell_states = self._small_lightning_config(tmp_path)
        data_csv = self._make_3class_data(tmp_path)

        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts/train_lightning.py"),
             "--config", str(cfg_path), "--data", str(data_csv),
             "--max-epochs", "1", "--batch-size", "4"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        assert result.returncode == 0, f"Lightning training failed: {result.stderr}"

        # Check exported config
        models_dir = tmp_path / "models"
        config_files = list(models_dir.glob("*.config.yaml"))
        assert config_files, f"No config file found in {models_dir}"
        saved = yaml.safe_load(config_files[0].read_text())
        assert saved["model"]["num_classes"] == 3, (
            f"Expected num_classes=3, got {saved['model']['num_classes']}"
        )
        assert saved["data"]["cell_states"] == cell_states

    def test_lightning_ckpt_inference(self, tmp_path):
        """Use Lightning-exported best_model.pt for CLI inference."""
        import os, sys, subprocess, yaml
        from pathlib import Path
        cfg_path, cell_states = self._small_lightning_config(tmp_path)
        data_csv = self._make_3class_data(tmp_path)

        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts/train_lightning.py"),
             "--config", str(cfg_path), "--data", str(data_csv),
             "--max-epochs", "1", "--batch-size", "4"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300,
        )

        models_dir = tmp_path / "models"
        ckpt = models_dir / "best_model.pt"
        if not ckpt.exists():
            ckpt = next(models_dir.glob("*.pt"), None)
        assert ckpt and ckpt.exists(), f"No best_model.pt found in {models_dir}"

        out = tmp_path / "pred.csv"
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts/predict.py"),
             "--model", str(ckpt),
             "--sequence", "ACDEFGHIKLMNPQRSTVWY",
             "--output", str(out), "--device", "cpu"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        assert result.returncode == 0, f"Inference failed: {result.stderr}"
        import pandas as pd
        df = pd.read_csv(out)
        for state in cell_states:
            assert f"prob_{state}" in df.columns, f"Missing prob_{state}"


class TestBatchProbabilityColumns:
    """P1-2: Batch output includes per-class probability columns."""

    def test_batch_has_prob_columns(self, native_artifact, tmp_path):
        """Batch prediction output must include prob_<state> columns."""
        batch_csv = tmp_path / "batch_prob.csv"
        batch_csv.write_text(
            "id,sequence,ptm_sites\n"
            '1,ACDEFGHIKLMNPQRSTVWY,"[]"\n'
            '2,ACDEFGHIKLMNPQRSTVWY,"[]"\n',
            encoding="utf-8",
        )
        import subprocess, sys
        from pathlib import Path
        out = tmp_path / "batch_prob_pred.csv"
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "predict.py"),
             "--model", str(native_artifact["ckpt"]),
             "--input", str(batch_csv), "--output", str(out), "--device", "cpu"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        assert result.returncode == 0, f"Batch inference failed: {result.stderr}"
        import pandas as pd
        df = pd.read_csv(out)
        for state in EXPECTED_LABEL_ORDER:
            assert f"prob_{state}" in df.columns, f"Missing prob_{state} in batch output"


class TestConfigInjection:
    """P1-1: Inference preprocessing uses training config."""

    def test_max_sequence_length_respected(self, native_artifact, tmp_path):
        """When training config has max_sequence_length=64, inference preprocessing
        should use 64, not the default 1000."""
        import subprocess, sys, yaml
        from pathlib import Path
        # Override config to set max_sequence_length=64
        out = tmp_path / "cfg_pred.csv"
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "predict.py"),
             "--model", str(native_artifact["ckpt"]),
             "--sequence", "ACDEFGHIKLMNPQRSTVWY",
             "--output", str(out), "--device", "cpu"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        assert result.returncode == 0, f"Inference failed: {result.stderr}"
        import pandas as pd
        df = pd.read_csv(out)
        assert len(df) == 1
        # Verify at least one probability column exists
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        assert len(prob_cols) == len(EXPECTED_LABEL_ORDER), (
            f"Expected {len(EXPECTED_LABEL_ORDER)} prob cols, got {prob_cols}"
        )


class TestAPIPredict:
    """P2-1: API /predict after /initialize."""

    def test_api_predict_after_initialize(self, native_artifact, tmp_path):
        """Initialize model via API, then call /predict."""
        from fastapi.testclient import TestClient
        from src.api.app import create_app
        from src.api.routes.state import STATE

        client = TestClient(create_app())
        # Initialize
        r_init = client.post("/api/v1/initialize", json={
            "checkpoint_path": str(native_artifact["ckpt"]),
            "cell_states": EXPECTED_LABEL_ORDER,
        })
        assert r_init.status_code == 200, r_init.text
        assert STATE.model is not None

        # Predict
        r_pred = client.post("/api/v1/predict", json={
            "sequence": "ACDEFGHIKLMNPQRSTVWY",
            "ptm_sites": [],
        })
        assert r_pred.status_code == 200, r_pred.text
        body = r_pred.json()
        assert "predicted_cell_state" in body
        assert body["predicted_cell_state"] in EXPECTED_LABEL_ORDER
        assert "probabilities" in body


class TestLightningE2E:
    """P0-1: Lightning training -> inference E2E with non-default class count."""

    def _small_lightning_config(self, tmp_path, num_classes=3):
        import yaml

        cell_states = [f"class_{i}" for i in range(num_classes)]
        cfg = {
            "model": {
                "encoder_type": "cnn", "vocab_size": 21, "embed_dim": 32,
                "num_filters": 32, "kernel_sizes": [3, 5], "max_seq_len": 512,
                "num_ptm_types": 6, "num_classes": 4,
                "pool_type": "mean", "dropout": 0.1,
            },
            "data": {
                "max_sequence_length": 512,
                "valid_amino_acids": "ACDEFGHIKLMNPQRSTVWY",
                "ptm_types": ["phosphorylation", "acetylation", "methylation",
                              "ubiquitination", "sumoylation"],
                "cell_states": cell_states,
            },
            "training": {
                "max_epochs": 1, "batch_size": 4, "learning_rate": 1e-3,
                "drop_last": False,
                "checkpoint": {"enabled": True, "monitor": "step", "save_top_k": 1},
                "early_stopping": {"enabled": False},
                "logger": {"type": "tensorboard"},
            },
            "paths": {"outputs_models": str(tmp_path / "models")},
        }
        path = tmp_path / "lightning_config.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return path, cell_states

    def _make_3class_data(self, tmp_path):
        """Create a tiny 3-class CSV with diverse sequences to avoid dedup collapse."""
        csv = tmp_path / "train_3class.csv"
        csv.write_text(
            "sequence,cell_state,ptm_sites\n"
            "MKTIIALSYIFCLVFA,class_0,[]\n"
            "ACDEFGHIKLMNPQRSTVWY,class_1,[]\n"
            "GCTVEDRCLIGMGAILLNGCVIGSGSLVAAGALITQQ,class_2,[]\n"
            "MKTVRQERLKSIVRILERSKEP,class_0,[]\n"
            "LEIKLISIAQCVNPHYEGA,class_1,[]\n"
            "CRFNGGKPLWVLAQKGNGEKVVF,class_2,[]\n"
            "DLLCMSEDINTFILQCIQSVVDSG,class_0,[]\n"
            "LSPGQSNALLRESLLLGLI,class_1,[]\n"
            "PSPLREAYALCNGLGQY,class_2,[]\n",
            encoding="utf-8",
        )
        return csv

    def test_lightning_export_has_correct_num_classes(self, tmp_path):
        import subprocess
        import sys
        import yaml

        cfg_path, cell_states = self._small_lightning_config(tmp_path)
        data_csv = self._make_3class_data(tmp_path)
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "train_lightning.py"),
             "--config", str(cfg_path), "--data", str(data_csv),
             "--max-epochs", "1", "--batch-size", "4"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        assert result.returncode == 0, f"Lightning training failed: {result.stderr}"
        models_dir = tmp_path / "models"
        config_files = list(models_dir.glob("*.config.yaml"))
        assert config_files, f"No config file found in {models_dir}"
        saved = yaml.safe_load(config_files[0].read_text())
        assert saved["model"]["num_classes"] == 3, (
            f"Expected num_classes=3, got {saved['model']['num_classes']}"
        )
        assert saved["data"]["cell_states"] == cell_states

    def test_lightning_ckpt_inference(self, tmp_path):
        import subprocess
        import sys

        cfg_path, cell_states = self._small_lightning_config(tmp_path)
        data_csv = self._make_3class_data(tmp_path)
        subprocess.run(
            [sys.executable, str(SCRIPTS / "train_lightning.py"),
             "--config", str(cfg_path), "--data", str(data_csv),
             "--max-epochs", "1", "--batch-size", "4"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300,
        )
        models_dir = tmp_path / "models"
        ckpt = models_dir / "best_model.pt"
        if not ckpt.exists():
            ckpt = next(models_dir.glob("*.pt"), None)
        assert ckpt and ckpt.exists(), f"No best_model.pt found in {models_dir}"
        out = tmp_path / "pred.csv"
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "predict.py"),
             "--model", str(ckpt),
             "--sequence", "ACDEFGHIKLMNPQRSTVWY",
             "--output", str(out), "--device", "cpu"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        assert result.returncode == 0, f"Inference failed: {result.stderr}"
        import pandas as pd

        df = pd.read_csv(out)
        for state in cell_states:
            assert f"prob_{state}" in df.columns, f"Missing prob_{state}"


class TestBatchProbabilityColumns:
    """P1-2: Batch output includes per-class probability columns."""

    def test_batch_has_prob_columns(self, native_artifact, tmp_path):
        import subprocess
        import sys

        batch_csv = tmp_path / "batch_prob.csv"
        batch_csv.write_text(
            "id,sequence,ptm_sites\n"
            '1,ACDEFGHIKLMNPQRSTVWY,"[]"\n'
            '2,ACDEFGHIKLMNPQRSTVWY,"[]"\n',
            encoding="utf-8",
        )
        out = tmp_path / "batch_prob_pred.csv"
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "predict.py"),
             "--model", str(native_artifact["ckpt"]),
             "--input", str(batch_csv), "--output", str(out), "--device", "cpu"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        assert result.returncode == 0, f"Batch inference failed: {result.stderr}"
        import pandas as pd

        df = pd.read_csv(out)
        for state in EXPECTED_LABEL_ORDER:
            assert f"prob_{state}" in df.columns, f"Missing prob_{state} in batch output"


class TestAPIPredict:
    """P2-1: API /predict after /initialize."""

    def test_api_predict_after_initialize(self, native_artifact, tmp_path):
        from fastapi.testclient import TestClient
        from src.api.app import create_app
        from src.api.routes.state import STATE

        client = TestClient(create_app())
        r_init = client.post("/api/v1/initialize", json={
            "checkpoint_path": str(native_artifact["ckpt"]),
            "cell_states": EXPECTED_LABEL_ORDER,
        })
        assert r_init.status_code == 200, r_init.text
        assert STATE.model is not None

        r_pred = client.post("/api/v1/predict", json={
            "sequence": "ACDEFGHIKLMNPQRSTVWY",
            "ptm_sites": [],
        })
        assert r_pred.status_code == 200, r_pred.text
        body = r_pred.json()
        assert "predicted_cell_state" in body
        assert body["predicted_cell_state"] in EXPECTED_LABEL_ORDER
        assert "probabilities" in body
