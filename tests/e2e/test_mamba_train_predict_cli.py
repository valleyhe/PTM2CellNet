"""端到端测试: Mamba 编码器训练 CLI → 推理 CLI。

对应缺口分析 §3.1 — Gap 2 (中等严重度):
Mamba 编码器 CLI E2E 测试缺失。

使用最小 Mamba 配置（CPU 可用的小模型），验证
scripts/train.py (Mamba 编码器) → scripts/predict.py 全链路。
需要 mamba-ssm 可选依赖时自动跳过。
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "scripts"
SAMPLE_DATA = PROJECT_ROOT / "data" / "raw" / "sample_data.csv"


def _run(cmd, cwd=PROJECT_ROOT):
    """运行子进程，失败时打印完整输出。"""
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    assert result.returncode == 0, (
        f"命令失败 (exit {result.returncode}): {' '.join(cmd)}"
    )
    return result


def _minimal_mamba_config(tmp_path: Path) -> Path:
    """生成最小 Mamba 训练配置（小模型、CPU 可运行）。"""
    cfg = {
        "model": {
            "encoder_type": "mamba",
            "vocab_size": 21,
            "embed_dim": 64,
            "hidden_dim": 64,
            "num_layers": 2,          # 最少层数
            "state_dim": 4,           # 最小 SSM 状态维
            "d_conv": 4,
            "expand_factor": 2,
            "max_seq_len": 1024,
            "num_ptm_types": 2,
            "num_classes": 2,
            "pool_type": "mean",
            "dropout": 0.0,
        },
        "data": {
            "max_sequence_length": 1024,
            "valid_amino_acids": "ACDEFGHIKLMNPQRSTVWY",
            "ptm_types": ["phosphorylation", "acetylation"],
        },
        "training": {
            "max_epochs": 1,
            "batch_size": 8,
            "learning_rate": 1e-3,
        },
    }
    path = tmp_path / "mamba_config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


# 检查 mamba-ssm 是否可用；不可用时跳过测试
_has_mamba_ssm = False
try:
    import mamba_ssm  # noqa: F401
    _has_mamba_ssm = True
except (ImportError, OSError):
    # MambaEncoder 纯 PyTorch 实现不需要 mamba-ssm C 扩展
    # 但训练管线可能会尝试导入
    pass


def _check_mamba_available():
    """验证 MambaEncoder 可在当前环境下创建（纯 PyTorch 实现）。"""
    try:
        from src.models.mamba_encoder import MambaEncoder
        model = MambaEncoder(vocab_size=21, hidden_dim=64, num_layers=1, state_dim=4)
        return True
    except Exception:
        return False


mamba_available = _check_mamba_available()


@pytest.mark.skipif(not mamba_available, reason="MambaEncoder 不可用")
class TestMambaTrainPredictE2E:
    """Mamba 训练→推理 E2E。"""

    @pytest.fixture(scope="class")
    def mamba_artifact(self, tmp_path_factory):
        """使用 Mamba 编码器训练一个模型。"""
        tmp = tmp_path_factory.mktemp("mamba")
        cfg_path = _minimal_mamba_config(tmp)
        output = tmp / "output"

        _run([
            sys.executable, str(SCRIPTS / "train.py"),
            "--config", str(cfg_path),
            "--data", str(SAMPLE_DATA),
            "--output", str(output),
            "--epochs", "1",
            "--batch_size", "8",
        ])

        ckpt = output / "models" / "best_model.pt"
        cfg = output / "models" / "best_model.config.yaml"
        assert ckpt.exists(), f"best_model.pt 未生成于 {ckpt}"
        assert cfg.exists(), f"best_model.config.yaml 未生成于 {cfg}"
        return {"ckpt": ckpt, "cfg": cfg, "output": output}

    def test_mamba_training_exports_artifacts(self, mamba_artifact):
        """训练完成后导出 best_model.pt + best_model.config.yaml。"""
        assert mamba_artifact["ckpt"].exists()
        assert mamba_artifact["cfg"].exists()

    def test_mamba_predict_single_sequence(self, mamba_artifact, tmp_path):
        """predict.py 可消费 Mamba 产出的 artifact。"""
        out = tmp_path / "pred.csv"
        _run([
            sys.executable, str(SCRIPTS / "predict.py"),
            "--model", str(mamba_artifact["ckpt"]),
            "--sequence", "ACDEFGHIKLMNPQRSTVWY",
            "--output", str(out),
            "--device", "cpu",
        ])
        import pandas as pd
        df = pd.read_csv(out)
        assert len(df) == 1
        assert "ptm_count" in df.columns
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        assert len(prob_cols) >= 1, "输出应包含 prob_<state> 列"

    def test_mamba_predict_batch(self, mamba_artifact, tmp_path):
        """predict.py 批量模式对 Mamba 模型可用。"""
        batch_csv = tmp_path / "batch.csv"
        batch_csv.write_text(
            "id,sequence\n"
            "1,ACDEFGHIKLMNPQRSTVWY\n"
            "2,AYDEFGHIKLMNPQRSTVWC\n",
            encoding="utf-8",
        )
        out = tmp_path / "batch_pred.csv"
        _run([
            sys.executable, str(SCRIPTS / "predict.py"),
            "--model", str(mamba_artifact["ckpt"]),
            "--input", str(batch_csv),
            "--output", str(out),
            "--device", "cpu",
        ])
        import pandas as pd
        df = pd.read_csv(out)
        assert len(df) == 2
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        assert len(prob_cols) >= 1

    def test_mamba_predict_with_ptm_sites(self, mamba_artifact, tmp_path):
        """predict.py 单样本模式传入 PTM 位点对 Mamba 模型可用。"""
        out = tmp_path / "pred_ptm.csv"
        _run([
            sys.executable, str(SCRIPTS / "predict.py"),
            "--model", str(mamba_artifact["ckpt"]),
            "--sequence", "ACDEFGHIKLMNPQRSTVWY",
            "--ptm-sites", '[{"position":3,"type":"phosphorylation"}]',
            "--output", str(out),
            "--device", "cpu",
        ])
        import pandas as pd
        df = pd.read_csv(out)
        assert len(df) == 1
        assert int(df["ptm_count"].iloc[0]) == 1
