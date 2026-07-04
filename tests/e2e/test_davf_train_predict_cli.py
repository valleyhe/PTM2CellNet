"""端到端测试: DAVF 微调训练 CLI → 推理 CLI。

对应缺口分析 §3.1 — Gap 1 (中等严重度):
DAVF 训练→推理 E2E 测试缺失。

使用不存在的 DAVF checkpoint（DAVFInferenceModule 优雅降级为零特征）
进行最小训练，验证 finetune_davf.py → predict.py 全链路可用。
"""

import csv
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "scripts"


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


def _make_minimal_davf_config(tmp_path: Path, num_classes: int = 2) -> Path:
    """创建最小 DAVF 训练配置（CNN + DAVF 启用）。"""
    cfg = {
        "model": {
            "encoder_type": "cnn",
            "vocab_size": 21,
            "embed_dim": 16,
            "num_filters": 8,
            "kernel_sizes": [3],
            "max_seq_len": 64,
            "num_ptm_types": 2,
            "num_classes": num_classes,
            "pool_type": "mean",
            "dropout": 0.0,
            "use_davf": True,
            "davf_config": {  # DAVF 配置，build_model() 会覆盖 checkpoint_path/freeze
                "feature_dim": 16,
                "hidden_dim": 32,
                "freeze": True,
                "state_space": "gene",
                "num_genes": 100,
            },
        },
        "data": {
            "max_sequence_length": 64,
            "valid_amino_acids": "ACDEFGHIKLMNPQRSTVWY",
            "ptm_types": ["phosphorylation", "acetylation"],
            "cell_states": ["apoptosis", "proliferation"],
        },
        "training": {
            "max_epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-3,
        },
    }
    path = tmp_path / "davf_config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _make_davf_train_csv(tmp_path: Path, num_rows: int = 8) -> Path:
    """创建最小训练 CSV（sequence + ptm_sites + cell_state）。
    使用 csv.writer 确保 JSON 字段内的双引号被正确转义。
    """
    path = tmp_path / "train.csv"
    sequences = [
        "ACDEFGHIKLMNPQRSTVWY",
        "AYDEFGHIKLMNPQRSTVWC",
        "ACDEFGHIKLMNPQRSTVYA",
        "AKDEFGHIKLMNPQRSTVWC",
        "ARCDEFGHIKLMNPQRSTVW",
        "ACDEFGHIKLMNPQASTVWY",
        "AYCDEFGHIKLMNPQRSTVW",
        "ACDEFGHIKLSNPQRSTVWY",
    ]
    cell_states = ["apoptosis", "proliferation"]
    rows = []
    for i in range(min(num_rows, len(sequences))):
        if i % 3 == 0:
            ptm = '[{"position":3,"type":"phosphorylation","amino_acid":"S"}]'
        else:
            ptm = "[]"
        rows.append({
            "id": i,
            "sequence": sequences[i],
            "ptm_sites": ptm,
            "cell_state": cell_states[i % len(cell_states)],
        })
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "sequence", "ptm_sites", "cell_state"])
        writer.writeheader()
        writer.writerows(rows)
    return path


class TestDAVFTrainPredictE2E:
    """DAVF 训练→推理 E2E。"""

    @pytest.fixture(scope="class")
    def davf_artifact(self, tmp_path_factory):
        """训练一个 DAVF 模型并返回 final_model.pt / config 路径。"""
        tmp = tmp_path_factory.mktemp("davf")
        cfg_path = _make_minimal_davf_config(tmp)
        data_path = _make_davf_train_csv(tmp, num_rows=8)
        output_dir = tmp / "output"
        output_dir.mkdir()

        # 使用不存在的 checkpoint 路径（DAVFInferenceModule 优雅降级）
        fake_checkpoint = tmp / "nonexistent_checkpoint.pt"

        _run([
            sys.executable, str(SCRIPTS / "finetune_davf.py"),
            "--config", str(cfg_path),
            "--data", str(data_path),
            "--checkpoint", str(fake_checkpoint),
            "--output", str(output_dir),
            "--stage1-epochs", "1",
            "--stage2-epochs", "1",
            "--batch-size", "4",
        ])

        final_ckpt = output_dir / "final_model.pt"
        assert final_ckpt.exists(), f"final_model.pt 未生成于 {output_dir}"

        results_json = output_dir / "finetune_results.json"
        assert results_json.exists(), f"finetune_results.json 未生成于 {output_dir}"

        return {"ckpt": final_ckpt, "cfg": cfg_path, "output": output_dir}

    def test_davf_training_exports_artifacts(self, davf_artifact):
        """训练完成后同时导出 final_model.pt 和 finetune_results.json。"""
        assert davf_artifact["ckpt"].exists()
        results = davf_artifact["output"] / "finetune_results.json"
        assert results.exists()

    def test_davf_model_predict_single(self, davf_artifact, tmp_path):
        """predict.py 可加载 DAVF 产出的 final_model.pt 并推理。"""
        out = tmp_path / "pred.csv"
        _run([
            sys.executable, str(SCRIPTS / "predict.py"),
            "--model", str(davf_artifact["ckpt"]),
            "--config", str(davf_artifact["cfg"]),
            "--sequence", "ACDEFGHIKLMNPQRSTVWY",
            "--ptm-sites", '[{"position":3,"type":"phosphorylation"}]',
            "--output", str(out),
            "--device", "cpu",
        ])
        import pandas as pd
        df = pd.read_csv(out)
        assert len(df) == 1
        assert "ptm_count" in df.columns
        assert int(df["ptm_count"].iloc[0]) == 1
        # 验证输出包含概率列
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        assert len(prob_cols) >= 1, "输出应包含 prob_<state> 列"

    def test_davf_model_predict_batch(self, davf_artifact, tmp_path):
        """predict.py 批量模式对 DAVF 模型同样可用。"""
        batch_csv = tmp_path / "batch.csv"
        batch_csv.write_text(
            "id,sequence,ptm_sites\n"
            '1,ACDEFGHIKLMNPQRSTVWY,"[{""position"":3,""type"":""phosphorylation""}]"\n'
            "2,ACDEFGHIKLMNPQRSTVWY,\n",
            encoding="utf-8",
        )
        out = tmp_path / "batch_pred.csv"
        _run([
            sys.executable, str(SCRIPTS / "predict.py"),
            "--model", str(davf_artifact["ckpt"]),
            "--config", str(davf_artifact["cfg"]),
            "--input", str(batch_csv),
            "--output", str(out),
            "--device", "cpu",
        ])
        import pandas as pd
        df = pd.read_csv(out)
        assert len(df) == 2
        assert int(df["ptm_count"].iloc[0]) == 1
        assert int(df["ptm_count"].iloc[1]) == 0
