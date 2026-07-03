"""
训练 artifact 导出单元测试

验证 src/training/artifacts.py:
  - write_artifact_manifest 写入 artifact_manifest.json, 格式/字段符合契约
  - export_inference_artifact 导出 best_model.pt + best_model.config.yaml
  - evaluate_release_gate / dataset_hash 辅助函数行为

使用 tmp_path + 小型 MockModel / dict config, 不依赖真实数据或 GPU。
"""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.training import artifacts as art
from src.utils.config import Config


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

class MockModel(nn.Module):
    """最小可保存 nn.Module, 供 state_dict 导出测试使用"""

    def __init__(self, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        return self.fc(x)


def _make_config(num_classes=3, max_seq_len=1000):
    """构造一份最小可用 config dict"""
    return Config({
        "model": {"num_classes": num_classes, "hidden_dim": 8},
        "data": {
            "max_sequence_length": max_seq_len,
            "ptm_types": ["phosphorylation"],
            "cell_states": [f"state_{i}" for i in range(num_classes)],
        },
    })


# ---------------------------------------------------------------------------
# write_artifact_manifest
# ---------------------------------------------------------------------------

class TestWriteArtifactManifest:
    """artifact_manifest.json 写入格式测试"""

    def test_manifest_file_written(self, tmp_path):
        """write_artifact_manifest 写出 artifact_manifest.json"""
        cfg = _make_config()
        manifest = art.write_artifact_manifest(
            tmp_path,
            model_class="PTM2CellNet",
            cell_states=["state_0", "state_1", "state_2"],
            config=cfg,
            training_entrypoint="scripts/train.py",
        )
        manifest_path = tmp_path / "artifact_manifest.json"
        assert manifest_path.exists()

        on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
        # 返回值与磁盘内容一致
        assert on_disk == manifest

    def test_manifest_required_fields(self, tmp_path):
        """manifest 包含契约要求的核心字段"""
        cfg = _make_config()
        manifest = art.write_artifact_manifest(
            tmp_path,
            model_class="PTM2CellNet",
            cell_states=["s0", "s1"],
            config=cfg,
            training_entrypoint="scripts/train.py",
        )
        for key in (
            "checkpoint_path",
            "config_path",
            "training_entrypoint",
            "model_class",
            "num_classes",
            "cell_states",
            "git_commit",
            "created_at",
            "model_kind",
            "model_card",
            "data_provenance",
            "deployable",
        ):
            assert key in manifest, f"manifest 缺少字段: {key}"

    def test_manifest_paths_resolve_under_output_dir(self, tmp_path):
        """checkpoint/config 路径指向 output_dir 内"""
        cfg = _make_config(num_classes=2)
        manifest = art.write_artifact_manifest(
            tmp_path,
            model_class="PTM2CellNet",
            cell_states=["a", "b"],
            config=cfg,
            training_entrypoint="scripts/train.py",
        )
        assert Path(manifest["checkpoint_path"]).parent == tmp_path
        assert Path(manifest["config_path"]).parent == tmp_path

    def test_manifest_is_valid_json(self, tmp_path):
        """磁盘文件是合法 JSON, 可被 json.load 解析"""
        cfg = _make_config()
        art.write_artifact_manifest(
            tmp_path,
            model_class="PTM2CellNet",
            cell_states=["s0", "s1"],
            config=cfg,
            training_entrypoint="scripts/train.py",
        )
        with open(tmp_path / "artifact_manifest.json", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_demo_data_marks_not_deployable(self, tmp_path):
        """is_demo_data=True 时 model_kind=demo 且 deployable=False"""
        cfg = _make_config()
        manifest = art.write_artifact_manifest(
            tmp_path,
            model_class="PTM2CellNet",
            cell_states=["s0", "s1"],
            config=cfg,
            training_entrypoint="scripts/train.py",
            is_demo_data=True,
        )
        assert manifest["model_kind"] == "demo"
        assert manifest["deployable"] is False
        assert manifest["model_card"]["not_for_biological_use"] is True

    def test_real_data_with_passing_gate_is_deployable(self, tmp_path):
        """真实数据 + metrics 满足阈值 → deployable=True"""
        cfg = _make_config()
        manifest = art.write_artifact_manifest(
            tmp_path,
            model_class="PTM2CellNet",
            cell_states=["s0", "s1"],
            config=cfg,
            training_entrypoint="scripts/train.py",
            is_demo_data=False,
            metrics={"accuracy": 0.95, "f1": 0.90},
            release_thresholds={"accuracy": 0.8, "f1": 0.8},
        )
        assert manifest["deployable"] is True
        assert manifest["release_gate"]["passed"] is True

    def test_real_data_with_failing_gate_not_deployable(self, tmp_path):
        """真实数据 + metrics 不满足阈值 → deployable=False"""
        cfg = _make_config()
        manifest = art.write_artifact_manifest(
            tmp_path,
            model_class="PTM2CellNet",
            cell_states=["s0", "s1"],
            config=cfg,
            training_entrypoint="scripts/train.py",
            is_demo_data=False,
            metrics={"accuracy": 0.5},
            release_thresholds={"accuracy": 0.8},
        )
        assert manifest["deployable"] is False
        assert manifest["release_gate"]["passed"] is False

    def test_dataset_hash_and_stats_recorded(self, tmp_path):
        """传入 train_df 时记录 dataset_hash + dataset_stats"""
        cfg = _make_config()
        df = pd.DataFrame({"sequence": ["ACDE", "FGHI"], "cell_state": ["s0", "s1"]})
        manifest = art.write_artifact_manifest(
            tmp_path,
            model_class="PTM2CellNet",
            cell_states=["s0", "s1"],
            config=cfg,
            training_entrypoint="scripts/train.py",
            train_df=df,
        )
        assert "dataset_hash" in manifest
        assert manifest["dataset_stats"]["row_count"] == 2
        assert manifest["dataset_stats"]["column_count"] == 2


# ---------------------------------------------------------------------------
# export_inference_artifact
# ---------------------------------------------------------------------------

class TestExportInferenceArtifact:
    """best_model.pt + best_model.config.yaml 导出测试"""

    def test_exports_checkpoint_and_config(self, tmp_path):
        """导出 best_model.pt + best_model.config.yaml 两个文件"""
        model = MockModel(num_classes=3)
        cfg = _make_config(num_classes=3)
        paths = art.export_inference_artifact(
            model, cfg, tmp_path, cell_states=["s0", "s1", "s2"],
        )
        assert Path(paths["checkpoint_path"]).exists()
        assert Path(paths["config_path"]).exists()
        assert (tmp_path / "best_model.pt").exists()
        assert (tmp_path / "best_model.config.yaml").exists()

    def test_checkpoint_loadable_state_dict(self, tmp_path):
        """导出的 best_model.pt 可被 torch.load 读为 state_dict"""
        model = MockModel(num_classes=2)
        cfg = _make_config(num_classes=2)
        art.export_inference_artifact(
            model, cfg, tmp_path, cell_states=["s0", "s1"],
        )
        state = torch.load(tmp_path / "best_model.pt", weights_only=True)
        assert isinstance(state, dict)
        assert "fc.weight" in state

    def test_config_yaml_contains_cell_states(self, tmp_path):
        """导出的 config.yaml 自描述 cell_states"""
        import yaml
        model = MockModel(num_classes=2)
        cfg = _make_config(num_classes=2)
        art.export_inference_artifact(
            model, cfg, tmp_path, cell_states=["alpha", "beta"],
        )
        with open(tmp_path / "best_model.config.yaml", encoding="utf-8") as f:
            saved = yaml.safe_load(f)
        assert saved["data"]["cell_states"] == ["alpha", "beta"]
        assert saved["model"]["num_classes"] == 2

    def test_num_classes_mismatch_raises(self, tmp_path):
        """config.num_classes 与 cell_states 数量不一致 → ValueError"""
        model = MockModel(num_classes=3)
        cfg = _make_config(num_classes=3)  # 故意设为 3
        with pytest.raises(ValueError):
            art.export_inference_artifact(
                model, cfg, tmp_path, cell_states=["s0", "s1"],  # 只有 2 个
            )

    def test_missing_cell_states_raises(self, tmp_path):
        """无 cell_states 且 config 中也无 → ValueError"""
        model = MockModel(num_classes=2)
        cfg = Config({"model": {"num_classes": 2}, "data": {}})
        with pytest.raises(ValueError):
            art.export_inference_artifact(model, cfg, tmp_path)

    def test_lightning_module_unwrap(self, tmp_path):
        """传入带 .model 属性的 LightningModule 风格对象可正确解包"""
        class FakeLightningModule:
            def __init__(self, base):
                self.model = base

        base = MockModel(num_classes=2)
        cfg = _make_config(num_classes=2)
        paths = art.export_inference_artifact(
            FakeLightningModule(base), cfg, tmp_path, cell_states=["s0", "s1"],
        )
        assert Path(paths["checkpoint_path"]).exists()

    def test_unrecognized_model_raises(self, tmp_path):
        """无法识别的模型对象 → TypeError"""
        cfg = _make_config(num_classes=2)
        with pytest.raises(TypeError):
            art.export_inference_artifact(
                object(), cfg, tmp_path, cell_states=["s0", "s1"],
            )


# ---------------------------------------------------------------------------
# Helpers: evaluate_release_gate / dataset_hash
# ---------------------------------------------------------------------------

class TestEvaluateReleaseGate:
    """release gate 评估逻辑测试"""

    def test_all_metrics_pass(self):
        result = art.evaluate_release_gate(
            {"accuracy": 0.9, "f1": 0.85},
            {"accuracy": 0.8, "f1": 0.8},
        )
        assert result["passed"] is True
        assert result["missing"] == []

    def test_one_metric_below_threshold(self):
        result = art.evaluate_release_gate(
            {"accuracy": 0.9, "f1": 0.5},
            {"accuracy": 0.8, "f1": 0.8},
        )
        assert result["passed"] is False
        assert result["checked"]["f1"]["passed"] is False
        assert result["checked"]["accuracy"]["passed"] is True

    def test_missing_metric_counts_as_failure(self):
        result = art.evaluate_release_gate(
            {"accuracy": 0.9},
            {"accuracy": 0.8, "f1": 0.8},
        )
        assert result["passed"] is False
        assert "f1" in result["missing"]


class TestDatasetHash:
    """dataset_hash 稳定性测试"""

    def test_identical_data_same_hash(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        assert art.dataset_hash(df) == art.dataset_hash(df.copy())

    def test_different_data_different_hash(self):
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [1, 3]})
        assert art.dataset_hash(df1) != art.dataset_hash(df2)

    def test_hash_is_hex_string(self):
        df = pd.DataFrame({"a": [1]})
        h = art.dataset_hash(df)
        assert isinstance(h, str)
        assert all(c in "0123456789abcdef" for c in h)
        assert len(h) == 16
