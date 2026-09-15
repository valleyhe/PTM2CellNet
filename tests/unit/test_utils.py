"""
工具模块单元测试
"""

import os
import tempfile
import numpy as np

from src.utils.config import Config
from src.utils.io import (
    save_pickle,
    load_pickle,
    save_json,
    load_json,
    save_dataframe,
    load_dataframe,
    save_numpy,
    load_numpy,
)
from src.utils.helpers import (
    validate_sequence,
    validate_ptm_site,
    clean_sequence,
    get_amino_acid_counts,
    calculate_sequence_length_stats,
    validate_cell_state_label,
)


class TestConfig:
    """Config类测试"""

    def test_from_yaml(self):
        """测试从YAML加载配置"""
        config = Config.from_yaml("configs/default.yaml")
        assert config is not None
        assert config.get("project.name") == "PTM2CellNet"

    def test_default_data_loader_workers_explicit(self):
        """N09: default.yaml 显式声明 data.num_workers，train.py 读取该键。

        dataset_base.DatasetConfig 与 PTMPlainDataModule 默认 0（主进程加载，
        安全默认不变）；default.yaml 显式设为 4，保证开箱训练不静默单进程。
        """
        config = Config.from_yaml("configs/default.yaml")
        assert config.get("data.num_workers") == 4, "data.num_workers must be explicit in configs/default.yaml (N09)"
        assert config.get("data.persistent_workers") is False

    def test_get_and_set(self):
        """测试get和set方法"""
        config = Config({"a": {"b": 1}})
        assert config.get("a.b") == 1
        assert config.get("nonexistent", "default") == "default"

        config.set("a.c", 2)
        assert config.get("a.c") == 2

    def test_dict_access(self):
        """测试字典式访问"""
        config = Config({"x": 10})
        assert config["x"] == 10
        config["y"] = 20
        assert config["y"] == 20
        assert "x" in config
        assert "nonexistent" not in config


class TestHelpers:
    """辅助函数测试"""

    def test_validate_sequence_valid(self):
        """测试有效序列验证"""
        is_valid, msg = validate_sequence("ACDEFGHIKLMNPQRSTVWY")
        assert is_valid is True
        assert msg == ""

    def test_validate_sequence_invalid_char(self):
        """测试无效字符序列"""
        is_valid, msg = validate_sequence("ACXDE")
        assert is_valid is False
        assert "X" in msg

    def test_validate_sequence_empty(self):
        """测试空序列"""
        is_valid, msg = validate_sequence("")
        assert is_valid is False

    def test_validate_ptm_site_valid(self):
        """测试有效PTM位点"""
        is_valid, msg = validate_ptm_site({"position": 5, "type": "phosphorylation"})
        assert is_valid is True
        assert msg == ""

    def test_validate_ptm_site_invalid_position(self):
        """测试无效位置"""
        is_valid, msg = validate_ptm_site({"position": 0, "type": "phosphorylation"})
        assert is_valid is False

    def test_clean_sequence(self):
        """测试序列清理"""
        assert clean_sequence("  acde  \n") == "ACDE"

    def test_get_amino_acid_counts(self):
        """测试氨基酸计数"""
        counts = get_amino_acid_counts("AACD")
        assert counts["A"] == 2
        assert counts["C"] == 1
        assert counts["D"] == 1

    def test_calculate_sequence_length_stats(self):
        """测试序列长度统计"""
        stats = calculate_sequence_length_stats(["A", "AA", "AAA"])
        assert stats["mean"] == 2.0
        assert stats["min"] == 1
        assert stats["max"] == 3

    def test_validate_cell_state_label(self):
        """测试细胞状态标签验证"""
        is_valid, msg = validate_cell_state_label("proliferation", {"proliferation", "differentiation"})
        assert is_valid is True


class TestIO:
    """文件IO测试"""

    def test_save_and_load_pickle(self):
        """测试pickle保存和加载"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            temp_path = f.name

        try:
            data = {"key": "value", "list": [1, 2, 3]}
            save_pickle(data, temp_path)
            loaded = load_pickle(temp_path)
            assert loaded == data
        finally:
            os.unlink(temp_path)

    def test_save_and_load_json(self):
        """测试JSON保存和加载"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            data = {"key": "value", "list": [1, 2, 3]}
            save_json(data, temp_path)
            loaded = load_json(temp_path)
            assert loaded == data
        finally:
            os.unlink(temp_path)

    def test_save_and_load_dataframe(self):
        """测试DataFrame保存和加载"""
        import pandas as pd

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
            temp_path = f.name

        try:
            df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
            save_dataframe(df, temp_path)
            loaded = load_dataframe(temp_path)
            pd.testing.assert_frame_equal(df, loaded)
        finally:
            os.unlink(temp_path)

    def test_save_and_load_numpy(self):
        """测试NumPy数组保存和加载"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as f:
            temp_path = f.name

        try:
            arr = np.array([[1, 2], [3, 4]])
            save_numpy(arr, temp_path)
            loaded = load_numpy(temp_path)
            np.testing.assert_array_equal(arr, loaded)
        finally:
            os.unlink(temp_path)
