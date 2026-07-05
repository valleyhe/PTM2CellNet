"""
数据验证和数据集基类单元测试
"""

import tempfile
import pytest
import pandas as pd
import torch

from src.data.validation import DataValidator, DatasetCache, collate_sequences
from src.data.dataset_base import PTMDatasetBase, compute_class_weights


class TestDataValidator:
    """DataValidator类测试"""

    def test_validate_sequence_non_string(self):
        validator = DataValidator()
        is_valid, message = validator.validate_sequence(12345)
        assert is_valid is False
        assert "字符串" in message or "type" in message or "str" in message

    def test_validate_sequence_empty(self):
        validator = DataValidator()
        is_valid, message = validator.validate_sequence("")
        assert is_valid is False
        assert "为空" in message

    def test_validate_sequence_too_long(self):
        validator = DataValidator()
        is_valid, message = validator.validate_sequence("A" * 10001, max_length=10000)
        assert is_valid is False
        assert "10001" in message or "长度" in message or "超过" in message

    def test_validate_ptm_site_not_dict(self):
        validator = DataValidator()
        is_valid, message = validator.validate_ptm_site("not_a_dict")
        assert is_valid is False

    def test_validate_ptm_site_missing_position(self):
        validator = DataValidator()
        is_valid, message = validator.validate_ptm_site({"type": "phos"})
        assert is_valid is False
        assert "position" in message

    def test_validate_ptm_site_invalid_position_type(self):
        validator = DataValidator()
        is_valid, message = validator.validate_ptm_site({"position": "five", "type": "phos"})
        assert is_valid is False

    def test_validate_ptm_site_out_of_range(self):
        validator = DataValidator()
        is_valid, message = validator.validate_ptm_site({"position": 50, "type": "phos"}, seq_length=10)
        assert is_valid is False

    def test_validate_dataframe_empty(self):
        validator = DataValidator()
        is_valid, errors = validator.validate_dataframe(pd.DataFrame())
        assert is_valid is False
        assert "为空" in errors[0]

    def test_validate_dataframe_missing_columns(self):
        validator = DataValidator()
        df = pd.DataFrame({"sequence": ["ACDEFG"]})
        is_valid, errors = validator.validate_dataframe(df, required_columns=["sequence", "cell_state"])
        assert is_valid is False
        assert "缺少必需列" in errors[0]

    def test_validate_dataframe_invalid_sequence(self):
        validator = DataValidator()
        df = pd.DataFrame({"sequence": ["ACDEFG1"]})
        is_valid, errors = validator.validate_dataframe(df)
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_dataframe_ptm_json_decode_error(self):
        validator = DataValidator()
        df = pd.DataFrame({
            "sequence": ["ACDEFG"],
            "ptm_sites": ["not-valid-json"],
        })
        is_valid, errors = validator.validate_dataframe(df)
        assert is_valid is False
        assert "JSON解析失败" in errors[0]


class TestDatasetCache:
    """DatasetCache类测试"""

    def test_cache_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DatasetCache(cache_dir=tmpdir)
            df = pd.DataFrame({"a": [1, 2]})
            config = {"key": "value"}
            data = [ {"x": 1}, {"x": 2} ]
            cache.save(df, config, data)
            loaded = cache.load(df, config)
            assert loaded == data

    def test_cache_load_corrupted_returns_none(self, caplog):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DatasetCache(cache_dir=tmpdir)
            # Create a dummy cache file that is not valid pickle
            df = pd.DataFrame({"a": [1, 2]})
            config = {"key": "value"}
            cache_key = cache._get_cache_key(df, config)
            cache_path = cache._get_cache_path(cache_key)
            with open(cache_path, "wb") as f:
                f.write(b"not-a-pickle")
            result = cache.load(df, config)
            assert result is None

    def test_cache_clear_removes_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DatasetCache(cache_dir=tmpdir)
            df = pd.DataFrame({"a": [1]})
            config = {"key": "value"}
            cache.save(df, config, [1])
            cache.clear()
            assert len(list(cache.cache_dir.glob("*.pkl"))) == 0

    def test_cache_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DatasetCache(cache_dir=tmpdir)
            df = pd.DataFrame({"a": [1]})
            config = {"key": "value"}
            cache.save(df, config, [1])
            info = cache.get_cache_info()
            assert info["num_files"] == 1
            assert info["total_size_mb"] > 0


class TestCollateAndWeights:
    """collate_sequences和compute_class_weights测试"""

    def test_collate_sequences_1d_pads_correctly(self):
        batch = [
            {"sequence": torch.tensor([1, 2, 3])},
            {"sequence": torch.tensor([4, 5, 6, 7, 8])},
        ]
        result = collate_sequences(batch, pad_value=0)
        assert result["sequence"].shape == (2, 5)
        assert result["sequence"][0].tolist() == [1, 2, 3, 0, 0]
        assert result["sequence"][1].tolist() == [4, 5, 6, 7, 8]

    def test_compute_class_weights_inverse(self):
        weights = compute_class_weights({"A": 10, "B": 10}, mode="inverse")
        assert weights.shape == (2,)
        # Inverse weights for equal counts should be equal
        assert torch.allclose(weights[0], weights[1], atol=1e-5)
        assert torch.allclose(weights.sum(), torch.tensor(2.0), atol=1e-4)

    def test_compute_class_weights_effective(self):
        weights = compute_class_weights({"A": 10, "B": 10}, mode="effective")
        assert weights.shape == (2,)

    def test_compute_class_weights_unknown_mode(self):
        with pytest.raises(ValueError, match="未知的权重计算模式"):
            compute_class_weights({"A": 10}, mode="bad_mode")


class TestPTMDatasetBase:
    """PTMDatasetBase类测试"""

    def test_encode_label_unknown_raises(self):
        df = pd.DataFrame({"cell_state": ["A", "B"]})
        base = PTMDatasetBase(df)
        with pytest.raises(ValueError):
            base._encode_label("UNKNOWN")

    def test_parse_ptm_sites_invalid_json(self, caplog):
        df = pd.DataFrame({"cell_state": ["A"]})
        base = PTMDatasetBase(df)
        result = base._parse_ptm_sites("not json")
        assert result == []

    def test_get_statistics_no_sequence_column(self):
        df = pd.DataFrame({"cell_state": ["A", "B"]})
        base = PTMDatasetBase(df)
        stats = base.get_statistics()
        assert stats["num_samples"] == 2
        assert "sequence_length" not in stats
