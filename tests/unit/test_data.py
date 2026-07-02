"""
数据模块单元测试
"""

import json
import os
import tempfile
import pandas as pd
import numpy as np
import torch
import pytest

from src.data.loaders import DataLoader
from src.data.preprocess import DataPreprocessor
from src.data.features import FeatureExtractor
from src.data.datasets import PTMDataset, PTMPlainDataModule


class TestDataLoader:
    """DataLoader类测试"""

    def test_load_sample_data(self):
        """测试加载示例数据"""
        loader = DataLoader()
        df = loader.load_sample_data(num_samples=20)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20
        assert "sequence" in df.columns
        assert "ptm_sites" in df.columns
        assert "cell_state" in df.columns

    def test_load_fasta_lowercase(self):
        """测试FASTA小写序列加载"""
        fasta_content = ">seq1\nacdefg\n>seq2\nLMNPQ\n"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as f:
            f.write(fasta_content.encode("utf-8"))
            temp_path = f.name

        try:
            loader = DataLoader()
            sequences = loader.load_from_fasta(temp_path)
            assert sequences["seq1"] == "ACDEFG"
            assert sequences["seq2"] == "LMNPQ"
        finally:
            os.unlink(temp_path)


class TestDataPreprocessor:
    """DataPreprocessor类测试"""

    @pytest.fixture
    def sample_df(self):
        """示例数据fixture"""
        data = [
            {
                "id": "1",
                "sequence": "ACDEFGHIKLMNPQRSTVWY",
                "ptm_sites": json.dumps([{"position": 5, "type": "phosphorylation"}]),
                "cell_state": "proliferation",
            },
            {
                "id": "2",
                "sequence": "ACDEFGHIKL",
                "ptm_sites": json.dumps([]),
                "cell_state": "differentiation",
            },
        ]
        return pd.DataFrame(data)

    def test_clean_sequences(self, sample_df):
        """测试序列清洗"""
        preprocessor = DataPreprocessor()
        cleaned = preprocessor.clean_sequences(sample_df)
        assert len(cleaned) == 2

    def test_normalize_ptm_labels(self, sample_df):
        """测试PTM标签标准化"""
        preprocessor = DataPreprocessor()
        normalized = preprocessor.normalize_ptm_labels(sample_df)
        assert "ptm_sites" in normalized.columns

    def test_split_dataset(self, sample_df):
        """测试数据集划分"""
        preprocessor = DataPreprocessor()
        train, val, test = preprocessor.split_dataset(sample_df, ratios=(0.5, 0.25, 0.25))
        assert len(train) + len(val) + len(test) == len(sample_df)

    def test_preprocess_pipeline(self, sample_df):
        """测试完整预处理流程"""
        preprocessor = DataPreprocessor()
        train, val, test = preprocessor.preprocess_pipeline(sample_df)
        assert len(train) > 0

    def test_preprocess_pipeline_raises_on_empty_train_after_filter(self):
        """P1-3: 过滤后训练集为空时 fail-fast，抛 EmptyDatasetError 并带上下文。"""
        from src.data.preprocess import EmptyDatasetError

        # max_sequence_length=5 但所有序列都更长 -> clean_sequences 全部删除
        config = {"data": {"max_sequence_length": 5, "valid_amino_acids": "ACDEFGHIKLMNPQRSTVWY"}}
        preprocessor = DataPreprocessor(config)
        df = pd.DataFrame([
            {"sequence": "ACDEFGHIKLMNPQRSTVWY", "cell_state": "a"},
            {"sequence": "MKTVTASSFTWMKTVTASSFTW", "cell_state": "b"},
        ])
        with pytest.raises(EmptyDatasetError) as exc_info:
            preprocessor.preprocess_pipeline(df)
        msg = str(exc_info.value)
        # 错误信息应包含可操作上下文
        assert "训练集为空" in msg
        assert "max_sequence_length_config" in msg

    def test_preprocess_pipeline_empty_error_carries_filter_stats(self):
        """P1-3: EmptyDatasetError 携带原始样本数与过滤统计。"""
        from src.data.preprocess import EmptyDatasetError

        config = {"data": {"max_sequence_length": 3, "valid_amino_acids": "ACDEFGHIKLMNPQRSTVWY"}}
        preprocessor = DataPreprocessor(config)
        df = pd.DataFrame([
            {"sequence": "ACDEFG", "cell_state": "a"},
            {"sequence": "GHIJKL", "cell_state": "b"},  # 含非标准 J->L 后仍超长
        ])
        with pytest.raises(EmptyDatasetError) as exc_info:
            preprocessor.preprocess_pipeline(df)
        # 原始样本数应在错误信息中可追溯
        assert "original_sample_count" in str(exc_info.value)

    # ------------------------------------------------------------------
    # P1-2 strategy 4: validate_data_quality 数据质量预检
    # ------------------------------------------------------------------
    def test_validate_data_quality_passes_for_valid_frame(self, sample_df):
        """健康的样本数据应通过预检，返回 ok=True 且无 hard_failures。"""
        preprocessor = DataPreprocessor()
        report = preprocessor.validate_data_quality(sample_df)
        assert report["ok"] is True
        assert report["hard_failures"] == []
        assert report["row_count"] == 2
        assert "proliferation" in report["label_distribution"]
        assert report["sequence_length_stats"]["min"] > 0

    def test_validate_data_quality_detects_empty_frame(self):
        """空 DataFrame 是 hard failure。"""
        preprocessor = DataPreprocessor()
        report = preprocessor.validate_data_quality(pd.DataFrame())
        assert report["ok"] is False
        assert any("为空" in f for f in report["hard_failures"])

    def test_validate_data_quality_detects_single_class(self):
        """只有单一类别无法训练，是 hard failure。"""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame([
            {"sequence": "ACDEFGHIK", "cell_state": "proliferation"},
            {"sequence": "KMNPQRSTV", "cell_state": "proliferation"},
        ])
        report = preprocessor.validate_data_quality(df)
        assert report["ok"] is False
        assert any("类别" in f for f in report["hard_failures"])

    def test_validate_data_quality_flags_severe_imbalance(self):
        """类别分布严重不均衡（>10x）应触发 warning（非 hard failure）。"""
        preprocessor = DataPreprocessor()
        rows = [{"sequence": "ACDE", "cell_state": "proliferation"}] * 100
        rows += [{"sequence": "GHIK", "cell_state": "apoptosis"}]
        df = pd.DataFrame(rows)
        report = preprocessor.validate_data_quality(df)
        assert report["ok"] is True  # 不平衡只是 warning
        assert any("不均衡" in w for w in report["warnings"])

    def test_validate_data_quality_counts_invalid_ptm_sites(self):
        """越界 / 非法 PTM 位点应计入 invalid_ptm_site_count。"""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame([
            {"sequence": "ACDE", "cell_state": "a",
             "ptm_sites": json.dumps([{"position": 99, "type": "phosphorylation"}])},
            {"sequence": "ACDE", "cell_state": "b",
             "ptm_sites": json.dumps([{"position": 2, "type": "phosphorylation"}])},
        ])
        report = preprocessor.validate_data_quality(df)
        assert report["invalid_ptm_site_count"] >= 1
        assert any("PTM" in w for w in report["warnings"])

    def test_validate_data_quality_strict_raises(self, ):
        """strict=True 时 hard failure 抛 DataQualityError。"""
        from src.data.preprocess import DataQualityError

        preprocessor = DataPreprocessor()
        df = pd.DataFrame([{"sequence": "ACDE", "cell_state": "only"}])
        with pytest.raises(DataQualityError) as exc_info:
            preprocessor.validate_data_quality(df, strict=True)
        assert "类别" in str(exc_info.value)
        assert exc_info.value.report["ok"] is False


class TestFeatureExtractor:
    """FeatureExtractor类测试"""

    def test_extract_onehot_sequence(self):
        """测试one-hot序列编码"""
        extractor = FeatureExtractor()
        onehot = extractor.extract_onehot_sequence("ACDE")
        assert isinstance(onehot, np.ndarray)
        assert onehot.shape[0] == extractor.max_sequence_length
        assert onehot.shape[1] == 21

    def test_extract_physicochemical_features(self):
        """测试理化特征提取"""
        extractor = FeatureExtractor()
        features = extractor.extract_physicochemical_features("ACDEFGHIKL")
        assert isinstance(features, np.ndarray)
        assert len(features) > 0

    def test_extract_kmer_features(self):
        """测试k-mer特征提取"""
        extractor = FeatureExtractor()
        features = extractor.extract_kmer_features("ACDE", k=1)
        assert isinstance(features, np.ndarray)
        assert features.sum() == pytest.approx(1.0)
        assert features.max() > 0.0

    def test_extract_ptm_features(self):
        """测试PTM特征提取"""
        extractor = FeatureExtractor()
        ptm_sites = json.dumps([{"position": 5, "type": "phosphorylation"}])
        features = extractor.extract_ptm_features(ptm_sites, 100)
        assert "position_features" in features
        assert "count_features" in features

    def test_extract_sequence_features(self):
        """测试批量序列特征提取"""
        extractor = FeatureExtractor()
        sequences = ["ACDEFGHIKL", "LMNPQRSTVWY"]
        features = extractor.extract_sequence_features(sequences)
        assert isinstance(features, np.ndarray)
        assert len(features) == 2

    @pytest.mark.parametrize("encoding", ["onehot", "kmer", "both"])
    def test_extract_sequence_features_encoding_modes(self, encoding):
        ex = FeatureExtractor({"features": {"sequence_encoding": encoding, "include_physicochemical": False}})
        arr = ex.extract_sequence_features(["ACDEFGHIKL"])
        assert arr.size > 0

    def test_extract_ptm_features_invalid_json_and_out_of_range(self):
        ex = FeatureExtractor()

        invalid = ex.extract_ptm_features("not-json", sequence_length=10)
        assert invalid["count_features"].sum() == 0
        assert invalid["position_features"].sum() == 0

        out_of_range = ex.extract_ptm_features(
            json.dumps([{"position": 9999, "type": "phosphorylation"}]),
            sequence_length=10,
        )
        assert out_of_range["count_features"].sum() == 0
        assert out_of_range["position_features"].sum() == 0

    def test_extract_features_for_sample_without_ptm_features(self):
        ex = FeatureExtractor({"features": {"include_ptm_features": False}})
        seq = "ACDEFGHIKL"
        with_ptm_json = ex.extract_features_for_sample(seq, json.dumps([{"position": 2, "type": "phosphorylation"}]))
        without_ptm_json = ex.extract_features_for_sample(seq, None)
        assert with_ptm_json.shape == without_ptm_json.shape

    def test_combine_features_flattens_1d_and_2d_arrays(self):
        ex = FeatureExtractor()
        one = np.array([1.0, 2.0], dtype=np.float32)
        two = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        merged = ex.combine_features({"a": one, "b": two})
        assert np.array_equal(merged, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32))


class TestPTMDataset:
    """PTMDataset类测试"""

    @pytest.fixture
    def sample_df(self):
        """示例数据fixture"""
        data = [
            {
                "id": "1",
                "sequence": "ACDEFGHIKLMNPQRSTVWY",
                "ptm_sites": json.dumps([{"position": 5, "type": "phosphorylation"}]),
                "cell_state": "proliferation",
            },
            {
                "id": "2",
                "sequence": "LMNPQRSTVWY",
                "ptm_sites": json.dumps([{"position": 3, "type": "acetylation"}]),
                "cell_state": "differentiation",
            },
            {
                "id": "3",
                "sequence": "ACDEFGHIKL",
                "ptm_sites": json.dumps([]),
                "cell_state": "apoptosis",
            },
        ]
        return pd.DataFrame(data)

    def test_dataset_len(self, sample_df):
        """测试数据集长度"""
        dataset = PTMDataset(sample_df)
        assert len(dataset) == 3

    def test_dataset_getitem(self, sample_df):
        """测试获取样本"""
        dataset = PTMDataset(sample_df)
        sample = dataset[0]
        assert "sequence" in sample
        assert "ptm_mask" in sample
        assert "label" in sample
        assert isinstance(sample["sequence"], torch.Tensor)
        assert isinstance(sample["ptm_mask"], torch.Tensor)
        assert isinstance(sample["label"], torch.Tensor)

    def test_labels(self, sample_df):
        """测试标签处理"""
        dataset = PTMDataset(sample_df)
        assert len(dataset.labels) == 3
        assert "proliferation" in dataset.labels

    def test_unknown_label_raises(self, sample_df):
        """测试未知标签处理"""
        dataset = PTMDataset(sample_df)
        with pytest.raises(ValueError):
            dataset._encode_label("unknown_label")


class TestPTMPlainDataModule:
    """PTMPlainDataModule类测试"""

    @pytest.fixture
    def sample_dfs(self):
        """示例数据fixture"""
        train_data = [
            {
                "id": f"train_{i}",
                "sequence": "ACDEFGHIKLMNPQRSTVWY",
                "ptm_sites": json.dumps([]),
                "cell_state": "proliferation",
            }
            for i in range(20)
        ]
        val_data = [
            {"id": f"val_{i}", "sequence": "LMNPQRSTVWY", "ptm_sites": json.dumps([]), "cell_state": "differentiation"}
            for i in range(10)
        ]
        test_data = [
            {"id": f"test_{i}", "sequence": "ACDEFGHIKL", "ptm_sites": json.dumps([]), "cell_state": "apoptosis"}
            for i in range(10)
        ]
        return pd.DataFrame(train_data), pd.DataFrame(val_data), pd.DataFrame(test_data)

    def test_datamodule_setup(self, sample_dfs):
        """测试数据模块设置"""
        train_df, val_df, test_df = sample_dfs
        datamodule = PTMPlainDataModule(train_df, val_df, test_df, batch_size=8)

        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None
        assert datamodule.test_dataset is not None

    def test_train_dataloader(self, sample_dfs):
        """测试训练数据加载器"""
        train_df, val_df, test_df = sample_dfs
        datamodule = PTMPlainDataModule(train_df, val_df, test_df, batch_size=8)

        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        assert "sequence" in batch
        assert batch["sequence"].shape[0] == 8

    def test_val_dataloader(self, sample_dfs):
        """测试验证数据加载器"""
        train_df, val_df, test_df = sample_dfs
        datamodule = PTMPlainDataModule(train_df, val_df, test_df, batch_size=5)

        val_loader = datamodule.val_dataloader()
        assert val_loader is not None

    def test_test_dataloader(self, sample_dfs):
        """测试测试数据加载器"""
        train_df, val_df, test_df = sample_dfs
        datamodule = PTMPlainDataModule(train_df, val_df, test_df, batch_size=5)

        test_loader = datamodule.test_dataloader()
        assert test_loader is not None

    def test_get_labels(self, sample_dfs):
        """测试获取标签"""
        train_df, val_df, test_df = sample_dfs
        datamodule = PTMPlainDataModule(train_df, val_df, test_df)

        labels = datamodule.get_labels()
        assert len(labels) > 0
        assert datamodule.get_num_classes() == len(labels)
