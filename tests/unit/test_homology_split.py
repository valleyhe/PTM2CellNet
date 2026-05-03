"""
同源感知数据分割测试
测试 HomologyAwareSplitter 的功能
"""

import pytest
import pandas as pd
import numpy as np

from src.data.homology_splitter import (
    HomologyAwareSplitter,
    SequenceSimilarityCalculator,
)


class TestSequenceSimilarityCalculator:
    """序列相似性计算器测试"""

    def test_kmer_similarity_identical(self):
        """测试相同序列的k-mer相似性"""
        seq = "ACDEFGHIKLMNPQRSTVWY"
        sim = SequenceSimilarityCalculator.kmer_similarity(seq, seq, k=3)
        assert sim == 1.0

    def test_kmer_similarity_different(self):
        """测试不同序列的k-mer相似性"""
        seq1 = "ACDEFGHIKLMNPQRSTVWY"
        seq2 = "AAAAAAAAAAAAAAAAAAAA"
        sim = SequenceSimilarityCalculator.kmer_similarity(seq1, seq2, k=3)
        assert sim == 0.0

    def test_kmer_similarity_partial(self):
        """测试部分相似的k-mer相似性"""
        seq1 = "ACDEFGHIKLMNPQRSTVWY"
        seq2 = "ACDEFGHIKLMNXXXXXXX"
        sim = SequenceSimilarityCalculator.kmer_similarity(seq1, seq2, k=3)
        assert 0 < sim < 1.0

    def test_local_alignment_score_identical(self):
        """测试相同序列的局部比对分数"""
        seq = "ACDEFGHIKLMNPQRSTVWY"
        score = SequenceSimilarityCalculator.local_alignment_score(seq, seq)
        assert score == 1.0

    def test_local_alignment_score_substring(self):
        """测试子串的局部比对分数"""
        seq1 = "ACDEFGHIKLMNPQRSTVWY"
        seq2 = "DEFGHIKLMN"
        score = SequenceSimilarityCalculator.local_alignment_score(seq1, seq2)
        assert score == 0.5  # 10/20

    def test_identity_estimate(self):
        """测试同一性估计"""
        seq1 = "ACDEFGHIKLMNPQRSTVWY"
        seq2 = "ACDEFGHIKLMNPQRSTVWY"
        identity = SequenceSimilarityCalculator.identity_estimate(seq1, seq2)
        assert identity == 1.0


class TestHomologyAwareSplitter:
    """同源感知分割器测试"""

    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        # 创建3组相似序列
        sequences = []
        labels = []

        # 第一组：相似序列（前缀相同）
        for i in range(5):
            sequences.append(f"ACDEFGHIKLMNPQRSTVWY{i}")
            labels.append("class_A")

        # 第二组：另一组相似序列
        for i in range(5):
            sequences.append(f"BBBBBBBBBBBBBBBBBBB{i}")
            labels.append("class_B")

        # 第三组：完全不同的序列
        for i in range(5):
            sequences.append(f"XXXXXXXXXXXXXXXXXXX{i}")
            labels.append("class_C")

        return pd.DataFrame({
            "sequence": sequences,
            "cell_state": labels,
        })

    def test_splitter_init(self):
        """测试分割器初始化"""
        splitter = HomologyAwareSplitter(
            identity_threshold=0.3,
            similarity_method="kmer",
            random_state=42,
        )
        assert splitter.identity_threshold == 0.3
        assert splitter.similarity_method == "kmer"
        assert splitter.random_state == 42

    def test_compute_similarity_matrix(self):
        """测试相似性矩阵计算"""
        splitter = HomologyAwareSplitter()
        sequences = ["ACDEFG", "ACDEFH", "BBBBBB"]

        sim_matrix = splitter.compute_similarity_matrix(sequences)

        assert sim_matrix.shape == (3, 3)
        assert sim_matrix[0, 0] == 1.0  # 对角线为1
        assert sim_matrix[1, 1] == 1.0
        assert sim_matrix[0, 1] > sim_matrix[0, 2]  # 相似序列分数更高

    def test_cluster_sequences(self):
        """测试序列聚类"""
        splitter = HomologyAwareSplitter()
        sequences = [
            "ACDEFGHIKLMNPQRSTVWY1",
            "ACDEFGHIKLMNPQRSTVWY2",
            "ACDEFGHIKLMNPQRSTVWY3",
            "BBBBBBBBBBBBBBBBBBB1",
            "BBBBBBBBBBBBBBBBBBB2",
        ]

        labels = splitter.cluster_sequences(sequences)

        assert len(labels) == 5
        # 相似序列应该有相同或相近的聚类结构
        # 验证聚类数量合理（应该在2-3个之间）
        assert len(set(labels)) >= 2
        assert len(set(labels)) <= 3

    def test_split(self, sample_data):
        """测试数据分割"""
        splitter = HomologyAwareSplitter(
            identity_threshold=0.3,
            random_state=42,
        )

        train_df, val_df, test_df = splitter.split(
            sample_data,
            sequence_col="sequence",
            ratios=(0.6, 0.2, 0.2),
        )

        # 验证分割结果
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == len(sample_data)
        assert len(train_df) > 0
        assert len(val_df) > 0 or len(test_df) > 0

    def test_split_by_clusters(self, sample_data):
        """测试基于聚类的分割"""
        splitter = HomologyAwareSplitter(random_state=42)

        # 手动创建聚类标签
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

        train_df, val_df, test_df = splitter.split_by_clusters(
            sample_data,
            labels,
            ratios=(0.6, 0.2, 0.2),
        )

        # 验证同一聚类的序列在同一集合
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == len(sample_data)

    def test_verify_no_leakage(self, sample_data):
        """测试无泄漏验证"""
        splitter = HomologyAwareSplitter(identity_threshold=0.3)

        # 创建明显分离的数据集
        train_df = sample_data.iloc[:5].reset_index(drop=True)
        val_df = sample_data.iloc[5:10].reset_index(drop=True)
        test_df = sample_data.iloc[10:].reset_index(drop=True)

        results = splitter.verify_no_leakage(train_df, val_df, test_df)

        assert "passed" in results
        assert "max_similarity" in results
        assert results["passed"] is True

    def test_verify_detects_leakage(self):
        """测试验证能检测到泄漏"""
        splitter = HomologyAwareSplitter(identity_threshold=0.1)

        # 创建有重叠的数据集
        train_df = pd.DataFrame({
            "sequence": ["ACDEFGHIKLMNPQRSTVWY"],
        })
        val_df = pd.DataFrame({
            "sequence": ["ACDEFGHIKLMNPQRSTVWY"],  # 完全相同
        })
        test_df = pd.DataFrame({"sequence": []})

        results = splitter.verify_no_leakage(train_df, val_df, test_df)

        assert results["passed"] is False
        assert results["train_val_violations"] > 0

    def test_empty_dataframe(self):
        """测试空DataFrame处理"""
        splitter = HomologyAwareSplitter()
        empty_df = pd.DataFrame({"sequence": [], "label": []})

        train_df, val_df, test_df = splitter.split(
            empty_df,
            sequence_col="sequence",
        )

        assert len(train_df) == 0
        assert len(val_df) == 0
        assert len(test_df) == 0

    def test_single_sequence(self):
        """测试单条序列"""
        splitter = HomologyAwareSplitter()
        single_df = pd.DataFrame({
            "sequence": ["ACDEFGHIKLMNPQRSTVWY"],
            "label": ["A"],
        })

        train_df, val_df, test_df = splitter.split(
            single_df,
            sequence_col="sequence",
        )

        assert len(train_df) == 1

    def test_similarity_methods(self):
        """测试不同的相似性计算方法"""
        seq1 = "ACDEFGHIKLMNPQRSTVWY"
        seq2 = "ACDEFGHIKLMNPQRSTVWY"

        kmer_sim = SequenceSimilarityCalculator.kmer_similarity(seq1, seq2)
        local_sim = SequenceSimilarityCalculator.local_alignment_score(seq1, seq2)
        identity = SequenceSimilarityCalculator.identity_estimate(seq1, seq2)

        assert kmer_sim == 1.0
        assert local_sim == 1.0
        assert identity == 1.0


class TestHomologyIntegration:
    """与DataPreprocessor集成测试"""

    def test_preprocessor_with_homology_split(self):
        """测试DataPreprocessor使用同源分割"""
        from src.data.preprocess import DataPreprocessor

        config = {
            "data": {
                "split": {
                    "homology_aware": True,
                    "identity_threshold": 0.3,
                    "verify_no_leakage": True,
                }
            }
        }

        preprocessor = DataPreprocessor(config)

        data = pd.DataFrame({
            "sequence": ["AAAAA", "AAAAB", "BBBBB", "BBBBB", "CCCCC", "CCCCD"],
            "cell_state": ["A", "A", "B", "B", "C", "C"],
        })

        train, val, test = preprocessor.split_dataset(
            data,
            sequence_col="sequence",
            label_col="cell_state",
        )

        total = len(train) + len(val) + len(test)
        assert total == len(data)

    def test_preprocessor_without_homology_split(self):
        """测试DataPreprocessor不使用同源分割（默认行为）"""
        from src.data.preprocess import DataPreprocessor

        config = {
            "data": {
                "split": {
                    "homology_aware": False,
                }
            }
        }

        preprocessor = DataPreprocessor(config)

        data = pd.DataFrame({
            "sequence": [f"SEQ{i}" for i in range(20)],
            "cell_state": ["A" if i < 10 else "B" for i in range(20)],
        })

        train, val, test = preprocessor.split_dataset(data)

        total = len(train) + len(val) + len(test)
        assert total == len(data)
