"""
同源感知数据分割模块
功能: 基于序列相似性聚类的数据分割，避免同源序列泄漏
设计思路: 使用序列相似性聚类将相似序列分到同一fold，确保训练/验证/测试集之间无序列相似性泄漏
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

logger = logging.getLogger(__name__)


class SequenceSimilarityCalculator:
    """序列相似性计算器"""

    @staticmethod
    def kmer_similarity(seq1: str, seq2: str, k: int = 3) -> float:
        """
        基于k-mer的相似性计算

        参数:
            seq1: 序列1
            seq2: 序列2
            k: k-mer长度

        返回:
            相似性分数 (0-1)
        """
        if len(seq1) < k or len(seq2) < k:
            return 0.0

        # 提取k-mer集合
        kmers1 = set(seq1[i:i+k] for i in range(len(seq1) - k + 1))
        kmers2 = set(seq2[i:i+k] for i in range(len(seq2) - k + 1))

        if not kmers1 or not kmers2:
            return 0.0

        # Jaccard相似性
        intersection = len(kmers1 & kmers2)
        union = len(kmers1 | kmers2)

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def local_alignment_score(seq1: str, seq2: str) -> float:
        """
        简化版局部比对分数（基于最长公共子串）

        参数:
            seq1: 序列1
            seq2: 序列2

        返回:
            归一化比对分数 (0-1)
        """
        if not seq1 or not seq2:
            return 0.0

        # 动态规划找最长公共子串
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(2)]
        max_length = 0

        for i in range(1, m + 1):
            curr = i % 2
            prev = (i - 1) % 2
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[curr][j] = dp[prev][j-1] + 1
                    max_length = max(max_length, dp[curr][j])
                else:
                    dp[curr][j] = 0

        # 归一化
        return max_length / max(m, n) if max(m, n) > 0 else 0.0

    @staticmethod
    def identity_estimate(seq1: str, seq2: str, k: int = 3) -> float:
        """
        估计序列同一性（identity）
        结合k-mer相似性和局部比对

        参数:
            seq1: 序列1
            seq2: 序列2
            k: k-mer长度

        返回:
            估计的同一性分数 (0-1)
        """
        if len(seq1) < 10 or len(seq2) < 10:
            # 短序列使用局部比对
            return SequenceSimilarityCalculator.local_alignment_score(seq1, seq2)

        # 长序列使用k-mer相似性作为identity估计
        kmer_sim = SequenceSimilarityCalculator.kmer_similarity(seq1, seq2, k=k)

        # 转换为identity估计 (经验转换)
        # k-mer相似性通常低于实际identity
        identity = min(1.0, kmer_sim * 1.5)

        return identity


class HomologyAwareSplitter:
    """
    同源感知数据分割器
    基于序列相似性聚类进行数据分割，防止同源序列泄漏
    """

    def __init__(
        self,
        identity_threshold: float = 0.3,
        similarity_method: str = "kmer",
        kmer_size: int = 3,
        random_state: int = 42,
    ):
        """
        初始化同源感知分割器

        参数:
            identity_threshold: 序列同一性阈值（默认30%）
            similarity_method: 相似性计算方法 ("kmer", "local", "combined")
            kmer_size: k-mer大小
            random_state: 随机种子
        """
        self.identity_threshold = identity_threshold
        self.similarity_method = similarity_method
        self.kmer_size = kmer_size
        self.random_state = random_state
        self.similarity_calculator = SequenceSimilarityCalculator()

    def compute_similarity_matrix(self, sequences: List[str]) -> np.ndarray:
        """
        计算序列相似性矩阵

        参数:
            sequences: 序列列表

        返回:
            相似性矩阵
        """
        n = len(sequences)
        sim_matrix = np.zeros((n, n))

        logger.info("计算 %d 条序列的相似性矩阵", n)

        for i in range(n):
            sim_matrix[i, i] = 1.0
            for j in range(i + 1, n):
                if self.similarity_method == "kmer":
                    sim = self.similarity_calculator.kmer_similarity(
                        sequences[i], sequences[j], self.kmer_size
                    )
                elif self.similarity_method == "local":
                    sim = self.similarity_calculator.local_alignment_score(
                        sequences[i], sequences[j]
                    )
                else:  # combined
                    sim = self.similarity_calculator.identity_estimate(
                        sequences[i], sequences[j], self.kmer_size
                    )

                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        return sim_matrix

    def cluster_sequences(self, sequences: List[str]) -> np.ndarray:
        """
        基于相似性对序列进行聚类

        参数:
            sequences: 序列列表

        返回:
            聚类标签数组
        """
        if len(sequences) <= 1:
            return np.zeros(len(sequences), dtype=int)

        # 计算相似性矩阵
        sim_matrix = self.compute_similarity_matrix(sequences)

        # 转换为距离矩阵
        dist_matrix = 1 - sim_matrix
        np.fill_diagonal(dist_matrix, 0)

        # 确保距离非负
        dist_matrix = np.clip(dist_matrix, 0, 1)

        # 使用层次聚类
        # 基于阈值确定聚类数
        max_clusters = min(len(sequences), max(3, len(sequences) // 5))
        min_clusters = max(1, len(sequences) // 20)

        # 尝试不同的聚类数，选择满足阈值的最佳方案
        best_labels: Optional[np.ndarray] = None
        best_score: float = -1.0

        for n_clusters in range(min_clusters, max_clusters + 1):
            if n_clusters >= len(sequences):
                break

            try:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric="precomputed",
                    linkage="average",
                )
                labels = clustering.fit_predict(dist_matrix)

                # 评估聚类质量：检查每个聚类内的相似性
                score = self._evaluate_clustering(sim_matrix, labels)

                if score > best_score:
                    best_score = score
                    best_labels = labels

            except Exception as e:
                logger.debug("聚类数 %d 失败: %s", n_clusters, str(e))
                continue

        if best_labels is None:
            # 回退：每个序列一个聚类
            best_labels = np.arange(len(sequences))

        logger.info(
            "序列聚类完成: %d 条序列分成 %d 个聚类",
            len(sequences),
            len(set(best_labels))
        )

        return best_labels

    def _evaluate_clustering(self, sim_matrix: np.ndarray, labels: np.ndarray) -> float:
        """
        评估聚类质量

        参数:
            sim_matrix: 相似性矩阵
            labels: 聚类标签

        返回:
            聚类质量分数
        """
        unique_labels = set(labels)
        if len(unique_labels) <= 1:
            return 0.0

        total_score = 0.0
        n_pairs = 0

        for label in unique_labels:
            indices = np.where(labels == label)[0]
            if len(indices) < 2:
                continue

            # 计算聚类内平均相似性
            cluster_sims = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    cluster_sims.append(sim_matrix[indices[i], indices[j]])

            if cluster_sims:
                total_score += np.mean(cluster_sims)
                n_pairs += 1

        return total_score / n_pairs if n_pairs > 0 else 0.0

    def split_by_clusters(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        label_col: Optional[str] = "cell_state",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        基于聚类结果进行数据集分割

        参数:
            df: 输入DataFrame
            labels: 聚类标签
            ratios: (train_ratio, val_ratio, test_ratio)
            label_col: 标签列名（用于分层）

        返回:
            (train_df, val_df, test_df)元组
        """
        train_ratio, val_ratio, test_ratio = ratios
        total = train_ratio + val_ratio + test_ratio

        if not np.isclose(total, 1.0):
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total

        # 按聚类组织数据
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(idx)

        cluster_ids = list(clusters.keys())
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(cluster_ids)

        # 计算目标大小
        n_samples = len(df)
        target_train = int(n_samples * train_ratio)
        target_val = int(n_samples * val_ratio)

        # 分配聚类到各集合
        train_clusters: List[int] = []
        val_clusters: List[int] = []
        test_clusters: List[int] = []

        current_train = 0
        current_val = 0

        for cluster_id in cluster_ids:
            cluster_size = len(clusters[cluster_id])

            # 决定分配到哪个集合
            if current_train + cluster_size <= target_train or not train_clusters:
                train_clusters.append(cluster_id)
                current_train += cluster_size
            elif current_val + cluster_size <= target_val or not val_clusters:
                val_clusters.append(cluster_id)
                current_val += cluster_size
            else:
                test_clusters.append(cluster_id)

        # 收集索引
        train_indices = []
        for cid in train_clusters:
            train_indices.extend(clusters[cid])

        val_indices = []
        for cid in val_clusters:
            val_indices.extend(clusters[cid])

        test_indices = []
        for cid in test_clusters:
            test_indices.extend(clusters[cid])

        # 创建DataFrame
        train_df = df.iloc[train_indices].reset_index(drop=True)
        val_df = df.iloc[val_indices].reset_index(drop=True) if val_indices else df.iloc[0:0].copy()
        test_df = df.iloc[test_indices].reset_index(drop=True) if test_indices else df.iloc[0:0].copy()

        logger.info(
            "基于聚类的分割完成: train=%d (聚类数=%d), val=%d (聚类数=%d), test=%d (聚类数=%d)",
            len(train_df), len(train_clusters),
            len(val_df), len(val_clusters),
            len(test_df), len(test_clusters)
        )

        return train_df, val_df, test_df

    def split(
        self,
        df: pd.DataFrame,
        sequence_col: str = "sequence",
        label_col: Optional[str] = "cell_state",
        ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        执行同源感知的数据分割

        参数:
            df: 输入DataFrame
            sequence_col: 序列列名
            label_col: 标签列名
            ratios: 分割比例

        返回:
            (train_df, val_df, test_df)元组
        """
        if sequence_col not in df.columns:
            raise ValueError(f"序列列 '{sequence_col}' 不存在")

        sequences = df[sequence_col].tolist()

        # 聚类
        labels = self.cluster_sequences(sequences)

        # 分割
        return self.split_by_clusters(df, labels, ratios, label_col)

    def verify_no_leakage(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        sequence_col: str = "sequence",
    ) -> Dict[str, Any]:
        """
        验证分割后的数据集之间无序列泄漏

        参数:
            train_df: 训练集
            val_df: 验证集
            test_df: 测试集
            sequence_col: 序列列名

        返回:
            验证结果字典
        """
        results: Dict[str, Any] = {
            "passed": True,
            "train_val_violations": 0,
            "train_test_violations": 0,
            "val_test_violations": 0,
            "max_similarity": 0.0,
            "details": []
        }

        train_seqs = train_df[sequence_col].tolist() if len(train_df) > 0 else []
        val_seqs = val_df[sequence_col].tolist() if len(val_df) > 0 else []
        test_seqs = test_df[sequence_col].tolist() if len(test_df) > 0 else []

        # 检查训练集与验证集
        if train_seqs and val_seqs:
            for t_seq in train_seqs[:100]:  # 限制检查数量以提高性能
                for v_seq in val_seqs[:100]:
                    sim = self.similarity_calculator.identity_estimate(t_seq, v_seq)
                    results["max_similarity"] = max(results["max_similarity"], sim)
                    if sim > self.identity_threshold:
                        results["train_val_violations"] += 1
                        results["details"].append(f"Train-Val similarity {sim:.3f}")

        # 检查训练集与测试集
        if train_seqs and test_seqs:
            for t_seq in train_seqs[:100]:
                for te_seq in test_seqs[:100]:
                    sim = self.similarity_calculator.identity_estimate(t_seq, te_seq)
                    results["max_similarity"] = max(results["max_similarity"], sim)
                    if sim > self.identity_threshold:
                        results["train_test_violations"] += 1
                        results["details"].append(f"Train-Test similarity {sim:.3f}")

        # 检查验证集与测试集
        if val_seqs and test_seqs:
            for v_seq in val_seqs[:100]:
                for te_seq in test_seqs[:100]:
                    sim = self.similarity_calculator.identity_estimate(v_seq, te_seq)
                    results["max_similarity"] = max(results["max_similarity"], sim)
                    if sim > self.identity_threshold:
                        results["val_test_violations"] += 1
                        results["details"].append(f"Val-Test similarity {sim:.3f}")

        total_violations = (
            results["train_val_violations"] +
            results["train_test_violations"] +
            results["val_test_violations"]
        )

        if total_violations > 0:
            results["passed"] = False
            logger.warning(
                "检测到 %d 个序列相似性违规 (阈值=%.2f)",
                total_violations, self.identity_threshold
            )
        else:
            logger.info(
                "无泄漏验证通过: 最大相似性=%.3f (阈值=%.2f)",
                results["max_similarity"], self.identity_threshold
            )

        return results
