"""
同源感知数据分割器
基于序列相似性聚类进行数据分割，防止同源序列泄漏
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

from .similarity import SequenceSimilarityCalculator
from .matrix import (
    compute_kmer_jaccard_matrix,
    max_cross_similarity,
    pairwise_similarity_matrix_nonkmer,
    sampled_similarity_matrix,
)

logger = logging.getLogger(__name__)


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
        approximate: bool = False,
        approximate_threshold: int = 5000,
    ):
        """
        初始化同源感知分割器

        参数:
            identity_threshold: 序列同一性阈值（默认30%）
            similarity_method: 相似性计算方法 ("kmer", "local", "combined")
            kmer_size: k-mer大小
            random_state: 随机种子
            approximate: 是否对大规模序列集合启用近似相似性计算（kmer 模式下
                使用稀疏 k-mer 计数矩阵批量计算 Jaccard，将 O(n²) 的 Python
                双层循环替换为向量化稀疏矩阵运算；非 kmer 模式下对超大规模
                集合采样子集构建相似性矩阵并 warn）
            approximate_threshold: 触发近似路径的序列数量阈值
        """
        self.identity_threshold = identity_threshold
        self.similarity_method = similarity_method
        self.kmer_size = kmer_size
        self.random_state = random_state
        self.approximate = approximate
        self.approximate_threshold = approximate_threshold
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

        logger.info("计算 %d 条序列的相似性矩阵", n)

        # kmer 模式优先使用稀疏 k-mer 计数矩阵批量计算 Jaccard（向量化），
        # 将 O(n²) 的 Python 双层循环替换为稀疏矩阵运算。
        if self.similarity_method == "kmer" and n >= 2:
            return compute_kmer_jaccard_matrix(
                sequences,
                k=self.kmer_size,
                approximate=self.approximate,
                approximate_threshold=self.approximate_threshold,
            )

        # non-kmer (local/combined) 方法基于逐对序列比对，无法向量化。
        # 当 n 超过阈值时告警，若 approximate 开启则采样子集构建矩阵。
        if self.similarity_method != "kmer" and n > self.approximate_threshold:
            logger.warning(
                "序列数量 %d 超过阈值 %d，non-kmer 模式 (%s) 为 O(n²) 逐对比对，"
                "无法向量化，大规模序列集合下性能会显著下降。"
                "建议切换到 kmer 模式以启用向量化相似性计算；"
                "若确需保留 %s 模式，将尝试并行化（需安装 joblib）。",
                n,
                self.approximate_threshold,
                self.similarity_method,
                self.similarity_method,
            )
            if self.approximate:
                return sampled_similarity_matrix(
                    sequences,
                    approximate_threshold=self.approximate_threshold,
                    random_state=self.random_state,
                    method=self.similarity_method,
                    kmer_size=self.kmer_size,
                )

        # non-kmer 或小规模路径
        return pairwise_similarity_matrix_nonkmer(
            sequences,
            method=self.similarity_method,
            kmer_size=self.kmer_size,
        )

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

            except (ValueError, RuntimeError) as e:
                logger.debug("聚类数 %d 失败: %s", n_clusters, str(e))
                continue

        if best_labels is None:
            # 回退：每个序列一个聚类
            best_labels = np.arange(len(sequences))

        logger.info("序列聚类完成: %d 条序列分成 %d 个聚类", len(sequences), len(set(best_labels)))

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
            len(train_df),
            len(train_clusters),
            len(val_df),
            len(val_clusters),
            len(test_df),
            len(test_clusters),
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
        验证分割后的数据集之间无序列泄漏（全量检查，不截断）

        通过计算两组序列的跨集相似性矩阵取最大值来判断泄漏，避免之前 [:100]
        截断导致超出部分无法检测的问题。对大集合采用分块计算以控制内存。

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
            "details": [],
        }

        train_seqs = train_df[sequence_col].tolist() if len(train_df) > 0 else []
        val_seqs = val_df[sequence_col].tolist() if len(val_df) > 0 else []
        test_seqs = test_df[sequence_col].tolist() if len(test_df) > 0 else []

        # 全量检查两组之间的最大相似性与违规数。复用 max_cross_similarity，
        # 该方法对 kmer 模式预计算 k-mer 集合后只做集合运算，避免重复计算。
        if train_seqs and val_seqs:
            max_sim, violations, details = max_cross_similarity(
                train_seqs,
                val_seqs,
                "Train-Val",
                method=self.similarity_method,
                identity_threshold=self.identity_threshold,
                kmer_size=self.kmer_size,
            )
            results["max_similarity"] = max(results["max_similarity"], max_sim)
            results["train_val_violations"] = violations
            results["details"].extend(details)

        if train_seqs and test_seqs:
            max_sim, violations, details = max_cross_similarity(
                train_seqs,
                test_seqs,
                "Train-Test",
                method=self.similarity_method,
                identity_threshold=self.identity_threshold,
                kmer_size=self.kmer_size,
            )
            results["max_similarity"] = max(results["max_similarity"], max_sim)
            results["train_test_violations"] = violations
            results["details"].extend(details)

        if val_seqs and test_seqs:
            max_sim, violations, details = max_cross_similarity(
                val_seqs,
                test_seqs,
                "Val-Test",
                method=self.similarity_method,
                identity_threshold=self.identity_threshold,
                kmer_size=self.kmer_size,
            )
            results["max_similarity"] = max(results["max_similarity"], max_sim)
            results["val_test_violations"] = violations
            results["details"].extend(details)

        total_violations = (
            results["train_val_violations"] + results["train_test_violations"] + results["val_test_violations"]
        )

        if total_violations > 0:
            results["passed"] = False
            logger.warning("检测到 %d 个序列相似性违规 (阈值=%.2f)", total_violations, self.identity_threshold)
        else:
            logger.info(
                "无泄漏验证通过: 最大相似性=%.3f (阈值=%.2f)", results["max_similarity"], self.identity_threshold
            )

        return results
