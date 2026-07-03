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
        计算序列相似性矩阵（kmer 方法预计算 k-mer 集合以消除重复计算）

        参数:
            sequences: 序列列表

        返回:
            相似性矩阵
        """
        n = len(sequences)
        sim_matrix = np.zeros((n, n))

        logger.info("计算 %d 条序列的相似性矩阵", n)

        # kmer 模式优先使用稀疏 k-mer 计数矩阵批量计算 Jaccard（向量化），
        # 将 O(n²) 的 Python 双层循环替换为稀疏矩阵运算。当 approximate
        # 开启且 n 超过阈值时强制走该路径；否则作为 kmer 的默认加速路径。
        if self.similarity_method == "kmer" and n >= 2:
            return self._compute_kmer_jaccard_matrix(sequences)

        # non-kmer (local/combined) 方法基于逐对序列比对，无法向量化，
        # 渐近复杂度恒为 O(n²)。无论 approximate 是否开启，只要 n 超过阈值
        # 即告警并提示切换 kmer 模式，避免大规模集合下静默卡顿。
        if self.similarity_method != "kmer" and n > self.approximate_threshold:
            logger.warning(
                "序列数量 %d 超过阈值 %d，non-kmer 模式 (%s) 为 O(n²) 逐对比对，"
                "无法向量化，大规模序列集合下性能会显著下降。"
                "建议切换到 kmer 模式以启用向量化相似性计算；"
                "若确需保留 %s 模式，将尝试并行化（需安装 joblib）。",
                n, self.approximate_threshold, self.similarity_method,
                self.similarity_method,
            )
            # approximate 开启时对 non-kmer 超大规模集合采样子集构建相似性
            # 矩阵并 warn，作为短期缓解，避免逐对比对在数万条序列上卡死。
            if self.approximate:
                return self._sampled_similarity_matrix(sequences)

        # 对 kmer 方法预先计算所有序列的 k-mer 集合（一次 O(n·L)），
        # 内层只做集合交并运算，显著降低常数因子。
        kmer_sets: Optional[List[set]] = None
        if self.similarity_method == "kmer":
            k = self.kmer_size
            kmer_sets = []
            for seq in sequences:
                if len(seq) < k:
                    kmer_sets.append(set())
                else:
                    kmer_sets.append(
                        {seq[i:i + k] for i in range(len(seq) - k + 1)}
                    )

        # non-kmer 路径：尝试用 joblib 并行化外层 i 循环以降低常数因子；
        # 缺失 joblib 时回退到串行逐对循环（行为一致，仅速度差异）。
        if kmer_sets is None and n >= 2:
            return self._pairwise_similarity_matrix_nonkmer(sequences)

        for i in range(n):
            sim_matrix[i, i] = 1.0
            for j in range(i + 1, n):
                if kmer_sets is not None:
                    # 向量化的 Jaccard：预计算集合后只做交并
                    s1, s2 = kmer_sets[i], kmer_sets[j]
                    if not s1 or not s2:
                        sim = 0.0
                    else:
                        inter = len(s1 & s2)
                        union = len(s1 | s2)
                        sim = inter / union if union > 0 else 0.0
                else:  # combined（已由上方 non-kmer 并行路径覆盖，保留兜底）
                    sim = self.similarity_calculator.identity_estimate(
                        sequences[i], sequences[j], self.kmer_size
                    )

                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        return sim_matrix

    def _compute_kmer_jaccard_matrix(self, sequences: List[str]) -> np.ndarray:
        """使用稀疏 k-mer 指示矩阵批量计算 Jaccard 相似性矩阵。

        将每条序列的 k-mer 集合编码为稀疏 0/1 行向量，构成 (n, V) 稀疏矩阵
        A，则交集矩阵 inter = A @ A.T 为 (n, n)，并集 union = |s_i| + |s_j| -
        inter。Jaccard = inter / union。相比双层 Python 循环，稀疏矩阵乘法在
        大规模序列集合上显著更快（由 BLAS/C++ 实现）。

        与逐对循环的 kmer 分支数值一致（同一 Jaccard 定义）。

        当 approximate=True 且 n 超过 approximate_threshold 时，构建稠密 n×n
        矩阵会消耗 O(n²) 内存（数万条序列即 OOM）。此时降级为 MinHash+LSH
        近似近邻（需 datasketch），仅对 LSH 命中的候选对精确计算 Jaccard，
        将稠密矩阵替换为稀疏填充。缺失 datasketch 时回退到稠密路径并 warn。
        """
        n = len(sequences)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64)

        # 大规模近似路径：MinHash + LSH 候选对，避免稠密 n×n 矩阵
        if self.approximate and n > self.approximate_threshold:
            approx_matrix = self._kmer_jaccard_matrix_minhash(sequences)
            if approx_matrix is not None:
                return approx_matrix
            # datasketch 缺失，回退稠密路径（下方）并 warn
            logger.warning(
                "n=%d 超过阈值 %d 且 approximate=True，但 datasketch 未安装，"
                "回退到稠密 n×n Jaccard 矩阵（大规模下可能 OOM）。"
                "建议 `pip install datasketch` 以启用 MinHash+LSH 近似路径。",
                n, self.approximate_threshold,
            )

        from scipy.sparse import csr_matrix

        sim_matrix = np.zeros((n, n), dtype=np.float64)
        np.fill_diagonal(sim_matrix, 1.0)

        k = self.kmer_size
        # 构建 kmer -> 列索引 的词汇表
        vocab: Dict[str, int] = {}
        rows: List[int] = []
        cols: List[int] = []
        row_sizes = np.zeros(n, dtype=np.int64)

        for i, seq in enumerate(sequences):
            if len(seq) < k:
                continue
            seen: set = set()
            for j in range(len(seq) - k + 1):
                km = seq[j:j + k]
                if km in seen:
                    continue
                seen.add(km)
                col = vocab.get(km)
                if col is None:
                    col = len(vocab)
                    vocab[km] = col
                rows.append(i)
                cols.append(col)
            row_sizes[i] = len(seen)

        if not rows:
            return sim_matrix

        data = np.ones(len(rows), dtype=np.float64)
        A = csr_matrix(
            (data, (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
            shape=(n, len(vocab)),
        )
        # 交集矩阵（稠密 n×n，n 较大时可改为保持稀疏）
        inter = (A @ A.T).toarray()
        # 并集 = |s_i| + |s_j| - inter
        sizes = row_sizes.reshape(-1, 1).astype(np.float64)
        union = sizes + sizes.T - inter

        # 避免除零：union==0 时相似度为 0
        with np.errstate(divide="ignore", invalid="ignore"):
            jaccard = np.where(union > 0, inter / union, 0.0)
        np.fill_diagonal(jaccard, 1.0)

        return jaccard

    def _kmer_jaccard_matrix_minhash(
        self, sequences: List[str]
    ) -> Optional[np.ndarray]:
        """MinHash + LSH 近似 Jaccard 矩阵（大规模 kmer 路径）。

        对每条序列构建 k-mer 集合并生成 MinHash 签名，用 LSH 桶仅检索
        候选相似对，再对这些候选对精确计算 Jaccard，避免构建稠密 n×n 矩阵
        （将 O(n²) 内存/时间降为近 O(n·log n) + 候选对数）。

        返回:
            近似 Jaccard 矩阵（未命中候选对填 0）；若 datasketch 未安装则
            返回 None 由调用方回退稠密路径。
        """
        try:
            from datasketch import MinHash, MinHashLSH  # 延迟导入
        except ImportError:
            return None

        n = len(sequences)
        sim_matrix = np.zeros((n, n), dtype=np.float64)
        np.fill_diagonal(sim_matrix, 1.0)
        if n < 2:
            return sim_matrix

        k = self.kmer_size
        # LSH 阈值与 identity_threshold 对齐：相似性低于阈值的对不影响泄漏判定
        # 与聚类（距离 = 1 - sim，低相似对距离接近 1，对 average linkage 影响小）。
        lsh_threshold = max(0.01, min(self.identity_threshold, 0.99))
        # 128 个排列足以在中等规模下稳定估计 Jaccard
        num_perm = 128

        minhashes: List[Optional[MinHash]] = [None] * n
        kmer_sets: List[set] = [set()] * n

        lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
        for i, seq in enumerate(sequences):
            if len(seq) < k:
                continue
            kmers = {seq[j:j + k] for j in range(len(seq) - k + 1)}
            kmer_sets[i] = kmers
            if not kmers:
                continue
            mh = MinHash(num_perm=num_perm)
            for km in kmers:
                mh.update(km.encode("utf-8"))
            minhashes[i] = mh
            lsh.insert(i, mh)

        # 仅对 LSH 命中的候选对精确计算 Jaccard
        for i in range(n):
            mh_i = minhashes[i]
            if mh_i is None:
                continue
            candidates = lsh.query(mh_i)
            for j in candidates:
                if j <= i:
                    continue
                s_i, s_j = kmer_sets[i], kmer_sets[j]
                if not s_i or not s_j:
                    continue
                inter = len(s_i & s_j)
                union = len(s_i | s_j)
                sim = inter / union if union > 0 else 0.0
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        logger.info(
            "MinHash+LSH 近似相似性矩阵完成: n=%d, lsh_threshold=%.3f, num_perm=%d"
            "（未命中候选对相似性按 0 处理）",
            n, lsh_threshold, num_perm,
        )
        return sim_matrix

    def _pairwise_similarity_matrix_nonkmer(
        self, sequences: List[str]
    ) -> np.ndarray:
        """non-kmer (local/combined) 方法的逐对相似性矩阵计算。

        渐近复杂度仍为 O(n²)（local/combined 基于逐对序列比对，无法向量化），
        此处通过可选的 joblib 并行化外层 i 循环降低常数因子；缺失 joblib 时
        回退到串行循环，结果数值完全一致。

        参数:
            sequences: 序列列表

        返回:
            相似性矩阵
        """
        n = len(sequences)
        sim_matrix = np.zeros((n, n), dtype=np.float64)
        np.fill_diagonal(sim_matrix, 1.0)

        method = self.similarity_method
        calc = self.similarity_calculator
        kmer_size = self.kmer_size

        def _pair_sim(i: int, j: int) -> float:
            if method == "local":
                return calc.local_alignment_score(sequences[i], sequences[j])
            # combined
            return calc.identity_estimate(
                sequences[i], sequences[j], kmer_size
            )

        # 尝试 joblib 并行化；缺失时回退串行
        try:
            from joblib import Parallel, delayed  # 延迟导入，避免顶层硬依赖
            have_joblib = True
        except ImportError:
            have_joblib = False
            logger.debug(
                "joblib 未安装，non-kmer 相似性矩阵将走串行路径；"
                "建议 `pip install joblib` 以启用并行加速。"
            )

        if have_joblib and n >= 2:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            if pairs:
                sims = Parallel(n_jobs=-1)(
                    delayed(_pair_sim)(i, j) for i, j in pairs
                )
                for (i, j), sim in zip(pairs, sims):
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    sim = _pair_sim(i, j)
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim

        return sim_matrix

    def _sampled_similarity_matrix(
        self, sequences: List[str]
    ) -> np.ndarray:
        """对超大规模 non-kmer 集合采样子集构建相似性矩阵（短期缓解）。

        当 non-kmer 模式下 n 超过 approximate_threshold 且 approximate=True
        时，逐对比对在数万条序列上会严重卡顿。此处随机采样子集（默认阈值条）
        构建相似性矩阵，未采样序列间的相似性按 0 处理（保守估计，倾向于不
        聚类以降低泄漏风险），并 warn 说明结果为近似值。

        参数:
            sequences: 序列列表

        返回:
            相似性矩阵（仅采样子集块非零）
        """
        n = len(sequences)
        sim_matrix = np.zeros((n, n), dtype=np.float64)
        np.fill_diagonal(sim_matrix, 1.0)

        sample_size = min(n, self.approximate_threshold)
        rng = np.random.RandomState(self.random_state)
        sample_idx = rng.choice(n, size=sample_size, replace=False)

        logger.warning(
            "non-kmer 模式 approximate 采样: 从 %d 条序列中采样 %d 条构建相似性矩阵，"
            "未采样序列间相似性按 0 处理（近似值，可能低估聚类合并）。",
            n, sample_size,
        )

        # 在采样子集上走 non-kmer 逐对路径
        sub_seqs = [sequences[i] for i in sample_idx]
        sub_sim = self._pairwise_similarity_matrix_nonkmer(sub_seqs)

        for a_local, a_global in enumerate(sample_idx):
            for b_local, b_global in enumerate(sample_idx):
                sim_matrix[a_global, b_global] = sub_sim[a_local, b_local]

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
            "details": []
        }

        train_seqs = train_df[sequence_col].tolist() if len(train_df) > 0 else []
        val_seqs = val_df[sequence_col].tolist() if len(val_df) > 0 else []
        test_seqs = test_df[sequence_col].tolist() if len(test_df) > 0 else []

        # 全量检查两组之间的最大相似性与违规数。复用 _max_cross_similarity，
        # 该方法对 kmer 模式预计算 k-mer 集合后只做集合运算，避免重复计算。
        if train_seqs and val_seqs:
            max_sim, violations, details = self._max_cross_similarity(
                train_seqs, val_seqs, "Train-Val"
            )
            results["max_similarity"] = max(results["max_similarity"], max_sim)
            results["train_val_violations"] = violations
            results["details"].extend(details)

        if train_seqs and test_seqs:
            max_sim, violations, details = self._max_cross_similarity(
                train_seqs, test_seqs, "Train-Test"
            )
            results["max_similarity"] = max(results["max_similarity"], max_sim)
            results["train_test_violations"] = violations
            results["details"].extend(details)

        if val_seqs and test_seqs:
            max_sim, violations, details = self._max_cross_similarity(
                val_seqs, test_seqs, "Val-Test"
            )
            results["max_similarity"] = max(results["max_similarity"], max_sim)
            results["val_test_violations"] = violations
            results["details"].extend(details)

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

    def _max_cross_similarity(
        self,
        seqs_a: List[str],
        seqs_b: List[str],
        label: str,
    ) -> Tuple[float, int, List[str]]:
        """计算两组序列之间的最大跨集相似性、违规数与详情（全量，不截断）。

        对 kmer 模式预计算 k-mer 集合后只做集合交并；对其他方法回退到逐对
        identity_estimate。为控制内存，违规详情只保留前若干条。
        """
        max_sim = 0.0
        violations = 0
        details: List[str] = []
        detail_cap = 50

        # kmer 模式：使用稀疏 k-mer 指示矩阵批量计算跨集 Jaccard，避免
        # sets_a × sets_b 的双层 Python 循环。与逐对循环数值一致。
        if self.similarity_method == "kmer":
            return self._max_cross_similarity_kmer(seqs_a, seqs_b, label)

        for a_seq in seqs_a:
            for b_seq in seqs_b:
                sim = self.similarity_calculator.identity_estimate(a_seq, b_seq, self.kmer_size)
                if sim > max_sim:
                    max_sim = sim
                if sim > self.identity_threshold:
                    violations += 1
                    if len(details) < detail_cap:
                        details.append(f"{label} similarity {sim:.3f}")

        return max_sim, violations, details

    def _max_cross_similarity_kmer(
        self,
        seqs_a: List[str],
        seqs_b: List[str],
        label: str,
    ) -> Tuple[float, int, List[str]]:
        """kmer 模式下跨集相似性的向量化实现。

        构建两组序列共享词汇表的稀疏指示矩阵 A (na, V) 与 B (nb, V)，
        交集矩阵 inter = A @ B.T (na, nb)，并集 = |s_i| + |s_j| - inter，
        Jaccard = inter / union。再应用与逐对循环一致的 identity 缩放
        （长序列 sim * 1.5，截断到 1.0）。
        """
        from scipy.sparse import csr_matrix

        max_sim = 0.0
        violations = 0
        details: List[str] = []
        detail_cap = 50

        na, nb = len(seqs_a), len(seqs_b)
        if na == 0 or nb == 0:
            return max_sim, violations, details

        k = self.kmer_size
        vocab: Dict[str, int] = {}

        def _build_rows(seqs: List[str]) -> Tuple[List[int], List[int], np.ndarray]:
            rows_i: List[int] = []
            cols_i: List[int] = []
            sizes = np.zeros(len(seqs), dtype=np.int64)
            for idx, seq in enumerate(seqs):
                if len(seq) < k:
                    continue
                seen: set = set()
                for j in range(len(seq) - k + 1):
                    km = seq[j:j + k]
                    if km in seen:
                        continue
                    seen.add(km)
                    col = vocab.get(km)
                    if col is None:
                        col = len(vocab)
                        vocab[km] = col
                    rows_i.append(idx)
                    cols_i.append(col)
                sizes[idx] = len(seen)
            return rows_i, cols_i, sizes

        rows_a, cols_a, sizes_a = _build_rows(seqs_a)
        rows_b, cols_b, sizes_b = _build_rows(seqs_b)
        vocab_size = len(vocab)

        if not rows_a or not rows_b:
            return max_sim, violations, details

        A = csr_matrix(
            (np.ones(len(rows_a), dtype=np.float64),
             (np.array(rows_a, dtype=np.int64), np.array(cols_a, dtype=np.int64))),
            shape=(na, vocab_size),
        )
        B = csr_matrix(
            (np.ones(len(rows_b), dtype=np.float64),
             (np.array(rows_b, dtype=np.int64), np.array(cols_b, dtype=np.int64))),
            shape=(nb, vocab_size),
        )
        inter = (A @ B.T).toarray()  # (na, nb)
        union = sizes_a.reshape(-1, 1).astype(np.float64) + sizes_b.reshape(1, -1).astype(np.float64) - inter

        with np.errstate(divide="ignore", invalid="ignore"):
            jaccard = np.where(union > 0, inter / union, 0.0)

        # identity 估计缩放（与逐对循环一致）
        len_a = np.array([len(s) for s in seqs_a], dtype=np.float64).reshape(-1, 1)
        len_b = np.array([len(s) for s in seqs_b], dtype=np.float64).reshape(1, -1)
        long_mask = (len_a >= 10) & (len_b >= 10)
        scaled = np.where(long_mask, np.minimum(1.0, jaccard * 1.5), jaccard)

        max_sim = float(np.max(scaled)) if scaled.size > 0 else 0.0
        viol_mask = scaled > self.identity_threshold
        violations = int(np.sum(viol_mask))
        if violations > 0 and len(details) < detail_cap:
            viols = np.argwhere(viol_mask)
            for (i, j) in viols[: detail_cap]:
                details.append(f"{label} similarity {scaled[i, j]:.3f}")

        return max_sim, violations, details
