"""
序列相似性计算工具
"""

import logging

logger = logging.getLogger(__name__)

__all__ = ["SequenceSimilarityCalculator"]


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
