"""
同源感知数据分割模块
基于序列相似性聚类的数据分割，避免同源序列泄漏
"""

from .similarity import SequenceSimilarityCalculator
from .splitter import HomologyAwareSplitter
from .matrix import (
    compute_kmer_jaccard_matrix,
    kmer_jaccard_matrix_minhash,
    max_cross_similarity,
    max_cross_similarity_kmer,
    pairwise_similarity_matrix_nonkmer,
    sampled_similarity_matrix,
)

__all__ = [
    "HomologyAwareSplitter",
    "SequenceSimilarityCalculator",
    "compute_kmer_jaccard_matrix",
    "kmer_jaccard_matrix_minhash",
    "max_cross_similarity",
    "max_cross_similarity_kmer",
    "pairwise_similarity_matrix_nonkmer",
    "sampled_similarity_matrix",
]
