"""
稀疏k-mer Jaccard矩阵近似计算与跨集相似性验证工具
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .similarity import SequenceSimilarityCalculator

logger = logging.getLogger(__name__)

__all__ = [
    "compute_kmer_jaccard_matrix",
    "kmer_jaccard_matrix_minhash",
    "pairwise_similarity_matrix_nonkmer",
    "sampled_similarity_matrix",
    "max_cross_similarity",
    "max_cross_similarity_kmer",
]


def compute_kmer_jaccard_matrix(
    sequences: List[str],
    k: int = 3,
    approximate: bool = False,
    approximate_threshold: int = 5000,
) -> np.ndarray:
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

    参数:
        sequences: 序列列表
        k: k-mer 长度
        approximate: 是否启用 MinHash+LSH 近似以规避 O(n²) 内存
        approximate_threshold: 触发近似路径的序列数量阈值

    返回:
        Jaccard 相似性矩阵 (n×n)
    """
    n = len(sequences)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    # 大规模近似路径：MinHash + LSH 候选对，避免稠密 n×n 矩阵
    if approximate and n > approximate_threshold:
        approx_matrix = kmer_jaccard_matrix_minhash(sequences, k, identity_threshold=0.3)
        if approx_matrix is not None:
            return approx_matrix
        # datasketch 缺失，回退稠密路径（下方）并 warn
        logger.warning(
            "n=%d 超过阈值 %d 且 approximate=True，但 datasketch 未安装，"
            "回退到稠密 n×n Jaccard 矩阵（大规模下可能 OOM）。"
            "建议 `pip install datasketch` 以启用 MinHash+LSH 近似路径。",
            n, approximate_threshold,
        )

    from scipy.sparse import csr_matrix

    sim_matrix = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(sim_matrix, 1.0)

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


def kmer_jaccard_matrix_minhash(
    sequences: List[str],
    k: int = 3,
    identity_threshold: float = 0.3,
) -> Optional[np.ndarray]:
    """MinHash + LSH 近似 Jaccard 矩阵（大规模 kmer 路径）。

    对每条序列构建 k-mer 集合并生成 MinHash 签名，用 LSH 桶仅检索
    候选相似对，再对这些候选对精确计算 Jaccard，避免构建稠密 n×n 矩阵
    （将 O(n²) 内存/时间降为近 O(n·log n) + 候选对数）。

    参数:
        sequences: 序列列表
        k: k-mer 长度
        identity_threshold: 序列同一性阈值（对齐 LSH 阈值）

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

    # LSH 阈值与 identity_threshold 对齐：相似性低于阈值的对不影响泄漏判定
    # 与聚类（距离 = 1 - sim，低相似对距离接近 1，对 average linkage 影响小）。
    lsh_threshold = max(0.01, min(identity_threshold, 0.99))
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


def pairwise_similarity_matrix_nonkmer(
    sequences: List[str],
    method: str,
    kmer_size: int,
) -> np.ndarray:
    """non-kmer (local/combined) 方法的逐对相似性矩阵计算。

    渐近复杂度仍为 O(n²)（local/combined 基于逐对序列比对，无法向量化），
    此处通过可选的 joblib 并行化外层 i 循环降低常数因子；缺失 joblib 时
    回退到串行循环，结果数值完全一致。

    参数:
        sequences: 序列列表
        method: 相似性计算方法 ("local" 或 "combined")
        kmer_size: k-mer 大小（仅 combined 模式使用）

    返回:
        相似性矩阵
    """
    n = len(sequences)
    sim_matrix = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(sim_matrix, 1.0)

    calc = SequenceSimilarityCalculator()

    def _pair_sim(i: int, j: int) -> float:
        if method == "local":
            return calc.local_alignment_score(sequences[i], sequences[j])
        # combined
        return calc.identity_estimate(sequences[i], sequences[j], kmer_size)

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


def sampled_similarity_matrix(
    sequences: List[str],
    approximate_threshold: int,
    random_state: int,
    method: str,
    kmer_size: int,
) -> np.ndarray:
    """对超大规模 non-kmer 集合采样子集构建相似性矩阵（短期缓解）。

    当 non-kmer 模式下 n 超过 approximate_threshold 且 approximate=True
    时，逐对比对在数万条序列上会严重卡顿。此处随机采样子集（默认阈值条）
    构建相似性矩阵，未采样序列间的相似性按 0 处理（保守估计，倾向于不
    聚类以降低泄漏风险），并 warn 说明结果为近似值。

    参数:
        sequences: 序列列表
        approximate_threshold: 采样阈值（采样子集大小）
        random_state: 随机种子
        method: 相似性计算方法
        kmer_size: k-mer 大小

    返回:
        相似性矩阵（仅采样子集块非零）
    """
    n = len(sequences)
    sim_matrix = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(sim_matrix, 1.0)

    sample_size = min(n, approximate_threshold)
    rng = np.random.RandomState(random_state)
    sample_idx = rng.choice(n, size=sample_size, replace=False)

    logger.warning(
        "non-kmer 模式 approximate 采样: 从 %d 条序列中采样 %d 条构建相似性矩阵，"
        "未采样序列间相似性按 0 处理（近似值，可能低估聚类合并）。",
        n, sample_size,
    )

    # 在采样子集上走 non-kmer 逐对路径
    sub_seqs = [sequences[i] for i in sample_idx]
    sub_sim = pairwise_similarity_matrix_nonkmer(sub_seqs, method, kmer_size)

    for a_local, a_global in enumerate(sample_idx):
        for b_local, b_global in enumerate(sample_idx):
            sim_matrix[a_global, b_global] = sub_sim[a_local, b_local]

    return sim_matrix


def max_cross_similarity(
    seqs_a: List[str],
    seqs_b: List[str],
    label: str,
    method: str,
    identity_threshold: float,
    kmer_size: int,
) -> Tuple[float, int, List[str]]:
    """计算两组序列之间的最大跨集相似性、违规数与详情（全量，不截断）。

    对 kmer 模式预计算 k-mer 集合后只做集合交并；对其他方法回退到逐对
    identity_estimate。为控制内存，违规详情只保留前若干条。

    参数:
        seqs_a: 第一组序列
        seqs_b: 第二组序列
        label: 组标签（用于详情描述）
        method: 相似性计算方法
        identity_threshold: 序列同一性阈值
        kmer_size: k-mer 大小

    返回:
        (max_similarity, violations_count, details_list)
    """
    max_sim = 0.0
    violations = 0
    details: List[str] = []
    detail_cap = 50

    # kmer 模式：使用稀疏 k-mer 指示矩阵批量计算跨集 Jaccard，避免
    # sets_a × sets_b 的双层 Python 循环。与逐对循环数值一致。
    if method == "kmer":
        return max_cross_similarity_kmer(seqs_a, seqs_b, kmer_size, identity_threshold, label)

    calc = SequenceSimilarityCalculator()
    for a_seq in seqs_a:
        for b_seq in seqs_b:
            sim = calc.identity_estimate(a_seq, b_seq, kmer_size)
            if sim > max_sim:
                max_sim = sim
            if sim > identity_threshold:
                violations += 1
                if len(details) < detail_cap:
                    details.append(f"{label} similarity {sim:.3f}")

    return max_sim, violations, details


def max_cross_similarity_kmer(
    seqs_a: List[str],
    seqs_b: List[str],
    k: int,
    identity_threshold: float,
    label: str = "",
) -> Tuple[float, int, List[str]]:
    """kmer 模式下跨集相似性的向量化实现。

    构建两组序列共享词汇表的稀疏指示矩阵 A (na, V) 与 B (nb, V)，
    交集矩阵 inter = A @ B.T (na, nb)，并集 = |s_i| + |s_j| - inter，
    Jaccard = inter / union。再应用与逐对循环一致的 identity 缩放
    （长序列 sim * 1.5，截断到 1.0）。

    参数:
        seqs_a: 第一组序列
        seqs_b: 第二组序列
        k: k-mer 长度
        identity_threshold: 序列同一性阈值
        label: 组标签（用于详情描述）

    返回:
        (max_similarity, violations_count, details_list)
    """
    from scipy.sparse import csr_matrix

    max_sim = 0.0
    violations = 0
    details: List[str] = []
    detail_cap = 50

    na, nb = len(seqs_a), len(seqs_b)
    if na == 0 or nb == 0:
        return max_sim, violations, details

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
    viol_mask = scaled > identity_threshold
    violations = int(np.sum(viol_mask))
    if violations > 0 and len(details) < detail_cap:
        viols = np.argwhere(viol_mask)
        for (i, j) in viols[: detail_cap]:
            details.append(f"{label} similarity {scaled[i, j]:.3f}")

    return max_sim, violations, details
