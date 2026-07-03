"""
特征工程模块
功能概述: 从蛋白质序列和PTM数据中提取特征
设计思路: 支持多种特征提取方法，可灵活组合使用
"""

import json
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    PCA = TruncatedSVD = TSNE = None

from src.data.aa_constants import (
    AMINO_ACIDS,
    AMINO_ACIDS_STR,
    AA_TO_IDX,
    NON_STANDARD_AA_MAP,
    PAD_IDX,
    AA_PAD_CHAR,
)
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

DEFAULT_AMINO_ACIDS = AMINO_ACIDS_STR
DEFAULT_PTM_TYPES = [
    "phosphorylation",
    "acetylation",
    "methylation",
    "ubiquitination",
    "sumoylation",
]

PTM_TYPES = DEFAULT_PTM_TYPES
PTM_TO_IDX = {ptm: i for i, ptm in enumerate(PTM_TYPES)}


class FeatureExtractor:
    """
    特征提取器类
    负责从蛋白质序列和PTM数据中提取各种特征
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化特征提取器

        参数:
            config: 配置字典
        """
        self.config = config or {}
        feature_config = self.config.get("features", {})
        self.sequence_encoding = feature_config.get("sequence_encoding", "onehot")
        self.kmer_size = feature_config.get("kmer_size", 3)
        self.include_physicochemical = feature_config.get("include_physicochemical", True)
        self.include_ptm_features = feature_config.get("include_ptm_features", True)
        self.include_structural_features = feature_config.get("include_structural_features", False)
        self.structural_source = feature_config.get("structural_source", "chou_fasman")
        if self.structural_source not in {"chou_fasman", "alphafold", "psipred"}:
            logger.warning(
                "未知 structural_source=%s，回退到 chou_fasman",
                self.structural_source,
            )
            self.structural_source = "chou_fasman"
        self.use_feature_extractor = feature_config.get("use_feature_extractor", False)
        self.max_sequence_length = self.config.get("data", {}).get("max_sequence_length", 1000)
        self.amino_acids = self.config.get("data", {}).get("valid_amino_acids", DEFAULT_AMINO_ACIDS)
        # 序列索引从1开始，0保留给padding
        # 基于共享常量构建映射；若配置自定义字母表则覆盖
        if self.amino_acids == AMINO_ACIDS_STR:
            self.aa_to_idx = dict(AA_TO_IDX)
        else:
            self.aa_to_idx = {aa: i + 1 for i, aa in enumerate(self.amino_acids)}
        self.ptm_types = self.config.get("data", {}).get("ptm_types", DEFAULT_PTM_TYPES)
        self.ptm_to_idx = {ptm: i for i, ptm in enumerate(self.ptm_types)}

        self.chou_fasman_propensities = {
            "A": (1.45, 0.97, 0.66),
            "C": (0.77, 1.30, 1.19),
            "D": (0.98, 0.80, 1.46),
            "E": (1.53, 0.26, 0.74),
            "F": (1.12, 1.28, 0.59),
            "G": (0.53, 0.81, 1.56),
            "H": (1.24, 0.71, 0.95),
            "I": (1.00, 1.60, 0.47),
            "K": (1.07, 0.74, 1.01),
            "L": (1.34, 1.22, 0.59),
            "M": (1.20, 1.67, 0.60),
            "N": (0.73, 0.65, 1.56),
            "P": (0.59, 0.62, 1.52),
            "Q": (1.17, 1.23, 0.98),
            "R": (0.79, 0.90, 0.95),
            "S": (0.79, 0.72, 1.43),
            "T": (0.82, 1.20, 0.96),
            "V": (1.14, 1.65, 0.50),
            "W": (1.14, 1.19, 0.96),
            "Y": (0.61, 1.29, 1.14),
        }

    def extract_onehot_sequence(self, sequence: str) -> np.ndarray:
        """提取序列的 one-hot 编码（向量化实现）。

        参数:
            sequence: 蛋白质序列

        返回:
            one-hot 编码数组，形状为 (max_len, num_amino_acids + 1)
            第 0 列保留给 padding，实际氨基酸从第 1 列开始
        """
        seq_len = min(len(sequence), self.max_sequence_length)
        # +1 for padding column (index 0)
        onehot = np.zeros((self.max_sequence_length, len(self.amino_acids) + 1), dtype=np.float32)

        if seq_len > 0:
            # 向量化：一次性将前 seq_len 个字符映射到索引并置 1
            head = sequence[:seq_len]
            idx_arr = np.fromiter(
                (self.aa_to_idx.get(c, -1) for c in head),
                dtype=np.int64,
                count=seq_len,
            )
            valid = idx_arr >= 0
            if np.any(valid):
                positions = np.nonzero(valid)[0]
                onehot[positions, idx_arr[valid]] = 1.0

        return onehot

    def extract_kmer_features(self, sequence: str, k: Optional[int] = None) -> np.ndarray:
        """提取 k-mer 频率特征（向量化实现）。

        参数:
            sequence: 蛋白质序列
            k: k-mer 大小，默认使用配置中的值

        返回:
            k-mer 频率向量
        """
        if k is None:
            k = self.kmer_size

        base = len(self.amino_acids)
        feature_vector = np.zeros(base ** k, dtype=np.float32)

        if len(sequence) < k:
            return feature_vector

        # 用 collections.Counter 批量统计 k-mer，再一次性写入向量
        from collections import Counter

        kmers = (sequence[i:i + k] for i in range(len(sequence) - k + 1))
        counts = Counter(kmers)

        total = 0
        for kmer, count in counts.items():
            idx = 0
            valid = True
            for aa in kmer:
                aa_idx = self.aa_to_idx.get(aa)
                if aa_idx is None:
                    valid = False
                    break
                idx = idx * base + aa_idx
            if valid:
                feature_vector[idx] += float(count)
                total += count

        if total > 0:
            feature_vector /= float(total)

        return feature_vector

    def extract_physicochemical_features(self, sequence: str) -> np.ndarray:
        """
        提取理化特征

        参数:
            sequence: 蛋白质序列

        返回:
            理化特征向量
        """
        hydropathy = {
            "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
            "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
            "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
            "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
        }

        charge = {
            "R": 1, "K": 1, "D": -1, "E": -1, "H": 0.1,
        }

        features = []
        values = [hydropathy.get(aa, 0.0) for aa in sequence]
        if values:
            features.extend([np.mean(values), np.std(values), np.min(values), np.max(values)])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        charge_values = [charge.get(aa, 0.0) for aa in sequence]
        if charge_values:
            features.extend([np.sum(charge_values), np.mean(charge_values)])
        else:
            features.extend([0.0, 0.0])

        aa_counts = {aa: sequence.count(aa) for aa in self.amino_acids}
        total = max(1, len(sequence))
        features.extend([aa_counts[aa] / total for aa in self.amino_acids])

        return np.array(features, dtype=np.float32)

    def extract_batch_physicochemical_features(self, sequences: List[str]) -> np.ndarray:
        """批量理化特征（向量化，消除逐样本 Python 循环）。

        参考单样本实现 :meth:`extract_physicochemical_features`，但用查表矩阵
        一次性映射整条序列为 hydropathy/charge 向量再聚合，避免逐样本调用。
        返回形状 (n, P) 的数组，与逐样本调用逐行拼接结果一致。
        """
        hydropathy = {
            "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
            "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
            "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
            "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
        }
        charge = {
            "R": 1, "K": 1, "D": -1, "E": -1, "H": 0.1,
        }

        # 构建查表数组：将标准氨基酸按固定顺序堆叠为向量，按索引取值
        std_aas = "ACDEFGHIKLMNPQRSTVWY"
        hydropathy_arr = np.array(
            [hydropathy.get(aa, 0.0) for aa in std_aas], dtype=np.float32
        )
        charge_arr = np.array(
            [charge.get(aa, 0.0) for aa in std_aas], dtype=np.float32
        )
        aa_to_pos = {aa: i for i, aa in enumerate(std_aas)}

        n = len(sequences)
        num_aa = len(self.amino_acids)
        # 理化特征列数：hydropathy(4) + charge(2) + aa_freq(num_aa)
        n_cols = 4 + 2 + num_aa
        features = np.zeros((n, n_cols), dtype=np.float32)

        for i, seq in enumerate(sequences):
            if len(seq) == 0:
                continue
            # 将序列映射为索引数组（非标准氨基酸映射到 -1，后置取缺省 0）
            idx_arr = np.fromiter(
                (aa_to_pos.get(c, -1) for c in seq),
                dtype=np.int64,
                count=len(seq),
            )
            valid_mask = idx_arr >= 0
            if np.any(valid_mask):
                valid_idx = idx_arr[valid_mask]
                hydro_vals = hydropathy_arr[valid_idx]
                charge_vals = charge_arr[valid_idx]

                features[i, 0] = hydro_vals.mean()
                features[i, 1] = hydro_vals.std()
                features[i, 2] = hydro_vals.min()
                features[i, 3] = hydro_vals.max()
                features[i, 4] = charge_vals.sum()
                features[i, 5] = charge_vals.mean()

            # aa 频率：对配置字母表中每个 aa 统计 count/total
            total = max(1, len(seq))
            for j, aa in enumerate(self.amino_acids):
                features[i, 6 + j] = seq.count(aa) / total

        return features

    def _load_ptm_sites(self, ptm_sites_input: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """归一化PTM输入为列表。"""
        if isinstance(ptm_sites_input, list):
            return [site for site in ptm_sites_input if isinstance(site, dict)]

        try:
            ptm_sites = json.loads(ptm_sites_input)
        except (json.JSONDecodeError, TypeError):
            ptm_sites = []

        return ptm_sites if isinstance(ptm_sites, list) else []

    def extract_ptm_features(
        self,
        ptm_sites_input: Union[str, List[Dict[str, Any]]],
        sequence_length: int,
    ) -> Dict[str, np.ndarray]:
        """
        提取PTM特征

        参数:
            ptm_sites_input: PTM位点JSON字符串或位点字典列表
            sequence_length: 序列长度

        返回:
            PTM特征数组
        """
        ptm_sites = self._load_ptm_sites(ptm_sites_input)

        ptm_counts = {ptm: 0 for ptm in self.ptm_types}
        position_features = np.zeros((self.max_sequence_length, len(self.ptm_types)), dtype=np.float32)
        max_len = min(self.max_sequence_length, sequence_length) if sequence_length else self.max_sequence_length

        for site in ptm_sites:
            pos = site.get("position", 0) - 1
            ptm_type = site.get("type", "")

            if 0 <= pos < max_len and ptm_type in self.ptm_to_idx:
                ptm_idx = self.ptm_to_idx[ptm_type]
                position_features[pos, ptm_idx] = 1.0
                ptm_counts[ptm_type] += 1

        count_features = np.array([ptm_counts[ptm] for ptm in self.ptm_types], dtype=np.float32)

        return {
            "position_features": position_features,
            "count_features": count_features,
        }

    def extract_ptm_features_array(self, ptm_sites: List[Dict[str, Any]], sequence_length: int) -> np.ndarray:
        """返回 [position_features.flatten(), count_features] 的拼接向量。"""
        feature_dict = self.extract_ptm_features(ptm_sites, sequence_length)
        return np.concatenate(
            [
                feature_dict["position_features"].flatten(),
                feature_dict["count_features"],
            ]
        ).astype(np.float32)

    def extract_structural_features(self, sequence: str) -> np.ndarray:
        """生成结构倾向特征。

        根据 ``structural_source`` 配置选择来源：

        - ``chou_fasman``（默认，离线）：经验性 Chou-Fasman propensity 表近似
          helix/sheet/coil 倾向。
        - ``alphafold``：调用 ``src/models/external_tools.py`` 的
          :class:`AlphaFoldClient.predict_structure` 获取 pLDDT/二级结构，
          映射为 helix/sheet/coil + pLDDT 归一化列。
        - ``psipred``：调用 :class:`PSIPREDClient.predict_secondary_structure`
          获取二级结构字符串与置信度。

        当 external_tools 不可用或网络失败时，自动回退到 Chou-Fasman 并记录
        warning。返回形状为 ``(max_sequence_length, 3)`` 的数组（alphafold 时
        仍保持 3 列，pLDDT 信息归一化并入对应倾向列）。

        返回:
            形状为 (max_sequence_length, 3) 的数组，列依次表示 helix/sheet/coil
            倾向（归一化后）。超出序列长度的行为零向量。
        """
        if self.structural_source in {"alphafold", "psipred"}:
            external = self._extract_external_structural(sequence)
            if external is not None:
                return external
            # external_tools 不可用或调用失败，回退到 Chou-Fasman
            logger.warning(
                "structural_source=%s 不可用，回退到 Chou-Fasman 近似",
                self.structural_source,
            )

        return self._extract_chou_fasman_structural(sequence)

    def _extract_external_structural(self, sequence: str) -> Optional[np.ndarray]:
        """调用 external_tools 客户端获取真实结构特征。

        返回 ``(max_sequence_length, 3)`` 数组或 ``None``（不可用时）。
        延迟导入以避免 data 与 models 之间的循环依赖。
        """
        structural = np.zeros((self.max_sequence_length, 3), dtype=np.float32)
        seq_len = min(len(sequence), self.max_sequence_length)
        if seq_len == 0:
            return structural

        try:
            if self.structural_source == "alphafold":
                from src.models.external_tools import AlphaFoldClient

                af_client = AlphaFoldClient(self.config)
                result = af_client.predict_structure(sequence)
                plddt = float(result.get("confidence", 0.0))
                # AlphaFold 返回 PDB 字符串，无逐残基二级结构标签；
                # 以 pLDDT 作为整体置信度，coil 倾向 = pLDDT，helix/sheet 各占
                # 剩余置信度的一半（保守近似），保证三维向量非零且可区分。
                helix_prop = (1.0 - plddt) * 0.5
                sheet_prop = (1.0 - plddt) * 0.5
                coil_prop = plddt
                row = np.array([helix_prop, sheet_prop, coil_prop], dtype=np.float32)
                structural[:seq_len] = row
                return structural
            elif self.structural_source == "psipred":
                from src.models.external_tools import PSIPREDClient

                client = PSIPREDClient(self.config)
                result = client.predict_secondary_structure(sequence)
                ss_pred = result.get("ss_prediction", "")
                confs = result.get("confidence_scores", [])
                for i in range(min(seq_len, len(ss_pred))):
                    ch = ss_pred[i]
                    conf = float(confs[i]) if i < len(confs) else 0.5
                    if ch == "H":
                        structural[i] = [conf, 0.0, 0.0]
                    elif ch == "E":
                        structural[i] = [0.0, conf, 0.0]
                    else:  # 'C' or unknown
                        structural[i] = [0.0, 0.0, conf]
                return structural
        except ImportError as exc:
            logger.warning("无法导入 external_tools (%s)", exc)
        except Exception as exc:  # noqa: BLE001 - 网络或解析失败需回退
            logger.warning("external_tools 结构预测失败: %s", exc)
        return None

    def _extract_chou_fasman_structural(self, sequence: str) -> np.ndarray:
        """Chou-Fasman 经验近似结构特征（离线 fallback）。"""
        structural = np.zeros((self.max_sequence_length, 3), dtype=np.float32)
        seq_len = min(len(sequence), self.max_sequence_length)

        # 构建查表矩阵以消除逐字符 Python 循环：将 Chou-Fasman 表堆叠为
        # (20, 3) 的数组，按氨基酸索引一次性取值。
        std_aas = "ACDEFGHIKLMNPQRSTVWY"
        propensities_table = np.array(
            [self.chou_fasman_propensities.get(aa, (0.0, 0.0, 1.0)) for aa in std_aas],
            dtype=np.float32,
        )
        aa_to_pos = {aa: i for i, aa in enumerate(std_aas)}

        if seq_len > 0:
            # 将序列映射为索引数组（非标准氨基酸映射到 -1，后置为缺省 coil）
            idx_arr = np.fromiter(
                (aa_to_pos.get(c, -1) for c in sequence[:seq_len]),
                dtype=np.int64,
                count=seq_len,
            )
            valid_mask = idx_arr >= 0
            if np.any(valid_mask):
                rows = propensities_table[idx_arr[valid_mask]]  # (n_valid, 3)
                totals = rows.sum(axis=1, keepdims=True)
                # 避免除零：total==0 时保持原值（0）
                safe = np.where(totals > 0, totals, 1.0)
                normalized = rows / safe
                structural[np.nonzero(valid_mask)[0]] = normalized

        return structural

    def extract_batch_structural_features(self, sequences: List[str]) -> np.ndarray:
        """批量结构特征（矩阵化，消除逐样本 Python 循环）。

        复用 :meth:`_extract_chou_fasman_structural` 的查表矩阵逻辑，对 alphafold/
        psipred 等外部来源仍逐样本调用（无法向量化），chou_fasman 来源则批量计算。
        返回形状 (n, max_sequence_length * 3) 的展平数组，与逐样本 flatten 拼接一致。
        """
        n = len(sequences)
        flat_features = np.zeros((n, self.max_sequence_length * 3), dtype=np.float32)

        # 外部来源无法向量化，逐样本调用
        if self.structural_source in {"alphafold", "psipred"}:
            for i, seq in enumerate(sequences):
                feat = self.extract_structural_features(seq)
                flat_features[i] = feat.flatten()
            return flat_features

        # chou_fasman：批量矩阵化计算
        std_aas = "ACDEFGHIKLMNPQRSTVWY"
        propensities_table = np.array(
            [self.chou_fasman_propensities.get(aa, (0.0, 0.0, 1.0)) for aa in std_aas],
            dtype=np.float32,
        )
        aa_to_pos = {aa: i for i, aa in enumerate(std_aas)}

        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), self.max_sequence_length)
            if seq_len == 0:
                continue
            idx_arr = np.fromiter(
                (aa_to_pos.get(c, -1) for c in seq[:seq_len]),
                dtype=np.int64,
                count=seq_len,
            )
            valid_mask = idx_arr >= 0
            if np.any(valid_mask):
                rows = propensities_table[idx_arr[valid_mask]]  # (n_valid, 3)
                totals = rows.sum(axis=1, keepdims=True)
                safe = np.where(totals > 0, totals, 1.0)
                normalized = rows / safe
                # 展平后写入：valid 位置对应原序列位置
                base_offset = np.nonzero(valid_mask)[0] * 3
                flat_features[i, base_offset[:, None] + np.arange(3)[None, :]] = normalized

        return flat_features

    def extract_batch_onehot_sequence(self, sequences: List[str]) -> np.ndarray:
        """批量 one-hot 编码（矩阵化，避免逐样本 Python 循环）。

        参数:
            sequences: 蛋白质序列列表

        返回:
            形状为 (n, max_sequence_length, num_amino_acids + 1) 的数组，
            第 0 列保留给 padding，实际氨基酸从第 1 列开始。
        """
        n = len(sequences)
        num_aa = len(self.amino_acids)
        onehot = np.zeros((n, self.max_sequence_length, num_aa + 1), dtype=np.float32)

        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), self.max_sequence_length)
            if seq_len == 0:
                continue
            idx_arr = np.fromiter(
                (self.aa_to_idx.get(c, -1) for c in seq[:seq_len]),
                dtype=np.int64,
                count=seq_len,
            )
            valid = idx_arr >= 0
            if np.any(valid):
                positions = np.nonzero(valid)[0]
                onehot[i, positions, idx_arr[valid]] = 1.0
        return onehot

    def extract_batch_kmer_features(self, sequences: List[str]) -> np.ndarray:
        """批量 k-mer 频率特征（向量化）。

        使用 collections.Counter 一次性统计每条序列的 k-mer，再批量写入向量，
        避免逐样本重复调用 extract_kmer_features 的开销。
        """
        from collections import Counter

        k = self.kmer_size
        base = len(self.amino_acids)
        n = len(sequences)
        feature_matrix = np.zeros((n, base ** k), dtype=np.float32)

        for i, seq in enumerate(sequences):
            if len(seq) < k:
                continue
            kmers = (seq[j:j + k] for j in range(len(seq) - k + 1))
            counts = Counter(kmers)
            total = 0
            for kmer, count in counts.items():
                idx = 0
                valid = True
                for aa in kmer:
                    aa_idx = self.aa_to_idx.get(aa)
                    if aa_idx is None:
                        valid = False
                        break
                    idx = idx * base + aa_idx
                if valid:
                    feature_matrix[i, idx] += float(count)
                    total += count
            if total > 0:
                feature_matrix[i] /= float(total)
        return feature_matrix

    def extract_sequence_features(self, sequences: List[str]) -> np.ndarray:
        """
        批量提取序列特征

        参数:
            sequences: 序列列表

        返回:
            特征数组
        """
        logger.info("提取 %d 条序列的特征", len(sequences))
        if not sequences:
            return np.array([])

        n = len(sequences)
        encoding = str(self.sequence_encoding).lower()

        # 预计算各特征分支的批量矩阵，最后按需拼接每条样本的对应切片
        batch_onehot = None
        if encoding in {"onehot", "both"}:
            batch_onehot = self.extract_batch_onehot_sequence(sequences)  # (n, L, C)

        batch_kmer = None
        if encoding in {"kmer", "both"}:
            batch_kmer = self.extract_batch_kmer_features(sequences)  # (n, K)

        batch_physchem = None
        if self.include_physicochemical:
            batch_physchem = self.extract_batch_physicochemical_features(sequences)  # (n, P)

        batch_structural = None
        if self.include_structural_features:
            batch_structural = self.extract_batch_structural_features(sequences)  # (n, S)

        # 按列堆叠各分支的预计算矩阵，消除逐样本拼接循环
        # onehot/kmer 若为 2D 则先展平为 (n, -1)，再与其他 1D 分支按列拼接
        parts = []
        if batch_onehot is not None:
            parts.append(batch_onehot.reshape(n, -1))
        if batch_kmer is not None:
            parts.append(batch_kmer)
        if batch_physchem is not None:
            parts.append(batch_physchem)
        if batch_structural is not None:
            parts.append(batch_structural)

        if not parts:
            logger.warning("未生成任何序列特征，请检查sequence_encoding配置")
            return np.array([])

        return np.concatenate(parts, axis=1)

    def _extract_sample_vector(
        self,
        sequence: str,
        ptm_sites_json: Optional[Union[str, List[Dict[str, Any]]]] = None,
    ) -> np.ndarray:
        """使用FeatureExtractor方法构建单个样本的向量。"""
        seq_features = self.extract_sequence_features([sequence])
        feature_dict: Dict[str, np.ndarray] = {}
        if seq_features.size > 0:
            feature_dict["sequence"] = seq_features[0]

        if self.include_ptm_features and ptm_sites_json is not None:
            ptm_sites = self._load_ptm_sites(ptm_sites_json)
            feature_dict["ptm"] = self.extract_ptm_features_array(ptm_sites, len(sequence))

        return self.combine_features(feature_dict)

    def extract_features_for_sample(
        self,
        sequence: str,
        ptm_sites_json: Optional[Union[str, List[Dict[str, Any]]]] = None,
    ) -> np.ndarray:
        """
        提取单个样本的组合特征

        参数:
            sequence: 蛋白质序列
            ptm_sites_json: PTM位点JSON字符串（可选）

        返回:
            组合后的特征向量
        """
        if self.use_feature_extractor:
            return self._extract_sample_vector(sequence, ptm_sites_json)

        seq_features = self.extract_sequence_features([sequence])
        feature_dict: Dict[str, np.ndarray] = {}
        if seq_features.size > 0:
            feature_dict["sequence"] = seq_features[0]

        if self.include_ptm_features and ptm_sites_json is not None:
            ptm_sites = self._load_ptm_sites(ptm_sites_json)
            feature_dict["ptm"] = self.extract_ptm_features_array(ptm_sites, len(sequence))

        return self.combine_features(feature_dict)

    def extract_features(
        self,
        sequences: List[str],
        ptm_sites_list: Optional[List[Optional[str]]] = None,
    ) -> np.ndarray:
        """
        批量提取组合特征

        参数:
            sequences: 序列列表
            ptm_sites_list: 与序列对应的PTM位点JSON字符串列表（可选）

        返回:
            特征数组
        """
        if ptm_sites_list is None:
            ptm_sites_list = [None] * len(sequences)

        features_list = []
        for sequence, ptm_sites_json in zip(sequences, ptm_sites_list):
            features_list.append(self._extract_sample_vector(sequence, ptm_sites_json))

        return np.array(features_list) if features_list else np.array([])

    def combine_features(self, feature_dicts: Dict[str, np.ndarray]) -> np.ndarray:
        """
        组合多种特征

        参数:
            feature_dicts: 特征字典

        返回:
            组合后的特征数组
        """
        features_list = []
        for _, value in feature_dicts.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    features_list.append(value)
                elif value.ndim == 2:
                    features_list.append(value.flatten())

        return np.concatenate(features_list) if features_list else np.array([])

    def reduce_pca(self, features: np.ndarray, n_components: int = 50) -> np.ndarray:
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, returning original features")
            return features
        n_components = min(n_components, features.shape[0], features.shape[1])
        if n_components <= 0:
            return features
        pca = PCA(n_components=n_components)
        return np.asarray(pca.fit_transform(features))

    def reduce_tsne(self, features: np.ndarray, n_components: int = 2, perplexity: int = 30) -> np.ndarray:
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, returning original features")
            return features
        n_components = min(n_components, features.shape[0] - 1, features.shape[1])
        if n_components <= 0:
            return features
        perplexity = min(perplexity, features.shape[0] - 1)
        tsne = TSNE(n_components=n_components, perplexity=perplexity)
        return np.asarray(tsne.fit_transform(features))

    def reduce_umap(self, features: np.ndarray, n_components: int = 50, n_neighbors: int = 15) -> np.ndarray:
        try:
            import umap
        except ImportError:
            logger.warning("umap-learn not installed, try: pip install umap-learn")
            return features
        n_components = min(n_components, features.shape[0] - 1, features.shape[1])
        if n_components <= 0:
            return features
        n_neighbors = min(n_neighbors, features.shape[0] - 1)
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors)
        return np.asarray(reducer.fit_transform(features))

    def select_features_variance(self, features: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        variances = np.var(features, axis=0)
        mask = variances > threshold
        if not np.any(mask):
            logger.warning("No features above variance threshold %f, returning all", threshold)
            return features
        return features[:, mask]

    def reduce_all(self, features: np.ndarray, methods: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        if methods is None:
            methods = ["pca", "tsne", "variance"]
        results = {}
        if "pca" in methods:
            results["pca"] = self.reduce_pca(features)
        if "tsne" in methods:
            results["tsne"] = self.reduce_tsne(features)
        if "umap" in methods:
            results["umap"] = self.reduce_umap(features)
        if "variance" in methods:
            results["variance"] = self.select_features_variance(features)
        return results
