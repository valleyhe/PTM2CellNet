"""
特征工程模块
功能概述: 从蛋白质序列和PTM数据中提取特征
设计思路: 支持多种特征提取方法，可灵活组合使用
"""

import json
from typing import Dict, List, Optional

import numpy as np

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

DEFAULT_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_PTM_TYPES = [
    "phosphorylation",
    "acetylation",
    "methylation",
    "ubiquitination",
    "sumoylation",
]

AMINO_ACIDS = DEFAULT_AMINO_ACIDS
# 序列索引从1开始，0保留给padding
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
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
        self.max_sequence_length = self.config.get("data", {}).get("max_sequence_length", 1000)
        self.amino_acids = self.config.get("data", {}).get("valid_amino_acids", DEFAULT_AMINO_ACIDS)
        # 序列索引从1开始，0保留给padding
        self.aa_to_idx = {aa: i + 1 for i, aa in enumerate(self.amino_acids)}
        self.ptm_types = self.config.get("data", {}).get("ptm_types", DEFAULT_PTM_TYPES)
        self.ptm_to_idx = {ptm: i for i, ptm in enumerate(self.ptm_types)}

    def extract_onehot_sequence(self, sequence: str) -> np.ndarray:
        """
        提取序列的one-hot编码

        参数:
            sequence: 蛋白质序列

        返回:
            one-hot编码数组，形状为 (max_len, num_amino_acids + 1)
            第0列保留给padding，实际氨基酸从第1列开始
        """
        seq_len = min(len(sequence), self.max_sequence_length)
        # +1 for padding column (index 0)
        onehot = np.zeros((self.max_sequence_length, len(self.amino_acids) + 1), dtype=np.float32)

        for i in range(seq_len):
            aa = sequence[i]
            if aa in self.aa_to_idx:
                onehot[i, self.aa_to_idx[aa]] = 1.0

        return onehot

    def extract_kmer_features(self, sequence: str, k: Optional[int] = None) -> np.ndarray:
        """
        提取k-mer频率特征

        参数:
            sequence: 蛋白质序列
            k: k-mer大小，默认使用配置中的值

        返回:
            k-mer频率向量
        """
        if k is None:
            k = self.kmer_size

        base = len(self.amino_acids)
        feature_vector = np.zeros(base ** k, dtype=np.float32)

        total = 0
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i + k]
            idx = 0
            valid = True
            for aa in kmer:
                aa_idx = self.aa_to_idx.get(aa)
                if aa_idx is None:
                    valid = False
                    break
                idx = idx * base + aa_idx
            if valid:
                feature_vector[idx] += 1.0
                total += 1

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

    def extract_ptm_features(self, ptm_sites_json: str, sequence_length: int) -> Dict[str, np.ndarray]:
        """
        提取PTM特征

        参数:
            ptm_sites_json: PTM位点JSON字符串
            sequence_length: 序列长度

        返回:
            PTM特征数组
        """
        try:
            ptm_sites = json.loads(ptm_sites_json)
        except (json.JSONDecodeError, TypeError):
            ptm_sites = []

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

    def extract_sequence_features(self, sequences: List[str]) -> np.ndarray:
        """
        批量提取序列特征

        参数:
            sequences: 序列列表

        返回:
            特征数组
        """
        logger.info("提取 %d 条序列的特征", len(sequences))
        features_list = []

        encoding = str(self.sequence_encoding).lower()
        for seq in sequences:
            seq_features = []

            if encoding in {"onehot", "both"}:
                onehot = self.extract_onehot_sequence(seq)
                seq_features.append(onehot.flatten())

            if encoding in {"kmer", "both"}:
                kmer = self.extract_kmer_features(seq)
                seq_features.append(kmer)

            if self.include_physicochemical:
                physchem = self.extract_physicochemical_features(seq)
                seq_features.append(physchem)

            if seq_features:
                combined = np.concatenate(seq_features)
                features_list.append(combined)

        if not features_list:
            logger.warning("未生成任何序列特征，请检查sequence_encoding配置")
            return np.array([])

        return np.array(features_list)

    def extract_features_for_sample(self, sequence: str, ptm_sites_json: Optional[str] = None) -> np.ndarray:
        """
        提取单个样本的组合特征

        参数:
            sequence: 蛋白质序列
            ptm_sites_json: PTM位点JSON字符串（可选）

        返回:
            组合后的特征向量
        """
        seq_features = self.extract_sequence_features([sequence])
        feature_dict: Dict[str, np.ndarray] = {}
        if seq_features.size > 0:
            feature_dict["sequence"] = seq_features[0]

        if self.include_ptm_features and ptm_sites_json is not None:
            ptm_features = self.extract_ptm_features(ptm_sites_json, len(sequence))
            feature_dict.update(ptm_features)

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
            features_list.append(self.extract_features_for_sample(sequence, ptm_sites_json))

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
