"""
PyTorch数据集模块
功能概述: 定义PyTorch Dataset和DataModule
设计思路: 基于PyTorch Lightning DataModule，支持批量加载和多进程
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..utils.logging import setup_logger
from .features import FeatureExtractor, DEFAULT_AMINO_ACIDS
from .dataset_base import PTMDatasetBase

logger = setup_logger(__name__)


class PTMDataset(PTMDatasetBase):
    """
    PTM数据集类
    PyTorch Dataset实现，用于加载蛋白质PTM数据（使用自定义序列编码）
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_extractor: Optional[FeatureExtractor] = None,
        config: Optional[Dict[str, Any]] = None,
        return_sequence: bool = True,
        return_ptm: bool = True,
        return_label: bool = True,
    ):
        """
        初始化数据集

        参数:
            df: 数据DataFrame
            feature_extractor: 特征提取器实例
            config: 配置字典
            return_sequence: 是否返回序列数据
            return_ptm: 是否返回PTM数据
            return_label: 是否返回标签
        """
        super().__init__(df, config)
        self.feature_extractor = feature_extractor or FeatureExtractor(config)
        self.return_sequence = return_sequence
        self.return_ptm = return_ptm
        self.return_label = return_label

        self.max_sequence_length = self.config.get("data", {}).get("max_sequence_length", 1000)
        self.amino_acids = self.config.get("data", {}).get("valid_amino_acids", DEFAULT_AMINO_ACIDS)
        # 序列索引从1开始，0保留给padding
        self.aa_to_idx = {aa: i + 1 for i, aa in enumerate(self.amino_acids)}

        # 非标准氨基酸字符映射表 (与preprocess.py保持一致)
        self.non_standard_aa_map = {
            'U': 'C', 'X': 'A', 'J': 'L', 'B': 'D', 'Z': 'E', 'O': 'K',
        }

    def _standardize_sequence(self, sequence: str) -> str:
        """将非标准氨基酸字符映射为标准字符"""
        for old_char, new_char in self.non_standard_aa_map.items():
            sequence = sequence.replace(old_char, new_char)
        return sequence

    def _encode_sequence(self, sequence: str) -> torch.Tensor:
        """编码序列为tensor"""
        # 先标准化非标准氨基酸字符
        sequence = self._standardize_sequence(sequence)
        seq_len = min(len(sequence), self.max_sequence_length)
        seq_tensor = torch.zeros(self.max_sequence_length, dtype=torch.long)

        for i in range(seq_len):
            aa = sequence[i]
            if aa in self.aa_to_idx:
                seq_tensor[i] = self.aa_to_idx[aa]

        return seq_tensor

    def _encode_ptm(self, ptm_sites_json: str, sequence_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码PTM位点"""
        ptm_sites = self._parse_ptm_sites(ptm_sites_json)
        ptm_mask = torch.zeros(self.max_sequence_length, dtype=torch.float32)
        ptm_types = torch.zeros(self.max_sequence_length, dtype=torch.long)
        max_len = min(self.max_sequence_length, sequence_length) if sequence_length else self.max_sequence_length

        for site in ptm_sites:
            is_valid, error_msg = self._validate_ptm_site(site, max_len)
            if not is_valid:
                logger.debug(f"无效的PTM位点: {error_msg}")
                continue

            pos = site["position"] - 1  # 转为0-based
            ptm_type = site.get("type", "")

            ptm_mask[pos] = 1.0
            ptm_types[pos] = self.ptm_type_to_idx.get(ptm_type, 0)

        return ptm_mask, ptm_types

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        row = self.df.iloc[idx]
        sample = {}

        if self.return_sequence and "sequence" in row:
            sequence = str(row["sequence"])
            sample["sequence"] = self._encode_sequence(sequence)
            sample["sequence_length"] = torch.tensor(min(len(sequence), self.max_sequence_length), dtype=torch.long)

        if self.return_ptm and "ptm_sites" in row:
            seq_len = len(str(row.get("sequence", "")))
            ptm_mask, ptm_types = self._encode_ptm(str(row["ptm_sites"]), seq_len)
            sample["ptm_mask"] = ptm_mask
            sample["ptm_types"] = ptm_types

        if self.return_label and "cell_state" in row:
            sample["label"] = self._encode_label(str(row["cell_state"]))

        return sample


class PTMDataModule:
    """
    PTM数据模块类
    管理训练、验证和测试数据集的加载
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        """
        初始化数据模块

        参数:
            train_df: 训练集DataFrame
            val_df: 验证集DataFrame
            test_df: 测试集DataFrame
            config: 配置字典
            batch_size: 批次大小
            num_workers: 数据加载进程数
        """
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.config = config or {}
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.feature_extractor = FeatureExtractor(config)
        self.train_dataset: Optional[PTMDataset] = None
        self.val_dataset: Optional[PTMDataset] = None
        self.test_dataset: Optional[PTMDataset] = None
        self.labels: List[str] = []
        self.label_to_idx: Dict[str, int] = {}

        self._setup_datasets()

    def _setup_datasets(self) -> None:
        """设置数据集"""
        self.train_dataset = PTMDataset(
            self.train_df,
            self.feature_extractor,
            self.config,
        )

        if "cell_state" in self.train_df.columns:
            self.labels = self.train_dataset.labels
            self.label_to_idx = self.train_dataset.label_to_idx

        if self.val_df is not None:
            self.val_dataset = PTMDataset(
                self.val_df,
                self.feature_extractor,
                self.config,
            )
            # 传播训练集的标签映射到验证集，确保编码一致
            if self.label_to_idx:
                self.val_dataset.label_to_idx = self.label_to_idx
                self.val_dataset.labels = self.labels

        if self.test_df is not None:
            self.test_dataset = PTMDataset(
                self.test_df,
                self.feature_extractor,
                self.config,
            )
            # 传播训练集的标签映射到测试集，确保编码一致
            if self.label_to_idx:
                self.test_dataset.label_to_idx = self.label_to_idx
                self.test_dataset.labels = self.labels

        logger.info(
            "数据集设置完成: train=%d, val=%d, test=%d",
            len(self.train_dataset),
            len(self.val_dataset) if self.val_dataset else 0,
            len(self.test_dataset) if self.test_dataset else 0,
        )

    def train_dataloader(self) -> DataLoader[Dict[str, torch.Tensor]]:
        """获取训练数据加载器"""
        if self.train_dataset is None:
            raise ValueError("训练数据集未初始化")

        # 从配置读取优化参数
        persistent_workers = self.config.get("data", {}).get("persistent_workers", False)
        pin_memory = self.config.get("data", {}).get("pin_memory", False)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            persistent_workers=persistent_workers if self.num_workers > 0 else False,
            pin_memory=pin_memory,
        )

    def val_dataloader(self) -> Optional[DataLoader[Dict[str, torch.Tensor]]]:
        """获取验证数据加载器"""
        if self.val_dataset is None:
            return None

        pin_memory = self.config.get("data", {}).get("pin_memory", False)

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0 and self.config.get("data", {}).get("persistent_workers", False),
            pin_memory=pin_memory,
        )

    def test_dataloader(self) -> Optional[DataLoader[Dict[str, torch.Tensor]]]:
        """获取测试数据加载器"""
        if self.test_dataset is None:
            return None

        pin_memory = self.config.get("data", {}).get("pin_memory", False)

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0 and self.config.get("data", {}).get("persistent_workers", False),
            pin_memory=pin_memory,
        )

    def get_labels(self) -> List[str]:
        """获取标签列表"""
        return self.labels

    def get_num_classes(self) -> int:
        """获取类别数量"""
        return len(self.labels)


class ESMTokenizedDataset(PTMDatasetBase):
    """
    使用ESM tokenizer编码的PTM数据集
    正确处理ESM tokenizer的特殊token偏移（<cls>在位置0，氨基酸从位置1开始）
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 1024,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化ESMTokenizedDataset

        参数:
            df: 数据DataFrame，需含sequence、ptm_sites（可选）、cell_state（可选）列
            tokenizer: ESM tokenizer实例（来自ESM2Encoder.tokenizer）
            max_length: tokenizer最大长度（含特殊token），默认1024
            config: 配置字典
        """
        super().__init__(df, config)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 非标准氨基酸字符映射表 (与PTMDataset保持一致)
        self.non_standard_aa_map = {
            'U': 'C', 'X': 'A', 'J': 'L', 'B': 'D', 'Z': 'E', 'O': 'K',
        }

    def _standardize_sequence(self, sequence: str) -> str:
        """将非标准氨基酸字符映射为标准字符"""
        for old_char, new_char in self.non_standard_aa_map.items():
            sequence = sequence.replace(old_char, new_char)
        return sequence

    def _tokenize_sequence(self, sequence: str):
        """使用ESM tokenizer编码序列，返回input_ids和attention_mask（1D tensors）"""
        # 先标准化非标准氨基酸字符
        sequence = self._standardize_sequence(sequence)
        encoded = self.tokenizer(
            [sequence],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        return encoded["input_ids"][0], encoded["attention_mask"][0]

    def _encode_ptm_esm(
        self, ptm_sites_json: str, tokenized_length: int
    ) -> tuple:
        """
        编码PTM位点，位置偏移+1以对齐ESM tokenized序列（跳过<cls>）

        参数:
            ptm_sites_json: PTM位点JSON字符串
            tokenized_length: tokenized序列总长度（含<cls>和<eos>）

        返回:
            (ptm_mask, ptm_types) 长度均为tokenized_length的tensor
        """
        ptm_sites = self._parse_ptm_sites(ptm_sites_json)
        ptm_mask = torch.zeros(tokenized_length, dtype=torch.float32)
        ptm_types = torch.zeros(tokenized_length, dtype=torch.long)

        for site in ptm_sites:
            is_valid, error_msg = self._validate_ptm_site(site)
            if not is_valid:
                logger.debug(f"无效的PTM位点: {error_msg}")
                continue

            raw_pos_0based = site["position"] - 1  # 转为0-based
            tokenized_pos = raw_pos_0based + 1  # 跳过<cls>
            ptm_type = site.get("type", "")

            if 0 < tokenized_pos < tokenized_length - 1:  # 排除<cls>和<eos>位置
                ptm_mask[tokenized_pos] = 1.0
                ptm_types[tokenized_pos] = self.ptm_type_to_idx.get(ptm_type, 0)

        return ptm_mask, ptm_types

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        返回:
            包含 input_ids, attention_mask, ptm_mask, ptm_types, label（可选）的字典
        """
        row = self.df.iloc[idx]
        sequence = str(row.get("sequence", ""))

        input_ids, attention_mask = self._tokenize_sequence(sequence)
        tokenized_length = input_ids.shape[0]

        ptm_sites_json = str(row["ptm_sites"]) if "ptm_sites" in row else "[]"
        ptm_mask, ptm_types = self._encode_ptm_esm(ptm_sites_json, tokenized_length)

        sample: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ptm_mask": ptm_mask,
            "ptm_types": ptm_types,
        }

        if "cell_state" in row:
            sample["label"] = self._encode_label(str(row["cell_state"]))

        return sample
