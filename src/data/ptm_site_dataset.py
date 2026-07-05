"""
PTM位点预测数据集
功能: 用于PTM位点二分类预测的数据集类
输入: 序列窗口 + 位点位置 + 标签
"""

import logging
from typing import Dict, Optional, Union
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L

logger = logging.getLogger(__name__)

from src.data.aa_constants import (
    AMINO_ACIDS,
    AA_TO_IDX,
    AA_PAD_CHAR,
)


class PTMSiteDataset(Dataset):
    """
    PTM位点预测数据集

    输入数据格式 (CSV):
    - uniprot_id: UniProt accession
    - position: 修饰位置 (1-based)
    - aa: 修饰残基类型
    - sequence_window: 序列窗口 (中心为修饰位点)
    - label: 标签 (1=正样本, 0=负样本)
    - ptm_type: PTM类型
    """

    def __init__(
        self,
        csv_path: str,
        ptm_type: Optional[str] = None,
        window_size: int = 31,
        max_samples: Optional[int] = None,
        balance: bool = False,
        seed: int = 42,
    ):
        """
        初始化数据集

        参数:
            csv_path: CSV文件路径
            ptm_type: 筛选特定PTM类型 (None则加载全部)
            window_size: 序列窗口大小 (应为奇数，中心为修饰位点)
            max_samples: 最大样本数 (用于调试)
            balance: 是否平衡正负样本
            seed: 随机种子
        """
        self.window_size = window_size
        self.csv_path = csv_path

        # 加载数据
        df = pd.read_csv(csv_path)

        # 筛选PTM类型
        if ptm_type is not None:
            df = df[df['ptm_type'] == ptm_type]

        # 限制样本数
        if max_samples is not None and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=seed)

        # 平衡正负样本
        if balance:
            pos_df = df[df['label'] == 1]
            neg_df = df[df['label'] == 0]
            min_count = min(len(pos_df), len(neg_df))
            pos_df = pos_df.sample(n=min_count, random_state=seed)
            neg_df = neg_df.sample(n=min_count, random_state=seed)
            df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=seed)

        self.df = df.reset_index(drop=True)

        # 统计信息
        self.pos_count = (self.df['label'] == 1).sum()
        self.neg_count = (self.df['label'] == 0).sum()

        logger.info("加载数据集: %s", csv_path)
        logger.info("总样本数: %d", len(self.df))
        logger.info("正样本: %d, 负样本: %d", self.pos_count, self.neg_count)

    def __len__(self) -> int:
        return len(self.df)

    def _encode_sequence(self, sequence: str) -> torch.Tensor:
        """
        将序列窗口编码为one-hot tensor

        参数:
            sequence: 氨基酸序列窗口

        返回:
            (window_size, num_amino_acids + 1) 的one-hot tensor
            第0列保留给padding，实际氨基酸从第1列开始
        """
        # 确保序列长度正确
        if len(sequence) < self.window_size:
            sequence = sequence + AA_PAD_CHAR * (self.window_size - len(sequence))
        elif len(sequence) > self.window_size:
            # 居中截取
            start = (len(sequence) - self.window_size) // 2
            sequence = sequence[start:start + self.window_size]

        # One-hot编码: +1 for padding column (index 0)
        onehot = torch.zeros(self.window_size, len(AMINO_ACIDS) + 1, dtype=torch.float32)
        for i, aa in enumerate(sequence):
            if aa in AA_TO_IDX:
                onehot[i, AA_TO_IDX[aa]] = 1.0
            # 未知氨基酸编码为全零（padding列也为0）

        return onehot

    def _encode_sequence_indices(self, sequence: str) -> torch.Tensor:
        """
        将序列窗口编码为索引tensor (用于embedding)

        参数:
            sequence: 氨基酸序列窗口

        返回:
            (window_size,) 的索引tensor
        """
        if len(sequence) < self.window_size:
            sequence = sequence + '-' * (self.window_size - len(sequence))
        elif len(sequence) > self.window_size:
            start = (len(sequence) - self.window_size) // 2
            sequence = sequence[start:start + self.window_size]

        indices = torch.zeros(self.window_size, dtype=torch.long)
        for i, aa in enumerate(sequence):
            if aa in AA_TO_IDX:
                indices[i] = AA_TO_IDX[aa]
            else:
                indices[i] = 20  # 未知氨基酸

        return indices

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        返回:
            dict:
                - sequence_onehot: (window_size, 21) one-hot编码
                - sequence_indices: (window_size,) 索引编码
                - label: 标签 (0或1)
                - position: 位点位置
                - aa: 中心残基索引
        """
        row = self.df.iloc[idx]

        sequence_window = row['sequence_window']
        label = int(row['label'])
        position = int(row['position'])
        aa = row['aa']

        return {
            'sequence_onehot': self._encode_sequence(sequence_window),
            'sequence_indices': self._encode_sequence_indices(sequence_window),
            'label': torch.tensor(label, dtype=torch.long),
            'position': torch.tensor(position, dtype=torch.long),
            'aa_idx': torch.tensor(AA_TO_IDX.get(aa, 20), dtype=torch.long),
        }


class PTMSiteDataModule(L.LightningDataModule):
    """
    PTM位点预测数据模块
    管理训练/验证/测试数据集
    """

    def __init__(
        self,
        train_path: str,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        ptm_type: Optional[str] = None,
        window_size: int = 31,
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
    ):
        """
        初始化数据模块

        参数:
            train_path: 训练数据CSV路径
            val_path: 验证数据CSV路径 (None则从训练数据划分)
            test_path: 测试数据CSV路径 (None则从训练数据划分)
            ptm_type: PTM类型筛选
            window_size: 序列窗口大小
            batch_size: 批次大小
            num_workers: 数据加载进程数
            val_split: 验证集比例
            test_split: 测试集比例
            seed: 随机种子
        """
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.ptm_type = ptm_type
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        self.train_dataset: Optional[Union[PTMSiteDataset, Dataset]] = None
        self.val_dataset: Optional[Union[PTMSiteDataset, Dataset]] = None
        self.test_dataset: Optional[Union[PTMSiteDataset, Dataset]] = None
        self._full_dataset: Optional[PTMSiteDataset] = None

    def setup(self, stage: Optional[str] = None):
        """准备数据集"""
        if self.val_path and self.test_path:
            # 使用单独的数据文件
            self.train_dataset = PTMSiteDataset(
                self.train_path, self.ptm_type, self.window_size
            )
            self.val_dataset = PTMSiteDataset(
                self.val_path, self.ptm_type, self.window_size
            )
            self.test_dataset = PTMSiteDataset(
                self.test_path, self.ptm_type, self.window_size
            )
        else:
            # 从单个文件划分
            if self._full_dataset is None:
                self._full_dataset = PTMSiteDataset(
                    self.train_path, self.ptm_type, self.window_size
                )

            total_size = len(self._full_dataset)
            test_size = int(total_size * self.test_split)
            val_size = int(total_size * self.val_split)
            train_size = total_size - test_size - val_size

            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self._full_dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.seed)
            )

            logger.info("数据划分: train=%d, val=%d, test=%d", train_size, val_size, test_size)

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "train_dataset not initialized — call setup() first"
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None, "val_dataset not initialized — call setup() first"
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None, "test_dataset not initialized — call setup() first"
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
