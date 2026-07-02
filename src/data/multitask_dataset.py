# mypy: ignore-errors
"""
多任务PTM数据集
功能: 同时加载多种PTM类型的训练数据
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import logging
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


class MultiTaskPTMDataset(Dataset):
    """
    多任务PTM数据集

    同时处理多种PTM类型的样本
    """

    AA_TO_IDX = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
        'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
        'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
        '-': 20,  # padding
    }

    PTM_TO_IDX = {
        'Phosphorylation': 0,
        'Acetylation': 1,
        'Ubiquitination': 2,
        'Methylation': 3,
        'Sumoylation': 4,
        'Succinylation': 5,
    }

    def __init__(
        self,
        data_files: Dict[str, str],
        window_size: int = 15,
        max_samples_per_type: Optional[int] = None,
        balance_types: bool = True,
        augment_minority: bool = True,
    ):
        """
        初始化数据集

        参数:
            data_files: PTM类型到数据文件路径的映射
            window_size: 序列窗口大小（每侧）
            max_samples_per_type: 每种PTM类型的最大样本数
            balance_types: 是否平衡各类型样本数
            augment_minority: 是否增强少数类
        """
        self.window_size = window_size
        self.max_samples_per_type = max_samples_per_type
        self.balance_types = balance_types
        self.augment_minority = augment_minority

        # 加载所有数据
        self.samples = []
        self.ptm_types = list(data_files.keys())

        for ptm_type, file_path in data_files.items():
            df = pd.read_csv(file_path)
            logger.info(f"加载 {ptm_type}: {len(df)} 样本")

            # 限制样本数
            if max_samples_per_type and len(df) > max_samples_per_type:
                # 平衡正负样本
                pos_df = df[df['label'] == 1]
                neg_df = df[df['label'] == 0]
                n_per_class = max_samples_per_type // 2

                if len(pos_df) > n_per_class:
                    pos_df = pos_df.sample(n_per_class, random_state=42)
                if len(neg_df) > n_per_class:
                    neg_df = neg_df.sample(n_per_class, random_state=42)

                df = pd.concat([pos_df, neg_df])

            for _, row in df.iterrows():
                self.samples.append({
                    'uniprot_id': row['uniprot_id'],
                    'position': row['position'],
                    'aa': row['aa'],
                    'sequence_window': row['sequence_window'],
                    'label': int(row['label']),
                    'ptm_type': ptm_type,
                })

        # 打乱
        random.shuffle(self.samples)

        # 统计
        self.type_counts = defaultdict(int)
        for s in self.samples:
            self.type_counts[s['ptm_type']] += 1

        logger.info(f"总样本数: {len(self.samples)}")
        for ptm, count in self.type_counts.items():
            logger.info(f"  {ptm}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 编码序列
        seq_indices = self._encode_sequence(sample['sequence_window'])

        # PTM类型索引
        ptm_idx = self.PTM_TO_IDX.get(sample['ptm_type'], 0)

        return {
            'sequence_indices': torch.tensor(seq_indices, dtype=torch.long),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'ptm_type': sample['ptm_type'],
            'ptm_idx': torch.tensor(ptm_idx, dtype=torch.long),
        }

    def _encode_sequence(self, sequence: str) -> List[int]:
        """编码序列为索引"""
        return [self.AA_TO_IDX.get(aa, 20) for aa in sequence]

    def get_type_weights(self) -> Dict[str, float]:
        """计算类型权重（用于平衡）"""
        total = len(self.samples)
        weights = {}

        for ptm_type in self.ptm_types:
            count = self.type_counts[ptm_type]
            # 逆频率权重
            weights[ptm_type] = total / (len(self.ptm_types) * count) if count > 0 else 1.0

        return weights


class MultiTaskPTMDataModule:
    """
    多任务PTM数据模块

    管理训练/验证/测试数据划分
    """

    def __init__(
        self,
        data_dir: str,
        ptm_types: List[str],
        window_size: int = 15,
        batch_size: int = 256,
        max_samples_per_type: Optional[int] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        num_workers: int = 4,
        collate_fn: Optional[callable] = None,
    ):
        """
        初始化数据模块

        参数:
            data_dir: 数据目录
            ptm_types: PTM类型列表
            window_size: 序列窗口大小
            batch_size: 批大小
            max_samples_per_type: 每种类型最大样本数
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            num_workers: 数据加载线程数
            collate_fn: 自定义批处理函数。默认使用
                :func:`collate_multitask_batch`，以支持按 PTM 类型分组并
                输出 ``labels`` 字典（多任务训练脚本依赖该结构）。
        """
        self.data_dir = Path(data_dir)
        self.ptm_types = ptm_types
        self.window_size = window_size
        self.batch_size = batch_size
        self.max_samples_per_type = max_samples_per_type
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        # 默认使用多任务批处理函数；允许传入 None 回退到 PyTorch 默认 collate
        self.collate_fn = collate_fn if collate_fn is not None else collate_multitask_batch

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        """准备数据"""
        # 查找数据文件
        data_files = {}
        for ptm_type in self.ptm_types:
            file_path = self.data_dir / f"ptm_train_{ptm_type.lower()}.csv"
            if file_path.exists():
                data_files[ptm_type] = str(file_path)
            else:
                logger.warning(f"数据文件不存在: {file_path}")

        if not data_files:
            raise FileNotFoundError(f"未找到任何数据文件在 {self.data_dir}")

        # 加载全部数据
        all_samples = []
        for ptm_type, file_path in data_files.items():
            df = pd.read_csv(file_path)

            # 限制样本数
            if self.max_samples_per_type and len(df) > self.max_samples_per_type:
                pos_df = df[df['label'] == 1].sample(
                    self.max_samples_per_type // 2, random_state=42
                )
                neg_df = df[df['label'] == 0].sample(
                    self.max_samples_per_type // 2, random_state=42
                )
                df = pd.concat([pos_df, neg_df])

            for _, row in df.iterrows():
                all_samples.append({
                    'uniprot_id': row['uniprot_id'],
                    'position': row['position'],
                    'aa': row['aa'],
                    'sequence_window': row['sequence_window'],
                    'label': int(row['label']),
                    'ptm_type': ptm_type,
                })

        # 打乱
        random.shuffle(all_samples)

        # 划分
        n_total = len(all_samples)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        train_samples = all_samples[:n_train]
        val_samples = all_samples[n_train:n_train + n_val]
        test_samples = all_samples[n_train + n_val:]

        # 创建数据集
        self.train_dataset = SampleDataset(train_samples, self.window_size)
        self.val_dataset = SampleDataset(val_samples, self.window_size)
        self.test_dataset = SampleDataset(test_samples, self.window_size)

        logger.info(f"数据划分: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )


class SampleDataset(Dataset):
    """简单样本数据集"""

    AA_TO_IDX = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
        'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
        'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
        '-': 20,
    }

    PTM_TO_IDX = {
        'Phosphorylation': 0,
        'Acetylation': 1,
        'Ubiquitination': 2,
        'Methylation': 3,
        'Sumoylation': 4,
        'Succinylation': 5,
    }

    def __init__(self, samples, window_size=15):
        self.samples = samples
        self.window_size = window_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        seq_indices = [self.AA_TO_IDX.get(aa, 20) for aa in sample['sequence_window']]
        ptm_idx = self.PTM_TO_IDX.get(sample['ptm_type'], 0)

        return {
            'sequence_indices': torch.tensor(seq_indices, dtype=torch.long),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'ptm_type': sample['ptm_type'],
            'ptm_idx': torch.tensor(ptm_idx, dtype=torch.long),
        }


def collate_multitask_batch(batch):
    """
    自定义批处理函数

    按PTM类型分组，返回字典格式
    """
    # 按PTM类型分组
    type_batches = defaultdict(list)

    for item in batch:
        type_batches[item['ptm_type']].append(item)

    # 构建输出
    outputs = {
        'sequence_indices': [],
        'labels': {},
        'ptm_types': [],
        'ptm_indices': [],
    }

    for ptm_type, items in type_batches.items():
        for item in items:
            outputs['sequence_indices'].append(item['sequence_indices'])
            outputs['ptm_types'].append(ptm_type)
            outputs['ptm_indices'].append(item['ptm_idx'])

            if ptm_type not in outputs['labels']:
                outputs['labels'][ptm_type] = []
            outputs['labels'][ptm_type].append(item['label'])

    # 堆叠
    outputs['sequence_indices'] = torch.stack(outputs['sequence_indices'])
    outputs['ptm_indices'] = torch.stack(outputs['ptm_indices'])

    for ptm_type in outputs['labels']:
        outputs['labels'][ptm_type] = torch.stack(outputs['labels'][ptm_type])

    return outputs
