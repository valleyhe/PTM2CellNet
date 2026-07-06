"""
数据增强模块
功能概述: 提供蛋白质序列的数据增强技术
设计思路: 支持多种增强策略，包括序列截断、随机mask等
"""
import random
from typing import Dict, List, Optional, Tuple, Any

import torch

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class SequenceAugmenter:
    """
    序列数据增强器
    提供多种蛋白质序列增强技术
    """

    def __init__(
        self,
        augment_prob: float = 0.5,
        max_truncate_ratio: float = 0.1,
        mask_token_id: int = 0,
        mask_prob: float = 0.0,
        random_swap_prob: float = 0.0,
    ):
        """
        初始化序列增强器

        参数:
            augment_prob: 应用增强的概率
            max_truncate_ratio: 最大截断比例
            mask_token_id: mask的token ID
            mask_prob: 随机mask的概率
            random_swap_prob: 随机交换氨基酸的概率
        """
        self.augment_prob = augment_prob
        self.max_truncate_ratio = max_truncate_ratio
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob
        self.random_swap_prob = random_swap_prob

    def __call__(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        对序列应用增强

        参数:
            sequence: 输入序列tensor

        返回:
            增强后的序列
        """
        if random.random() > self.augment_prob:
            return sequence

        seq = sequence.clone()

        # 随机截断（从头部或尾部）
        if random.random() < 0.5:
            seq = self._random_truncate(seq)

        # 随机mask
        if self.mask_prob > 0:
            seq = self._random_mask(seq)

        # 随机交换
        if self.random_swap_prob > 0:
            seq = self._random_swap(seq)

        return seq

    def _random_truncate(self, sequence: torch.Tensor) -> torch.Tensor:
        """随机截断序列头部或尾部"""
        seq_len = int((sequence != 0).sum().item())
        if seq_len <= 1:
            return sequence

        max_truncate = max(1, int(seq_len * self.max_truncate_ratio))
        truncate_len = random.randint(1, max_truncate)

        if random.random() < 0.5:
            # 截断头部
            truncated = sequence[truncate_len:]
            # 补齐长度
            result = torch.zeros_like(sequence)
            result[:len(truncated)] = truncated
        else:
            # 截断尾部
            result = sequence.clone()
            result[seq_len - truncate_len:seq_len] = 0

        return result

    def _random_mask(self, sequence: torch.Tensor) -> torch.Tensor:
        """随机mask氨基酸"""
        mask = torch.rand_like(sequence.float()) < self.mask_prob
        result = sequence.clone()
        result[mask] = self.mask_token_id
        return result

    def _random_swap(self, sequence: torch.Tensor) -> torch.Tensor:
        """随机交换相邻氨基酸"""
        seq_len = int((sequence != 0).sum().item())
        if seq_len < 2:
            return sequence

        result = sequence.clone()
        for i in range(seq_len - 1):
            if random.random() < self.random_swap_prob:
                result[i], result[i+1] = result[i+1].clone(), result[i].clone()

        return result


class PTMAugmenter:
    """
    PTM数据增强器
    对PTM位点信息进行增强（用于训练鲁棒性）
    """

    def __init__(
        self,
        drop_prob: float = 0.0,
        add_noise_prob: float = 0.0,
        noise_radius: int = 1,
    ):
        """
        初始化PTM增强器

        参数:
            drop_prob: 随机丢弃PTM的概率
            add_noise_prob: 添加假阳性PTM的概率
            noise_radius: 噪声位置范围（±radius）
        """
        self.drop_prob = drop_prob
        self.add_noise_prob = add_noise_prob
        self.noise_radius = noise_radius

    def __call__(
        self,
        ptm_mask: torch.Tensor,
        ptm_types: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对PTM信息进行增强

        参数:
            ptm_mask: PTM掩码
            ptm_types: PTM类型

        返回:
            增强后的 (ptm_mask, ptm_types)
        """
        mask = ptm_mask.clone()
        types = ptm_types.clone()

        # 随机丢弃PTM
        if self.drop_prob > 0:
            drop_mask = torch.rand_like(mask) < self.drop_prob
            mask[drop_mask] = 0
            types[drop_mask] = 0

        # 添加假阳性（在已有PTM附近）
        if self.add_noise_prob > 0:
            ptm_positions = torch.where(mask > 0)[0]
            for pos in ptm_positions:
                if random.random() < self.add_noise_prob:
                    offset = random.randint(-self.noise_radius, self.noise_radius)
                    new_pos = pos + offset
                    if 0 < new_pos < len(mask) and mask[new_pos] == 0:
                        mask[new_pos] = 1.0
                        types[new_pos] = types[pos].item()  # 复制相同类型

        return mask, types


class DAVFSiteAugmenter:
    """DAVF 位点协同增强器。

    对 ``davf_sites`` / ``davf_type_names`` / ``davf_attention_mask`` /
    ``davf_gene_names`` 四个变长 Python list 字段做协同增强：

    - 随机丢弃部分位点（同步删除四个列表对应索引）
    - 随机扰动位置（±noise_radius），扰动后的 attention_mask 同步保持为 1

    这些字段在 collate 阶段统一 padding，增强后保持四者长度一致。
    """

    def __init__(
        self,
        drop_prob: float = 0.0,
        noise_prob: float = 0.0,
        noise_radius: int = 1,
    ):
        """
        参数:
            drop_prob: 随机丢弃 DAVF 位点的概率
            noise_prob: 对保留位点添加位置噪声的概率
            noise_radius: 位置噪声范围（±radius）
        """
        self.drop_prob = drop_prob
        self.noise_prob = noise_prob
        self.noise_radius = noise_radius

    def __call__(
        self,
        davf_sites: List[int],
        davf_type_names: List[str],
        davf_attention_mask: List[int],
        davf_gene_names: Optional[List[str]] = None,
        max_position: Optional[int] = None,
    ) -> Tuple[List[int], List[str], List[int], Optional[List[str]]]:
        """对 DAVF 位点三元组（+ 可选 gene_names）做协同增强。

        参数:
            davf_sites: PTM 位点位置列表
            davf_type_names: 对应位点的 PTM 类型名
            davf_attention_mask: 对应位点的注意力掩码
            davf_gene_names: 对应位点的基因名（可选）
            max_position: 允许的最大位置（位置扰动后越界则丢弃扰动）

        返回:
            增强后的 (davf_sites, davf_type_names, davf_attention_mask,
            davf_gene_names)。davf_gene_names 为 None 时返回 None。
        """
        n = len(davf_sites)
        if n == 0:
            return davf_sites, davf_type_names, davf_attention_mask, davf_gene_names

        keep_indices = []
        for i in range(n):
            if self.drop_prob > 0 and random.random() < self.drop_prob:
                continue
            keep_indices.append(i)

        if not keep_indices:
            # 至少保留一个位点，避免空增强
            keep_indices = [random.randint(0, n - 1)]

        new_sites: List[int] = []
        new_types: List[str] = []
        new_mask: List[int] = []
        new_genes: Optional[List[str]] = [] if davf_gene_names is not None else None

        for i in keep_indices:
            pos = int(davf_sites[i])
            if self.noise_prob > 0 and random.random() < self.noise_prob:
                offset = random.randint(-self.noise_radius, self.noise_radius)
                pos = pos + offset
                if max_position is not None:
                    pos = max(1, min(pos, max_position))
                elif pos < 1:
                    pos = 1
            new_sites.append(pos)
            new_types.append(davf_type_names[i])
            new_mask.append(1)
            if new_genes is not None and davf_gene_names is not None:
                new_genes.append(davf_gene_names[i])

        return new_sites, new_types, new_mask, new_genes


def compute_sample_weights(labels: List[int], mode: str = "balanced") -> torch.Tensor:
    """
    计算样本权重用于类别平衡

    参数:
        labels: 标签列表
        mode: 权重计算模式，"balanced" 或 "effective"

    返回:
        每个样本的权重
    """
    from collections import Counter

    label_counts = Counter(labels)
    num_classes = len(label_counts)

    if mode == "balanced":
        # 倒数频率权重
        weights = {label: 1.0 / count for label, count in label_counts.items()}
    elif mode == "effective":
        # 有效样本数权重
        beta = 0.9999
        weights = {}
        for label, count in label_counts.items():
            effective_num = (1.0 - beta ** count) / (1.0 - beta)
            weights[label] = 1.0 / effective_num
    else:
        raise ValueError(f"未知的权重模式: {mode}")

    # 归一化
    total = sum(weights.values())
    weights = {k: v * num_classes / total for k, v in weights.items()}

    sample_weights = torch.tensor([weights[label] for label in labels], dtype=torch.float32)
    return sample_weights


class WeightedRandomSampler:
    """
    加权随机采样器
    用于类别不平衡的数据集
    """

    def __init__(self, labels: List[int], num_samples: Optional[int] = None, replacement: bool = True):
        """
        初始化采样器

        参数:
            labels: 标签列表
            num_samples: 采样数量（默认等于数据集大小）
            replacement: 是否放回采样
        """
        self.labels = labels
        self.num_samples = num_samples or len(labels)
        self.replacement = replacement

        # 计算样本权重
        self.weights = compute_sample_weights(labels)

    def __iter__(self):
        """返回采样索引的迭代器"""
        indices = torch.multinomial(
            self.weights,
            num_samples=self.num_samples,
            replacement=self.replacement,
        ).tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples


def get_augmentation_config(aug_type: str = "light") -> Dict[str, Any]:
    """
    获取预定义的增强配置

    参数:
        aug_type: 增强类型，"light", "medium", "heavy"

    返回:
        增强配置字典
    """
    configs = {
        "light": {
            "sequence_augment_prob": 0.3,
            "max_truncate_ratio": 0.05,
            "mask_prob": 0.0,
            "random_swap_prob": 0.0,
            "ptm_drop_prob": 0.0,
            "ptm_noise_prob": 0.0,
        },
        "medium": {
            "sequence_augment_prob": 0.5,
            "max_truncate_ratio": 0.1,
            "mask_prob": 0.02,
            "random_swap_prob": 0.01,
            "ptm_drop_prob": 0.1,
            "ptm_noise_prob": 0.05,
        },
        "heavy": {
            "sequence_augment_prob": 0.8,
            "max_truncate_ratio": 0.2,
            "mask_prob": 0.05,
            "random_swap_prob": 0.03,
            "ptm_drop_prob": 0.2,
            "ptm_noise_prob": 0.1,
        },
    }

    return configs.get(aug_type, configs["light"])
