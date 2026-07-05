"""
数据集基类和工具函数
功能概述: 提供共享功能和工具，减少ESMTokenizedDataset和PTMDataset之间的重复代码
"""
import dataclasses
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing_extensions import TypedDict

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# TypedDict definitions for structured dicts used in this module
# ---------------------------------------------------------------------------

class PTMSiteDict(TypedDict, total=False):
    """Shape of a single PTM site dict used in dataset processing."""

    position: int
    type: str
    amino_acid: Optional[str]


class DatasetStatistics(TypedDict, total=False):
    """Shape of the dict returned by ``PTMDatasetBase.get_statistics``."""

    num_samples: int
    num_features: int
    columns: List[str]
    sequence_length: Dict[str, Union[int, float]]
    num_classes: int
    labels: List[str]
    label_distribution: Dict[str, int]
    ptm_per_sample: Dict[str, Union[int, float]]
    ptm_type_distribution: Dict[str, int]


# ---------------------------------------------------------------------------
# DatasetConfig — flat dataclass replacing the nested-dict config pattern.
# Consumers that still pass dicts use from_dict() for backward compatibility.
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Configuration for dataset construction.

    All fields have sensible defaults so existing code that passes partial
    dicts continues to work after converting via :meth:`from_dict`.
    """

    # Core
    max_sequence_length: int = 1024
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = True

    # PTM
    max_ptm_sites: int = 10
    ptm_types: Optional[List[str]] = None
    ptm_position_mode: str = "relative"  # "relative" | "absolute"
    ptm_type_embed_dim: int = 32
    use_ptm_attention: bool = True

    # Feature
    feature_type: str = "one_hot"  # "one_hot" | "kmers" | "physicochemical" | "combined"
    kmer_size: int = 3
    use_physicochemical: bool = False
    use_feature_extractor: bool = False

    # Label
    label_column: str = "label"
    num_classes: int = 2
    task_type: str = "classification"  # "classification" | "regression" | "multitask"

    # Augmentation — kept as a nested block so string presets (light/medium/heavy)
    # continue to work without forcing every field onto the top level.
    augmentation: Optional[Union[str, Dict[str, Any]]] = None
    augment_prob: float = 0.0
    max_truncate_ratio: float = 0.1
    mask_prob: float = 0.0
    random_swap_prob: float = 0.0

    # Split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Tokenizer / Pretrained
    tokenizer_name: Optional[str] = None
    pretrained_model_name: Optional[str] = None

    # DAVF
    use_davf: bool = False
    davf_feature_dim: int = 128
    davf_checkpoint_path: Optional[str] = None

    # Other
    seed: int = 42
    data_path: Optional[str] = None
    cache_dir: Optional[str] = None

    # Data section keys (previously config["data"])
    valid_amino_acids: Optional[Union[str, List[str]]] = None
    cell_states: Optional[List[str]] = None
    label_to_idx: Optional[Dict[str, int]] = None
    persistent_workers: bool = False
    strict_load: bool = False
    use_hdf5: bool = False

    # Section containers for non-flattened subsections (FeatureExtractor etc.)
    features: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    training: Optional[Dict[str, Any]] = None
    paths: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]] = None) -> "DatasetConfig":
        """Create a DatasetConfig from a dict, for backward compatibility.

        Handles three input shapes:

        * ``None`` → returns ``cls()`` (all defaults).
        * A flat dict with keys matching field names → used directly.
        * A nested dict with ``data`` / ``features`` / ``augmentation`` /
          ``training`` / ``paths`` sections → sections are unwrapped and
          recognised fields are extracted.
        """
        if d is None:
            return cls()

        # Sentinel section keys that exist on the dataclass but should NOT
        # short-circuit the nested-dict unwrapping below.
        # NOTE: ``augmentation`` is intentionally omitted — it can carry a
        # string preset (e.g. "medium") that must be set directly as a flat key.
        _SECTION_KEYS = frozenset({"data", "features", "training", "paths"})

        valid_keys: set = {f.name for f in dataclasses.fields(cls)}
        kwargs: Dict[str, Any] = {}

        # 1) Try flat keys first — skip sentinel section containers.
        for k in list(d.keys()):
            if k in valid_keys and k not in _SECTION_KEYS:
                kwargs[k] = d[k]

        # 2) If *no* flat keys matched, try nested sections
        if not kwargs:
            for section_key in _SECTION_KEYS:
                section = d.get(section_key)
                if isinstance(section, dict):
                    for k, v in section.items():
                        if k in valid_keys and k not in kwargs:
                            kwargs[k] = v

        return cls(**kwargs)


class PTMDatasetBase(Dataset[Dict[str, torch.Tensor]]):
    """
    PTM数据集基类
    提供标签编码、PTM解析等通用功能
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[Union[Dict[str, Any], DatasetConfig]] = None,
    ):
        self.df = df.reset_index(drop=True)

        # Keep the original dict for callers (e.g. FeatureExtractor) that
        # expect the nested-dict format.
        self._raw_config: Dict[str, Any] = config if isinstance(config, dict) else {}

        # Convert to typed dataclass for our own use.
        if isinstance(config, DatasetConfig):
            self.config = config
        else:
            self.config = DatasetConfig.from_dict(config)

        # 设置PTM类型映射
        self.ptm_types = self._get_ptm_types()
        self.ptm_type_to_idx = {ptm: i + 1 for i, ptm in enumerate(self.ptm_types)}

        # 设置标签映射
        if "cell_state" in self.df.columns:
            configured_labels = self.config.cell_states or []
            configured_mapping = self.config.label_to_idx or {}
            if configured_labels:
                self.labels = [str(label) for label in configured_labels]
                self.label_to_idx = {
                    str(label): int(configured_mapping.get(label, i))
                    for i, label in enumerate(self.labels)
                }
            else:
                self.labels = sorted(self.df["cell_state"].unique())
                self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        else:
            self.labels = []
            self.label_to_idx = {}

        # 接入 DatasetCache（将特征预计算结果接入缓存主链路）。
        # 仅当 cache_dir 非空时实例化，避免无配置时产生目录副作用。
        self.dataset_cache = None
        if self.config.cache_dir:
            try:
                from .validation import DatasetCache
                self.dataset_cache = DatasetCache(self.config.cache_dir)
            except (ImportError, TypeError, ValueError) as exc:  # 缓存初始化失败不阻塞数据加载
                logger.warning("DatasetCache 初始化失败 (%s)，将不使用缓存", exc)

    def _get_ptm_types(self) -> List[str]:
        """从配置或默认值获取PTM类型列表"""
        default_ptm_types = [
            "phosphorylation",
            "acetylation",
            "methylation",
            "ubiquitination",
            "sumoylation",
        ]
        if self.config.ptm_types:
            return self.config.ptm_types
        return default_ptm_types

    def _encode_label(self, label: str) -> torch.Tensor:
        """编码标签字符串为整数tensor

        如果使用了从训练集传播的标签映射，验证/测试集中的未知标签将触发ValueError。
        """
        if label in self.label_to_idx:
            return torch.tensor(self.label_to_idx[label], dtype=torch.long)
        # 检查是否是传播的训练标签映射（有labels属性）导致的未知标签
        if hasattr(self, 'labels') and self.labels and label not in self.label_to_idx:
            raise ValueError(
                f"验证/测试集中存在训练集中未见过的标签: '{label}'。 "
                f"训练集标签: {list(self.label_to_idx.keys())}"
            )
        raise ValueError(f"未知标签: {label}，可用标签: {list(self.label_to_idx.keys())}")

    def _parse_ptm_sites(self, ptm_sites_json: str) -> List[PTMSiteDict]:
        """解析PTM位点JSON字符串为列表"""
        try:
            if pd.isna(ptm_sites_json):
                return []
            if isinstance(ptm_sites_json, str):
                ptm_sites = json.loads(ptm_sites_json)
            elif isinstance(ptm_sites_json, list):
                ptm_sites = ptm_sites_json
            else:
                ptm_sites = []
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"PTM JSON解析错误: {e}，数据: {ptm_sites_json}")
            ptm_sites = []

        return ptm_sites if isinstance(ptm_sites, list) else []

    def _validate_ptm_site(self, site: PTMSiteDict, seq_length: Optional[int] = None) -> Tuple[bool, str]:
        """
        验证单个PTM位点的有效性

        参数:
            site: PTM位点字典
            seq_length: 序列长度（用于验证位置范围）

        返回:
            (是否有效, 错误信息)
        """
        if not isinstance(site, dict):
            return False, "PTM位点必须是字典"

        # 检查必需字段
        if "position" not in site:
            return False, "PTM位点缺少position字段"

        pos = site.get("position")
        if not isinstance(pos, int) or pos < 1:
            return False, f"无效的position值: {pos}"

        # 验证位置在序列范围内
        if seq_length is not None and pos > seq_length:
            return False, f"position {pos} 超出序列长度 {seq_length}"

        # 验证PTM类型
        ptm_type = site.get("type", "")
        if ptm_type and self.ptm_type_to_idx.get(ptm_type, 0) == 0:
            logger.debug(f"未知的PTM类型: {ptm_type}，将使用默认类型")

        return True, ""

    def __len__(self) -> int:
        return len(self.df)

    def get_label_distribution(self) -> Dict[str, int]:
        """获取标签分布统计"""
        if "cell_state" not in self.df.columns:
            return {}
        return cast(Dict[str, int], self.df["cell_state"].value_counts().to_dict())

    def get_ptm_type_distribution(self) -> Dict[str, int]:
        """获取PTM类型分布统计"""
        ptm_counts: Dict[str, int] = {}

        if "ptm_sites" not in self.df.columns:
            return ptm_counts

        for ptm_json in self.df["ptm_sites"]:
            ptm_sites = self._parse_ptm_sites(str(ptm_json))
            for site in ptm_sites:
                ptm_type = site.get("type", "unknown")
                ptm_counts[ptm_type] = ptm_counts.get(ptm_type, 0) + 1

        return ptm_counts

    def get_statistics(self) -> DatasetStatistics:
        """
        获取数据集统计信息

        返回:
            包含各种统计信息的字典
        """
        stats = {
            "num_samples": len(self),
            "num_features": len(self.df.columns),
            "columns": list(self.df.columns),
        }

        # 序列长度统计
        if "sequence" in self.df.columns:
            seq_lengths = self.df["sequence"].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
            stats["sequence_length"] = {
                "min": int(seq_lengths.min()),
                "max": int(seq_lengths.max()),
                "mean": float(seq_lengths.mean()),
                "median": float(seq_lengths.median()),
            }

        # 标签分布
        if self.labels:
            stats["num_classes"] = len(self.labels)
            stats["labels"] = self.labels
            stats["label_distribution"] = self.get_label_distribution()

        # PTM统计
        if "ptm_sites" in self.df.columns:
            ptm_counts = []
            for ptm_json in self.df["ptm_sites"]:
                ptm_sites = self._parse_ptm_sites(str(ptm_json))
                ptm_counts.append(len(ptm_sites))

            if ptm_counts:
                stats["ptm_per_sample"] = {
                    "min": min(ptm_counts),
                    "max": max(ptm_counts),
                    "mean": sum(ptm_counts) / len(ptm_counts),
                    "samples_with_ptm": sum(1 for c in ptm_counts if c > 0),
                }
                stats["ptm_type_distribution"] = self.get_ptm_type_distribution()

        return cast(DatasetStatistics, stats)

    def print_statistics(self) -> None:
        """打印数据集统计信息"""
        stats = self.get_statistics()

        logger.info("")
        logger.info("=" * 50)
        logger.info("数据集统计信息")
        logger.info("=" * 50)
        logger.info("样本总数: %d", stats['num_samples'])
        logger.info("特征列数: %d", stats['num_features'])
        logger.info("列名: %s", ', '.join(stats['columns']))

        if "sequence_length" in stats:
            seq_stats = stats["sequence_length"]
            logger.info("序列长度统计:")
            logger.info("  最小值: %d", seq_stats['min'])
            logger.info("  最大值: %d", seq_stats['max'])
            logger.info("  平均值: %.2f", seq_stats['mean'])
            logger.info("  中位数: %.2f", seq_stats['median'])

        if "num_classes" in stats:
            logger.info("类别信息:")
            logger.info("  类别数: %d", stats['num_classes'])
            logger.info("  标签分布:")
            for label, count in stats["label_distribution"].items():
                percentage = (count / stats['num_samples']) * 100
                logger.info("    %s: %d (%.1f%%)", label, count, percentage)

        if "ptm_per_sample" in stats:
            ptm_stats = stats["ptm_per_sample"]
            logger.info("PTM统计:")
            logger.info("  每样本PTM数: %d - %d", ptm_stats['min'], ptm_stats['max'])
            logger.info("  平均每样本PTM: %.2f", ptm_stats['mean'])
            logger.info("  含PTM的样本数: %d", ptm_stats['samples_with_ptm'])

            if stats.get("ptm_type_distribution"):
                logger.info("  PTM类型分布:")
                for ptm_type, count in sorted(stats["ptm_type_distribution"].items(), key=lambda x: -x[1]):
                    logger.info("    %s: %d", ptm_type, count)

        logger.info("=" * 50)


def compute_class_weights(label_distribution: Dict[str, int], mode: str = "inverse") -> torch.Tensor:
    """
    计算类别权重用于类别平衡

    参数:
        label_distribution: 标签到数量的映射
        mode: 权重计算模式，可选 "inverse"（倒数）或 "effective"（有效样本数）

    返回:
        类别权重tensor
    """
    labels = sorted(label_distribution.keys())
    counts = torch.tensor([label_distribution[label] for label in labels], dtype=torch.float32)

    if mode == "inverse":
        # 倒数权重
        weights = 1.0 / counts
    elif mode == "effective":
        # 有效样本数权重
        beta = 0.9999
        effective_num = (1.0 - torch.pow(beta, counts)) / (1.0 - beta)
        weights = 1.0 / effective_num
    else:
        raise ValueError(f"未知的权重计算模式: {mode}")

    # 归一化权重
    weights = weights / weights.sum() * len(labels)

    return cast(torch.Tensor, weights)
