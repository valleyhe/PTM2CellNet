"""
PyTorch数据集模块
功能概述: 定义PyTorch Dataset和DataModule
设计思路: 基于PyTorch Lightning DataModule，支持批量加载和多进程
"""

import json
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, cast

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..utils.logging import setup_logger
from .aa_constants import (
    AMINO_ACIDS_STR,
    AA_TO_IDX,
    NON_STANDARD_AA_MAP,
)
from .augmentation import DAVFSiteAugmenter, PTMAugmenter, SequenceAugmenter, get_augmentation_config
from .dataset_base import DatasetConfig, PTMDatasetBase, PTMSiteDict
from .features import FeatureExtractor, DEFAULT_AMINO_ACIDS

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# TypedDict definitions replacing Dict[str, Any] annotations
# ---------------------------------------------------------------------------


class _PtmSiteDict(PTMSiteDict, total=False):
    """Shape of a single PTM site dictionary in data loading.

    ``position`` is always present at runtime but declared total=False so
    that partially-constructed dicts still type-check without casts.
    """

    # Extra fields beyond PTMSiteDict
    residue: str
    confidence: float
    gene_name: str
    gene_symbol: str
    gene: str


class _DataConfigSection(TypedDict, total=False):
    """Nested ``config["data"]`` section."""

    max_sequence_length: int
    valid_amino_acids: List[str]
    ptm_types: List[str]
    cell_states: List[str]
    label_to_idx: Dict[str, int]
    cache_dir: str
    use_davf: bool
    persistent_workers: bool
    pin_memory: bool


class _FeaturesConfigSection(TypedDict, total=False):
    """Nested ``config["features"]`` section."""

    sequence_encoding: str
    kmer_size: int
    include_physicochemical: bool
    include_ptm_features: bool
    include_structural_features: bool
    structural_source: str
    use_feature_extractor: bool


class _AugmentationConfigSection(TypedDict, total=False):
    """Nested ``config["augmentation"]`` section (or preset dict)."""

    sequence_augment_prob: float
    max_truncate_ratio: float
    mask_token_id: int
    mask_prob: float
    random_swap_prob: float
    ptm_drop_prob: float
    ptm_noise_prob: float
    noise_radius: int
    davf_drop_prob: float
    davf_noise_prob: float


class PTMDataset(PTMDatasetBase):
    """
    PTM数据集类
    PyTorch Dataset实现，用于加载蛋白质PTM数据（使用自定义序列编码）
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_extractor: Optional[FeatureExtractor] = None,
        config: Optional[Union[Dict[str, Any], DatasetConfig]] = None,
        return_sequence: bool = True,
        return_ptm: bool = True,
        return_label: bool = True,
        use_feature_extractor: bool = False,
        sequence_augmenter: Optional[SequenceAugmenter] = None,
        ptm_augmenter: Optional[PTMAugmenter] = None,
        training: bool = True,
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

        # Pre-parse PTM sites JSON to avoid repeated json.loads in __getitem__
        # 接入 DatasetCache：当 cache_dir 配置时，缓存预解析的 PTM 位点列表。
        self._parsed_ptm_sites: List[Optional[List[_PtmSiteDict]]] = []
        cache_key_config = {
            "ptm_types": self.ptm_types,
            "max_sequence_length": self.config.max_sequence_length,
        }
        cached_parsed = None
        if self.dataset_cache is not None and "ptm_sites" in self.df.columns:
            cached_parsed = self.dataset_cache.load(self.df, {"parse_ptm_sites": cache_key_config})

        if cached_parsed is not None and isinstance(cached_parsed, list):
            self._parsed_ptm_sites = cached_parsed
        else:
            if "ptm_sites" in self.df.columns:
                for idx in range(len(self.df)):
                    ptm_data = self.df.iloc[idx].get("ptm_sites")
                    try:
                        if isinstance(ptm_data, str) and ptm_data:
                            parsed = json.loads(ptm_data)
                        elif isinstance(ptm_data, list):
                            parsed = ptm_data
                        else:
                            parsed = None
                        if isinstance(parsed, list):
                            self._parsed_ptm_sites.append(parsed)
                        else:
                            self._parsed_ptm_sites.append([])
                    except (json.JSONDecodeError, TypeError):
                        self._parsed_ptm_sites.append([])
            else:
                self._parsed_ptm_sites = [None] * len(self.df)

            if self.dataset_cache is not None and "ptm_sites" in self.df.columns:
                self.dataset_cache.save(
                    self.df, {"parse_ptm_sites": cache_key_config}, self._parsed_ptm_sites
                )

        self.feature_extractor = feature_extractor or FeatureExtractor(self._raw_config or {})
        self.return_sequence = return_sequence
        self.return_ptm = return_ptm
        self.return_label = return_label
        self.use_feature_extractor = bool(
            use_feature_extractor or self.config.use_feature_extractor
        )
        self.training = training
        self.use_davf = self.config.use_davf

        self.max_sequence_length = self.config.max_sequence_length
        self.amino_acids = self.config.valid_amino_acids or DEFAULT_AMINO_ACIDS
        # 序列索引从1开始，0保留给padding
        # 基于共享常量构建映射；若配置自定义字母表则覆盖
        if self.amino_acids == AMINO_ACIDS_STR:
            self.aa_to_idx = dict(AA_TO_IDX)
        else:
            self.aa_to_idx = {aa: i + 1 for i, aa in enumerate(self.amino_acids)}

        # 非标准氨基酸字符映射表 (统一从aa_constants导入)
        self.non_standard_aa_map = dict(NON_STANDARD_AA_MAP)
        self.sequence_augmenter = sequence_augmenter
        self.ptm_augmenter = ptm_augmenter

        if self.sequence_augmenter is None or self.ptm_augmenter is None:
            self._initialize_augmenters_from_config()

        # DAVF 位点协同增强器（仅 use_davf 时生效）
        self.davf_augmenter: Optional[DAVFSiteAugmenter] = None
        if self.use_davf:
            self.davf_augmenter = self._build_davf_augmenter()

    def _initialize_augmenters_from_config(self) -> None:
        """按需从配置构建数据增强器。支持字符串预设（light/medium/heavy）或参数字典。"""
        augmentation_config = self.config.augmentation
        if not augmentation_config:
            return

        if isinstance(augmentation_config, str):
            augmentation_config = get_augmentation_config(augmentation_config)

        if self.sequence_augmenter is None:
            self.sequence_augmenter = SequenceAugmenter(
                augment_prob=augmentation_config.get("sequence_augment_prob", 0.0),
                max_truncate_ratio=augmentation_config.get("max_truncate_ratio", 0.1),
                mask_token_id=augmentation_config.get("mask_token_id", 0),
                mask_prob=augmentation_config.get("mask_prob", 0.0),
                random_swap_prob=augmentation_config.get("random_swap_prob", 0.0),
            )

        if self.ptm_augmenter is None:
            self.ptm_augmenter = PTMAugmenter(
                drop_prob=augmentation_config.get("ptm_drop_prob", 0.0),
                add_noise_prob=augmentation_config.get("ptm_noise_prob", 0.0),
                noise_radius=augmentation_config.get("noise_radius", 1),
            )

    def _build_davf_augmenter(self) -> Optional[DAVFSiteAugmenter]:
        """按配置构建 DAVF 位点协同增强器。"""
        augmentation_config = self.config.augmentation
        if not augmentation_config:
            return None
        if isinstance(augmentation_config, str):
            augmentation_config = get_augmentation_config(augmentation_config)
        return DAVFSiteAugmenter(
            drop_prob=augmentation_config.get("davf_drop_prob", augmentation_config.get("ptm_drop_prob", 0.0)),
            noise_prob=augmentation_config.get("davf_noise_prob", augmentation_config.get("ptm_noise_prob", 0.0)),
            noise_radius=augmentation_config.get("noise_radius", 1),
        )

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

    def _encode_ptm(self, ptm_sites: List[_PtmSiteDict], sequence_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码PTM位点（ptm_sites已是解析后的列表）"""
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

    # DAVF gene name column candidates, checked in priority order
    _DAVF_GENE_COLUMNS = ("gene_name", "gene_symbol", "gene", "gene_names", "gene_symbols")

    def _extract_davf_gene_names(self, row: pd.Series, valid_sites: List[_PtmSiteDict]) -> List[str]:
        """Extract gene names for DAVF from PTM site dicts or row-level column.

        Checks each PTM site dict for gene_name/gene_symbol/gene keys first.
        Falls back to a row-level column matching the same naming patterns.
        Returns a list of gene name strings, one per valid site.
        """
        # Try per-site gene fields first (site dicts may carry their own gene info)
        site_gene_keys = ("gene_name", "gene_symbol", "gene")
        gene_names = []
        for site in valid_sites:
            name = None
            for key in site_gene_keys:
                val = site.get(key)
                if val and isinstance(val, str):
                    name = val
                    break
            gene_names.append(name)

        # If all sites have gene names, return them directly
        if all(g is not None for g in gene_names):
            return [g for g in gene_names if g is not None]

        # Fall back to row-level column
        row_gene = None
        for col in self._DAVF_GENE_COLUMNS:
            if col in self.df.columns:
                val = row.get(col)
                if pd.notna(val):
                    row_gene = str(val)
                    break

        if row_gene is not None:
            # Row-level gene applies to all sites
            return [row_gene if g is None else g for g in gene_names]

        # Fill remaining None with empty string
        return [g if g is not None else "" for g in gene_names]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        row = self.df.iloc[idx]
        # Builder accumulates mixed types; cast to Dict[str, torch.Tensor] at return
        _SampleValue = Union[torch.Tensor, List[int], List[str]]
        sample: Dict[str, _SampleValue] = {}

        if self.return_sequence and "sequence" in row:
            sequence = str(row["sequence"])
            sample["sequence"] = self._encode_sequence(sequence)
            sample["sequence_length"] = torch.tensor(min(len(sequence), self.max_sequence_length), dtype=torch.long)

        if self.return_ptm and "ptm_sites" in row:
            seq_len = len(str(row.get("sequence", "")))
            parsed_ptm = self._parsed_ptm_sites[idx]
            if parsed_ptm is None:
                parsed_ptm = []
            ptm_mask, ptm_types = self._encode_ptm(parsed_ptm, seq_len)
            sample["ptm_mask"] = ptm_mask
            sample["ptm_types"] = ptm_types

            if self.use_davf:
                ptm_sites = parsed_ptm
                max_len = min(self.max_sequence_length, seq_len) if seq_len else self.max_sequence_length
                valid_sites = []
                for site in ptm_sites:
                    is_valid, _ = self._validate_ptm_site(site, max_len)
                    if is_valid:
                        valid_sites.append(site)

                davf_sites = [site["position"] for site in valid_sites]
                davf_type_names = [site.get("type", "") for site in valid_sites]

                # Extract gene names from PTM site dicts or row-level column
                davf_gene_names = self._extract_davf_gene_names(row, valid_sites)

                sample["davf_sites"] = davf_sites
                sample["davf_gene_names"] = davf_gene_names
                sample["davf_type_names"] = davf_type_names
                sample["davf_attention_mask"] = [1] * len(valid_sites)

        if self.use_feature_extractor and self.return_sequence and "sequence" in row:
            sequence_features = self.feature_extractor.extract_sequence_features([sequence])
            if sequence_features.size > 0:
                sample["sequence_features"] = torch.tensor(sequence_features[0], dtype=torch.float32)

            if self.feature_extractor.include_ptm_features and "ptm_sites" in row:
                feat_ptm_sites = self._parsed_ptm_sites[idx]
                if feat_ptm_sites is None:
                    feat_ptm_sites = []
                sample["ptm_features"] = torch.tensor(
                    self.feature_extractor.extract_ptm_features_array(
                        cast("List[Any]", feat_ptm_sites), len(sequence)
                    ),
                    dtype=torch.float32,
                )

        if self.training:
            if "sequence" in sample and self.sequence_augmenter is not None:
                sample["sequence"] = self.sequence_augmenter(cast(torch.Tensor, sample["sequence"]))
            if "ptm_mask" in sample and "ptm_types" in sample and self.ptm_augmenter is not None:
                sample["ptm_mask"], sample["ptm_types"] = self.ptm_augmenter(
                    cast(torch.Tensor, sample["ptm_mask"]),
                    cast(torch.Tensor, sample["ptm_types"]),
                )
            # DAVF 位点字段协同增强（仅 use_davf 且存在相关字段时）
            if (
                self.davf_augmenter is not None
                and "davf_sites" in sample
                and "davf_type_names" in sample
                and "davf_attention_mask" in sample
            ):
                _davf_result = self.davf_augmenter(
                    cast(List[int], sample["davf_sites"]),
                    cast(List[str], sample["davf_type_names"]),
                    cast(List[int], sample["davf_attention_mask"]),
                    cast(Optional[List[str]], sample.get("davf_gene_names")),
                    max_position=self.max_sequence_length,
                )
                sample["davf_sites"] = _davf_result[0]
                sample["davf_type_names"] = _davf_result[1]
                sample["davf_attention_mask"] = _davf_result[2]
                sample["davf_gene_names"] = cast(
                    "Union[List[int], List[str]]", _davf_result[3] or []
                )

        if self.return_label and "cell_state" in row:
            sample["label"] = self._encode_label(str(row["cell_state"]))

        return cast(Dict[str, torch.Tensor], sample)


class PTMPlainDataModule:
    """
    PTM数据模块类
    管理训练、验证和测试数据集的加载
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        config: Optional[Union[Dict[str, Any], DatasetConfig]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        use_feature_extractor: bool = False,
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
        self._raw_config: Dict[str, Any] = config if isinstance(config, dict) else {}
        if isinstance(config, DatasetConfig):
            self.config = config
        else:
            self.config = DatasetConfig.from_dict(config)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_feature_extractor = bool(
            use_feature_extractor or self.config.use_feature_extractor
        )

        self.feature_extractor = FeatureExtractor(self._raw_config)
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
            use_feature_extractor=self.use_feature_extractor,
            training=True,
        )

        if "cell_state" in self.train_df.columns:
            self.labels = self.train_dataset.labels
            self.label_to_idx = self.train_dataset.label_to_idx

        if self.val_df is not None:
            self.val_dataset = PTMDataset(
                self.val_df,
                self.feature_extractor,
                self.config,
                use_feature_extractor=self.use_feature_extractor,
                training=False,
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
                use_feature_extractor=self.use_feature_extractor,
                training=False,
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
        persistent_workers = self.config.persistent_workers
        pin_memory = self.config.pin_memory

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

        pin_memory = self.config.pin_memory

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0 and self.config.persistent_workers,
            pin_memory=pin_memory,
        )

    def test_dataloader(self) -> Optional[DataLoader[Dict[str, torch.Tensor]]]:
        """获取测试数据加载器"""
        if self.test_dataset is None:
            return None

        pin_memory = self.config.pin_memory

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0 and self.config.persistent_workers,
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
        config: Optional[Union[Dict[str, Any], DatasetConfig]] = None,
        training: bool = True,
        sequence_augmenter: Optional[SequenceAugmenter] = None,
        ptm_augmenter: Optional[PTMAugmenter] = None,
    ):
        """
        初始化ESMTokenizedDataset

        参数:
            df: 数据DataFrame，需含sequence、ptm_sites（可选）、cell_state（可选）列
            tokenizer: ESM tokenizer实例（来自ESM2Encoder.tokenizer）
            max_length: tokenizer最大长度（含特殊token），默认1024
            config: 配置字典
            training: 是否为训练模式（启用数据增强）
            sequence_augmenter: 序列增强器（可选，未提供时按配置构建）
            ptm_augmenter: PTM增强器（可选，未提供时按配置构建）
        """
        super().__init__(df, config)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.training = training

        # 非标准氨基酸字符映射表 (统一从aa_constants导入)
        self.non_standard_aa_map = dict(NON_STANDARD_AA_MAP)

        # 数据增强器
        self.sequence_augmenter = sequence_augmenter
        self.ptm_augmenter = ptm_augmenter
        if self.sequence_augmenter is None or self.ptm_augmenter is None:
            self._initialize_augmenters_from_config()

        # 预 tokenization 缓存：避免每次 __getitem__ 重复调用 tokenizer。
        # 内存权衡——对大 df 可改用磁盘缓存(DatasetCache)按 sequence 哈希存 token 结果。
        self._token_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for row_idx in range(len(self.df)):
            sequence = str(self.df.iloc[row_idx].get("sequence", ""))
            self._token_cache.append(self._tokenize_sequence(sequence))

    def _initialize_augmenters_from_config(self) -> None:
        """按需从配置构建数据增强器（与 PTMDataset 一致）。"""
        augmentation_config = self.config.augmentation
        if not augmentation_config:
            return

        if isinstance(augmentation_config, str):
            augmentation_config = get_augmentation_config(augmentation_config)

        if self.sequence_augmenter is None:
            self.sequence_augmenter = SequenceAugmenter(
                augment_prob=augmentation_config.get("sequence_augment_prob", 0.0),
                max_truncate_ratio=augmentation_config.get("max_truncate_ratio", 0.1),
                mask_token_id=augmentation_config.get(
                    "mask_token_id",
                    getattr(self.tokenizer, "mask_token_id", 0) or 0,
                ),
                mask_prob=augmentation_config.get("mask_prob", 0.0),
                random_swap_prob=augmentation_config.get("random_swap_prob", 0.0),
            )

        if self.ptm_augmenter is None:
            self.ptm_augmenter = PTMAugmenter(
                drop_prob=augmentation_config.get("ptm_drop_prob", 0.0),
                add_noise_prob=augmentation_config.get("ptm_noise_prob", 0.0),
                noise_radius=augmentation_config.get("noise_radius", 1),
            )

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

        # 查表获取预 tokenization 结果（避免每次 __getitem__ 重复调用 tokenizer）
        input_ids, attention_mask = self._token_cache[idx]
        tokenized_length = input_ids.shape[0]

        ptm_sites_json = str(row["ptm_sites"]) if "ptm_sites" in row else "[]"
        ptm_mask, ptm_types = self._encode_ptm_esm(ptm_sites_json, tokenized_length)

        sample: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ptm_mask": ptm_mask,
            "ptm_types": ptm_types,
        }

        # 训练模式下应用数据增强（与 PTMDataset.__getitem__ 一致）
        if self.training:
            if "ptm_mask" in sample and "ptm_types" in sample and self.ptm_augmenter is not None:
                sample["ptm_mask"], sample["ptm_types"] = self.ptm_augmenter(
                    sample["ptm_mask"],
                    sample["ptm_types"],
                )
            # sequence 增强作用于 input_ids（ESM tokenizer 的 mask_token_id 已传入）
            if "input_ids" in sample and self.sequence_augmenter is not None:
                sample["input_ids"] = self.sequence_augmenter(sample["input_ids"])

        if "cell_state" in row:
            sample["label"] = self._encode_label(str(row["cell_state"]))

        return sample

