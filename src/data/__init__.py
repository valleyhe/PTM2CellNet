"""数据处理模块 - 数据加载、预处理、特征工程和数据集定义"""

from .loaders import DataLoader
from .preprocess import DataPreprocessor
from .features import FeatureExtractor
from .datasets import PTMDataset, PTMPlainDataModule, ESMTokenizedDataset
from .lightning_datamodule import PTMLightningDataModule

# PTMDataModule is the primary DataModule — points to the Lightning-compatible version.
# Use PTMPlainDataModule if you need the non-Lightning plain DataModule.
PTMDataModule = PTMLightningDataModule
from .dataset_base import PTMDatasetBase, compute_class_weights
from .augmentation import (
    SequenceAugmenter,
    PTMAugmenter,
    WeightedRandomSampler,
    get_augmentation_config,
)
from .validation import (
    DataValidator,
    DatasetCache,
    validate_and_report,
    collate_sequences,
)
from .data_manifest import (
    DataManifestError,
    load_manifest,
    validate_manifest,
    sha256_file,
    manifest_digest,
)

__all__ = [
    "DataLoader",
    "DataPreprocessor",
    "FeatureExtractor",
    "PTMDataset",
    "PTMPlainDataModule",
    "PTMDataModule",
    "ESMTokenizedDataset",
    "PTMDatasetBase",
    "compute_class_weights",
    "SequenceAugmenter",
    "PTMAugmenter",
    "WeightedRandomSampler",
    "get_augmentation_config",
    "DataValidator",
    "DatasetCache",
    "validate_and_report",
    "collate_sequences",
    "PTMLightningDataModule",
    "DataManifestError",
    "load_manifest",
    "validate_manifest",
    "sha256_file",
    "manifest_digest",
]
from .cross_scale_dataset import (
    CROSS_SCALE_DATA_SCHEMA_VERSION,
    CrossScaleDataContractError,
    CrossScaleNPZDataset,
    cross_scale_collate,
    EpochShuffleSampler,
    CrossScaleDataModule,
)

__all__ += [
    "CROSS_SCALE_DATA_SCHEMA_VERSION",
    "CrossScaleDataContractError",
    "CrossScaleNPZDataset",
    "cross_scale_collate",
    "EpochShuffleSampler",
    "CrossScaleDataModule",
]
