"""
Lightning数据模块
功能概述: 封装PTMDataset，提供PyTorch Lightning兼容的数据接口
设计思路: 复用现有PTMDataset，支持分布式训练和多进程数据加载
"""

from typing import Any, Dict, Optional

import pandas as pd
import lightning as L
from torch.utils.data import DataLoader

from .datasets import PTMDataset
from .features import FeatureExtractor
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class PTMLightningDataModule(L.LightningDataModule):
    """
    Lightning数据模块（Lightning data module）
    封装PTMDataset，提供训练/验证/测试数据加载器
    """

    train_dataset: Any
    val_dataset: Any
    test_dataset: Any

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        config: Dict[str, Any],
        feature_extractor: Optional[FeatureExtractor] = None,
        tokenizer: Optional[Any] = None,
    ):
        """
        初始化数据模块

        参数:
            train_df: 训练数据DataFrame
            val_df: 验证数据DataFrame
            test_df: 测试数据DataFrame
            config: 配置字典
            feature_extractor: 特征提取器实例
        """
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.config = config
        self.feature_extractor = feature_extractor or FeatureExtractor(config)
        self.tokenizer = tokenizer

        # 从配置中提取参数
        training_config = config.get("training", {})
        data_config = config.get("data", {})

        self.batch_size = training_config.get("batch_size", 32)
        self.num_workers = data_config.get("num_workers", 4)
        self.pin_memory = data_config.get("pin_memory", True)
        self.drop_last = training_config.get("drop_last", True)

        # 数据集实例（在setup中初始化）
        self.train_dataset: Optional[PTMDataset] = None
        self.val_dataset: Optional[PTMDataset] = None
        self.test_dataset: Optional[PTMDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        准备数据集

        参数:
            stage: 当前阶段（"fit", "test", 或 None）
        """
        if self.tokenizer is not None:
            from .datasets import ESMTokenizedDataset

            if stage == "fit" or stage is None:
                logger.info("准备训练和验证数据集...")
                train_dataset = ESMTokenizedDataset(
                    df=self.train_df,
                    tokenizer=self.tokenizer,
                    config=self.config,
                )
                val_dataset = ESMTokenizedDataset(
                    df=self.val_df,
                    tokenizer=self.tokenizer,
                    config=self.config,
                )
                # 传播训练集的标签映射到验证集，确保编码一致
                if train_dataset.label_to_idx:
                    val_dataset.label_to_idx = train_dataset.label_to_idx
                    val_dataset.labels = train_dataset.labels
                self.train_dataset = train_dataset
                self.val_dataset = val_dataset
                logger.info(f"训练集大小: {len(train_dataset)}")
                logger.info(f"验证集大小: {len(val_dataset)}")

            if stage == "test" or stage is None:
                logger.info("准备测试数据集...")
                test_dataset = ESMTokenizedDataset(
                    df=self.test_df,
                    tokenizer=self.tokenizer,
                    config=self.config,
                )
                # 传播训练集的标签映射到测试集，确保编码一致
                train_ds = self.train_dataset
                if train_ds is not None and train_ds.label_to_idx:
                    test_dataset.label_to_idx = train_ds.label_to_idx
                    test_dataset.labels = train_ds.labels
                self.test_dataset = test_dataset
                logger.info(f"测试集大小: {len(test_dataset)}")
        else:
            if stage == "fit" or stage is None:
                logger.info("准备训练和验证数据集...")
                ptm_train = PTMDataset(
                    df=self.train_df,
                    feature_extractor=self.feature_extractor,
                    config=self.config,
                    training=True,
                )
                ptm_val = PTMDataset(
                    df=self.val_df,
                    feature_extractor=self.feature_extractor,
                    config=self.config,
                    training=False,
                )
                # 传播训练集的标签映射到验证集，确保编码一致
                if ptm_train.label_to_idx:
                    ptm_val.label_to_idx = ptm_train.label_to_idx
                    ptm_val.labels = ptm_train.labels
                self.train_dataset = ptm_train
                self.val_dataset = ptm_val
                logger.info(f"训练集大小: {len(ptm_train)}")
                logger.info(f"验证集大小: {len(ptm_val)}")

            if stage == "test" or stage is None:
                logger.info("准备测试数据集...")
                ptm_test = PTMDataset(
                    df=self.test_df,
                    feature_extractor=self.feature_extractor,
                    config=self.config,
                    training=False,
                )
                # 传播训练集的标签映射到测试集，确保编码一致
                train_ds = self.train_dataset
                if train_ds is not None and train_ds.label_to_idx:
                    ptm_test.label_to_idx = train_ds.label_to_idx
                    ptm_test.labels = train_ds.labels
                self.test_dataset = ptm_test
                logger.info(f"测试集大小: {len(ptm_test)}")

    def train_dataloader(self) -> DataLoader:
        """
        创建训练数据加载器

        返回:
            训练DataLoader
        """
        if self.train_dataset is None:
            raise RuntimeError("必须先调用setup()方法")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,  # 训练时丢弃不完整的批次
        )

    def val_dataloader(self) -> DataLoader:
        """
        创建验证数据加载器

        返回:
            验证DataLoader
        """
        if self.val_dataset is None:
            raise RuntimeError("必须先调用setup()方法")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """
        创建测试数据加载器

        返回:
            测试DataLoader
        """
        if self.test_dataset is None:
            raise RuntimeError("必须先调用setup()方法")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
