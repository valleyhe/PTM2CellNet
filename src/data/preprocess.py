"""
数据预处理模块
功能概述: 数据清洗、标准化、格式转换和数据集划分
设计思路: 提供可配置的预处理流程，支持多种数据清洗策略
"""

import json
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.logging import setup_logger
from src.utils.helpers import validate_sequence, validate_ptm_site, clean_sequence

logger = setup_logger(__name__)


class DataPreprocessor:
    """
    数据预处理器类
    负责数据清洗、标准化、PTM标签处理和数据集划分
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化数据预处理器

        参数:
            config: 配置字典
        """
        self.config = config or {}
        data_config = self.config.get("data", {})
        self.max_sequence_length = data_config.get("max_sequence_length", 1000)
        self.valid_amino_acids = set(data_config.get("valid_amino_acids", "ACDEFGHIKLMNPQRSTVWY"))
        self.remove_duplicates = data_config.get("preprocessing", {}).get("remove_duplicates", True)
        self.handle_missing = data_config.get("preprocessing", {}).get("handle_missing", "drop")

    # 非标准氨基酸字符映射表
    # U (selenocysteine) → C (cysteine, 生化性质最接近)
    # X (unknown) → A (alanine, 最常见氨基酸)
    # J (leucine/isoleucine ambiguous) → L (leucine)
    # B (asx: aspartic acid/asparagine) → D (aspartic acid)
    # Z (glx: glutamic acid/glutamine) → E (glutamic acid)
    # O (pyrrolysine) → K (lysine)
    NON_STANDARD_AA_MAP = {
        'U': 'C', 'X': 'A', 'J': 'L', 'B': 'D', 'Z': 'E', 'O': 'K',
    }

    def standardize_amino_acids(
        self, df: pd.DataFrame, sequence_col: str = "sequence"
    ) -> pd.DataFrame:
        """
        将非标准氨基酸字符映射为标准字符

        映射规则:
            U (selenocysteine) → C (cysteine, 生化性质最接近)
            X (unknown) → A (alanine, 最常见氨基酸)
            J (leucine/isoleucine) → L (leucine)
            B (asx) → D (aspartic acid)
            Z (glx) → E (glutamic acid)
            O (pyrrolysine) → K (lysine)

        参数:
            df: 输入DataFrame
            sequence_col: 序列列名

        返回:
            标准化后的DataFrame
        """
        logger.info("开始标准化氨基酸字符")
        df = df.copy()
        original_count = 0

        for old_char, new_char in self.NON_STANDARD_AA_MAP.items():
            # 统计替换前的数量
            count = df[sequence_col].apply(lambda s: s.count(old_char)).sum()
            if count > 0:
                original_count += count
                df[sequence_col] = df[sequence_col].str.replace(
                    old_char, new_char, regex=False
                )
                logger.info(
                    "将 %d 个 '%s' 替换为 '%s'", count, old_char, new_char
                )

        if original_count > 0:
            logger.info(
                "氨基酸标准化完成: 共替换 %d 个非标准字符", original_count
            )
        else:
            logger.info("无需标准化，所有字符均为标准氨基酸")

        return df

    def clean_sequences(self, df: pd.DataFrame, sequence_col: str = "sequence") -> pd.DataFrame:
        """
        清洗序列数据

        参数:
            df: 输入DataFrame
            sequence_col: 序列列名

        返回:
            清洗后的DataFrame
        """
        logger.info("开始清洗序列数据")
        df = df.copy()

        valid_mask = []

        for idx, seq in df[sequence_col].items():
            if pd.isna(seq):
                valid_mask.append(False)
                continue

            cleaned_seq = clean_sequence(str(seq))
            is_valid, _ = validate_sequence(
                cleaned_seq,
                self.max_sequence_length,
                self.valid_amino_acids,
            )

            if is_valid:
                df.at[idx, sequence_col] = cleaned_seq
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        num_invalid = len(valid_mask) - sum(valid_mask)
        if num_invalid > 0:
            logger.warning("发现 %d 条无效序列", num_invalid)
            if self.handle_missing == "drop":
                df = df[valid_mask].reset_index(drop=True)
                logger.info("已删除无效序列，剩余 %d 条", len(df))

        return df

    def normalize_ptm_labels(self, df: pd.DataFrame, ptm_col: str = "ptm_sites") -> pd.DataFrame:
        """
        标准化PTM标签

        参数:
            df: 输入DataFrame
            ptm_col: PTM列名

        返回:
            处理后的DataFrame
        """
        logger.info("标准化PTM标签")
        df = df.copy()

        for idx, ptm_data in df[ptm_col].items():
            if pd.isna(ptm_data):
                df.at[idx, ptm_col] = json.dumps([])
                continue

            try:
                if isinstance(ptm_data, str):
                    ptm_sites = json.loads(ptm_data)
                elif isinstance(ptm_data, list):
                    ptm_sites = ptm_data
                else:
                    ptm_sites = []
            except json.JSONDecodeError:
                ptm_sites = []

            if "sequence" in df.columns:
                seq_length = len(str(df.at[idx, "sequence"]))
            else:
                seq_length = None

            valid_sites = []
            for site in ptm_sites:
                is_valid, _ = validate_ptm_site(site, seq_length)
                if is_valid:
                    valid_sites.append(site)

            df.at[idx, ptm_col] = json.dumps(valid_sites)
        return df

    def remove_duplicate_sequences(self, df: pd.DataFrame, sequence_col: str = "sequence") -> pd.DataFrame:
        """
        移除重复序列

        参数:
            df: 输入DataFrame
            sequence_col: 序列列名

        返回:
            去重后的DataFrame
        """
        if not self.remove_duplicates:
            return df

        logger.info("移除重复序列")
        initial_count = len(df)
        df = df.drop_duplicates(subset=[sequence_col], keep="first").reset_index(drop=True)
        removed_count = initial_count - len(df)

        if removed_count > 0:
            logger.info("移除了 %d 条重复序列", removed_count)

        return df

    def split_dataset(
        self,
        df: pd.DataFrame,
        label_col: str = "cell_state",
        ratios: Optional[Tuple[float, float, float]] = None,
        random_state: int = 42,
        sequence_col: str = "sequence",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        划分训练集、验证集和测试集

        参数:
            df: 输入DataFrame
            label_col: 标签列名
            ratios: (train_ratio, val_ratio, test_ratio)元组
            random_state: 随机种子
            sequence_col: 序列列名（用于同源分割）

        返回:
            (train_df, val_df, test_df)元组
        """
        # 检查是否启用同源感知分割
        split_config = self.config.get("data", {}).get("split", {})
        homology_aware = split_config.get("homology_aware", False)

        if homology_aware and sequence_col in df.columns:
            logger.info("使用同源感知分割")
            from src.data.homology_splitter import HomologyAwareSplitter

            identity_threshold = split_config.get("identity_threshold", 0.3)
            splitter = HomologyAwareSplitter(
                identity_threshold=identity_threshold,
                random_state=random_state,
            )

            train_df, val_df, test_df = splitter.split(
                df,
                sequence_col=sequence_col,
                label_col=label_col if label_col in df.columns else None,
                ratios=ratios or (0.7, 0.15, 0.15),
            )

            # 验证无泄漏
            verify = split_config.get("verify_no_leakage", True)
            if verify:
                results = splitter.verify_no_leakage(train_df, val_df, test_df, sequence_col)
                if not results["passed"]:
                    logger.warning("同源分割验证发现问题，但继续执行")

            return train_df, val_df, test_df

        # 原有的随机分割逻辑
        if ratios is None:
            ratios = (
                split_config.get("train_ratio", 0.7),
                split_config.get("val_ratio", 0.15),
                split_config.get("test_ratio", 0.15),
            )

        train_ratio, val_ratio, test_ratio = ratios
        stratified = self.config.get("data", {}).get("split", {}).get("stratified", True)

        logger.info(
            "划分数据集: train=%.2f, val=%.2f, test=%.2f",
            train_ratio,
            val_ratio,
            test_ratio,
        )

        if len(df) < 3:
            logger.warning("样本量过小，使用简化划分策略")
            if len(df) == 1:
                return df.reset_index(drop=True), df.iloc[0:0].copy(), df.iloc[0:0].copy()
            train = df.iloc[:1].reset_index(drop=True)
            val = df.iloc[1:2].reset_index(drop=True)
            test = df.iloc[0:0].copy()
            return train, val, test

        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            logger.warning("比例总和不为1.0: %.6f，将进行归一化", total)
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total

        stratify = df[label_col] if (stratified and label_col in df.columns) else None
        if stratify is not None:
            value_counts = df[label_col].value_counts()
            if (value_counts < 2).any():
                logger.warning("样本量过小，无法进行分层划分，已切换为非分层")
                stratify = None

        try:
            train_val, test = train_test_split(
                df,
                test_size=test_ratio,
                random_state=random_state,
                stratify=stratify,
            )
        except ValueError:
            logger.warning("分层划分失败，已切换为非分层")
            train_val, test = train_test_split(
                df,
                test_size=test_ratio,
                random_state=random_state,
                stratify=None,
            )

        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        if len(train_val) < 2:
            logger.warning("训练/验证样本过少，跳过二次划分")
            return train_val.reset_index(drop=True), train_val.iloc[0:0].copy(), test.reset_index(drop=True)
        stratify_train_val = train_val[label_col] if (stratified and label_col in train_val.columns) else None
        if stratify_train_val is not None:
            value_counts = train_val[label_col].value_counts()
            if (value_counts < 2).any():
                logger.warning("训练/验证样本量过小，无法分层，已切换为非分层")
                stratify_train_val = None

        try:
            train, val = train_test_split(
                train_val,
                test_size=val_ratio_adjusted,
                random_state=random_state,
                stratify=stratify_train_val,
            )
        except ValueError:
            logger.warning("分层划分失败，已切换为非分层")
            train, val = train_test_split(
                train_val,
                test_size=val_ratio_adjusted,
                random_state=random_state,
                stratify=None,
            )

        logger.info("数据集划分完成: train=%d, val=%d, test=%d", len(train), len(val), len(test))
        return train, val, test

    def preprocess_pipeline(
        self,
        df: pd.DataFrame,
        sequence_col: str = "sequence",
        ptm_col: str = "ptm_sites",
        label_col: str = "cell_state",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        完整的预处理流程

        参数:
            df: 原始DataFrame
            sequence_col: 序列列名
            ptm_col: PTM列名
            label_col: 标签列名

        返回:
            (train_df, val_df, test_df)元组
        """
        logger.info("开始完整的预处理流程")

        if sequence_col in df.columns:
            # 先标准化非标准氨基酸字符
            df = self.standardize_amino_acids(df, sequence_col)
            # 再清洗序列
            df = self.clean_sequences(df, sequence_col)
            df = self.remove_duplicate_sequences(df, sequence_col)

        if ptm_col in df.columns:
            df = self.normalize_ptm_labels(df, ptm_col)

        train, val, test = self.split_dataset(df, label_col)

        logger.info("预处理流程完成")
        return train, val, test
