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
from src.data.aa_constants import NON_STANDARD_AA_MAP as _NON_STANDARD_AA_MAP

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

    # 非标准氨基酸字符映射表 (统一从aa_constants导入)
    NON_STANDARD_AA_MAP = dict(_NON_STANDARD_AA_MAP)

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

    def remove_duplicate_sequences(
        self,
        df: pd.DataFrame,
        sequence_col: str = "sequence",
        subset_columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
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
        subset = subset_columns or [sequence_col]
        df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
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

    def validate_data_quality(
        self,
        df: pd.DataFrame,
        sequence_col: str = "sequence",
        ptm_col: str = "ptm_sites",
        label_col: str = "cell_state",
        *,
        strict: bool = False,
    ) -> Dict[str, object]:
        """Aggregate data-quality checks *before* the expensive preprocess pipeline.

        Implements audit P1-2 strategy 4: surface label-distribution, sequence
        and PTM-site problems early so users do not train a model on
        degenerate data and then mistake the resulting weights for useful ones.

        The method is non-destructive: it only inspects ``df`` and never mutates
        it. It reports counts/distributions and, when ``strict=True``, raises
        :class:`DataQualityError` on the first hard failure.

        Args:
            df: Raw input DataFrame.
            sequence_col: Name of the sequence column.
            ptm_col: Name of the PTM sites column (JSON string or list).
            label_col: Name of the cell-state label column.
            strict: When True, raise :class:`DataQualityError` on the first
                hard failure instead of just collecting warnings.

        Returns:
            A dict with keys:

            * ``ok`` (bool): True when no hard failures were found.
            * ``warnings`` (list[str]): Soft issues (e.g. imbalance).
            * ``hard_failures`` (list[str]): Issues that would make training
              meaningless (no labels, single class, empty frame, etc.).
            * ``label_distribution`` (dict): label -> count.
            * ``sequence_length_stats`` (dict): min/max/mean length.
            * ``invalid_ptm_site_count`` (int): sites failing positional/type
              validation.
            * ``row_count`` (int): rows inspected.

        Raises:
            DataQualityError: If ``strict`` is True and a hard failure occurs.
        """
        warnings: list[str] = []
        hard_failures: list[str] = []

        report: Dict[str, object] = {
            "ok": True,
            "warnings": warnings,
            "hard_failures": hard_failures,
            "label_distribution": {},
            "sequence_length_stats": {},
            "invalid_ptm_site_count": 0,
            "row_count": int(len(df)),
        }

        if len(df) == 0:
            hard_failures.append("输入 DataFrame 为空，无法训练。")
            report["ok"] = False
            if strict:
                raise DataQualityError(hard_failures[0], report)
            return report

        # --- Label column -------------------------------------------------
        if label_col not in df.columns:
            hard_failures.append(
                f"缺少标签列 '{label_col}'；训练数据必须包含 cell_state 标签。"
            )
        else:
            non_null = df[label_col].dropna()
            n_null = len(df) - len(non_null)
            if n_null > 0:
                warnings.append(f"标签列有 {n_null} 个空值（占比 {n_null / len(df):.1%}）。")
            dist = non_null.astype(str).value_counts().to_dict()
            report["label_distribution"] = {str(k): int(v) for k, v in dist.items()}
            if len(dist) < 2:
                hard_failures.append(
                    f"标签列仅含 {len(dist)} 个类别（需 ≥ 2 才能训练）：" f"{list(dist)}"
                )
            elif len(dist) == 2:
                # Binary is fine, but note it for awareness.
                warnings.append("标签为二分类，将走二分类训练路径。")
            # Imbalance check applies to binary and multi-class alike.
            if len(dist) >= 2:
                counts = list(dist.values())
                max_c, min_c = max(counts), min(counts)
                ratio = max_c / min_c if min_c > 0 else float("inf")
                if ratio > 10:
                    warnings.append(
                        f"标签分布严重不均衡（最大/最小 = {ratio:.1f}），"
                        "建议使用分层采样或类别加权。"
                    )

        # --- Sequence column ----------------------------------------------
        if sequence_col not in df.columns:
            hard_failures.append(
                f"缺少序列列 '{sequence_col}'；训练数据必须包含蛋白质序列。"
            )
        else:
            seqs = df[sequence_col].dropna().astype(str)
            empty_seqs = int((seqs.str.len() == 0).sum())
            if empty_seqs > 0:
                warnings.append(f"序列列有 {empty_seqs} 条空字符串。")
            too_long = int((seqs.str.len() > self.max_sequence_length).sum())
            if too_long > 0:
                warnings.append(
                    f"{too_long} 条序列超过 max_sequence_length="
                    f"{self.max_sequence_length}，预处理时会被删除。"
                )
            invalid_chars = 0
            for s in seqs:
                if s:
                    invalid_chars += len(set(s) - self.valid_amino_acids)
            if seqs.any():
                lens = seqs.str.len()
                report["sequence_length_stats"] = {
                    "min": int(lens.min()),
                    "max": int(lens.max()),
                    "mean": float(lens.mean()),
                }

        # --- PTM sites column ---------------------------------------------
        if ptm_col in df.columns:
            invalid_sites = 0
            for _, row in df.iterrows():
                raw = row.get(ptm_col)
                if pd.isna(raw):
                    continue
                try:
                    sites = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(sites, list):
                    continue
                seq_len = len(str(row[sequence_col])) if sequence_col in df.columns and pd.notna(row.get(sequence_col)) else None
                for site in sites:
                    ok, _ = validate_ptm_site(site, seq_len)
                    if not ok:
                        invalid_sites += 1
            report["invalid_ptm_site_count"] = invalid_sites
            if invalid_sites > 0:
                warnings.append(
                    f"{invalid_sites} 个 PTM 位点未通过校验（位置越界或类型未知），"
                    "预处理时会被丢弃。"
                )

        report["ok"] = len(hard_failures) == 0
        if hard_failures and strict:
            raise DataQualityError(
                "数据质量预检未通过：\n  - " + "\n  - ".join(hard_failures),
                report,
            )
        return report

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

        异常:
            EmptyDatasetError: 预处理后任一划分为空（尤其训练集），包含原始样本数、
                过滤统计与当前 max_sequence_length，便于定位配置/数据问题。
        """
        logger.info("开始完整的预处理流程")

        original_count = len(df)
        seq_len_stats = None
        if sequence_col in df.columns and original_count > 0:
            seq_lens = df[sequence_col].astype(str).str.len()
            seq_len_stats = {
                "min": int(seq_lens.min()),
                "max": int(seq_lens.max()),
                "mean": float(seq_lens.mean()),
            }

        if sequence_col in df.columns:
            # 先标准化非标准氨基酸字符
            df = self.standardize_amino_acids(df, sequence_col)
            # 再清洗序列
            after_clean = None
            if self.handle_missing == "drop":
                # clean_sequences 在 drop 模式下会删除无效序列；记录过滤量
                before_clean = len(df)
                df = self.clean_sequences(df, sequence_col)
                after_clean = len(df)
            else:
                df = self.clean_sequences(df, sequence_col)
                after_clean = len(df)
            before_dedup = len(df)
            dedup_subset = [sequence_col]
            for col in (label_col, ptm_col):
                if col in df.columns and col not in dedup_subset:
                    dedup_subset.append(col)
            df = self.remove_duplicate_sequences(df, sequence_col, dedup_subset)
            after_dedup = len(df)
        else:
            before_clean = after_clean = before_dedup = after_dedup = len(df)

        if ptm_col in df.columns:
            df = self.normalize_ptm_labels(df, ptm_col)

        train, val, test = self.split_dataset(df, label_col)

        # P1-3: 预处理后空数据集 fail-fast，带可操作上下文
        filter_stats = {
            "original_sample_count": original_count,
            "after_clean_sequences": after_clean,
            "after_dedup": after_dedup,
            "removed_by_clean": before_clean - after_clean,
            "removed_by_dedup": before_dedup - after_dedup,
            "sequence_length_at_intake": seq_len_stats,
            "max_sequence_length_config": self.max_sequence_length,
            "split_sizes": {
                "train": len(train),
                "val": len(val),
                "test": len(test),
            },
        }
        if len(train) == 0:
            raise EmptyDatasetError(
                "预处理后训练集为空，无法继续训练。这通常意味着数据被全部过滤掉。"
                f" 过滤统计: {filter_stats}。"
                " 常见原因：max_sequence_length 过小（序列均超长被删）、"
                "valid_amino_acids 不覆盖数据中出现的字符、或数据文件列名不匹配。"
            )
        if len(val) == 0 or len(test) == 0:
            logger.warning(
                "预处理后验证/测试集为空（但仍可训练）。过滤统计: %s", filter_stats
            )

        logger.info("预处理流程完成")
        return train, val, test


class EmptyDatasetError(ValueError):
    """预处理后数据集为空（通常因配置/数据不匹配导致样本被全部过滤）。

    抛出时携带原始样本数、各阶段过滤量与当前 ``max_sequence_length``，
    供 CLI 捕获并打印可操作错误（对应审计报告 P1-3）。
    """


class DataQualityError(ValueError):
    """原始数据未通过质量预检（审计 P1-2 策略 4）。

    由 :meth:`DataPreprocessor.validate_data_quality` 在 ``strict=True`` 时抛出，
    携带结构化 ``report`` 字典，供 CLI 打印聚合统计。
    """

    def __init__(self, message: str, report: Optional[Dict[str, object]] = None):
        super().__init__(message)
        self.report = report or {}
