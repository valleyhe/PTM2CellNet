"""
数据验证和缓存模块
功能概述: 提供数据验证、统计信息收集和缓存功能
设计思路: 确保数据质量，提高数据加载性能
"""
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd
import torch

from ..utils.io import safe_pickle_load
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class DataValidator:
    """
    数据验证器
    验证数据格式和内容的正确性
    """

    def __init__(self, valid_amino_acids: str = "ACDEFGHIKLMNPQRSTVWY"):
        self.valid_amino_acids = set(valid_amino_acids)

    def validate_sequence(self, sequence: str, max_length: int = 10000) -> Tuple[bool, str]:
        """
        验证序列有效性

        参数:
            sequence: 氨基酸序列
            max_length: 最大允许长度

        返回:
            (是否有效, 错误信息)
        """
        if not isinstance(sequence, str):
            return False, f"序列必须是字符串，实际类型: {type(sequence)}"

        if len(sequence) == 0:
            return False, "序列为空"

        if len(sequence) > max_length:
            return False, f"序列长度 {len(sequence)} 超过最大限制 {max_length}"

        invalid_chars = set(sequence) - self.valid_amino_acids
        if invalid_chars:
            return False, f"包含无效氨基酸字符: {invalid_chars}"

        return True, ""

    def validate_ptm_site(
        self,
        site: Dict[str, Any],
        seq_length: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        验证单个PTM位点

        参数:
            site: PTM位点字典
            seq_length: 序列长度

        返回:
            (是否有效, 错误信息)
        """
        if not isinstance(site, dict):
            return False, "PTM位点必须是字典"

        if "position" not in site:
            return False, "PTM位点缺少position字段"

        pos = site["position"]
        if not isinstance(pos, int):
            return False, f"position必须是整数，实际类型: {type(pos)}"

        if pos < 1:
            return False, f"position必须>=1，实际值: {pos}"

        if seq_length is not None and pos > seq_length:
            return False, f"position {pos} 超出序列长度 {seq_length}"

        if "type" not in site:
            return False, "PTM位点缺少type字段"

        return True, ""

    def validate_ptm_sites(
        self,
        ptm_sites: List[Dict[str, Any]],
        seq_length: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        验证PTM位点列表

        参数:
            ptm_sites: PTM位点列表
            seq_length: 序列长度

        返回:
            (有效位点列表, 错误信息列表)
        """
        valid_sites = []
        errors = []

        for i, site in enumerate(ptm_sites):
            is_valid, error_msg = self.validate_ptm_site(site, seq_length)
            if is_valid:
                valid_sites.append(site)
            else:
                errors.append(f"位点 {i}: {error_msg}")

        return valid_sites, errors

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        验证DataFrame格式

        参数:
            df: 输入DataFrame
            required_columns: 必需的列名列表

        返回:
            (是否有效, 错误信息列表)
        """
        errors = []

        if df.empty:
            errors.append("DataFrame为空")

        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                errors.append(f"缺少必需列: {missing_cols}")

        # 验证序列列（如果存在）
        if "sequence" in df.columns:
            for idx, seq in df["sequence"].items():
                if pd.isna(seq):
                    errors.append(f"行 {idx}: sequence为空")
                    continue

                is_valid, error_msg = self.validate_sequence(str(seq))
                if not is_valid:
                    errors.append(f"行 {idx}: sequence无效 - {error_msg}")

        # 验证PTM列（如果存在）
        if "ptm_sites" in df.columns:
            for idx, ptm_data in df["ptm_sites"].items():
                if pd.isna(ptm_data):
                    continue

                try:
                    if isinstance(ptm_data, str):
                        ptm_sites = json.loads(ptm_data)
                    elif isinstance(ptm_data, list):
                        ptm_sites = ptm_data
                    else:
                        errors.append(f"行 {idx}: ptm_sites格式无效")
                        continue

                    seq_length = None
                    if "sequence" in df.columns and not pd.isna(df.at[idx, "sequence"]):
                        seq_length = len(str(df.at[idx, "sequence"]))

                    _, ptm_errors = self.validate_ptm_sites(ptm_sites, seq_length)
                    for error in ptm_errors:
                        errors.append(f"行 {idx}: {error}")

                except json.JSONDecodeError:
                    errors.append(f"行 {idx}: ptm_sites JSON解析失败")

        return len(errors) == 0, errors


class DatasetCache:
    """
    数据集缓存管理器
    缓存预处理后的数据以提高加载速度

    安全措施:
        - 缓存数据包含版本号，版本不匹配时自动失效
        - 缓存数据包含源数据校验和，检测数据损坏或源变更
        - 加载时使用 try/except 捕获反序列化异常并给出明确错误信息

    # TODO: 生产环境应考虑迁移到 safetensors 格式，彻底消除 pickle 反序列化风险。
    #       safetensors 不支持任意 Python 对象，天然防止代码注入。
    #       参考: https://github.com/huggingface/safetensors
    """

    CACHE_VERSION = 1

    def __init__(self, cache_dir: str = ".cache/ptm_dataset"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, df: pd.DataFrame, config: Dict[str, Any]) -> str:
        """生成缓存key"""
        # 基于数据内容和配置的哈希
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()[:16]

        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        return f"{data_hash}_{config_hash}"

    def _compute_data_checksum(self, df: pd.DataFrame) -> str:
        """计算源数据的校验和，用于检测缓存与源数据不一致或数据损坏"""
        return hashlib.sha256(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        ).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.pkl"

    def load(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[List[Any]]:
        """
        尝试从缓存加载数据

        验证步骤:
            1. 缓存版本号必须与当前 CACHE_VERSION 一致，否则失效
            2. 源数据校验和必须匹配，否则失效（源数据已变更）
            3. 反序列化异常会被捕获并记录，不会传播

        参数:
            df: 数据DataFrame
            config: 配置字典

        返回:
            缓存的数据列表，如果不存在或验证失败则返回None
        """
        cache_key = self._get_cache_key(df, config)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                cached = safe_pickle_load(f)
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            logger.warning(
                "缓存反序列化失败（文件可能已损坏）: %s — 将重新计算。路径: %s",
                e, cache_path.name,
            )
            return None
        except Exception as e:
            logger.warning(
                "缓存加载出现意外错误: %s: %s — 将重新计算。路径: %s",
                type(e).__name__, e, cache_path.name,
            )
            return None

        # 验证缓存结构
        if not isinstance(cached, dict) or "cache_version" not in cached or "data" not in cached:
            logger.warning(
                "缓存格式无效（缺少 cache_version 或 data 字段）— 将重新计算。路径: %s",
                cache_path.name,
            )
            return None

        # 验证版本号
        if cached["cache_version"] != self.CACHE_VERSION:
            logger.info(
                "缓存版本不匹配（当前=%d, 缓存=%d）— 将重新计算。路径: %s",
                self.CACHE_VERSION, cached["cache_version"], cache_path.name,
            )
            return None

        # 验证源数据校验和
        expected_checksum = self._compute_data_checksum(df)
        stored_checksum = cached.get("data_checksum", "")
        if stored_checksum != expected_checksum:
            logger.info(
                "源数据校验和不匹配（缓存可能过期）— 将重新计算。路径: %s",
                cache_path.name,
            )
            return None

        logger.info("从缓存加载数据: %s", cache_path.name)
        return cast(List[Any], cached["data"])

    def save(self, df: pd.DataFrame, config: Dict[str, Any], data: List[Any]) -> None:
        """
        保存数据到缓存

        保存结构包含版本号和源数据校验和，供 load() 验证使用。

        参数:
            df: 数据DataFrame
            config: 配置字典
            data: 要缓存的数据列表
        """
        cache_key = self._get_cache_key(df, config)
        cache_path = self._get_cache_path(cache_key)

        cache_payload = {
            "cache_version": self.CACHE_VERSION,
            "data_checksum": self._compute_data_checksum(df),
            "data": data,
        }

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cache_payload, f)
            logger.info("数据已缓存: %s", cache_path.name)
        except Exception as e:
            logger.warning("缓存保存失败: %s", e)

    def clear(self) -> None:
        """清除所有缓存"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("缓存已清除")

    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "cache_dir": str(self.cache_dir),
            "num_files": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
        }


def collate_sequences(batch: List[Dict[str, torch.Tensor]], pad_value: int = 0) -> Dict[str, torch.Tensor]:
    """
    自定义collate函数，处理变长序列

    参数:
        batch: 样本列表
        pad_value: 填充值

    返回:
        填充后的批次字典
    """
    keys = batch[0].keys()
    result = {}

    for key in keys:
        values = [sample[key] for sample in batch]

        if values[0].dim() == 0:
            # 标量tensor（如label）
            result[key] = torch.stack(values)
        elif values[0].dim() == 1:
            # 1D tensor（如序列）
            max_len = max(v.shape[0] for v in values)
            padded = torch.full((len(values), max_len), pad_value, dtype=values[0].dtype)
            for i, v in enumerate(values):
                padded[i, :v.shape[0]] = v
            result[key] = padded
        elif values[0].dim() == 2:
            # 2D tensor
            max_len = max(v.shape[0] for v in values)
            padded = torch.full((len(values), max_len, values[0].shape[1]), pad_value, dtype=values[0].dtype)
            for i, v in enumerate(values):
                padded[i, :v.shape[0], :] = v
            result[key] = padded
        else:
            result[key] = torch.stack(values)

    return result


def validate_and_report(df: pd.DataFrame, dataset_name: str = "数据集") -> bool:
    """
    验证数据并打印报告

    参数:
        df: 数据DataFrame
        dataset_name: 数据集名称

    返回:
        是否通过验证
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("%s 验证报告", dataset_name)
    logger.info("=" * 60)

    validator = DataValidator()
    is_valid, errors = validator.validate_dataframe(df)

    if is_valid:
        logger.info("✓ 数据验证通过")
        logger.info("  - 样本数: %d", len(df))
        logger.info("  - 列名: %s", list(df.columns))
        if "sequence" in df.columns:
            seq_min = df['sequence'].apply(len).min()
            seq_max = df['sequence'].apply(len).max()
            logger.info("  - 序列长度范围: %d - %d", seq_min, seq_max)
    else:
        logger.info("✗ 发现 %d 个错误:", len(errors))
        for error in errors[:10]:  # 只显示前10个错误
            logger.info("  - %s", error)
        if len(errors) > 10:
            logger.info("  ... 还有 %d 个错误", len(errors) - 10)

    logger.info("=" * 60)
    return is_valid
