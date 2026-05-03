"""
数据验证和缓存模块
功能概述: 提供数据验证、统计信息收集和缓存功能
设计思路: 确保数据质量，提高数据加载性能
"""
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

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
    """

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

    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.pkl"

    def load(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[List[Any]]:
        """
        尝试从缓存加载数据

        参数:
            df: 数据DataFrame
            config: 配置字典

        返回:
            缓存的数据列表，如果不存在则返回None
        """
        cache_key = self._get_cache_key(df, config)
        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                logger.info(f"从缓存加载数据: {cache_path.name}")
                return data
            except Exception as e:
                logger.warning(f"缓存加载失败: {e}")

        return None

    def save(self, df: pd.DataFrame, config: Dict[str, Any], data: List[Any]) -> None:
        """
        保存数据到缓存

        参数:
            df: 数据DataFrame
            config: 配置字典
            data: 要缓存的数据列表
        """
        cache_key = self._get_cache_key(df, config)
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"数据已缓存: {cache_path.name}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")

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
    print(f"\n{'='*60}")
    print(f"{dataset_name} 验证报告")
    print(f"{'='*60}")

    validator = DataValidator()
    is_valid, errors = validator.validate_dataframe(df)

    if is_valid:
        print("✓ 数据验证通过")
        print(f"  - 样本数: {len(df)}")
        print(f"  - 列名: {list(df.columns)}")
        if "sequence" in df.columns:
            print(f"  - 序列长度范围: {df['sequence'].apply(len).min()} - {df['sequence'].apply(len).max()}")
    else:
        print(f"✗ 发现 {len(errors)} 个错误:")
        for error in errors[:10]:  # 只显示前10个错误
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors) - 10} 个错误")

    print(f"{'='*60}\n")
    return is_valid
