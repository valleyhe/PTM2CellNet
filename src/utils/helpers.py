"""
辅助函数模块
功能概述: 提供通用的辅助函数，包括序列验证、PTM位点验证等
设计思路: 提供可复用的工具函数，集中处理通用的验证和转换操作
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def validate_sequence(
    sequence: str,
    max_length: Optional[int] = None,
    valid_amino_acids: Optional[Set[str]] = None,
) -> Tuple[bool, str]:
    """
    验证蛋白质序列的有效性

    参数:
        sequence: 蛋白质序列字符串
        max_length: 最大允许长度，None表示不限制

    返回:
        (是否有效, 错误信息)元组

    示例:
        >>> validate_sequence("ACDEFGHIKLMNPQRSTVWY")
        (True, "")
        >>> validate_sequence("ACXDE")
        (False, "包含无效氨基酸: X")
    """
    if not sequence or not isinstance(sequence, str):
        return False, "序列不能为空"

    if max_length is not None and len(sequence) > max_length:
        return False, f"序列长度超过限制: {len(sequence)} > {max_length}"

    allowed = valid_amino_acids if valid_amino_acids is not None else VALID_AMINO_ACIDS
    invalid_chars = set(sequence) - allowed
    if invalid_chars:
        return False, f"包含无效氨基酸: {', '.join(sorted(invalid_chars))}"

    return True, ""


def validate_ptm_site(ptm_site: Dict[str, Any], sequence_length: Optional[int] = None) -> Tuple[bool, str]:
    """
    验证PTM位点的有效性

    参数:
        ptm_site: PTM位点字典，需包含"position"和"type"字段
        sequence_length: 关联序列的长度，用于验证位置有效性

    返回:
        (是否有效, 错误信息)元组

    示例:
        >>> validate_ptm_site({"position": 5, "type": "phosphorylation"})
        (True, "")
    """
    if not isinstance(ptm_site, dict):
        return False, "PTM位点必须是字典类型"

    required_fields = ["position", "type"]
    for field in required_fields:
        if field not in ptm_site:
            return False, f"缺少必需字段: {field}"

    position = ptm_site["position"]
    if not isinstance(position, int) or position < 1:
        return False, "position必须是正整数"

    if sequence_length is not None and position > sequence_length:
        return False, f"position超出序列长度: {position} > {sequence_length}"

    ptm_type = ptm_site["type"]
    if not isinstance(ptm_type, str) or not ptm_type:
        return False, "type必须是非空字符串"

    return True, ""


def clean_sequence(sequence: str) -> str:
    """
    清理蛋白质序列，移除空白字符并转为大写

    参数:
        sequence: 原始序列

    返回:
        清理后的序列
    """
    return re.sub(r"\s+", "", sequence).upper()


def get_amino_acid_counts(sequence: str) -> Dict[str, int]:
    """
    统计序列中各氨基酸的出现次数

    参数:
        sequence: 蛋白质序列

    返回:
        氨基酸计数字典
    """
    counts: Dict[str, int] = {}
    for aa in sequence:
        counts[aa] = counts.get(aa, 0) + 1
    return counts


def calculate_sequence_length_stats(sequences: List[str]) -> Dict[str, float]:
    """
    计算序列长度统计信息

    参数:
        sequences: 序列列表

    返回:
        统计信息字典，包含mean, std, min, max
    """
    import numpy as np

    lengths = [len(seq) for seq in sequences if seq]
    if not lengths:
        return {"mean": 0.0, "std": 0.0, "min": 0, "max": 0}

    return {
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "count": len(lengths),
    }


def validate_cell_state_label(label: str, valid_labels: Optional[Set[str]] = None) -> Tuple[bool, str]:
    """
    验证细胞状态标签的有效性

    参数:
        label: 细胞状态标签
        valid_labels: 有效标签集合，None表示不限制

    返回:
        (是否有效, 错误信息)元组
    """
    if not label or not isinstance(label, str):
        return False, "标签不能为空"

    if valid_labels is not None and label not in valid_labels:
        return False, f"无效标签: {label}，有效标签: {sorted(valid_labels)}"

    return True, ""
