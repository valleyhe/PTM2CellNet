# mypy: disable-error-code="annotation-unchecked"
"""
模型工具函数
功能概述: 提供模型配置验证、参数量统计、计算量分析等工具
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

# --- 估算常量（解释内存/批次估算中出现的魔数）---
# float32 参数每个占 4 字节。
BYTES_PER_PARAM = 4
# 1 MB 的字节数，用于将字节换算为兆字节。
BYTES_PER_MB = 1024 ** 2
# 为优化器状态与中间激活预留的内存比例（可用内存的 50%）。
MEMORY_RESERVE_RATIO = 0.5
# 无法从模型探测 hidden_dim 时的保守默认值。
DEFAULT_HIDDEN_DIM = 128
# 批次大小估算的默认可用 GPU 显存（GB）。
DEFAULT_AVAILABLE_MEMORY_GB = 8.0
# 估算最大批次大小后施加的上限，避免推荐过大批次。
MAX_BATCH_SIZE_CAP = 512


# ---------------------------------------------------------------------------
# TypedDict definitions for structured dicts used in this module
# ---------------------------------------------------------------------------

class _ModuleParamStats(TypedDict):
    """Per-module parameter statistics."""

    total: int
    trainable: int
    percentage: float


class ParameterStats(TypedDict):
    """Shape of the dict returned by ``count_parameters``."""

    total_params: int
    total_params_m: float
    trainable_params: int
    trainable_params_m: float
    frozen_params: int
    modules: Dict[str, _ModuleParamStats]


class MemoryEstimate(TypedDict):
    """Shape of the dict returned by ``get_model_memory_usage``."""

    params_memory_mb: float
    activation_memory_mb: float
    total_memory_mb: float
    recommended_batch_size: int


def validate_model_config(config: Dict[str, Union[str, int, float, bool, List[Any]]]) -> Tuple[bool, List[str]]:
    """
    验证模型配置的有效性

    参数:
        config: 模型配置字典

    返回:
        (是否有效, 错误信息列表)
    """
    errors = []
    model_cfg = config.get("model", {})

    # 必需字段检查
    required_fields = ["encoder_type"]
    for field in required_fields:
        if field not in model_cfg:
            errors.append(f"缺少必需字段: model.{field}")

    # 编码器类型检查
    valid_encoders = [
        "cnn", "transformer", "lstm", "gru", "mamba",
        "esm2", "esm2_8m", "esm2_35m", "esm2_70m", "esm2_150m", "esm2_650m", "esm2_3b",
        "protbert", "prott5"
    ]
    encoder_type = model_cfg.get("encoder_type", "").lower()
    if encoder_type and encoder_type not in valid_encoders:
        errors.append(f"无效的编码器类型: {encoder_type}，可用: {valid_encoders}")

    # 数值范围检查
    hidden_dim = model_cfg.get("hidden_dim", 128)
    if not isinstance(hidden_dim, int) or hidden_dim <= 0:
        errors.append(f"hidden_dim必须是正整数，当前: {hidden_dim}")

    num_layers = model_cfg.get("num_layers", 2)
    if not isinstance(num_layers, int) or num_layers <= 0:
        errors.append(f"num_layers必须是正整数，当前: {num_layers}")

    num_heads = model_cfg.get("num_heads", 4)
    if not isinstance(num_heads, int) or num_heads <= 0:
        errors.append(f"num_heads必须是正整数，当前: {num_heads}")

    dropout = model_cfg.get("dropout", 0.1)
    if not isinstance(dropout, (int, float)) or dropout < 0 or dropout > 1:
        errors.append(f"dropout必须在[0, 1]范围内，当前: {dropout}")

    num_classes = model_cfg.get("num_classes", 4)
    if not isinstance(num_classes, int) or num_classes <= 0:
        errors.append(f"num_classes必须是正整数，当前: {num_classes}")

    return len(errors) == 0, errors


def count_parameters(model: nn.Module, trainable_only: bool = False) -> ParameterStats:
    """
    统计模型参数量

    参数:
        model: PyTorch模型
        trainable_only: 是否只统计可训练参数

    返回:
        参数量统计字典
    """
    if trainable_only:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())

    # 按模块统计
    module_params = {}
    for name, module in model.named_children():
        module_count = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        module_params[name] = {
            "total": module_count,
            "trainable": module_trainable,
            "percentage": module_count / total_params * 100 if total_params > 0 else 0
        }

    return {
        "total_params": total_params,
        "total_params_m": total_params / 1e6,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_params_m": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
        "frozen_params": total_params - sum(p.numel() for p in model.parameters() if p.requires_grad),
        "modules": module_params
    }


def get_model_memory_usage(
    model: nn.Module,
    batch_size: int = 1,
    seq_len: int = 1000,
    config: Optional[Dict[str, Union[str, int, float]]] = None,
) -> MemoryEstimate:
    """
    估算模型内存使用量

    参数:
        model: PyTorch模型
        batch_size: 批次大小
        seq_len: 序列长度
        config: 可选模型配置字典，用于提供 num_layers 等参数

    返回:
        内存使用量估算（MB）
    """
    param_count = sum(p.numel() for p in model.parameters())
    param_memory_mb = param_count * BYTES_PER_PARAM / BYTES_PER_MB  # 假设float32

    # 估算激活内存（简化计算）
    # 每层大约需要 batch_size * seq_len * hidden_dim * BYTES_PER_PARAM bytes
    hidden_dim = DEFAULT_HIDDEN_DIM
    if hasattr(model, 'embed_dim'):
        embed_dim = model.embed_dim
        # 确保是整数（如果是tensor，取item()）
        if isinstance(embed_dim, torch.Tensor):
            hidden_dim = int(embed_dim.item())
        else:
            hidden_dim = int(embed_dim)

    # 假设10层 → 改为从模型实际探测层数（避免对 base/large 变体误估）。
    hidden_dim, num_layers = _infer_activation_depth(model, config=config)
    activation_memory_mb = batch_size * seq_len * hidden_dim * num_layers * BYTES_PER_PARAM / BYTES_PER_MB

    return {
        "params_memory_mb": param_memory_mb,
        "activation_memory_mb": activation_memory_mb,
        "total_memory_mb": param_memory_mb + activation_memory_mb,
        "recommended_batch_size": estimate_max_batch_size(model, seq_len, config=config)
    }


def _infer_activation_depth(
    model: nn.Module,
    config: Optional[Dict[str, Union[str, int, float]]] = None,
) -> tuple:
    """Best-effort inference of (hidden_dim, num_layers) for activation memory.

    Previously this logic hardcoded ``num_layers = 10``, which wildly
    over-estimated memory for the small base model (2 layers) and
    under-estimated for the large variant (4+ layers). We now probe the model
    for the most common layer-count attributes, falling back to counting
    transformer/LSTM blocks when no explicit attribute is exposed.

    An optional *config* dict (typically the model config) is consulted first;
    keys ``num_layers``, ``n_layers``, or ``num_encoder_layers`` override
    any heuristic. This lets callers pass the authoritative layer count
    without relying on introspection.

    Returns ``(hidden_dim, num_layers)``. Both default to conservative values
    when nothing can be inferred so the caller still gets a usable estimate.
    """
    hidden_dim = getattr(model, "embed_dim", None) or getattr(model, "hidden_dim", None) or DEFAULT_HIDDEN_DIM

    # 0. Explicit config override — highest priority when available.
    if config is not None:
        for key in ("num_layers", "n_layers", "num_encoder_layers"):
            val = config.get(key)
            if isinstance(val, int) and val > 0:
                return int(hidden_dim), int(val)

    # 1. Explicit attributes on the model object.
    for attr in ("num_layers", "n_layers", "num_encoder_layers"):
        val = getattr(model, attr, None)
        if isinstance(val, int) and val > 0:
            return int(hidden_dim), int(val)

    # 2. Common submodule names that hold stacked blocks.
    for attr in ("layers", "encoder", "transformer", "blocks"):
        sub = getattr(model, attr, None)
        if sub is not None and hasattr(sub, "layers") and isinstance(sub.layers, (list, nn.ModuleList)):
            return int(hidden_dim), max(1, len(sub.layers))

    # 3. Heuristic: count TransformerEncoderLayer / LSTM / GRU / MambaBlock
    # submodules anywhere in the model tree.
    layer_like = (
        nn.TransformerEncoderLayer,
        nn.LSTM,
        nn.GRU,
        nn.ModuleList,
    )
    try:
        # Walk named_modules once; the deepest ModuleList of consistent length
        # is a reasonable proxy for the stack depth.
        candidate_depths = []
        for _name, mod in model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > 0:
                candidate_depths.append(len(mod))
        if candidate_depths:
            # Pick the largest stack we saw — usually the encoder stack.
            return int(hidden_dim), max(candidate_depths)
    except (AttributeError, TypeError, ValueError) as e:
        logger.warning("Failed to infer model depth from named_modules: %s", e)
        pass

    # 4. Conservative fallback (matches historical default).
    return int(hidden_dim), 10


def estimate_max_batch_size(model: nn.Module, seq_len: int = 1000,
                            available_memory_gb: float = DEFAULT_AVAILABLE_MEMORY_GB,
                            config: Optional[Dict[str, Union[str, int, float]]] = None) -> int:
    """
    估算最大批次大小

    参数:
        model: PyTorch模型
        seq_len: 序列长度
        available_memory_gb: 可用GPU内存（GB）
        config: 可选模型配置字典，用于提供 num_layers 等参数

    返回:
        估算的最大批次大小
    """
    available_memory_mb = available_memory_gb * 1024

    # 参数内存
    param_memory_mb = sum(p.numel() for p in model.parameters()) * BYTES_PER_PARAM / BYTES_PER_MB

    # 为优化器状态和其他开销预留 MEMORY_RESERVE_RATIO 比例内存
    usable_memory_mb = available_memory_mb * MEMORY_RESERVE_RATIO - param_memory_mb

    # 每个样本的激活内存。层数从模型实际探测而非硬编码（旧实现固定为
    # ``num_layers = 10``，对 base/large 变体都不准）。可由 config 显式指定。
    hidden_dim, num_layers = _infer_activation_depth(model, config=config)
    activation_per_sample_mb = seq_len * hidden_dim * num_layers * BYTES_PER_PARAM / BYTES_PER_MB

    if activation_per_sample_mb <= 0:
        return 1

    max_batch_size = int(usable_memory_mb / activation_per_sample_mb)
    return max(1, min(max_batch_size, MAX_BATCH_SIZE_CAP))


def print_model_summary(model: nn.Module, detailed: bool = False) -> None:
    """
    打印模型摘要信息

    参数:
        model: PyTorch模型
        detailed: 是否打印详细信息
    """
    stats = count_parameters(model)

    logger.info("")
    logger.info("=" * 60)
    logger.info("模型摘要")
    logger.info("=" * 60)
    logger.info("总参数量: %.2fM (%s)", stats['total_params_m'], f"{stats['total_params']:,}")
    logger.info("可训练参数: %.2fM (%s)", stats['trainable_params_m'], f"{stats['trainable_params']:,}")
    logger.info("冻结参数: %s", f"{stats['frozen_params']:,}")

    if detailed:
        logger.info("各模块参数量:")
        for name, module_stats in stats['modules'].items():
            logger.info("  %s: %.2fM (%.1f%%)", name, module_stats['total'] / 1e6, module_stats['percentage'])

    memory = get_model_memory_usage(model)
    logger.info("内存估算:")
    logger.info("  参数内存: %.2f MB", memory['params_memory_mb'])
    logger.info("  激活内存: %.2f MB", memory['activation_memory_mb'])
    logger.info("  推荐批次大小: %d", memory['recommended_batch_size'])
    logger.info("=" * 60)


def compare_models(model1: nn.Module, model2: nn.Module,
                   names: Tuple[str, str] = ("Model1", "Model2")) -> None:
    """
    比较两个模型的参数量和内存使用

    参数:
        model1: 第一个模型
        model2: 第二个模型
        names: 模型名称
    """
    stats1 = count_parameters(model1)
    stats2 = count_parameters(model2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("模型对比")
    logger.info("=" * 60)
    logger.info("%-30s %-15s %-15s", "指标", names[0], names[1])
    logger.info("-" * 60)
    logger.info("%-30s %-15.2f %-15.2f", "总参数量 (M)", stats1['total_params_m'], stats2['total_params_m'])
    logger.info("%-30s %-15.2f %-15.2f", "可训练参数 (M)", stats1['trainable_params_m'], stats2['trainable_params_m'])
    logger.info("%-30s %-15s %-15s", "冻结参数", f"{stats1['frozen_params']:,}", f"{stats2['frozen_params']:,}")
    logger.info("=" * 60)
