# mypy: ignore-errors
"""
模型工具函数
功能概述: 提供模型配置验证、参数量统计、计算量分析等工具
"""

from typing import Any, Dict, List, Tuple, Union
import torch
from torch import nn


def validate_model_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
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
        "cnn", "transformer", "lstm", "mamba",
        "esm2_8m", "esm2_150m", "esm2_650m", "esm2_3b",
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


def count_parameters(model: nn.Module, trainable_only: bool = False) -> Dict[str, Any]:
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


def get_model_memory_usage(model: nn.Module, batch_size: int = 1, seq_len: int = 1000) -> Dict[str, Union[float, int]]:
    """
    估算模型内存使用量

    参数:
        model: PyTorch模型
        batch_size: 批次大小
        seq_len: 序列长度

    返回:
        内存使用量估算（MB）
    """
    param_count = sum(p.numel() for p in model.parameters())
    param_memory_mb = param_count * 4 / (1024 ** 2)  # 假设float32

    # 估算激活内存（简化计算）
    # 每层大约需要 batch_size * seq_len * hidden_dim * 4 bytes
    hidden_dim = 128
    if hasattr(model, 'embed_dim'):
        embed_dim = model.embed_dim
        # 确保是整数（如果是tensor，取item()）
        if isinstance(embed_dim, torch.Tensor):
            hidden_dim = int(embed_dim.item())
        else:
            hidden_dim = int(embed_dim)

    # 假设10层
    num_layers = 10
    activation_memory_mb = batch_size * seq_len * hidden_dim * num_layers * 4 / (1024 ** 2)

    return {
        "params_memory_mb": param_memory_mb,
        "activation_memory_mb": activation_memory_mb,
        "total_memory_mb": param_memory_mb + activation_memory_mb,
        "recommended_batch_size": estimate_max_batch_size(model, seq_len)
    }


def estimate_max_batch_size(model: nn.Module, seq_len: int = 1000,
                            available_memory_gb: float = 8.0) -> int:
    """
    估算最大批次大小

    参数:
        model: PyTorch模型
        seq_len: 序列长度
        available_memory_gb: 可用GPU内存（GB）

    返回:
        估算的最大批次大小
    """
    available_memory_mb = available_memory_gb * 1024

    # 参数内存
    param_memory_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)

    # 为优化器状态和其他开销预留50%内存
    usable_memory_mb = available_memory_mb * 0.5 - param_memory_mb

    # 每个样本的激活内存
    hidden_dim = getattr(model, 'embed_dim', 128)
    num_layers = 10
    activation_per_sample_mb = seq_len * hidden_dim * num_layers * 4 / (1024 ** 2)

    if activation_per_sample_mb <= 0:
        return 1

    max_batch_size = int(usable_memory_mb / activation_per_sample_mb)
    return max(1, min(max_batch_size, 512))


def print_model_summary(model: nn.Module, detailed: bool = False) -> None:
    """
    打印模型摘要信息

    参数:
        model: PyTorch模型
        detailed: 是否打印详细信息
    """
    stats = count_parameters(model)

    print(f"\n{'='*60}")
    print("模型摘要")
    print(f"{'='*60}")
    print(f"总参数量: {stats['total_params_m']:.2f}M ({stats['total_params']:,})")
    print(f"可训练参数: {stats['trainable_params_m']:.2f}M ({stats['trainable_params']:,})")
    print(f"冻结参数: {stats['frozen_params']:,}")

    if detailed:
        print("\n各模块参数量:")
        for name, module_stats in stats['modules'].items():
            print(f"  {name}: {module_stats['total']/1e6:.2f}M "
                  f"({module_stats['percentage']:.1f}%)")

    memory = get_model_memory_usage(model)
    print("\n内存估算:")
    print(f"  参数内存: {memory['params_memory_mb']:.2f} MB")
    print(f"  激活内存: {memory['activation_memory_mb']:.2f} MB")
    print(f"  推荐批次大小: {memory['recommended_batch_size']}")
    print(f"{'='*60}\n")


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

    print(f"\n{'='*60}")
    print("模型对比")
    print(f"{'='*60}")
    print(f"{'指标':<30} {names[0]:<15} {names[1]:<15}")
    print(f"{'-'*60}")
    print(f"{'总参数量 (M)':<30} {stats1['total_params_m']:<15.2f} {stats2['total_params_m']:<15.2f}")
    print(f"{'可训练参数 (M)':<30} {stats1['trainable_params_m']:<15.2f} {stats2['trainable_params_m']:<15.2f}")
    print(f"{'冻结参数':<30} {stats1['frozen_params']:<15,} {stats2['frozen_params']:<15,}")
    print(f"{'='*60}\n")
