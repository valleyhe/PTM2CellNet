"""
PEFT/LoRA配置模块
功能概述: 提供LoRA配置和应用于ESM-2编码器的功能
设计思路: 使用peft库实现高效的参数高效微调

主要组件:
    - get_lora_config: 创建LoRA配置
    - apply_lora_to_encoder: 将LoRA应用于编码器
    - get_trainable_parameters: 获取可训练参数统计
"""

from typing import Any, List, Optional, Tuple, cast

import torch.nn as nn

from ..utils.logging import setup_logger

logger = setup_logger(__name__)

# 尝试导入peft，如果不可用则提供警告
_LoraConfig_cls: Any = None
_TaskType_cls: Any = None
_get_peft_model_fn: Any = None
try:
    from peft import LoraConfig as _LoraConfig_cls
    from peft import TaskType as _TaskType_cls
    from peft import get_peft_model as _get_peft_model_fn
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("peft库未安装，LoRA功能不可用。请运行: pip install peft>=0.8.0")

    class _LoraConfig_cls:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            raise ImportError("peft库未安装，无法创建LoraConfig")

    class _TaskType_cls:  # type: ignore[no-redef]
        FEATURE_EXTRACTION: str = "FEATURE_EXTRACTION"

    def _get_peft_model_fn(model: Any, config: Any) -> Any:
        raise ImportError("peft库未安装，无法应用LoRA")


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
) -> Any:
    """
    创建LoRA配置

    参数:
        r: LoRA秩，默认16
        lora_alpha: LoRA缩放参数，默认32
        lora_dropout: LoRA dropout率，默认0.05
        target_modules: 目标模块列表，默认["query", "key", "value", "dense"]
        bias: 偏置训练模式，默认"none"

    返回:
        LoraConfig实例

    示例:
        >>> config = get_lora_config(r=8, lora_alpha=16)
        >>> print(config.r)  # 8
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "peft库未安装，无法创建LoRA配置。"
            "请运行: pip install peft>=0.8.0"
        )

    if target_modules is None:
        target_modules = ["query", "key", "value", "dense"]

    config = _LoraConfig_cls(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=_TaskType_cls.FEATURE_EXTRACTION,
    )

    logger.info(
        f"创建LoRA配置: r={r}, alpha={lora_alpha}, "
        f"target_modules={target_modules}, dropout={lora_dropout}"
    )

    return config


def apply_lora_to_encoder(
    encoder: nn.Module,
    lora_config: Optional[Any] = None,
) -> nn.Module:
    """
    将LoRA应用于编码器

    参数:
        encoder: 编码器实例（如ESM2Encoder）
        lora_config: LoRA配置，如果为None则使用默认配置

    返回:
        应用了LoRA的编码器

    示例:
        >>> encoder = ESM2Encoder(model_size="150M")
        >>> encoder = apply_lora_to_encoder(encoder)
        >>> print(f"可训练参数: {get_trainable_parameters(encoder)}")
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "peft库未安装，无法应用LoRA。"
            "请运行: pip install peft>=0.8.0"
        )

    # 如果未提供配置，使用默认配置
    if lora_config is None:
        lora_config = get_lora_config()

    # 获取编码器的底层模型
    if hasattr(encoder, "model"):
        base_model = encoder.model
    else:
        base_model = encoder

    # 应用LoRA
    lora_model = _get_peft_model_fn(base_model, lora_config)

    # 替换编码器中的模型
    if hasattr(encoder, "model"):
        encoder.model = lora_model
    else:
        encoder = lora_model

    # 记录可训练参数信息
    trainable_params, total_params, percentage = get_trainable_parameters(encoder)
    logger.info(
        f"LoRA应用完成: 可训练参数 {trainable_params:,} / {total_params:,} "
        f"({percentage:.2f}%)"
    )

    # 验证可训练参数比例<1%
    if percentage >= 1.0:
        logger.warning(
            f"可训练参数比例({percentage:.2f}%) >= 1%，"
            f"可能影响参数高效微调的效果"
        )

    return encoder


def get_trainable_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """
    获取模型的可训练参数统计

    参数:
        model: PyTorch模型

    返回:
        (可训练参数数量, 总参数数量, 可训练参数百分比)

    示例:
        >>> trainable, total, pct = get_trainable_parameters(model)
        >>> print(f"可训练参数: {trainable:,} ({pct:.2f}%)")
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    percentage = (trainable_params / total_params * 100) if total_params > 0 else 0.0

    return trainable_params, total_params, percentage


def freeze_base_model(model: nn.Module) -> None:
    """
    冻结模型的基础参数（保留LoRA参数可训练）

    参数:
        model: PyTorch模型或PEFT模型

    示例:
        >>> freeze_base_model(model)
        >>> # 只有LoRA参数可训练
    """
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 解冻LoRA参数（如果存在）
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    logger.info("基础模型已冻结，LoRA参数保持可训练")


def save_lora_adapters(model: nn.Module, save_path: str) -> None:
    """
    保存LoRA适配器权重

    参数:
        model: 应用了LoRA的模型
        save_path: 保存路径

    示例:
        >>> save_lora_adapters(model, "outputs/lora_adapters")
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "peft库未安装，无法保存LoRA适配器。"
            "请运行: pip install peft>=0.8.0"
        )

    # 检查模型是否应用了LoRA
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(save_path)
        logger.info("LoRA适配器已保存到: %s", save_path)
    elif hasattr(model, "model") and hasattr(model.model, "save_pretrained"):
        model.model.save_pretrained(save_path)
        logger.info("LoRA适配器已保存到: %s", save_path)
    else:
        raise ValueError("模型未应用LoRA或不是PEFT模型")


def load_lora_adapters(model: nn.Module, load_path: str) -> nn.Module:
    """
    加载LoRA适配器权重

    参数:
        model: 基础模型
        load_path: 适配器加载路径

    返回:
        加载了适配器的模型

    示例:
        >>> model = load_lora_adapters(encoder.model, "outputs/lora_adapters")
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "peft库未安装，无法加载LoRA适配器。"
            "请运行: pip install peft>=0.8.0"
        )

    from peft import PeftModel

    # 加载适配器
    model = PeftModel.from_pretrained(model, load_path)
    logger.info("LoRA适配器已从 %s 加载", load_path)

    return model


def merge_lora_adapters(model: nn.Module) -> nn.Module:
    """
    合并LoRA适配器到基础模型

    合并后的模型不再依赖PEFT库，可以直接用于推理

    参数:
        model: 应用了LoRA的PEFT模型

    返回:
        合并后的基础模型

    示例:
        >>> merged_model = merge_lora_adapters(lora_model)
        >>> # merged_model可以像普通PyTorch模型一样使用
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "peft库未安装，无法合并LoRA适配器。"
            "请运行: pip install peft>=0.8.0"
        )

    if hasattr(model, "merge_and_unload"):
        merged_model = model.merge_and_unload()
        logger.info("LoRA适配器已合并到基础模型")
        return cast(nn.Module, merged_model)
    elif hasattr(model, "model") and hasattr(model.model, "merge_and_unload"):
        merged_model = model.model.merge_and_unload()
        logger.info("LoRA适配器已合并到基础模型")
        return cast(nn.Module, merged_model)
    else:
        raise ValueError("模型不支持合并LoRA适配器")
