"""
优化器配置模块
功能概述: 提供优化器和学习率调度器的配置接口
设计思路: 支持多种优化器和调度器，提供统一的配置接口
"""

from typing import Any, Mapping, Optional, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

# 兼容PyTorch 1.x和2.x的LRScheduler导入
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


def configure_optimizer(
    model: nn.Module,
    config: Optional[Mapping[str, Any]] = None,
) -> Tuple[Optimizer, Optional[LRScheduler]]:
    """
    配置优化器和学习率调度器

    参数:
        model: PyTorch模型
        config: 配置字典，包含以下可选键:
            - optimizer: 优化器类型，可选 "adam", "adamw", "sgd"
            - learning_rate: 学习率
            - weight_decay: 权重衰减
            - scheduler: 调度器类型，可选 "cosine", "plateau", "step"
            - scheduler_params: 调度器参数字典

    返回:
        (optimizer, scheduler)元组
    """
    cfg = config or {}
    training_config = cfg.get("training", {})

    optimizer_type = training_config.get("optimizer", "adamw")
    learning_rate = float(training_config.get("learning_rate", 0.001))
    weight_decay = training_config.get("weight_decay", 0.0001)
    momentum = training_config.get("momentum", 0.9)
    nesterov = training_config.get("nesterov", False)

    scheduler_type = training_config.get("scheduler", None)
    scheduler_params = training_config.get("scheduler_params", {})
    max_epochs = training_config.get("max_epochs", 100)

    optimizer = _create_optimizer(
        model,
        optimizer_type,
        learning_rate,
        weight_decay,
        momentum,
        nesterov,
    )

    scheduler = None
    if scheduler_type is not None:
        scheduler = _create_scheduler(
            optimizer,
            scheduler_type,
            max_epochs,
            scheduler_params,
        )

    return optimizer, scheduler


def _create_optimizer(
    model: nn.Module,
    optimizer_type: str,
    learning_rate: float,
    weight_decay: float,
    momentum: float = 0.9,
    nesterov: bool = False,
) -> Optimizer:
    """
    创建优化器

    参数:
        model: PyTorch模型
        optimizer_type: 优化器类型
        learning_rate: 学习率
        weight_decay: 权重衰减
        momentum: SGD动量（仅optimizer_type='sgd'时生效）
        nesterov: 是否启用Nesterov动量（仅optimizer_type='sgd'时生效）

    返回:
        优化器实例
    """
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer: Optimizer
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
        )
    else:
        raise ValueError(f"未知的优化器类型: {optimizer_type}")

    return optimizer


def _create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    max_epochs: int,
    scheduler_params: Mapping[str, Any],
) -> LRScheduler:
    """
    创建学习率调度器

    参数:
        optimizer: 优化器
        scheduler_type: 调度器类型
        max_epochs: 最大训练轮数
        scheduler_params: 调度器参数字典

    返回:
        学习率调度器实例
    """
    scheduler: LRScheduler
    if scheduler_type == "cosine":
        T_max = scheduler_params.get("T_max", max_epochs)
        eta_min = scheduler_params.get("eta_min", 0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
        )
    elif scheduler_type == "plateau":
        mode = scheduler_params.get("mode", "min")
        factor = scheduler_params.get("factor", 0.1)
        patience = scheduler_params.get("patience", 10)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
        )
    elif scheduler_type == "step":
        step_size = scheduler_params.get("step_size", 30)
        gamma = scheduler_params.get("gamma", 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    else:
        raise ValueError(f"未知的调度器类型: {scheduler_type}")

    return scheduler
