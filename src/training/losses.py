"""
损失函数模块
功能概述: 提供多种损失函数，包括Focal Loss、Dice Loss和多任务损失
设计思路: 定义统一的损失函数接口，支持灵活组合和配置
"""

from typing import Dict, List, Optional, cast

import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss
    用于处理类别不平衡问题，通过降低易分类样本的权重来关注难分类样本

    特性:
        - 支持alpha类别权重，可自动从类别数量计算
        - 数值稳定性优化，避免NaN和梯度爆炸
        - 支持ignore_index忽略特定标签
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
        eps: float = 1e-6,
    ):
        """
        初始化Focal Loss

        参数:
            alpha: 类别权重，形状为 (num_classes,)
            gamma: 聚焦参数，默认2.0
            reduction: 损失聚合方式，可选 "mean", "sum", "none"
            ignore_index: 忽略的类别索引
            eps: 数值稳定性epsilon值
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.eps = eps

    @classmethod
    def from_class_counts(
        cls,
        class_counts: List[int],
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> "FocalLoss":
        """
        从类别计数自动计算alpha权重的工厂方法

        计算方式: alpha = total_samples / (num_classes * class_counts)
        这样频率低的类别获得更高的权重

        参数:
            class_counts: 每个类别的样本数量列表
            gamma: 聚焦参数，默认2.0
            reduction: 损失聚合方式
            ignore_index: 忽略的类别索引

        返回:
            配置好的FocalLoss实例

        示例:
            >>> # 二分类：99个负样本，1个正样本
            >>> focal_loss = FocalLoss.from_class_counts([99, 1])
            >>> # alpha将为 [1.0, 99.0]
        """
        class_counts_tensor = torch.tensor(class_counts, dtype=torch.float32)
        total_samples = class_counts_tensor.sum()
        num_classes = len(class_counts)

        # 计算逆频率权重: total / (num_classes * count)
        # 添加eps避免除零
        alpha = total_samples / (num_classes * class_counts_tensor + 1e-8)

        # 归一化使平均权重为1
        alpha = alpha / alpha.mean()

        return cls(alpha=alpha, gamma=gamma, reduction=reduction, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            logits: 模型输出logits，形状为 (batch_size, num_classes) 或 (batch_size, num_classes, ...)
            targets: 真实标签，形状为 (batch_size,) 或 (batch_size, ...)

        返回:
            Focal Loss值
        """
        num_classes = logits.shape[1]

        # 数值稳定性：限制logits范围避免极端值
        logits_clamped = torch.clamp(logits, min=-10, max=10)

        # 计算log概率和概率（数值稳定）
        log_probs = F.log_softmax(logits_clamped, dim=1)
        probs = torch.exp(log_probs)

        # 限制概率范围避免log(0)
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)

        # 处理ignore_index
        valid_mask = None
        safe_targets = targets
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            if not valid_mask.any():
                return logits.new_tensor(0.0)
            safe_targets = targets.clone()
            safe_targets[~valid_mask] = 0

        # 创建one-hot编码
        targets_one_hot = F.one_hot(safe_targets, num_classes=num_classes).float()
        targets_one_hot = targets_one_hot.permute(0, -1, *range(1, targets_one_hot.dim() - 1))

        # 计算pt（正确类别的概率）
        pt = torch.sum(targets_one_hot * probs, dim=1)
        pt = torch.clamp(pt, min=self.eps, max=1.0)

        # 计算focal权重: (1 - pt)^gamma
        focal_weight = (1.0 - pt + self.eps).pow(self.gamma)

        # 获取正确类别的log概率
        log_pt = torch.sum(targets_one_hot * log_probs, dim=1)

        # 应用alpha权重
        if self.alpha is not None:
            # 将alpha移到与targets相同的设备
            if isinstance(self.alpha, float):
                alpha = torch.tensor(self.alpha, device=logits.device)
            else:
                alpha = self.alpha.to(logits.device)
            alpha_t = torch.sum(targets_one_hot * alpha.view(1, -1, *([1] * (targets_one_hot.dim() - 2))), dim=1)
            loss = -alpha_t * focal_weight * log_pt
        else:
            loss = -focal_weight * log_pt

        # 应用ignore mask
        if valid_mask is not None:
            mask = valid_mask.float()
            while mask.dim() < loss.dim():
                mask = mask.unsqueeze(-1)
            loss = loss * mask

        # 应用reduction
        if self.reduction == "mean":
            if valid_mask is not None:
                return cast(torch.Tensor, loss.sum() / (valid_mask.sum().float() + self.eps))
            return cast(torch.Tensor, loss.mean())
        if self.reduction == "sum":
            return cast(torch.Tensor, loss.sum())
        return cast(torch.Tensor, loss)


class DiceLoss(nn.Module):
    """
    Dice Loss
    用于语义分割等任务，衡量预测与真实的重叠程度
    """

    def __init__(
        self,
        smooth: float = 1.0,
        reduction: str = "mean",
    ):
        """
        初始化Dice Loss

        参数:
            smooth: 平滑因子，防止除零
            reduction: 损失聚合方式
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            logits: 模型输出logits，形状为 (batch_size, num_classes, ...)
            targets: 真实标签，形状为 (batch_size, ...)

        返回:
            Dice Loss值
        """
        num_classes = logits.shape[1]

        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        targets_one_hot = targets_one_hot.permute(0, -1, *range(1, targets_one_hot.dim() - 1))

        intersection = torch.sum(probs * targets_one_hot, dim=tuple(range(2, probs.dim())))
        union = torch.sum(probs, dim=tuple(range(2, probs.dim()))) + torch.sum(
            targets_one_hot, dim=tuple(range(2, targets_one_hot.dim()))
        )

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice

        if self.reduction == "mean":
            return cast(torch.Tensor, loss.mean())
        if self.reduction == "sum":
            return cast(torch.Tensor, loss.sum())
        return cast(torch.Tensor, loss)


class MultiTaskLoss(nn.Module):
    """
    多任务损失
    组合多个损失函数，支持加权组合
    """

    def __init__(
        self,
        loss_dict: Dict[str, nn.Module],
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        初始化多任务损失

        参数:
            loss_dict: 损失函数字典，键为任务名
            weights: 各任务的权重字典，默认为均匀权重
        """
        super().__init__()
        self.loss_dict = nn.ModuleDict(loss_dict)

        if weights is None:
            self.weights = {key: 1.0 / len(loss_dict) for key in loss_dict.keys()}
        else:
            self.weights = weights

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            predictions: 预测结果字典，键为任务名
            targets: 真实标签字典，键为任务名

        返回:
            损失字典，包含各任务损失和总损失
        """
        loss_dict = {}
        total_loss = 0.0

        for task_name, loss_fn in self.loss_dict.items():
            if task_name in predictions and task_name in targets:
                task_loss = loss_fn(predictions[task_name], targets[task_name])
                loss_dict[task_name] = task_loss
                total_loss += self.weights.get(task_name, 1.0) * task_loss

        loss_dict["total"] = total_loss
        return loss_dict
