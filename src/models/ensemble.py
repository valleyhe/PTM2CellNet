"""
集成模型 — 聚合多个PTM2CellNet子模型的预测结果
功能概述: 支持均值、投票、加权三种聚合策略
"""

from typing import Any, Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from .model_utils import count_parameters
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class PTM2CellNetEnsemble(nn.Module):
    """PTM2CellNetEnsemble — 多模型集成预测器。

    聚合多个训练好的 PTM2CellNet 子模型的输出，支持三种策略:
      - mean: 对各模型 logits 取平均
      - voting: 各模型独立 argmax 后多数投票
      - weighted: 按指定权重对各模型 logits 加权平均
    """

    _VALID_AGGREGATIONS = {"mean", "voting", "weighted"}

    def __init__(
        self,
        models: List[nn.Module],
        aggregation: str = "mean",
        weights: Optional[List[float]] = None,
    ):
        super().__init__()

        if not models:
            raise ValueError("models 列表不能为空，至少需要一个子模型")

        if aggregation not in self._VALID_AGGREGATIONS:
            raise ValueError(
                f"aggregation 必须是 {self._VALID_AGGREGATIONS} 之一，"
                f"收到: {aggregation!r}"
            )

        if aggregation == "weighted" and weights is None:
            raise ValueError("aggregation='weighted' 时必须提供 weights")

        if weights is not None and len(weights) != len(models):
            raise ValueError(
                f"weights 长度 ({len(weights)}) 必须等于 models 长度 ({len(models)})"
            )

        self.aggregation = aggregation
        self.num_models = len(models)

        # 将子模型注册为 ModuleList，以便 .to() / .eval() 等递归生效
        self.models = nn.ModuleList(models)

        # 注册权重为 buffer（随设备迁移，但不参与梯度更新）
        if weights is not None:
            self.register_buffer(
                "_weights",
                torch.tensor(weights, dtype=torch.float32),
            )
        else:
            self.register_buffer(
                "_weights",
                torch.ones(self.num_models, dtype=torch.float32) / self.num_models,
            )

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """运行所有子模型并聚合预测。

        Args:
            batch: 与单个 PTM2CellNet 输入格式相同的字典。

        Returns:
            包含 logits / predictions / probabilities / individual_predictions 的字典。
        """
        all_logits: List[torch.Tensor] = []
        individual_predictions: List[torch.Tensor] = []

        for model in self.models:
            output = model(batch)
            logits = output["logits"]
            all_logits.append(logits)
            individual_predictions.append(logits.argmax(dim=-1))

        stacked = torch.stack(all_logits, dim=0)  # (num_models, B, C)

        # ---- 聚合 ----
        if self.aggregation == "mean":
            aggregated = stacked.mean(dim=0)  # (B, C)

        elif self.aggregation == "voting":
            # 多数投票: 统计每个类别得票数，取最多票的类别
            votes = torch.stack(individual_predictions, dim=0)  # (num_models, B)
            num_classes = stacked.shape[-1]
            # one-hot 累加得票
            one_hot = F.one_hot(votes, num_classes=num_classes).float()  # (num_models, B, C)
            vote_counts = one_hot.sum(dim=0)  # (B, C)
            aggregated = vote_counts

        elif self.aggregation == "weighted":
            w = self._weights.to(stacked.device)  # (num_models,)
            # 扩展维度以广播: (num_models, 1, 1) * (num_models, B, C)
            aggregated = (stacked * w.view(-1, 1, 1)).sum(dim=0)  # (B, C)

        else:
            # 不应到达此处，构造函数已做校验
            raise ValueError(f"未知的聚合策略: {self.aggregation!r}")

        predictions = aggregated.argmax(dim=-1)
        probabilities = F.softmax(aggregated, dim=-1)

        return {
            "logits": aggregated,
            "predictions": predictions,
            "probabilities": probabilities,
            "individual_predictions": individual_predictions,
        }

    # ------------------------------------------------------------------
    # 模型信息
    # ------------------------------------------------------------------

    def get_model_info(self) -> Dict[str, Any]:
        """返回集成模型的元信息。"""
        param_stats = count_parameters(self)

        model_summaries = []
        for i, model in enumerate(self.models):
            if hasattr(model, "get_model_info"):
                info = model.get_model_info()
            else:
                info = {"type": type(model).__name__}
            info["index"] = i
            model_summaries.append(info)

        return {
            "model_type": "PTM2CellNetEnsemble",
            "aggregation": self.aggregation,
            "num_models": self.num_models,
            "weights": self._weights.tolist(),
            "total_params": param_stats["total_params"],
            "trainable_params": param_stats["trainable_params"],
            "models": model_summaries,
        }
