"""
预测器模块
功能概述: 从特征预测细胞状态（分类或回归）
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


def _build_mlp(input_dim: int, hidden_dims: Optional[List[int]], dropout: float) -> Tuple[nn.Sequential, int]:
    layers: List[nn.Module] = []
    prev_dim = input_dim
    for dim in hidden_dims or []:
        layers.append(nn.Linear(prev_dim, dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        prev_dim = dim
    return nn.Sequential(*layers), prev_dim


class CellStatePredictor(nn.Module):
    """预测器基类"""

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:  # pragma: no cover - interface
        raise NotImplementedError


class ClassificationPredictor(CellStatePredictor):
    """分类预测器"""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        mlp, last_dim = _build_mlp(input_dim, hidden_dims, dropout)
        self.mlp = mlp
        self.classifier = nn.Linear(last_dim, num_classes)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.mlp(features)
        logits = self.classifier(x)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
        }


class RegressionPredictor(CellStatePredictor):
    """回归预测器"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        mlp, last_dim = _build_mlp(input_dim, hidden_dims, dropout)
        self.mlp = mlp
        self.regressor = nn.Linear(last_dim, output_dim)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.mlp(features)
        predictions = self.regressor(x)
        return {
            "predictions": predictions,
        }
