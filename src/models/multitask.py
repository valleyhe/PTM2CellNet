"""
多任务预测器模块
功能概述: 支持同时预测多个细胞状态属性
"""

from typing import Dict, List, Optional

import torch
from torch import nn
from typing_extensions import TypedDict

from .predictors import _build_mlp


# ---------------------------------------------------------------------------
# TypedDict definitions for task configurations
# ---------------------------------------------------------------------------

class TaskConfig(TypedDict, total=False):
    """Shape of a single task configuration dict.

    ``name`` and ``type`` are required in practice but marked optional
    here because the validator checks for their presence.
    """

    name: str
    type: str
    num_classes: int
    output_dim: int


class MultiTaskPredictor(nn.Module):
    """
    多任务预测器

    同时预测多个细胞状态属性，支持分类和回归任务混合。
    使用共享的底层表示和任务特定的预测头。

    Args:
        input_dim: 输入特征维度
        task_configs: 任务配置列表，每个任务包含:
            - name: 任务名称
            - type: "classification" 或 "regression"
            - num_classes: 分类任务的类别数（分类任务必需）
            - output_dim: 回归任务的输出维度（回归任务必需，默认1）
        hidden_dims: 共享MLP的隐藏层维度
        task_specific_layers: 每个任务特定层的维度
        dropout: Dropout概率
    """

    def __init__(
        self,
        input_dim: int,
        task_configs: List[TaskConfig],
        hidden_dims: Optional[List[int]] = None,
        task_specific_layers: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.task_configs = task_configs
        self.num_tasks = len(task_configs)

        # 验证任务配置
        self._validate_task_configs()

        # 共享特征提取层
        self.shared_backbone, backbone_dim = _build_mlp(
            input_dim, hidden_dims, dropout
        )

        # 任务特定的表示层（可选）
        self.task_specific_layers = nn.ModuleDict()
        task_input_dim = backbone_dim

        if task_specific_layers:
            # 如果指定了任务特定层，为每个任务创建
            for task in task_configs:
                task_name = task["name"]
                layers, task_input_dim = _build_mlp(
                    backbone_dim, task_specific_layers, dropout
                )
                self.task_specific_layers[task_name] = layers

        # 任务输出头
        self.task_heads = nn.ModuleDict()
        for task in task_configs:
            task_name = task["name"]
            task_type = task["type"]

            if task_type == "classification":
                num_classes = task["num_classes"]
                self.task_heads[task_name] = nn.Linear(task_input_dim, num_classes)
            elif task_type == "regression":
                output_dim = task.get("output_dim", 1)
                self.task_heads[task_name] = nn.Linear(task_input_dim, output_dim)
            else:
                raise ValueError(f"未知的任务类型: {task_type}")

    def _validate_task_configs(self) -> None:
        """验证任务配置"""
        for task in self.task_configs:
            if "name" not in task:
                raise ValueError("任务配置必须包含'name'字段")
            if "type" not in task:
                raise ValueError("任务配置必须包含'type'字段")

            task_type = task["type"]
            if task_type == "classification" and "num_classes" not in task:
                raise ValueError(f"分类任务 '{task['name']}' 必须包含'num_classes'")

    def forward(self, features: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        前向传播

        参数:
            features: [batch_size, input_dim] 输入特征

        返回:
            字典，键为任务名称，值为任务输出
            分类任务输出: {"logits": ..., "probabilities": ..., "predictions": ...}
            回归任务输出: {"predictions": ...}
        """
        # 共享特征
        shared_features = self.shared_backbone(features)

        outputs = {}
        for task in self.task_configs:
            task_name = task["name"]
            task_type = task["type"]

            # 任务特定表示（如果有）
            if task_name in self.task_specific_layers:
                task_features = self.task_specific_layers[task_name](shared_features)
            else:
                task_features = shared_features

            # 任务输出
            logits = self.task_heads[task_name](task_features)

            if task_type == "classification":
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
                outputs[task_name] = {
                    "logits": logits,
                    "probabilities": probabilities,
                    "predictions": predictions,
                }
            else:  # regression
                outputs[task_name] = {
                    "predictions": logits,
                }

        return outputs

    def get_task_names(self) -> List[str]:
        """获取所有任务名称"""
        return [task["name"] for task in self.task_configs]

    def get_classification_tasks(self) -> List[str]:
        """获取分类任务名称列表"""
        return [
            task["name"] for task in self.task_configs
            if task["type"] == "classification"
        ]

    def get_regression_tasks(self) -> List[str]:
        """获取回归任务名称列表"""
        return [
            task["name"] for task in self.task_configs
            if task["type"] == "regression"
        ]


class HierarchicalMultiTaskPredictor(nn.Module):
    """
    分层多任务预测器

    支持主任务和子任务的分层预测结构，主任务的输出可以作为子任务的输入。

    Args:
        input_dim: 输入特征维度
        primary_task: 主任务配置
        sub_tasks: 子任务配置列表
        hidden_dims: 共享层维度
        dropout: Dropout概率
    """

    def __init__(
        self,
        input_dim: int,
        primary_task: TaskConfig,
        sub_tasks: List[TaskConfig],
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.primary_task = primary_task
        self.sub_tasks = sub_tasks

        # 共享特征提取
        self.shared_backbone, backbone_dim = _build_mlp(
            input_dim, hidden_dims, dropout
        )

        # 主任务头
        primary_type = primary_task["type"]
        if primary_type == "classification":
            self.primary_head = nn.Linear(
                backbone_dim, primary_task["num_classes"]
            )
        else:
            self.primary_head = nn.Linear(
                backbone_dim, primary_task.get("output_dim", 1)
            )

        # 子任务头（接收主任务输出作为额外输入）
        self.sub_task_heads = nn.ModuleDict()
        primary_output_dim = (
            primary_task["num_classes"]
            if primary_type == "classification"
            else primary_task.get("output_dim", 1)
        )
        sub_task_input_dim = backbone_dim + primary_output_dim

        for task in sub_tasks:
            task_name = task["name"]
            task_type = task["type"]

            if task_type == "classification":
                self.sub_task_heads[task_name] = nn.Linear(
                    sub_task_input_dim, task["num_classes"]
                )
            else:
                self.sub_task_heads[task_name] = nn.Linear(
                    sub_task_input_dim, task.get("output_dim", 1)
                )

    def forward(self, features: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        前向传播

        参数:
            features: [batch_size, input_dim]

        返回:
            所有任务的输出字典
        """
        # 共享特征
        shared_features = self.shared_backbone(features)

        # 主任务预测
        primary_logits = self.primary_head(shared_features)

        outputs = {}
        if self.primary_task["type"] == "classification":
            primary_probs = torch.softmax(primary_logits, dim=-1)
            primary_preds = torch.argmax(primary_probs, dim=-1)
            outputs[self.primary_task["name"]] = {
                "logits": primary_logits,
                "probabilities": primary_probs,
                "predictions": primary_preds,
            }
            primary_output = primary_probs
        else:
            outputs[self.primary_task["name"]] = {
                "predictions": primary_logits,
            }
            primary_output = primary_logits

        # 子任务预测（使用主任务输出）
        combined_features = torch.cat([shared_features, primary_output], dim=-1)

        for task in self.sub_tasks:
            task_name = task["name"]
            task_type = task["type"]

            logits = self.sub_task_heads[task_name](combined_features)

            if task_type == "classification":
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
                outputs[task_name] = {
                    "logits": logits,
                    "probabilities": probabilities,
                    "predictions": predictions,
                }
            else:
                outputs[task_name] = {
                    "predictions": logits,
                }

        return outputs
