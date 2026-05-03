"""
评估器模块
功能概述: 提供完整的模型评估流程
设计思路: 封装评估指标，支持批量评估和交叉验证
"""

from typing import Any, Dict, List, Optional

from typing_extensions import TypeAlias

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..utils.logging import setup_logger
from .metrics import (
    calculate_accuracy,
    calculate_auc_pr,
    calculate_auc_roc,
    calculate_classification_report,
    calculate_confusion_matrix,
    calculate_f1_score,
    calculate_mae,
    calculate_mcc,
    calculate_mse,
    calculate_per_ptm_type_metrics,
    calculate_precision,
    calculate_r2,
    calculate_recall,
    calculate_rmse,
)

logger = setup_logger(__name__)

Array: TypeAlias = NDArray[Any]
Batch: TypeAlias = Dict[str, torch.Tensor]


class Evaluator:
    """评估器类"""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        task_type: str = "classification",
    ) -> None:
        self.model = model
        self.config = config or {}
        self.task_type = task_type

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

    def evaluate(
        self,
        dataloader: DataLoader[Batch],
        return_predictions: bool = False,
    ) -> Dict[str, Any]:
        """评估模型并返回指标。"""
        logger.info("开始评估")

        all_predictions: List[Array] = []
        all_probabilities: List[Array] = []
        all_targets: List[Array] = []
        all_ptm_types: List[str] = []

        with torch.no_grad():
            for batch in dataloader:
                if not isinstance(batch, dict):
                    raise TypeError("Batch must be a dict")

                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                outputs = self.model(batch)

                if isinstance(outputs, dict):
                    predictions = outputs.get("predictions")
                    probabilities = outputs.get("probabilities")
                else:
                    if not isinstance(outputs, torch.Tensor):
                        raise TypeError("Model output must be tensor or dict")
                    probabilities = torch.softmax(outputs, dim=-1)
                    predictions = torch.argmax(probabilities, dim=-1)

                if not isinstance(predictions, torch.Tensor):
                    raise TypeError("Model output must contain tensor predictions")

                probabilities_tensor: Optional[torch.Tensor]
                if isinstance(probabilities, torch.Tensor):
                    probabilities_tensor = probabilities
                elif probabilities is None:
                    probabilities_tensor = None
                else:
                    raise TypeError("Model output probabilities must be a tensor")

                targets = batch.get("label")
                if not isinstance(targets, torch.Tensor):
                    raise TypeError("Batch must contain tensor label")

                all_predictions.append(np.asarray(predictions.cpu().numpy()))
                if probabilities_tensor is not None:
                    all_probabilities.append(np.asarray(probabilities_tensor.cpu().numpy()))
                all_targets.append(np.asarray(targets.cpu().numpy()))

                ptm_type = batch.get("ptm_type")
                if ptm_type is not None:
                    if isinstance(ptm_type, list):
                        all_ptm_types.extend([str(x) for x in ptm_type])
                    elif isinstance(ptm_type, torch.Tensor):
                        ptm_type_names = self.config.get("ptm_type_names") if self.config else None
                        for x in ptm_type.tolist():
                            if ptm_type_names is not None and isinstance(ptm_type_names, (list, dict)):
                                if isinstance(ptm_type_names, list) and 0 <= int(x) < len(ptm_type_names):
                                    all_ptm_types.append(str(ptm_type_names[int(x)]))
                                elif isinstance(ptm_type_names, dict) and str(x) in ptm_type_names:
                                    all_ptm_types.append(str(ptm_type_names[str(x)]))
                                else:
                                    all_ptm_types.append(str(int(x)))
                            else:
                                all_ptm_types.append(str(int(x)))

        predictions_array = np.concatenate(all_predictions, axis=0)
        targets_array = np.concatenate(all_targets, axis=0)

        probabilities_array: Optional[Array]
        if all_probabilities:
            probabilities_array = np.concatenate(all_probabilities, axis=0)
        else:
            probabilities_array = None

        metrics: Dict[str, Any]
        if self.task_type == "classification":
            metrics = self._calculate_classification_metrics(
                targets_array,
                predictions_array,
                probabilities_array,
                all_ptm_types if all_ptm_types else None,
            )
        else:
            metrics = self._calculate_regression_metrics(
                targets_array,
                predictions_array,
            )

        logger.info("评估完成: %s", metrics)

        if return_predictions:
            result: Dict[str, Any] = {
                "metrics": metrics,
                "predictions": predictions_array,
                "targets": targets_array,
            }
            if probabilities_array is not None:
                result["probabilities"] = probabilities_array
            return result

        return {"metrics": metrics}

    def _calculate_classification_metrics(
        self,
        y_true: Array,
        y_pred: Array,
        y_score: Optional[Array] = None,
        ptm_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """计算分类指标。"""
        metrics: Dict[str, Any] = {
            "accuracy": calculate_accuracy(y_true, y_pred),
            "precision_macro": calculate_precision(y_true, y_pred, average="macro"),
            "precision_micro": calculate_precision(y_true, y_pred, average="micro"),
            "precision_weighted": calculate_precision(y_true, y_pred, average="weighted"),
            "recall_macro": calculate_recall(y_true, y_pred, average="macro"),
            "recall_micro": calculate_recall(y_true, y_pred, average="micro"),
            "recall_weighted": calculate_recall(y_true, y_pred, average="weighted"),
            "f1_macro": calculate_f1_score(y_true, y_pred, average="macro"),
            "f1_micro": calculate_f1_score(y_true, y_pred, average="micro"),
            "f1_weighted": calculate_f1_score(y_true, y_pred, average="weighted"),
            "mcc": calculate_mcc(y_true, y_pred),
        }

        if y_score is not None:
            metrics["auc_roc"] = calculate_auc_roc(y_true, y_score)
            metrics["auc_pr"] = calculate_auc_pr(y_true, y_score)

        if y_score is not None and ptm_types is not None and len(ptm_types) == len(y_true):
            metrics["per_ptm_type_metrics"] = calculate_per_ptm_type_metrics(
                y_true, y_pred, y_score, ptm_types
            )

        metrics["confusion_matrix"] = calculate_confusion_matrix(y_true, y_pred).tolist()
        metrics["classification_report"] = calculate_classification_report(y_true, y_pred)
        return metrics

    def _calculate_regression_metrics(
        self,
        y_true: Array,
        y_pred: Array,
    ) -> Dict[str, float]:
        """计算回归指标。"""
        return {
            "mae": calculate_mae(y_true, y_pred),
            "mse": calculate_mse(y_true, y_pred),
            "rmse": calculate_rmse(y_true, y_pred),
            "r2": calculate_r2(y_true, y_pred),
        }
