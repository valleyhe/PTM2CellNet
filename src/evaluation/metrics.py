"""
评估指标模块
功能概述: 提供分类和回归任务的各种评估指标
设计思路: 封装scikit-learn的指标，提供统一的接口
"""

from typing import Any, Dict, List, Optional
import warnings

from typing_extensions import TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    matthews_corrcoef,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.exceptions import UndefinedMetricWarning
from torchmetrics.functional.classification import binary_average_precision

Array: TypeAlias = NDArray[Any]


def calculate_accuracy(y_true: Array, y_pred: Array) -> float:
    """计算准确率"""
    return float(accuracy_score(y_true, y_pred))


def calculate_precision(
    y_true: Array,
    y_pred: Array,
    average: str = "macro",
) -> float:
    """计算精确率"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        return float(precision_score(y_true, y_pred, average=average))


def calculate_recall(
    y_true: Array,
    y_pred: Array,
    average: str = "macro",
) -> float:
    """计算召回率"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        return float(recall_score(y_true, y_pred, average=average))


def calculate_f1_score(
    y_true: Array,
    y_pred: Array,
    average: str = "macro",
) -> float:
    """计算F1分数"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        return float(f1_score(y_true, y_pred, average=average))


def calculate_auc_roc(
    y_true: Array,
    y_score: Array,
    multi_class: str = "ovr",
) -> float:
    """计算AUC-ROC"""
    if len(y_score.shape) == 1:
        # 一维数组，直接使用
        return float(roc_auc_score(y_true, y_score))

    n_classes = y_score.shape[1]
    if n_classes == 1:
        # 单列，直接使用
        return float(roc_auc_score(y_true, y_score[:, 0]))
    elif n_classes == 2:
        # 二分类：使用正类（索引1）的概率
        return float(roc_auc_score(y_true, y_score[:, 1]))
    else:
        # 多分类
        return float(roc_auc_score(y_true, y_score, multi_class=multi_class))


def calculate_auc_pr(
    y_true: Array,
    y_score: Array,
) -> float:
    """计算AUC-PR（平均精确率）"""
    if len(y_score.shape) == 1:
        return float(average_precision_score(y_true, y_score))

    n_classes = y_score.shape[1]
    if n_classes == 1:
        return float(average_precision_score(y_true, y_score[:, 0]))
    elif n_classes == 2:
        # 二分类：使用正类（索引1）的概率
        return float(average_precision_score(y_true, y_score[:, 1]))
    else:
        # 多分类
        y_true_onehot = np.zeros((len(y_true), n_classes))
        y_true_onehot[np.arange(len(y_true)), y_true.astype(int)] = 1
        return float(average_precision_score(y_true_onehot, y_score, average="macro"))


def calculate_confusion_matrix(
    y_true: Array,
    y_pred: Array,
    labels: Optional[List[int]] = None,
) -> Array:
    """计算混淆矩阵"""
    return np.asarray(confusion_matrix(y_true, y_pred, labels=labels))


def calculate_classification_report(
    y_true: Array,
    y_pred: Array,
    labels: Optional[List[int]] = None,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """计算分类报告"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
        )
    if not isinstance(report, dict):
        raise TypeError("Expected dict report")
    return report


def calculate_mae(y_true: Array, y_pred: Array) -> float:
    """计算平均绝对误差（MAE）"""
    return float(mean_absolute_error(y_true, y_pred))


def calculate_mse(y_true: Array, y_pred: Array) -> float:
    """计算均方误差（MSE）"""
    return float(mean_squared_error(y_true, y_pred))


def calculate_rmse(y_true: Array, y_pred: Array) -> float:
    """计算均方根误差（RMSE）"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_r2(y_true: Array, y_pred: Array) -> float:
    """计算R²分数"""
    return float(r2_score(y_true, y_pred))


# MCC, AUPR, and per-PTM-type metrics (EVAL-01)


def calculate_mcc(y_true: Array, y_pred: Array) -> float:
    """
    Calculate Matthews Correlation Coefficient.

    Best metric for imbalanced binary classification.
    Range: -1 (perfect inverse) to +1 (perfect prediction)

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        MCC score, or NaN if undefined (single class)
    """
    # Check for single-class edge case (Pitfall 4 per RESEARCH.md)
    if len(set(y_true)) < 2 or len(set(y_pred)) < 2:
        return float('nan')

    return float(matthews_corrcoef(y_true, y_pred))


def calculate_aupr_torch(
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    thresholds: Optional[int] = None,
) -> float:
    """
    Calculate Area Under Precision-Recall curve using torchmetrics.

    Better than AUROC for imbalanced data (PTM: 1-5% positive).
    torchmetrics handles logits automatically.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_score: Prediction scores/probabilities
        thresholds: Number of thresholds (None for exact, int for approximate)

    Returns:
        AUPR score
    """
    # Ensure tensors
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.long)
    if not isinstance(y_score, torch.Tensor):
        y_score = torch.tensor(y_score, dtype=torch.float)

    # Calculate AUPR
    aupr = binary_average_precision(y_score, y_true, thresholds=thresholds)
    return float(aupr.item())


def calculate_per_ptm_type_metrics(
    y_true: Array,
    y_pred: Array,
    y_score: Array,
    ptm_types: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics separately for each PTM type.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_score: Prediction scores/probabilities
        ptm_types: PTM type for each sample

    Returns:
        Dictionary mapping PTM type to metrics dict:
        {
            'Phosphorylation': {
                'auc_roc': float,
                'auc_pr': float,
                'mcc': float,
                'f1': float,
                'precision': float,
                'recall': float,
                'support': int,
            },
            ...
        }
    """
    from collections import defaultdict

    # Group by PTM type
    type_indices = defaultdict(list)
    for i, ptm_type in enumerate(ptm_types):
        type_indices[ptm_type].append(i)

    results = {}
    for ptm_type, indices in type_indices.items():
        y_true_ptm = y_true[indices]
        y_pred_ptm = y_pred[indices]
        y_score_ptm = y_score[indices]

        # Skip if too few samples
        if len(y_true_ptm) < 5:
            continue

        # Calculate metrics
        metrics = {
            'auc_roc': calculate_auc_roc(y_true_ptm, y_score_ptm),
            'auc_pr': calculate_auc_pr(y_true_ptm, y_score_ptm),
            'mcc': calculate_mcc(y_true_ptm, y_pred_ptm),
            'f1': calculate_f1_score(y_true_ptm, y_pred_ptm),
            'precision': calculate_precision(y_true_ptm, y_pred_ptm),
            'recall': calculate_recall(y_true_ptm, y_pred_ptm),
            'support': len(y_true_ptm),
        }

        results[ptm_type] = metrics

    return results
