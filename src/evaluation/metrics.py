"""
评估指标模块
功能概述: 提供分类和回归任务的各种评估指标
设计思路: 封装scikit-learn的指标，提供统一的接口
"""

from typing import Any, Dict, List, Optional, Tuple
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
    if len(y_score.shape) > 1:
        unique_labels = np.unique(y_true)
        n_score_columns = y_score.shape[1]
        if len(unique_labels) > n_score_columns:
            raise ValueError(
                f"标签类别数与模型输出类别数不一致: "
                f"y_true中包含{len(unique_labels)}个类别，但y_score只有{n_score_columns}列。"
                f"请确保评估数据与模型输出类别数匹配。"
            )

    # ROC AUC is undefined when y_true contains a single class; return NaN
    # rather than triggering sklearn's UndefinedMetricWarning on degenerate
    # (e.g. single-class cross-validation folds) inputs.
    if np.unique(y_true).size < 2:
        return float("nan")

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
    if len(y_score.shape) > 1:
        unique_labels = np.unique(y_true)
        n_score_columns = y_score.shape[1]
        if len(unique_labels) > n_score_columns:
            raise ValueError(
                f"标签类别数与模型输出类别数不一致: "
                f"y_true中包含{len(unique_labels)}个类别，但y_score只有{n_score_columns}列。"
                f"请确保评估数据与模型输出类别数匹配。"
            )

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


# ── Ranking metrics (NDCG, MAP) ──────────────────────────────────────────


def calculate_ndcg(
    y_true: Array,
    y_score: Array,
    k: Optional[int] = None,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG).

    Supports binary and graded relevance.  DCG = sum((2^rel - 1) / log2(i+1))
    where *i* is the 1-based rank position.  NDCG = DCG / IDCG.

    Args:
        y_true: Array of true relevance scores (binary or graded).
        y_score: Array of predicted scores used for ranking.
        k: Optional cutoff position.  None means use all items.

    Returns:
        NDCG score between 0 and 1.  Returns 0.0 for degenerate cases
        (empty arrays, single element, or all-zero relevance).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)

    # Edge cases
    if y_true.size == 0 or y_score.size == 0:
        return 0.0
    if y_true.size == 1:
        return 0.0

    # Sort by predicted score descending
    ranked_indices = np.argsort(-y_score)
    ranked_relevance = y_true[ranked_indices]

    # Apply cutoff
    if k is not None:
        k = min(k, len(ranked_relevance))
        ranked_relevance = ranked_relevance[:k]

    # Compute DCG
    positions = np.arange(1, len(ranked_relevance) + 1, dtype=np.float64)
    discounts = np.log2(positions + 1)  # log2(i+1) where i is 1-based
    gains = (2.0 ** ranked_relevance) - 1.0
    dcg = np.sum(gains / discounts)

    # Compute IDCG (ideal ordering: sort relevance descending)
    ideal_relevance = np.sort(y_true)[::-1]
    if k is not None:
        ideal_relevance = ideal_relevance[:k]

    ideal_gains = (2.0 ** ideal_relevance) - 1.0
    ideal_positions = np.arange(1, len(ideal_relevance) + 1, dtype=np.float64)
    ideal_discounts = np.log2(ideal_positions + 1)
    idcg = np.sum(ideal_gains / ideal_discounts)

    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)


def calculate_map(
    y_true: Array,
    y_score: Array,
) -> float:
    """
    Compute Mean Average Precision (MAP).

    Sorts items by predicted score descending, then computes the average
    of precisions at each position where a relevant item appears.

    Args:
        y_true: Binary array of true labels (0 or 1).
        y_score: Array of predicted scores used for ranking.

    Returns:
        MAP score between 0 and 1.  Returns 0.0 for degenerate cases
        (empty arrays, no positive labels).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)

    # Edge cases
    if y_true.size == 0 or y_score.size == 0:
        return 0.0

    n_positives = int(np.sum(y_true > 0))
    if n_positives == 0:
        return 0.0

    # Sort by predicted score descending
    ranked_indices = np.argsort(-y_score)
    ranked_labels = y_true[ranked_indices]

    # Compute average precision
    cumsum = np.cumsum(ranked_labels)
    positions = np.arange(1, len(ranked_labels) + 1, dtype=np.float64)
    precisions = cumsum / positions

    # Average precision at positions where label is positive
    ap = np.sum(precisions * ranked_labels) / n_positives

    return float(ap)


def calculate_ranking_metrics(
    y_true: Array,
    y_score: Array,
    k: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute ranking metrics (NDCG and MAP) in a single call.

    Args:
        y_true: Array of true relevance scores / binary labels.
        y_score: Array of predicted scores used for ranking.
        k: Optional cutoff position for NDCG.  None means use all items.

    Returns:
        Dictionary with keys ``"ndcg"`` and ``"map"`` mapping to float scores.
    """
    return {
        "ndcg": calculate_ndcg(y_true, y_score, k=k),
        "map": calculate_map(y_true, y_score),
    }


# ── Confidence interval functions ─────────────────────────────────────────


def calculate_bootstrap_ci(
    values: Array,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a list/array of metric values.

    Resamples *values* with replacement, computes the mean of each resample,
    and returns the percentile-based confidence interval.

    Args:
        values: Array-like of observed metric values.
        confidence: Confidence level (default 0.95).
        n_bootstrap: Number of bootstrap resamples (default 1000).
        seed: Random seed for reproducibility (default 42).

    Returns:
        Tuple of (lower, upper) percentile bounds.
        Returns (0.0, 0.0) for empty arrays and (value, value) for a
        single-element array.
    """
    values = np.asarray(values, dtype=np.float64)

    # Edge cases
    if values.size == 0:
        return (0.0, 0.0)
    if values.size == 1:
        v = float(values[0])
        return (v, v)

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        sample = rng.choice(values, size=values.size, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return (lower, upper)


def calculate_metric_ci(
    y_true: Array,
    y_pred: Array,
    metric_fn: Any,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute a metric with bootstrap confidence interval.

    Resamples (y_true, y_pred) pairs with replacement, evaluates *metric_fn*
    on each resample, and returns the point estimate plus CI.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels or scores.
        metric_fn: Callable ``(y_true, y_pred) -> float`` (e.g.
            :func:`calculate_accuracy`).
        confidence: Confidence level (default 0.95).
        n_bootstrap: Number of bootstrap resamples (default 1000).
        seed: Random seed for reproducibility (default 42).

    Returns:
        Dictionary with keys ``"point_estimate"``, ``"ci_lower"``,
        ``"ci_upper"``, and ``"confidence"``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Edge case: empty arrays
    if y_true.size == 0 or y_pred.size == 0:
        return {
            "point_estimate": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "confidence": confidence,
        }

    # Point estimate on full data
    point_estimate = float(metric_fn(y_true, y_pred))

    # Single sample: CI collapses to the point estimate
    if y_true.size == 1:
        return {
            "point_estimate": point_estimate,
            "ci_lower": point_estimate,
            "ci_upper": point_estimate,
            "confidence": confidence,
        }

    rng = np.random.RandomState(seed)
    n = y_true.size
    boot_scores = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        boot_scores[i] = metric_fn(y_true[indices], y_pred[indices])

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_scores, 100 * alpha / 2))
    upper = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))

    return {
        "point_estimate": point_estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": confidence,
    }


def calculate_cross_validation_ci(
    fold_values: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute t-distribution based confidence interval for cross-validation folds.

    Uses the Student t critical value when scipy is available; falls back to
    a normal approximation (z = 1.96 for 95%) otherwise.

    Args:
        fold_values: List of metric values from each CV fold.
        confidence: Confidence level (default 0.95).

    Returns:
        Tuple of (lower, upper) bounds.
        Returns (0.0, 0.0) for empty input and (value, value) for a single
        fold.
    """
    fold_values_arr = np.asarray(fold_values, dtype=np.float64)

    # Edge cases
    if fold_values_arr.size == 0:
        return (0.0, 0.0)
    if fold_values_arr.size == 1:
        v = float(fold_values_arr[0])
        return (v, v)

    mean = float(np.mean(fold_values_arr))
    std = float(np.std(fold_values_arr, ddof=1))
    n = fold_values_arr.size
    se = std / np.sqrt(n)

    alpha = 1.0 - confidence

    try:
        from scipy.stats import t as t_dist

        critical_value = float(t_dist.ppf(1 - alpha / 2, df=n - 1))
    except ImportError:
        # Normal approximation fallback
        # Inverse normal: z_{1-alpha/2} via erfc approximation
        # For common confidence levels, use a simple lookup
        _z_lookup = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        critical_value = _z_lookup.get(confidence, 1.96)

    margin = critical_value * se
    return (float(mean - margin), float(mean + margin))
