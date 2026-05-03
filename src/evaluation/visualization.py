"""
结果可视化模块
功能概述: 提供各种评估结果的可视化图表
设计思路: 使用matplotlib和seaborn生成专业的可视化图表
"""

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

plt.switch_backend("Agg")
sns.set_style("whitegrid")


def _ensure_parent_dir(file_path: str) -> None:
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: str,
    title: str = "ROC Curve",
    labels: Optional[List[str]] = None,
) -> None:
    """
    绘制ROC曲线

    参数:
        y_true: 真实标签
        y_score: 预测概率分数
        save_path: 保存路径
        title: 图表标题
        labels: 类别标签列表
    """
    _ensure_parent_dir(save_path)

    plt.figure(figsize=(8, 6))

    if len(y_score.shape) == 1 or y_score.shape[1] == 1:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    else:
        n_classes = y_score.shape[1]
        y_true_onehot = np.zeros((len(y_true), n_classes))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            label = labels[i] if labels and i < len(labels) else f"Class {i}"
            plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("ROC曲线已保存到: %s", save_path)


def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: str,
    title: str = "Precision-Recall Curve",
    labels: Optional[List[str]] = None,
) -> None:
    """
    绘制PR曲线

    参数:
        y_true: 真实标签
        y_score: 预测概率分数
        save_path: 保存路径
        title: 图表标题
        labels: 类别标签列表
    """
    _ensure_parent_dir(save_path)

    plt.figure(figsize=(8, 6))

    if len(y_score.shape) == 1 or y_score.shape[1] == 1:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f"PR (AUC = {pr_auc:.4f})")
    else:
        n_classes = y_score.shape[1]
        y_true_onehot = np.zeros((len(y_true), n_classes))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1

        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_score[:, i])
            pr_auc = auc(recall, precision)
            label = labels[i] if labels and i < len(labels) else f"Class {i}"
            plt.plot(recall, precision, lw=2, label=f"{label} (AUC = {pr_auc:.4f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("PR曲线已保存到: %s", save_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    title: str = "Confusion Matrix",
    labels: Optional[List[str]] = None,
    normalize: bool = True,
) -> None:
    """
    绘制混淆矩阵

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        save_path: 保存路径
        title: 图表标题
        labels: 类别标签列表
        normalize: 是否归一化
    """
    _ensure_parent_dir(save_path)

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=labels if labels else "auto",
        yticklabels=labels if labels else "auto",
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("混淆矩阵已保存到: %s", save_path)


def plot_training_curves(
    logs: Dict[str, List[float]],
    save_path: str,
    title: str = "Training Curves",
) -> None:
    """
    绘制训练曲线

    参数:
        logs: 训练日志字典，键为指标名，值为各epoch的值列表
        save_path: 保存路径
        title: 图表标题
    """
    _ensure_parent_dir(save_path)

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    if "train_loss" in logs:
        axes[0].plot(logs["train_loss"], label="Train Loss", lw=2)
    if "val_loss" in logs:
        axes[0].plot(logs["val_loss"], label="Val Loss", lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "train_accuracy" in logs:
        axes[1].plot(logs["train_accuracy"], label="Train Accuracy", lw=2)
    if "val_accuracy" in logs:
        axes[1].plot(logs["val_accuracy"], label="Val Accuracy", lw=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, y=1.02, fontsize=14)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("训练曲线已保存到: %s", save_path)


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    save_path: str,
    title: str = "Feature Importance",
    top_n: int = 20,
) -> None:
    """
    绘制特征重要性

    参数:
        importances: 特征重要性数组
        feature_names: 特征名称列表
        save_path: 保存路径
        title: 图表标题
        top_n: 显示前N个重要特征
    """
    _ensure_parent_dir(save_path)

    indices = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[indices]
    top_features = [feature_names[i] for i in indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_importances)), top_importances, color="steelblue")
    plt.yticks(range(len(top_importances)), top_features)
    plt.xlabel("Importance")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("特征重要性图已保存到: %s", save_path)


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    save_path: str,
    title: str = "Attention Heatmap",
    xticklabels: Optional[List[str]] = None,
    yticklabels: Optional[List[str]] = None,
) -> None:
    """
    绘制注意力热图

    参数:
        attention_weights: 注意力权重矩阵
        save_path: 保存路径
        title: 图表标题
        xticklabels: x轴标签
        yticklabels: y轴标签
    """
    _ensure_parent_dir(save_path)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights,
        cmap="viridis",
        xticklabels=xticklabels if xticklabels else "auto",
        yticklabels=yticklabels if yticklabels else "auto",
    )
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.title(title)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("注意力热图已保存到: %s", save_path)
