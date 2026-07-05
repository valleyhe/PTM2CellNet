"""
可视化模块单元测试
使用matplotlib Agg后端进行无头测试
使用tmp_path进行文件输出隔离
"""

import os
import sys
from unittest.mock import MagicMock

import matplotlib
import numpy as np

matplotlib.use("Agg")

# Mock seaborn before importing visualization module
mock_sns = MagicMock()
mock_sns.set_style = MagicMock()
mock_sns.heatmap = MagicMock(return_value=MagicMock())
sys.modules["seaborn"] = mock_sns

from src.evaluation.visualization import (
    plot_attention_heatmap,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curve,
    plot_roc_curve,
    plot_training_curves,
)


class TestPlotConfusionMatrix:
    """混淆矩阵图测试"""

    def test_plot_confusion_matrix(self, tmp_path):
        """基本绘图功能"""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 0, 2])
        save_path = str(tmp_path / "confusion_matrix.png")
        plot_confusion_matrix(y_true, y_pred, save_path)
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_plot_with_labels(self, tmp_path):
        """带标签的混淆矩阵"""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 0, 2])
        save_path = str(tmp_path / "cm_labels.png")
        labels = ["Class A", "Class B", "Class C"]
        plot_confusion_matrix(y_true, y_pred, save_path, labels=labels)
        assert os.path.exists(save_path)

    def test_save_figure(self, tmp_path):
        """保存图像到文件"""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        save_path = str(tmp_path / "sub_dir" / "cm.png")
        plot_confusion_matrix(y_true, y_pred, save_path)
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_normalize_false(self, tmp_path):
        """不归一化的混淆矩阵"""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        save_path = str(tmp_path / "cm_raw.png")
        plot_confusion_matrix(y_true, y_pred, save_path, normalize=False)
        assert os.path.exists(save_path)


class TestPlotROCCurve:
    """ROC曲线测试"""

    def test_plot_roc_curve_binary(self, tmp_path):
        """二分类ROC曲线"""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])
        save_path = str(tmp_path / "roc_binary.png")
        plot_roc_curve(y_true, y_score, save_path)
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_plot_roc_curve_multiclass(self, tmp_path):
        """多分类ROC曲线"""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_score = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
                [0.9, 0.05, 0.05],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ]
        )
        save_path = str(tmp_path / "roc_multiclass.png")
        plot_roc_curve(y_true, y_score, save_path)
        assert os.path.exists(save_path)

    def test_auc_annotation(self, tmp_path):
        """AUC值标注"""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])
        save_path = str(tmp_path / "roc_auc.png")
        labels = ["Negative", "Positive"]
        plot_roc_curve(y_true, y_score, save_path, labels=labels)
        assert os.path.exists(save_path)

    def test_single_column_score(self, tmp_path):
        """单列概率分数"""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([[0.1], [0.4], [0.35], [0.8]])
        save_path = str(tmp_path / "roc_single_col.png")
        plot_roc_curve(y_true, y_score, save_path)
        assert os.path.exists(save_path)


class TestPlotPrecisionRecall:
    """PR曲线测试"""

    def test_plot_precision_recall_curve(self, tmp_path):
        """PR曲线绘制"""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])
        save_path = str(tmp_path / "pr_curve.png")
        plot_pr_curve(y_true, y_score, save_path)
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_ap_annotation(self, tmp_path):
        """AP值标注"""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])
        save_path = str(tmp_path / "pr_ap.png")
        labels = ["Negative", "Positive"]
        plot_pr_curve(y_true, y_score, save_path, labels=labels)
        assert os.path.exists(save_path)

    def test_multiclass_pr(self, tmp_path):
        """多分类PR曲线"""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_score = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
                [0.9, 0.05, 0.05],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ]
        )
        save_path = str(tmp_path / "pr_multiclass.png")
        plot_pr_curve(y_true, y_score, save_path)
        assert os.path.exists(save_path)


class TestPlotTrainingHistory:
    """训练历史测试"""

    def test_plot_training_history(self, tmp_path):
        """训练和验证损失曲线"""
        logs = {
            "train_loss": [0.9, 0.7, 0.5, 0.3],
            "val_loss": [0.8, 0.6, 0.5, 0.4],
        }
        save_path = str(tmp_path / "training_curves.png")
        plot_training_curves(logs, save_path)
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_plot_metrics(self, tmp_path):
        """指标历史曲线"""
        logs = {
            "train_loss": [0.9, 0.7, 0.5],
            "val_loss": [0.8, 0.6, 0.5],
            "train_accuracy": [0.6, 0.75, 0.85],
            "val_accuracy": [0.65, 0.72, 0.8],
        }
        save_path = str(tmp_path / "metrics_curves.png")
        plot_training_curves(logs, save_path)
        assert os.path.exists(save_path)

    def test_partial_logs(self, tmp_path):
        """只有部分日志键"""
        logs = {"train_loss": [0.9, 0.7, 0.5]}
        save_path = str(tmp_path / "partial_curves.png")
        plot_training_curves(logs, save_path)
        assert os.path.exists(save_path)


class TestPlotFeatureImportance:
    """特征重要性图测试"""

    def test_plot_feature_importance(self, tmp_path):
        """特征重要性绘图"""
        importances = np.array([0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.03, 0.01, 0.01])
        feature_names = [f"feat_{i}" for i in range(10)]
        save_path = str(tmp_path / "feature_importance.png")
        plot_feature_importance(importances, feature_names, save_path)
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_top_n(self, tmp_path):
        """top_n参数"""
        importances = np.array([0.3, 0.2, 0.15, 0.1, 0.08])
        feature_names = [f"feat_{i}" for i in range(5)]
        save_path = str(tmp_path / "top_n.png")
        plot_feature_importance(importances, feature_names, save_path, top_n=3)
        assert os.path.exists(save_path)


class TestPlotAttentionHeatmap:
    """注意力热图测试"""

    def test_plot_attention_heatmap(self, tmp_path):
        """基本注意力热图"""
        attention = np.random.rand(5, 5)
        save_path = str(tmp_path / "attention.png")
        plot_attention_heatmap(attention, save_path)
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_with_labels(self, tmp_path):
        """带标签的注意力热图"""
        attention = np.random.rand(3, 3)
        save_path = str(tmp_path / "attention_labels.png")
        xlabels = ["A", "B", "C"]
        ylabels = ["1", "2", "3"]
        plot_attention_heatmap(attention, save_path, xticklabels=xlabels, yticklabels=ylabels)
        assert os.path.exists(save_path)
