"""
visualization.py 直接测试
快照式测试: 验证每个公开函数调用不抛异常 + 返回类型正确 + 文件落盘
使用 matplotlib Agg 后端避免显示, 用 tmp_path 鐦离输出文件
不校验图像像素内容, 仅校验调用稳定性与产物存在性
"""

import os

import matplotlib

matplotlib.use("Agg")

import numpy as np

from src.evaluation.visualization import (
    plot_attention_heatmap,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curve,
    plot_roc_curve,
    plot_training_curves,
)


# ---------------------------------------------------------------------------
# plot_roc_curve
# ---------------------------------------------------------------------------


class TestPlotRocCurve:
    def test_binary_returns_none_and_saves(self, tmp_path):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])
        save_path = str(tmp_path / "roc_binary.png")

        result = plot_roc_curve(y_true, y_score, save_path)

        assert result is None
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_multiclass(self, tmp_path):
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

        result = plot_roc_curve(y_true, y_score, save_path, labels=["A", "B", "C"])

        assert result is None
        assert os.path.exists(save_path)

    def test_single_column_score(self, tmp_path):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([[0.1], [0.4], [0.35], [0.8]])
        save_path = str(tmp_path / "roc_single_col.png")

        result = plot_roc_curve(y_true, y_score, save_path)

        assert result is None
        assert os.path.exists(save_path)


# ---------------------------------------------------------------------------
# plot_pr_curve
# ---------------------------------------------------------------------------


class TestPlotPrCurve:
    def test_binary(self, tmp_path):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])
        save_path = str(tmp_path / "pr_binary.png")

        result = plot_pr_curve(y_true, y_score, save_path)

        assert result is None
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_multiclass_with_labels(self, tmp_path):
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

        result = plot_pr_curve(y_true, y_score, save_path, labels=["A", "B", "C"])

        assert result is None
        assert os.path.exists(save_path)


# ---------------------------------------------------------------------------
# plot_confusion_matrix
# ---------------------------------------------------------------------------


class TestPlotConfusionMatrix:
    def test_normalized(self, tmp_path):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 0, 2])
        save_path = str(tmp_path / "cm_normalized.png")

        result = plot_confusion_matrix(y_true, y_pred, save_path)

        assert result is None
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_not_normalized(self, tmp_path):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        save_path = str(tmp_path / "cm_raw.png")

        result = plot_confusion_matrix(y_true, y_pred, save_path, normalize=False)

        assert result is None
        assert os.path.exists(save_path)

    def test_with_labels_and_subdir(self, tmp_path):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 0, 2])
        save_path = str(tmp_path / "sub" / "cm_labels.png")
        labels = ["Class A", "Class B", "Class C"]

        result = plot_confusion_matrix(y_true, y_pred, save_path, labels=labels)

        assert result is None
        assert os.path.exists(save_path)


# ---------------------------------------------------------------------------
# plot_training_curves
# ---------------------------------------------------------------------------


class TestPlotTrainingCurves:
    def test_loss_only(self, tmp_path):
        logs = {
            "train_loss": [0.9, 0.7, 0.5, 0.3],
            "val_loss": [0.8, 0.6, 0.5, 0.4],
        }
        save_path = str(tmp_path / "training_loss.png")

        result = plot_training_curves(logs, save_path)

        assert result is None
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_full_metrics(self, tmp_path):
        logs = {
            "train_loss": [0.9, 0.7, 0.5],
            "val_loss": [0.8, 0.6, 0.5],
            "train_accuracy": [0.6, 0.75, 0.85],
            "val_accuracy": [0.65, 0.72, 0.8],
        }
        save_path = str(tmp_path / "training_full.png")

        result = plot_training_curves(logs, save_path)

        assert result is None
        assert os.path.exists(save_path)

    def test_partial_logs_no_exception(self, tmp_path):
        logs = {"train_loss": [0.9, 0.7, 0.5]}
        save_path = str(tmp_path / "training_partial.png")

        result = plot_training_curves(logs, save_path)

        assert result is None
        assert os.path.exists(save_path)


# ---------------------------------------------------------------------------
# plot_feature_importance
# ---------------------------------------------------------------------------


class TestPlotFeatureImportance:
    def test_default_top_n(self, tmp_path):
        importances = np.array([0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.03, 0.01, 0.01])
        feature_names = [f"feat_{i}" for i in range(10)]
        save_path = str(tmp_path / "feature_importance.png")

        result = plot_feature_importance(importances, feature_names, save_path)

        assert result is None
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_top_n_smaller(self, tmp_path):
        importances = np.array([0.3, 0.2, 0.15, 0.1, 0.08])
        feature_names = [f"feat_{i}" for i in range(5)]
        save_path = str(tmp_path / "feature_topn.png")

        result = plot_feature_importance(importances, feature_names, save_path, top_n=3)

        assert result is None
        assert os.path.exists(save_path)


# ---------------------------------------------------------------------------
# plot_attention_heatmap
# ---------------------------------------------------------------------------


class TestPlotAttentionHeatmap:
    def test_basic(self, tmp_path):
        rng = np.random.default_rng(42)
        attention = rng.random((5, 5))
        save_path = str(tmp_path / "attention.png")

        result = plot_attention_heatmap(attention, save_path)

        assert result is None
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_with_labels(self, tmp_path):
        rng = np.random.default_rng(7)
        attention = rng.random((3, 3))
        save_path = str(tmp_path / "attention_labels.png")
        xlabels = ["A", "B", "C"]
        ylabels = ["1", "2", "3"]

        result = plot_attention_heatmap(attention, save_path, xticklabels=xlabels, yticklabels=ylabels)

        assert result is None
        assert os.path.exists(save_path)
