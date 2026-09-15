"""评估模块 - 评估指标、评估器和结果可视化"""

import logging

from src.utils.lazy_import import LazyImport

_logger = logging.getLogger(__name__)

from .metrics import (
    calculate_accuracy,
    calculate_precision,
    calculate_recall,
    calculate_f1_score,
    calculate_auc_roc,
    calculate_auc_pr,
    calculate_confusion_matrix,
    calculate_classification_report,
    calculate_mae,
    calculate_mse,
    calculate_rmse,
    calculate_r2,
    calculate_ndcg,
    calculate_map,
    calculate_ranking_metrics,
    calculate_bootstrap_ci,
    calculate_metric_ci,
    calculate_cross_validation_ci,
)

# Optional submodules are imported lazily so that a missing optional
# dependency does not break importing src.evaluation. Accessing a symbol
# whose import failed re-raises the original ImportError.
_Evaluator: LazyImport = LazyImport(f"{__name__}.evaluators", "Evaluator")
_LeaveOnePTMOutScorer: LazyImport = LazyImport(f"{__name__}.explainers", "LeaveOnePTMOutScorer")
_TwoStageExplanationPipeline: LazyImport = LazyImport(f"{__name__}.explainers", "TwoStageExplanationPipeline")
_aggregate_by_protein: LazyImport = LazyImport(f"{__name__}.explainers", "aggregate_by_protein")
_plot_roc_curve: LazyImport = LazyImport(f"{__name__}.visualization", "plot_roc_curve")
_plot_pr_curve: LazyImport = LazyImport(f"{__name__}.visualization", "plot_pr_curve")
_plot_confusion_matrix: LazyImport = LazyImport(f"{__name__}.visualization", "plot_confusion_matrix")
_plot_training_curves: LazyImport = LazyImport(f"{__name__}.visualization", "plot_training_curves")
_plot_feature_importance: LazyImport = LazyImport(f"{__name__}.visualization", "plot_feature_importance")
_plot_attention_heatmap: LazyImport = LazyImport(f"{__name__}.visualization", "plot_attention_heatmap")

Evaluator: LazyImport = _Evaluator
LeaveOnePTMOutScorer: LazyImport = _LeaveOnePTMOutScorer
TwoStageExplanationPipeline: LazyImport = _TwoStageExplanationPipeline
aggregate_by_protein: LazyImport = _aggregate_by_protein
plot_roc_curve: LazyImport = _plot_roc_curve
plot_pr_curve: LazyImport = _plot_pr_curve
plot_confusion_matrix: LazyImport = _plot_confusion_matrix
plot_training_curves: LazyImport = _plot_training_curves
plot_feature_importance: LazyImport = _plot_feature_importance
plot_attention_heatmap: LazyImport = _plot_attention_heatmap

__all__ = [
    "calculate_accuracy",
    "calculate_precision",
    "calculate_recall",
    "calculate_f1_score",
    "calculate_auc_roc",
    "calculate_auc_pr",
    "calculate_confusion_matrix",
    "calculate_classification_report",
    "calculate_mae",
    "calculate_mse",
    "calculate_rmse",
    "calculate_r2",
    "calculate_ndcg",
    "calculate_map",
    "calculate_ranking_metrics",
    "calculate_bootstrap_ci",
    "calculate_metric_ci",
    "calculate_cross_validation_ci",
    "Evaluator",
    "LeaveOnePTMOutScorer",
    "TwoStageExplanationPipeline",
    "aggregate_by_protein",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_confusion_matrix",
    "plot_training_curves",
    "plot_feature_importance",
    "plot_attention_heatmap",
]
