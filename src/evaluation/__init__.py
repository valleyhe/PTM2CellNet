"""评估模块 - 评估指标、评估器和结果可视化"""
from typing import Any

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
)

_Evaluator: Any = None
_LeaveOnePTMOutScorer: Any = None
_TwoStageExplanationPipeline: Any = None
_aggregate_by_protein: Any = None
_plot_roc_curve: Any = None
_plot_pr_curve: Any = None
_plot_confusion_matrix: Any = None
_plot_training_curves: Any = None
_plot_feature_importance: Any = None
_plot_attention_heatmap: Any = None

try:
    from .evaluators import Evaluator as _Evaluator
except ImportError:
    pass

try:
    from .explainers import (
        LeaveOnePTMOutScorer as _LeaveOnePTMOutScorer,
        TwoStageExplanationPipeline as _TwoStageExplanationPipeline,
        aggregate_by_protein as _aggregate_by_protein,
    )
except ImportError:
    pass

try:
    from .visualization import (
        plot_roc_curve as _plot_roc_curve,
        plot_pr_curve as _plot_pr_curve,
        plot_confusion_matrix as _plot_confusion_matrix,
        plot_training_curves as _plot_training_curves,
        plot_feature_importance as _plot_feature_importance,
        plot_attention_heatmap as _plot_attention_heatmap,
    )
except ImportError:
    pass

Evaluator: Any = _Evaluator
LeaveOnePTMOutScorer: Any = _LeaveOnePTMOutScorer
TwoStageExplanationPipeline: Any = _TwoStageExplanationPipeline
aggregate_by_protein: Any = _aggregate_by_protein
plot_roc_curve: Any = _plot_roc_curve
plot_pr_curve: Any = _plot_pr_curve
plot_confusion_matrix: Any = _plot_confusion_matrix
plot_training_curves: Any = _plot_training_curves
plot_feature_importance: Any = _plot_feature_importance
plot_attention_heatmap: Any = _plot_attention_heatmap

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
