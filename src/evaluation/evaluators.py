"""
评估器模块
功能概述: 提供完整的模型评估流程
设计思路: 封装评估指标，支持批量评估和交叉验证
"""

from typing import Any, Callable, Dict, List, Optional, Type

from typing_extensions import TypeAlias

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

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

    def cross_validate(
        self,
        dataset: torch.utils.data.Dataset,
        model_class: Optional[Type] = None,
        n_splits: int = 5,
        model_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,
        return_predictions: bool = False,
        train_fn: Optional[Callable[..., Any]] = None,
        train_fn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """K-fold cross-validation.

        Splits the dataset into *n_splits* roughly equal partitions, then
        iteratively holds out one partition as the validation set while
        training on the rest (via ``train_fn`` when provided, otherwise only
        evaluating ``self.model`` / a fresh ``model_class`` instance).

        Args:
            dataset: A torch Dataset (not DataLoader) to split.
            model_class: Optional model class to re-initialize per fold.
                If None, uses ``self.model`` for all folds.
            n_splits: Number of splits.  Clamped to ``len(dataset)`` when it
                exceeds the dataset size so that every split has at least one
                sample.
            model_kwargs: Keyword arguments passed to ``model_class`` when
                re-initializing per fold.
            batch_size: Batch size for the DataLoaders created from each
                train/val subset.
            shuffle: Whether to shuffle indices before splitting.
            seed: Random seed used when *shuffle* is True.
            return_predictions: If True, include per-fold prediction arrays
                under the ``"fold_predictions"`` key.
            train_fn: Optional callable invoked as
                ``train_fn(model, train_loader, **train_fn_kwargs)`` to train
                the freshly initialized model on each fold's training subset.
                When provided, each fold truly trains before evaluation
                (historically only evaluation was performed, which made the
                name "cross_validate" misleading). When omitted, behavior is
                unchanged (evaluation-only) and a warning is logged.
            train_fn_kwargs: Extra keyword arguments forwarded to ``train_fn``.

        Returns:
            Dict with keys:
                - ``"fold_metrics"``: list of per-fold metric dicts.
                - ``"mean_metrics"``: dict of metric name -> mean across folds.
                - ``"std_metrics"``: dict of metric name -> std across folds.
                - ``"n_splits"``: int, the actual number of splits used.
                - ``"fold_predictions"`` (optional): list of prediction arrays,
                  one per fold, only when *return_predictions* is True.
        """
        n_samples = len(dataset)
        if n_samples == 0:
            raise ValueError("Cannot cross-validate on an empty dataset")

        effective_splits = min(n_splits, n_samples)

        # 1. Generate and optionally shuffle indices
        indices = np.arange(n_samples)
        if shuffle:
            rng = np.random.RandomState(seed)
            rng.shuffle(indices)

        # 2. Split into roughly equal chunks
        fold_sizes = np.full(effective_splits, n_samples // effective_splits, dtype=int)
        fold_sizes[: n_samples % effective_splits] += 1
        fold_indices = np.split(indices, np.cumsum(fold_sizes)[:-1])

        if train_fn is None:
            logger.warning(
                "cross_validate 在未提供 train_fn 时仅对未训练模型做评估，"
                "结果不能反映训练后性能。请传入 train_fn 回调以启用真正的交叉验证训练。"
            )

        fold_metrics: List[Dict[str, Any]] = []
        fold_predictions: List[Array] = []

        # 3. Iterate folds
        for fold_idx in range(effective_splits):
            val_idx = fold_indices[fold_idx]
            train_idx = np.concatenate(
                [fold_indices[j] for j in range(effective_splits) if j != fold_idx]
            )

            train_subset = Subset(dataset, train_idx.tolist())
            val_subset = Subset(dataset, val_idx.tolist())

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            logger.info("交叉验证 fold %d/%d", fold_idx + 1, effective_splits)

            if model_class is not None:
                kw = model_kwargs or {}
                fold_model = model_class(**kw)
                fold_model.to(self.device)
                self.model = fold_model

            # 真正训练当前 fold 的模型（当 train_fn 提供时）
            if train_fn is not None:
                try:
                    train_fn(self.model, train_loader, **(train_fn_kwargs or {}))
                except Exception as exc:
                    logger.error("fold %d 训练失败: %s", fold_idx + 1, exc)
                    raise
                # 确保训练后切回评估模式
                if hasattr(self.model, "eval"):
                    self.model.eval()

            result = self.evaluate(val_loader, return_predictions=return_predictions)
            fold_metrics.append(result["metrics"])

            if return_predictions and "predictions" in result:
                fold_predictions.append(result["predictions"])

        # 4. Aggregate mean and std across folds
        all_keys = sorted(
            key
            for key in set().union(*fold_metrics)
            if isinstance(fold_metrics[0].get(key), (int, float, np.integer, np.floating))
        )
        mean_metrics: Dict[str, float] = {}
        std_metrics: Dict[str, float] = {}
        for key in all_keys:
            values = [float(fm[key]) for fm in fold_metrics]
            mean_metrics[key] = float(np.mean(values))
            std_metrics[key] = float(np.std(values))

        # 5. Build result dict
        cv_result: Dict[str, Any] = {
            "fold_metrics": fold_metrics,
            "mean_metrics": mean_metrics,
            "std_metrics": std_metrics,
            "n_splits": effective_splits,
        }
        if return_predictions:
            cv_result["fold_predictions"] = fold_predictions

        logger.info("交叉验证完成: mean=%s", mean_metrics)
        return cv_result

    def _calculate_classification_metrics(
        self,
        y_true: Array,
        y_pred: Array,
        y_score: Optional[Array] = None,
        ptm_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """计算分类指标。"""
        if y_score is not None and len(y_score.shape) > 1:
            n_labels = len(np.unique(y_true))
            n_score_columns = y_score.shape[1]
            if n_labels > n_score_columns:
                raise ValueError(
                    f"标签类别数与模型输出类别数不一致: "
                    f"y_true中包含{n_labels}个类别，但y_score只有{n_score_columns}列。"
                    f"请确保评估数据与模型输出类别数匹配。"
                )

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
