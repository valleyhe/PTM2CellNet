"""
训练器模块
功能概述: 提供模型训练、验证和评估的完整流程
设计思路: 封装训练循环，支持回调机制，灵活配置
"""

from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# 兼容PyTorch 1.x和2.x的LRScheduler导入
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from ..utils.logging import setup_logger
from .optimizers import configure_optimizer
from .callbacks import Callback

logger = setup_logger(__name__)


class Trainer:
    """
    训练器类
    负责模型的训练、验证和评估
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        初始化训练器

        参数:
            model: PyTorch模型
            config: 配置字典
            device: 设备，如 "cuda" 或 "cpu"
        """
        self.model = model
        self.config = config or {}
        self.training_config = self.config.get("training", {})

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[LRScheduler] = None
        self.loss_fn: Optional[nn.Module] = None
        self.callbacks: List[Callback] = []
        self.epoch = 0
        self.global_step = 0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        # S8: 训练曲线可视化依赖 accuracy 指标，与 train_losses/val_losses 对应记录
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []
        self.best_val_loss = float("inf")

    def compile(
        self,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        """
        编译训练器，设置损失函数、优化器和回调

        参数:
            loss_fn: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            callbacks: 回调列表
        """
        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn

        if optimizer is None:
            self.optimizer, self.scheduler = configure_optimizer(self.model, self.config)
        else:
            self.optimizer = optimizer
            self.scheduler = scheduler

        if callbacks is not None:
            self.callbacks = callbacks

    def add_callback(self, callback: Callback):
        """
        添加回调

        参数:
            callback: 回调实例
        """
        self.callbacks.append(callback)

    def _train_epoch(self, train_loader: DataLoader[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        训练一个epoch

        参数:
            train_loader: 训练数据加载器

        返回:
            训练指标字典
        """
        if self.optimizer is None or self.loss_fn is None:
            raise RuntimeError("Trainer must be compiled before training")

        self.model.train()
        total_loss = 0.0
        num_samples = 0
        # S8: 同时累积训练准确率，供 plot_training_curves 使用
        correct_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            if not isinstance(batch, dict):
                raise TypeError("Batch must be a dict")
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(batch)

            if isinstance(outputs, dict):
                logits = outputs.get("logits")
                predictions = outputs.get("predictions")
            else:
                logits = outputs
                predictions = None

            if logits is None or not isinstance(logits, torch.Tensor):
                raise TypeError("Model output must contain tensor logits")

            targets = batch.get("label")
            if not isinstance(targets, torch.Tensor):
                raise TypeError("Batch must contain tensor label")
            loss = self.loss_fn(logits, targets)

            loss.backward()
            self.optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size

            # 计算训练批准确率（若模型未直接给出 predictions，则从 logits 取 argmax）
            if predictions is None:
                predictions = torch.argmax(logits, dim=1)
            if isinstance(predictions, torch.Tensor) and predictions.dim() == 1:
                correct_samples += (predictions == targets).sum().item()

            for callback in self.callbacks:
                callback.on_batch_end(self, batch_idx, {"loss": loss.item()})

            self.global_step += 1

        avg_loss = total_loss / num_samples
        avg_acc = correct_samples / num_samples if num_samples > 0 else 0.0
        return {"loss": avg_loss, "accuracy": avg_acc}

    def _val_epoch(self, val_loader: DataLoader[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        验证一个epoch

        参数:
            val_loader: 验证数据加载器

        返回:
            验证指标字典
        """
        if self.loss_fn is None:
            raise RuntimeError("Trainer must be compiled before validation")

        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        all_predictions: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in val_loader:
                if not isinstance(batch, dict):
                    raise TypeError("Batch must be a dict")
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                outputs = self.model(batch)

                if isinstance(outputs, dict):
                    logits = outputs.get("logits")
                    predictions = outputs.get("predictions")
                else:
                    logits = outputs
                    predictions = torch.argmax(logits, dim=1)

                if logits is None or not isinstance(logits, torch.Tensor):
                    raise TypeError("Model output must contain tensor logits")
                if predictions is None or not isinstance(predictions, torch.Tensor):
                    raise TypeError("Model output must contain tensor predictions")

                targets = batch.get("label")
                if not isinstance(targets, torch.Tensor):
                    raise TypeError("Batch must contain tensor label")
                loss = self.loss_fn(logits, targets)

                total_loss += loss.item() * targets.size(0)
                num_samples += targets.size(0)

                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

        avg_loss = total_loss / num_samples

        all_predictions_tensor = torch.cat(all_predictions, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)

        accuracy = (all_predictions_tensor == all_targets_tensor).float().mean().item()

        return {"loss": avg_loss, "accuracy": accuracy}

    def fit(
        self,
        train_loader: DataLoader[Dict[str, torch.Tensor]],
        val_loader: Optional[DataLoader[Dict[str, torch.Tensor]]] = None,
        max_epochs: Optional[int] = None,
    ) -> None:
        """
        训练模型

        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            max_epochs: 最大训练轮数
        """
        if max_epochs is None:
            max_epochs = self.training_config.get("max_epochs", 100)
        if not isinstance(max_epochs, int):
            max_epochs = 100
        epochs = max_epochs

        logger.info("开始训练，最大轮数: %d", epochs)

        for callback in self.callbacks:
            callback.on_train_start(self)

        early_stop = False

        for epoch in range(epochs):
            self.epoch = epoch

            for callback in self.callbacks:
                callback.on_epoch_start(self, epoch)

            train_logs = self._train_epoch(train_loader)
            self.train_losses.append(train_logs["loss"])
            if "accuracy" in train_logs:
                self.train_accuracies.append(train_logs["accuracy"])

            val_logs: Dict[str, float] = {}
            if val_loader is not None:
                val_logs = self._val_epoch(val_loader)
                self.val_losses.append(val_logs["loss"])
                if "accuracy" in val_logs:
                    self.val_accuracies.append(val_logs["accuracy"])

            logs = {
                "epoch": epoch,
                "train_loss": train_logs["loss"],
                **{f"val_{k}": v for k, v in val_logs.items()},
            }

            val_message = ""
            if val_logs:
                val_message = ", " + ", ".join([f"val_{k}={v:.4f}" for k, v in val_logs.items()])
            logger.info(
                "Epoch %3d: train_loss=%.4f%s",
                epoch,
                train_logs["loss"],
                val_message,
            )

            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, logs)

            for callback in self.callbacks:
                if bool(getattr(callback, "should_stop", False)):
                    early_stop = True
                    break

            if early_stop:
                logger.info("早停触发，停止训练")
                break

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_logs.get("loss", train_logs["loss"]))
                else:
                    self.scheduler.step()

        for callback in self.callbacks:
            callback.on_train_end(self)

        logger.info("训练完成")

    def validate(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """
        将给定模型移动到当前设备，运行一个验证epoch并返回损失与准确率。

        参数:
            model: 待验证的PyTorch模型
            dataloader: 验证数据加载器

        返回:
            包含 ``loss`` 和 ``accuracy`` 的验证指标字典
        """
        logger.info("开始验证")
        self.model = model.to(self.device)
        val_logs = self._val_epoch(dataloader)
        metrics = {
            "loss": float(val_logs["loss"]),
            "accuracy": float(val_logs["accuracy"]),
        }
        logger.info("验证完成: %s", metrics)
        return metrics

    def predict(self, test_loader: DataLoader[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """
        预测

        参数:
            test_loader: 测试数据加载器

        返回:
            预测结果列表
        """
        self.model.eval()
        predictions: List[Dict[str, torch.Tensor]] = []

        with torch.no_grad():
            for batch in test_loader:
                if not isinstance(batch, dict):
                    raise TypeError("Batch must be a dict")
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                outputs = self.model(batch)
                if not isinstance(outputs, dict):
                    raise TypeError("Predict output must be a dict")
                predictions.append(outputs)

        return predictions
