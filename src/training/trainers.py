"""
训练器模块
功能概述: 提供模型训练、验证和评估的完整流程
设计思路: 封装训练循环，支持回调机制，灵活配置
"""

from typing import Any, Dict, List, Optional, Union, cast

import random

import numpy as np
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
from .amp_compat import make_grad_scaler, amp_autocast

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
        grad_clip_norm: Optional[float] = None,
        use_amp: bool = False,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 0,
    ) -> None:
        """
        初始化训练器

        参数:
            model: PyTorch模型
            config: 配置字典
            device: 设备，如 "cuda" 或 "cpu"
            grad_clip_norm: 梯度裁剪的最大范数，None表示不裁剪
            use_amp: 是否使用自动混合精度训练
            gradient_accumulation_steps: 梯度累积步数，每N步更新一次优化器
            warmup_steps: 学习率预热步数，线性从0增长到目标学习率
        """
        self.model = model
        self.config = config or {}
        self.training_config = self.config.get("training", {})

        # 可复现性: 从配置中读取 seed 并设置所有 RNG，使训练器在被任意入口
        # 调用时也能保证可复现（脚本层如 scripts/train.py 仍可单独设置 seed）。
        seed = self.training_config.get("seed")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            logger.info("已从配置设置随机种子: %s", seed)

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

        # 生产训练特性
        self.grad_clip_norm = grad_clip_norm if grad_clip_norm is not None else self.training_config.get("grad_clip_norm", None)
        self.use_amp = use_amp or self.training_config.get("use_amp", False)
        self.gradient_accumulation_steps = gradient_accumulation_steps if gradient_accumulation_steps > 1 else self.training_config.get("gradient_accumulation_steps", 1)
        self.warmup_steps = warmup_steps if warmup_steps > 0 else self.training_config.get("warmup_steps", 0)

        # AMP scaler (uses the modern torch.amp API when available; see
        # ``_make_grad_scaler`` for the version-compat shim).
        self.scaler: Optional[Any] = None
        if self.use_amp and self.device == "cuda":
            self.scaler = make_grad_scaler()

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

        # 为warmup保存初始学习率
        if self.warmup_steps > 0 and self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group["initial_lr"] = param_group.get("initial_lr", param_group["lr"])

    def add_callback(self, callback: Callback):
        """
        添加回调

        参数:
            callback: 回调实例
        """
        self.callbacks.append(callback)

    def _get_warmup_lr_scale(self) -> float:
        """
        计算warmup阶段的学习率缩放因子

        返回:
            0.0~1.0之间的缩放因子，warmup完成后为1.0
        """
        if self.warmup_steps <= 0 or self.global_step >= self.warmup_steps:
            return 1.0
        return float(self.global_step) / float(self.warmup_steps)

    def _apply_warmup(self) -> None:
        """应用warmup学习率缩放到优化器参数组"""
        if self.warmup_steps <= 0 or self.optimizer is None:
            return
        scale = self._get_warmup_lr_scale()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = param_group.get("initial_lr", param_group["lr"]) * scale

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
        correct_samples: int | float = 0

        for batch_idx, batch in enumerate(train_loader):
            if not isinstance(batch, dict):
                raise TypeError("Batch must be a dict")
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)

            # Forward pass with optional AMP autocast
            use_autocast = self.use_amp and self.device == "cuda"

            if use_autocast:
                with amp_autocast():
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
                    loss = loss / self.gradient_accumulation_steps
            else:
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
                loss = loss / self.gradient_accumulation_steps

            # Backward pass with AMP support
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Only step optimizer every gradient_accumulation_steps
            is_accumulation_boundary = (batch_idx + 1) % self.gradient_accumulation_steps == 0

            if is_accumulation_boundary:
                # Gradient clipping
                if self.grad_clip_norm is not None:
                    if self.use_amp and self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                # Apply warmup learning rate
                self._apply_warmup()

                # Optimizer step
                if self.use_amp and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Report unscaled loss for logging
            unscaled_loss = loss.item() * self.gradient_accumulation_steps
            batch_size = targets.size(0)
            total_loss += unscaled_loss * batch_size
            num_samples += batch_size

            # 计算训练批准确率（若模型未直接给出 predictions，则从 logits 取 argmax）
            if predictions is None:
                predictions = torch.argmax(logits, dim=1)
            if isinstance(predictions, torch.Tensor) and predictions.dim() == 1:
                correct_samples += (predictions == targets).sum().item()

            for callback in self.callbacks:
                callback.on_batch_end(self, batch_idx, {"loss": unscaled_loss})

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

    def _is_datamodule(self, obj: Any) -> bool:
        """
        检测对象是否为LightningDataModule（duck typing）

        参数:
            obj: 待检测对象

        返回:
            True如果对象具有train_dataloader方法
        """
        return hasattr(obj, "train_dataloader") and callable(obj.train_dataloader)

    def fit(
        self,
        train_loader_or_datamodule: Union[DataLoader[Dict[str, torch.Tensor]], Any],
        val_loader: Optional[DataLoader[Dict[str, torch.Tensor]]] = None,
        max_epochs: Optional[int] = None,
    ) -> None:
        """
        训练模型

        支持两种调用方式:
            1. trainer.fit(train_loader, val_loader) — 传统DataLoader方式
            2. trainer.fit(datamodule) — LightningDataModule方式

        参数:
            train_loader_or_datamodule: 训练数据加载器或LightningDataModule实例
            val_loader: 验证数据加载器（仅在DataLoader方式下使用）
            max_epochs: 最大训练轮数
        """
        # 检测是否为LightningDataModule
        if self._is_datamodule(train_loader_or_datamodule):
            datamodule = train_loader_or_datamodule
            train_loader = cast(Any, datamodule).train_dataloader()
            if val_loader is None and hasattr(datamodule, "val_dataloader") and callable(datamodule.val_dataloader):
                val_loader = datamodule.val_dataloader()
        else:
            train_loader = train_loader_or_datamodule

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
                "train_acc": train_logs.get("accuracy", 0.0),
                **{f"val_{k}": v for k, v in val_logs.items()},
            }
            # TensorBoardCallback 期望的键名为 'val_acc' 而非 'val_accuracy'，
            # 因此显式补充该别名；'learning_rate' 仅在 optimizer 已编译时可得。
            if "accuracy" in val_logs:
                logs["val_acc"] = val_logs["accuracy"]
            if self.optimizer is not None:
                logs["learning_rate"] = self.optimizer.param_groups[0]["lr"]

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


def train(
    model: nn.Module,
    datamodule: Any,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    max_epochs: Optional[int] = None,
    loss_fn: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    callbacks: Optional[List[Any]] = None,
) -> nn.Module:
    """
    Train a model using the provided data module and configuration.

    This is a convenience function that wraps the Trainer class with minimal
    boilerplate. It supports both PyTorch DataLoader and LightningDataModule
    inputs, matching the interface contract in AGENTS.md.

    Args:
        model: PyTorch model to train
        datamodule: Training data source — either a DataLoader or
            LightningDataModule with train_dataloader()/val_dataloader() methods
        config: Optional configuration dict (used for optimizer/scheduler setup)
        device: Device string (e.g., 'cuda', 'cpu'). Auto-detected if None.
        max_epochs: Maximum training epochs. Defaults to config value or 100.
        loss_fn: Loss function. Defaults to CrossEntropyLoss.
        optimizer: Pre-configured optimizer. Auto-created from config if None.
        callbacks: List of training callbacks.

    Returns:
        The trained model (nn.Module)

    Example:
        >>> model = PTM2CellNet(config)
        >>> datamodule = PTMDataModule(...)
        >>> trained_model = train(model, datamodule, config)
    """
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
    )
    trainer.compile(
        loss_fn=loss_fn,
        optimizer=optimizer,
        callbacks=callbacks,
    )
    trainer.fit(datamodule, max_epochs=max_epochs)
    return model
