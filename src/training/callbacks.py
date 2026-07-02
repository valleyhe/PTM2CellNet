"""
训练回调模块
功能概述: 提供模型检查点、早停等训练过程回调
设计思路: 使用回调模式，在训练的不同阶段执行特定操作
"""

import os
from typing import Dict, List

import torch

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

try:
    from lightning.pytorch.callbacks import Callback as LightningCallback
except ImportError:
    LightningCallback = object


class Callback(LightningCallback):
    """
    回调基类
    所有回调都应继承此类
    """

    def on_train_start(self, _trainer) -> None:
        """训练开始时调用"""
        return None

    def on_train_end(self, _trainer) -> None:
        """训练结束时调用"""
        return None

    def on_epoch_start(self, _trainer, _epoch: int) -> None:
        """每个epoch开始时调用"""
        return None

    def on_epoch_end(self, _trainer, _epoch: int, _logs: Dict) -> None:
        """每个epoch结束时调用"""
        return None

    def on_batch_start(self, _trainer, _batch_idx: int) -> None:
        """每个batch开始时调用"""
        return None

    def on_batch_end(self, _trainer, _batch_idx: int, _logs: Dict) -> None:
        """每个batch结束时调用"""
        return None


class ModelCheckpoint(Callback):
    """
    模型检查点回调
    保存最佳模型检查点
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_last: bool = False,
        verbose: int = 1,
    ):
        """
        初始化模型检查点

        参数:
            filepath: 保存路径
            monitor: 监控的指标
            mode: "min" 或 "max"
            save_best_only: 是否只保存最佳模型
            save_last: 是否保存最新模型
            verbose: 日志级别
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.verbose = verbose

        if mode == "min":
            self.best_value = float("inf")
            self.is_better = lambda a, b: a < b
        else:
            self.best_value = -float("inf")
            self.is_better = lambda a, b: a > b

        self.epochs_since_improvement = 0

    def on_epoch_end(self, trainer, epoch: int, logs: Dict) -> None:
        """
        epoch结束时调用

        参数:
            trainer: 训练器实例
            epoch: 当前epoch
            logs: 日志字典
        """
        if self.monitor not in logs:
            logger.warning("监控指标 %s 不存在于日志中", self.monitor)
            return

        current_value = logs[self.monitor]

        dir_path = os.path.dirname(self.filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        if self.save_last:
            last_filepath = self.filepath.replace(".pt", "_last.pt")
            torch.save(trainer.model.state_dict(), last_filepath)
            if self.verbose > 0:
                logger.info("保存最新模型到 %s", last_filepath)

        if self.save_best_only:
            if self.is_better(current_value, self.best_value):
                self.best_value = current_value
                self.epochs_since_improvement = 0
                torch.save(trainer.model.state_dict(), self.filepath)
                if self.verbose > 0:
                    logger.info(
                        "保存最佳模型到 %s (epoch %d, %s=%.4f)",
                        self.filepath,
                        epoch,
                        self.monitor,
                        current_value,
                    )
            else:
                self.epochs_since_improvement += 1


class EarlyStopping(Callback):
    """
    早停回调
    当监控指标不再改善时提前停止训练
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 10,
        min_delta: float = 0.0,
        verbose: int = 1,
    ):
        """
        初始化早停

        参数:
            monitor: 监控的指标
            mode: "min" 或 "max"
            patience: 耐心值（连续多少个epoch不改善就停止）
            min_delta: 最小改善幅度
            verbose: 日志级别
        """
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        if mode == "min":
            self.best_value = float("inf")
            self.is_improved = lambda a, b: b < a - min_delta
        else:
            self.best_value = -float("inf")
            self.is_improved = lambda a, b: b > a + min_delta

        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False

    def on_train_start(self, _trainer) -> None:
        """训练开始时重置状态"""
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False

    def on_epoch_end(self, _trainer, epoch: int, logs: Dict) -> None:
        """
        epoch结束时调用

        参数:
            trainer: 训练器实例
            epoch: 当前epoch
            logs: 日志字典
        """
        if self.monitor not in logs:
            logger.warning("监控指标 %s 不存在于日志中", self.monitor)
            return

        current_value = logs[self.monitor]

        if self.is_improved(self.best_value, current_value):
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.verbose > 0:
                logger.info("早停等待中: %d/%d", self.wait, self.patience)
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.should_stop = True
                if self.verbose > 0:
                    logger.info("早停触发于 epoch %d", epoch)


class LearningRateMonitor(Callback):
    """
    学习率监控回调
    记录并日志输出训练过程中的学习率变化
    """

    def __init__(self, logging_interval: str = "epoch"):
        """
        初始化学习率监控

        参数:
            logging_interval: 记录间隔，"epoch" 或 "batch"
        """
        super().__init__()
        if logging_interval not in ("epoch", "batch"):
            raise ValueError("logging_interval 必须是 'epoch' 或 'batch'")
        self.logging_interval = logging_interval
        self._lrs: Dict[str, List[float]] = {"epoch": [], "batch": []}

    @property
    def lrs(self) -> Dict[str, List[float]]:
        """返回记录的学习率"""
        return self._lrs

    def on_train_start(self, _trainer) -> None:
        """训练开始时重置学习率记录"""
        self._lrs = {"epoch": [], "batch": []}

    def on_epoch_end(self, trainer, epoch: int, logs: Dict) -> None:
        """
        epoch结束时记录学习率

        参数:
            trainer: 训练器实例
            epoch: 当前epoch
            logs: 日志字典
        """
        current_lr = trainer.optimizer.param_groups[0]["lr"]
        self._lrs["epoch"].append(current_lr)
        logger.info("Epoch %d: lr=%.6f", epoch, current_lr)

    def on_batch_end(self, trainer, batch_idx: int, logs: Dict) -> None:
        """
        batch结束时记录学习率（仅当logging_interval为batch时）

        参数:
            trainer: 训练器实例
            batch_idx: 当前batch索引
            logs: 日志字典
        """
        if self.logging_interval != "batch":
            return
        current_lr = trainer.optimizer.param_groups[0]["lr"]
        self._lrs["batch"].append(current_lr)
        logger.info("Batch %d: lr=%.6f", batch_idx, current_lr)


try:
    from tqdm import tqdm as _tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


class ProgressBarCallback(Callback):
    """
    进度条回调
    使用tqdm显示训练进度，tqdm不可用时回退到logger输出
    """

    def __init__(self, verbose: int = 1):
        """
        初始化进度条回调

        参数:
            verbose: 日志级别，0为静默
        """
        super().__init__()
        self.verbose = verbose
        self.epoch_pbar = None
        self.batch_pbar = None

    def on_train_start(self, _trainer) -> None:
        """训练开始时初始化epoch进度条"""
        if self.verbose <= 0:
            return
        if _HAS_TQDM:
            self.epoch_pbar = _tqdm(desc="Training", unit="epoch")
        else:
            logger.info("Training started")

    def on_epoch_start(self, _trainer, epoch: int) -> None:
        """
        每个epoch开始时初始化batch进度条

        参数:
            trainer: 训练器实例
            epoch: 当前epoch
        """
        if self.verbose <= 0:
            return
        if _HAS_TQDM:
            self.batch_pbar = _tqdm(desc=f"Epoch {epoch}", unit="batch")
        else:
            logger.info("Epoch %d started", epoch)

    def on_batch_end(self, _trainer, _batch_idx: int, logs: Dict) -> None:
        """
        每个batch结束时更新进度条

        参数:
            trainer: 训练器实例
            batch_idx: 当前batch索引
            logs: 日志字典
        """
        if self.verbose <= 0:
            return
        if _HAS_TQDM and self.batch_pbar is not None:
            self.batch_pbar.update(1)
            self.batch_pbar.set_postfix(loss=logs.get("loss", 0))

    def on_epoch_end(self, _trainer, epoch: int, logs: Dict) -> None:
        """
        epoch结束时关闭batch进度条并更新epoch进度条

        参数:
            trainer: 训练器实例
            epoch: 当前epoch
            logs: 日志字典
        """
        if self.verbose <= 0:
            return
        if _HAS_TQDM:
            if self.batch_pbar is not None:
                self.batch_pbar.close()
                self.batch_pbar = None
            if self.epoch_pbar is not None:
                self.epoch_pbar.update(1)
                postfix = {}
                if "train_loss" in logs:
                    postfix["train_loss"] = logs["train_loss"]
                if "val_loss" in logs:
                    postfix["val_loss"] = logs["val_loss"]
                if postfix:
                    self.epoch_pbar.set_postfix(**postfix)
        else:
            msg = f"Epoch {epoch} ended"
            parts = []
            if "train_loss" in logs:
                parts.append(f"train_loss={logs['train_loss']:.4f}")
            if "val_loss" in logs:
                parts.append(f"val_loss={logs['val_loss']:.4f}")
            if parts:
                msg += " (" + ", ".join(parts) + ")"
            logger.info(msg)

    def on_train_end(self, _trainer) -> None:
        """训练结束时关闭epoch进度条"""
        if self.verbose <= 0:
            return
        if _HAS_TQDM and self.epoch_pbar is not None:
            self.epoch_pbar.close()
            self.epoch_pbar = None
        elif not _HAS_TQDM:
            logger.info("Training ended")
