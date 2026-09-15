"""
训练回调模块
功能概述: 提供模型检查点、早停等训练过程回调
设计思路: 使用回调模式，在训练的不同阶段执行特定操作
"""

import os
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

try:
    from lightning.pytorch.callbacks import Callback as _RealLightningCallback

    LightningCallback: type = _RealLightningCallback
except ImportError:
    LightningCallback = object


class Callback(LightningCallback):
    """
    回调基类
    所有回调都应继承此类
    """

    def on_train_start(self, _trainer: Any) -> None:
        """训练开始时调用"""
        return None

    def on_train_end(self, _trainer: Any) -> None:
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


def _capture_rng_state() -> Dict[str, Any]:
    """Capture python/numpy/torch RNG state for exact-resume checkpoints.

    Mirrors the semantics used by ``CrossScaleTrainer`` so both standard and
    cross-scale trainers serialize the same RNG contract.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state previously captured by :func:`_capture_rng_state`."""
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    cpu_state = state.get("torch_cpu")
    if isinstance(cpu_state, torch.Tensor):
        torch.set_rng_state(cpu_state.cpu())
    cuda_states = state.get("torch_cuda")
    if torch.cuda.is_available() and isinstance(cuda_states, list) and cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)


def _callback_state_keys(callbacks: Optional[Sequence[Any]]) -> Dict[int, str]:
    """为有 ``state_dict()`` 的 callback 生成稳定的 checkpoint key。

    同一类名出现多次时追加序号（``ModelCheckpoint``、``ModelCheckpoint_2``），
    保存与恢复两侧必须使用同一映射。
    """
    keys: Dict[int, str] = {}
    seen: Dict[str, int] = {}
    if callbacks is None:
        return keys
    for index, callback in enumerate(callbacks):
        if not callable(getattr(callback, "state_dict", None)):
            continue
        name = type(callback).__name__
        seen[name] = seen.get(name, 0) + 1
        keys[index] = f"{name}_{seen[name]}" if seen[name] > 1 else name
    return keys


def collect_callback_states(callbacks: Optional[Sequence[Any]]) -> Dict[str, Any]:
    """收集 stateful callback 的状态，用于写入 checkpoint（TD-M01）。

    只有实现了 ``state_dict()`` 的 callback 会进入 ``callback_states``；
    无状态的 callback（日志/进度条等）被跳过。
    """
    if callbacks is None:
        return {}
    states: Dict[str, Any] = {}
    for index, key in _callback_state_keys(callbacks).items():
        callback = callbacks[index]
        state_fn = getattr(callback, "state_dict", None)
        if callable(state_fn):
            state = state_fn()
            # 跳过空状态（如 Lightning 基类默认返回 {} 的无状态回调），
            # 避免无意义地占用 checkpoint 空间。
            if state:
                states[key] = state
    return states


class ModelCheckpoint(Callback):
    """
    模型检查点回调
    保存最佳模型检查点，支持完整检查点和top-k保存
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_last: bool = False,
        save_top_k: int = 1,
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
            save_top_k: 保留top-k个最佳检查点（仅save_best_only=False时生效）
            verbose: 日志级别
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.verbose = verbose

        if mode == "min":
            self.best_value = float("inf")
            self.is_better = lambda a, b: a < b
        else:
            self.best_value = -float("inf")
            self.is_better = lambda a, b: a > b

        self.epochs_since_improvement = 0

        # Top-k tracking: sorted list of (metric_value, filepath)
        # For mode="min", worst is the largest value; for mode="max", worst is the smallest
        self._top_k_checkpoints: List[Tuple[float, str]] = []

        # Optimizer and scheduler references (set externally by trainer)
        self.optimizer = None
        self.scheduler = None

    def state_dict(self) -> Dict[str, Any]:
        """返回可序列化的 callback 状态（best 语义与 top-k 记录）。

        TD-M01（2026-08-09 修复）: 使 ``--resume`` 能恢复 early stopping / best
        语义，而不是只恢复权重/优化器/RNG 后重新从空状态计数。
        """
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "best_value": self.best_value,
            "epochs_since_improvement": self.epochs_since_improvement,
            "top_k_checkpoints": [[float(value), path] for value, path in self._top_k_checkpoints],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """恢复 callback 状态；monitor/mode 与构造参数不一致时拒绝恢复。

        旧 checkpoint 无此字段时调用方应跳过，保持缺省状态（与
        optimizer/scheduler 的缺省路径一致）。
        """
        if state.get("monitor") != self.monitor or state.get("mode") != self.mode:
            raise ValueError(
                f"ModelCheckpoint 状态与当前配置不一致: "
                f"checkpoint monitor={state.get('monitor')!r} mode={state.get('mode')!r}, "
                f"当前 monitor={self.monitor!r} mode={self.mode!r}"
            )
        self.best_value = float(state.get("best_value", self.best_value))
        self.epochs_since_improvement = int(state.get("epochs_since_improvement", 0))
        raw_top_k = state.get("top_k_checkpoints", [])
        self._top_k_checkpoints = [(float(value), str(path)) for value, path in raw_top_k]
        # 恢复后按当前 mode 重新排序，防御旧数据未排序或模式漂移。
        self._top_k_checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

    def _ensure_dir(self, filepath: str) -> None:
        """确保文件所在目录存在"""
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    def _build_checkpoint(self, model, epoch: int, trainer: Optional[Any] = None) -> Dict:
        """构建完整检查点字典

        包含模型权重、epoch、optimizer/scheduler（若已挂载）、AMP scaler、
        RNG 状态（python/numpy/torch）与 stateful callback 状态，使
        ``--resume`` 成为精确续训（TD-M01）。
        """
        checkpoint: Dict = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "global_step": int(getattr(trainer, "global_step", 0)) if trainer is not None else 0,
            "rng_state": _capture_rng_state(),
        }
        if self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if trainer is not None:
            scaler = getattr(trainer, "scaler", None)
            if scaler is not None and hasattr(scaler, "state_dict"):
                checkpoint["scaler_state_dict"] = scaler.state_dict()
            callback_states = collect_callback_states(getattr(trainer, "callbacks", None))
            if callback_states:
                checkpoint["callback_states"] = callback_states
        return checkpoint

    def _save_checkpoint(self, filepath: str, model, epoch: int, trainer: Optional[Any] = None) -> None:
        """保存检查点到指定路径"""
        self._ensure_dir(filepath)
        checkpoint = self._build_checkpoint(model, epoch, trainer)
        torch.save(checkpoint, filepath)

    def _get_last_filepath(self) -> str:
        """生成save_last文件路径，使用os.path.splitext正确处理扩展名"""
        base, ext = os.path.splitext(self.filepath)
        return base + "_last" + ext

    def _cleanup_top_k(self) -> None:
        """当检查点数量超过save_top_k时，删除最差的检查点"""
        if self.save_top_k <= 0:
            return  # save_top_k=0 means keep all
        while len(self._top_k_checkpoints) > self.save_top_k:
            # The worst checkpoint is the last one after sorting
            if self.mode == "min":
                # Sorted ascending: worst (largest value) is last
                worst = self._top_k_checkpoints.pop(-1)
            else:
                # Sorted descending: worst (smallest value) is last
                worst = self._top_k_checkpoints.pop(-1)
            _, worst_path = worst
            if os.path.exists(worst_path):
                os.remove(worst_path)
                if self.verbose > 0:
                    logger.info("移除检查点 %s (超出save_top_k=%d)", worst_path, self.save_top_k)

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

        if self.save_best_only:
            if self.is_better(current_value, self.best_value):
                self.best_value = current_value
                self.epochs_since_improvement = 0
                self._save_checkpoint(self.filepath, trainer.model, epoch, trainer)
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
        else:
            # save_best_only=False: save every epoch with top-k management
            # Generate epoch-specific filepath
            base, ext = os.path.splitext(self.filepath)
            epoch_filepath = f"{base}_epoch{epoch:04d}{ext}"
            self._save_checkpoint(epoch_filepath, trainer.model, epoch, trainer)

            # Track in top-k list
            self._top_k_checkpoints.append((current_value, epoch_filepath))
            # Sort: ascending for mode="min" (best first), descending for mode="max"
            self._top_k_checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
            self._cleanup_top_k()

            if self.verbose > 0:
                logger.info(
                    "保存检查点到 %s (epoch %d, %s=%.4f)",
                    epoch_filepath,
                    epoch,
                    self.monitor,
                    current_value,
                )

        # Save last checkpoint after updating best/top-k state so resume from
        # the latest checkpoint sees the same callback state as this epoch.
        if self.save_last:
            last_filepath = self._get_last_filepath()
            self._save_checkpoint(last_filepath, trainer.model, epoch, trainer)
            if self.verbose > 0:
                logger.info("保存最新模型到 %s", last_filepath)


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
        # TD-M01: 由 load_state_dict 置位，on_train_start 消费后保留恢复的
        # wait/best 状态（精确续训）；未恢复时保持“训练开始时重置”语义。
        self._state_restored = False

    def state_dict(self) -> Dict[str, Any]:
        """返回可序列化的早停状态（TD-M01）。"""
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_value": self.best_value,
            "wait": self.wait,
            "stopped_epoch": self.stopped_epoch,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """恢复早停状态；monitor/mode/patience/min_delta 不一致时拒绝恢复。"""
        if (
            state.get("monitor") != self.monitor
            or state.get("mode") != self.mode
            or state.get("patience") != self.patience
            or state.get("min_delta") != self.min_delta
        ):
            raise ValueError(
                "EarlyStopping 状态与当前配置不一致: "
                f"checkpoint monitor={state.get('monitor')!r} mode={state.get('mode')!r} "
                f"patience={state.get('patience')!r} min_delta={state.get('min_delta')!r}, "
                f"当前 monitor={self.monitor!r} mode={self.mode!r} "
                f"patience={self.patience!r} min_delta={self.min_delta!r}"
            )
        self.best_value = float(state.get("best_value", self.best_value))
        self.wait = int(state.get("wait", 0))
        self.stopped_epoch = int(state.get("stopped_epoch", 0))
        self.should_stop = bool(state.get("should_stop", False))
        self._state_restored = True

    def on_train_start(self, _trainer: Any) -> None:
        """训练开始时重置状态；已通过 load_state_dict 恢复时保留恢复值。"""
        if self._state_restored:
            self._state_restored = False
            return
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

    def on_train_start(self, _trainer: Any) -> None:
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
    from torch.utils.tensorboard import SummaryWriter

    _HAS_TENSORBOARD = True
except ImportError:
    _HAS_TENSORBOARD = False


class TensorBoardCallback(Callback):
    """
    TensorBoard回调
    将训练指标写入TensorBoard用于可视化，不依赖Lightning的logger
    """

    def __init__(self, log_dir: str = "runs", flush_secs: int = 30):
        """
        初始化TensorBoard回调

        参数:
            log_dir: TensorBoard日志目录
            flush_secs: 刷新间隔（秒）
        """
        super().__init__()
        self.log_dir = log_dir
        self.flush_secs = flush_secs
        self._writer: Optional[Any] = None

        if not _HAS_TENSORBOARD:
            logger.warning(
                "torch.utils.tensorboard 不可用，TensorBoardCallback将被跳过。"
                "请安装 tensorboard: pip install tensorboard"
            )

    def on_train_start(self, trainer: Any) -> None:
        """训练开始时初始化SummaryWriter"""
        if not _HAS_TENSORBOARD:
            return
        self._writer = SummaryWriter(log_dir=self.log_dir, flush_secs=self.flush_secs)

    def on_epoch_end(self, trainer, epoch: int, logs: Dict) -> None:
        """
        epoch结束时将指标写入TensorBoard

        参数:
            trainer: 训练器实例
            epoch: 当前epoch
            logs: 日志字典
        """
        if self._writer is None:
            return

        for key in ("train_loss", "val_loss", "train_acc", "val_acc", "learning_rate"):
            if key in logs:
                self._writer.add_scalar(key, logs[key], epoch)

    def on_train_end(self, trainer: Any) -> None:
        """训练结束时关闭SummaryWriter"""
        if self._writer is not None:
            self._writer.close()
            self._writer = None


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

    def on_train_start(self, _trainer: Any) -> None:
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

    def on_train_end(self, _trainer: Any) -> None:
        """训练结束时关闭epoch进度条"""
        if self.verbose <= 0:
            return
        if _HAS_TQDM and self.epoch_pbar is not None:
            self.epoch_pbar.close()
            self.epoch_pbar = None
        elif not _HAS_TQDM:
            logger.info("Training ended")
