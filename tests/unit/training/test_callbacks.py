"""
训练回调模块单元测试
使用mock隔离文件系统和模型依赖
"""

import random

import pytest
import torch
from unittest.mock import MagicMock

from src.training.callbacks import (
    Callback,
    ModelCheckpoint,
    EarlyStopping,
    TensorBoardCallback,
    LearningRateMonitor,
    _capture_rng_state,
    _restore_rng_state,
)

try:
    from lightning.pytorch.callbacks import Callback as LightningCallback
except ImportError:  # pragma: no cover - depends on optional dependency
    LightningCallback = None


class MockTrainer:
    """模拟训练器"""

    def __init__(self):
        self.model = MagicMock()
        self.model.state_dict.return_value = {"weight": torch.tensor([1.0, 2.0])}


class MockModel:
    """模拟模型"""

    def __init__(self):
        self.state_dict_val = {"param": torch.tensor([1.0])}

    def state_dict(self):
        return self.state_dict_val


class TestCallbackBase:
    """回调基类测试"""

    def test_optional_lightning_inheritance(self):
        """有lightning时应继承其Callback"""
        if LightningCallback is None:
            assert Callback.__mro__[1] is object
        else:
            assert issubclass(Callback, LightningCallback)

    def test_on_train_start(self):
        """训练开始钩子"""
        callback = Callback()
        trainer = MockTrainer()
        result = callback.on_train_start(trainer)
        assert result is None

    def test_on_train_end(self):
        """训练结束钩子"""
        callback = Callback()
        trainer = MockTrainer()
        result = callback.on_train_end(trainer)
        assert result is None

    def test_on_epoch_start(self):
        """epoch开始钩子"""
        callback = Callback()
        trainer = MockTrainer()
        result = callback.on_epoch_start(trainer, 0)
        assert result is None

    def test_on_epoch_end(self):
        """epoch结束钩子"""
        callback = Callback()
        trainer = MockTrainer()
        result = callback.on_epoch_end(trainer, 0, {})
        assert result is None

    def test_on_batch_start(self):
        """batch开始钩子"""
        callback = Callback()
        trainer = MockTrainer()
        result = callback.on_batch_start(trainer, 0)
        assert result is None

    def test_on_batch_end(self):
        """batch结束钩子"""
        callback = Callback()
        trainer = MockTrainer()
        result = callback.on_batch_end(trainer, 0, {})
        assert result is None


class TestModelCheckpoint:
    """模型检查点测试"""

    def test_init_min_mode(self):
        """min模式初始化，best_value=inf"""
        checkpoint = ModelCheckpoint("model.pt", monitor="val_loss", mode="min")
        assert checkpoint.best_value == float("inf")
        assert checkpoint.mode == "min"
        assert checkpoint.is_better(0.5, 1.0) is True
        assert checkpoint.is_better(1.0, 0.5) is False

    def test_init_max_mode(self):
        """max模式初始化，best_value=-inf"""
        checkpoint = ModelCheckpoint("model.pt", monitor="val_acc", mode="max")
        assert checkpoint.best_value == -float("inf")
        assert checkpoint.mode == "max"
        assert checkpoint.is_better(0.8, 0.5) is True
        assert checkpoint.is_better(0.5, 0.8) is False

    def test_save_best_only_improvement(self, tmp_path):
        """改善时保存"""
        filepath = str(tmp_path / "best.pt")
        checkpoint = ModelCheckpoint(filepath, monitor="val_loss", mode="min", verbose=0)
        trainer = MockTrainer()

        checkpoint.on_epoch_end(trainer, epoch=0, logs={"val_loss": 0.5})
        assert (tmp_path / "best.pt").exists()
        assert checkpoint.best_value == 0.5
        assert checkpoint.epochs_since_improvement == 0

    def test_save_best_only_no_improvement(self, tmp_path):
        """不改善时不保存（覆盖）"""
        filepath = str(tmp_path / "best.pt")
        checkpoint = ModelCheckpoint(filepath, monitor="val_loss", mode="min", verbose=0)
        trainer = MockTrainer()

        checkpoint.on_epoch_end(trainer, epoch=0, logs={"val_loss": 0.5})
        checkpoint.on_epoch_end(trainer, epoch=1, logs={"val_loss": 0.6})
        assert checkpoint.best_value == 0.5
        assert checkpoint.epochs_since_improvement == 1

    def test_save_last(self, tmp_path):
        """配置save_last=True时保存_last.pt"""
        filepath = str(tmp_path / "model.pt")
        checkpoint = ModelCheckpoint(filepath, monitor="val_loss", save_last=True, verbose=0)
        trainer = MockTrainer()

        checkpoint.on_epoch_end(trainer, 0, {"val_loss": 0.5})
        assert (tmp_path / "model_last.pt").exists()

    def test_different_monitor(self, tmp_path):
        """监控不同指标(val_acc)"""
        filepath = str(tmp_path / "best.pt")
        checkpoint = ModelCheckpoint(filepath, monitor="val_acc", mode="max", verbose=0)
        trainer = MockTrainer()

        checkpoint.on_epoch_end(trainer, epoch=0, logs={"val_acc": 0.8})
        assert checkpoint.best_value == 0.8

    def test_epochs_since_improvement(self, tmp_path):
        """记录未改善epoch数"""
        filepath = str(tmp_path / "best.pt")
        checkpoint = ModelCheckpoint(filepath, monitor="val_loss", mode="min", verbose=0)
        trainer = MockTrainer()

        checkpoint.on_epoch_end(trainer, epoch=0, logs={"val_loss": 0.5})
        checkpoint.on_epoch_end(trainer, epoch=1, logs={"val_loss": 0.6})
        checkpoint.on_epoch_end(trainer, epoch=2, logs={"val_loss": 0.7})
        assert checkpoint.epochs_since_improvement == 2

    def test_file_creation(self, tmp_path):
        """使用tmp_path验证文件创建"""
        filepath = str(tmp_path / "checkpoints" / "best.pt")
        checkpoint = ModelCheckpoint(filepath, monitor="val_loss", verbose=0)
        trainer = MockTrainer()

        checkpoint.on_epoch_end(trainer, epoch=0, logs={"val_loss": 0.3})
        assert (tmp_path / "checkpoints" / "best.pt").exists()
        assert checkpoint.best_value == 0.3

    def test_monitor_missing_from_logs(self, tmp_path, caplog):
        """监控指标缺失时不保存"""
        import logging

        filepath = str(tmp_path / "best.pt")
        checkpoint = ModelCheckpoint(filepath, monitor="val_loss", verbose=1)
        trainer = MockTrainer()

        with caplog.at_level(logging.WARNING, logger="src.training.callbacks"):
            checkpoint.on_epoch_end(trainer, epoch=0, logs={"train_loss": 0.5})
        assert "不存在于日志中" in caplog.text

    def test_verbose_save_best(self, tmp_path, caplog):
        """verbose>0时保存最佳模型记录日志"""
        import logging

        filepath = str(tmp_path / "best.pt")
        checkpoint = ModelCheckpoint(filepath, monitor="val_loss", mode="min", verbose=1)
        trainer = MockTrainer()

        with caplog.at_level(logging.INFO, logger="src.training.callbacks"):
            checkpoint.on_epoch_end(trainer, epoch=0, logs={"val_loss": 0.5})
        assert "保存最佳模型" in caplog.text

    def test_verbose_save_last(self, tmp_path, caplog):
        """verbose>0时保存最新模型记录日志"""
        import logging

        filepath = str(tmp_path / "model.pt")
        checkpoint = ModelCheckpoint(filepath, monitor="val_loss", save_last=True, verbose=1)
        trainer = MockTrainer()

        with caplog.at_level(logging.INFO, logger="src.training.callbacks"):
            checkpoint.on_epoch_end(trainer, epoch=0, logs={"val_loss": 0.5})
        assert "保存最新模型" in caplog.text


class TestEarlyStopping:
    """早停测试"""

    def test_init_min_mode(self):
        """min模式初始化"""
        early_stop = EarlyStopping(monitor="val_loss", mode="min")
        assert early_stop.best_value == float("inf")
        assert early_stop.mode == "min"

    def test_init_max_mode(self):
        """max模式初始化"""
        early_stop = EarlyStopping(monitor="val_acc", mode="max")
        assert early_stop.best_value == -float("inf")
        assert early_stop.mode == "max"

    def test_no_stop_when_improving(self):
        """改善时不停止"""
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=2, verbose=0)
        trainer = MockTrainer()

        early_stop.on_epoch_end(trainer, epoch=0, logs={"val_loss": 0.5})
        early_stop.on_epoch_end(trainer, epoch=1, logs={"val_loss": 0.4})
        assert early_stop.should_stop is False
        assert early_stop.wait == 0

    def test_stop_after_patience(self):
        """patience次不改善后停止"""
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=2, verbose=0)
        trainer = MockTrainer()

        early_stop.on_epoch_end(trainer, epoch=0, logs={"val_loss": 0.5})
        early_stop.on_epoch_end(trainer, epoch=1, logs={"val_loss": 0.6})
        early_stop.on_epoch_end(trainer, epoch=2, logs={"val_loss": 0.7})
        assert early_stop.should_stop is True
        assert early_stop.stopped_epoch == 2

    def test_min_delta(self):
        """改善小于min_delta不算改善"""
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=2, min_delta=0.1, verbose=0)
        trainer = MockTrainer()

        early_stop.on_epoch_end(trainer, epoch=0, logs={"val_loss": 0.5})
        early_stop.on_epoch_end(trainer, epoch=1, logs={"val_loss": 0.49})
        assert early_stop.wait == 1  # 改善0.01 < min_delta 0.1，不算改善

    def test_should_stop_flag(self):
        """should_stop标志设置"""
        early_stop = EarlyStopping(monitor="val_loss", patience=1, verbose=0)
        trainer = MockTrainer()

        assert early_stop.should_stop is False
        early_stop.on_epoch_end(trainer, epoch=0, logs={"val_loss": 0.5})
        early_stop.on_epoch_end(trainer, epoch=1, logs={"val_loss": 0.6})
        assert early_stop.should_stop is True

    def test_stopped_epoch_recorded(self):
        """记录停止epoch"""
        early_stop = EarlyStopping(monitor="val_loss", patience=1, verbose=0)
        trainer = MockTrainer()

        early_stop.on_epoch_end(trainer, epoch=5, logs={"val_loss": 0.5})
        early_stop.on_epoch_end(trainer, epoch=6, logs={"val_loss": 0.6})
        assert early_stop.stopped_epoch == 6

    def test_on_train_start_resets(self):
        """训练开始时重置状态"""
        early_stop = EarlyStopping(monitor="val_loss", patience=2, verbose=0)
        trainer = MockTrainer()

        early_stop.on_epoch_end(trainer, epoch=0, logs={"val_loss": 0.5})
        early_stop.on_epoch_end(trainer, epoch=1, logs={"val_loss": 0.6})
        early_stop.on_epoch_end(trainer, epoch=2, logs={"val_loss": 0.7})
        assert early_stop.should_stop is True

        early_stop.on_train_start(trainer)
        assert early_stop.wait == 0
        assert early_stop.stopped_epoch == 0
        assert early_stop.should_stop is False

    def test_monitor_missing_from_logs(self, caplog):
        """监控指标缺失时忽略"""
        import logging

        early_stop = EarlyStopping(monitor="val_loss", verbose=1)
        trainer = MockTrainer()

        with caplog.at_level(logging.WARNING, logger="src.training.callbacks"):
            early_stop.on_epoch_end(trainer, epoch=0, logs={"train_loss": 0.5})
        assert "不存在于日志中" in caplog.text

    def test_verbose_early_stop_wait(self, caplog):
        """verbose>0时等待中记录日志"""
        import logging

        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=2, verbose=1)
        trainer = MockTrainer()

        early_stop.on_epoch_end(trainer, epoch=0, logs={"val_loss": 0.5})
        with caplog.at_level(logging.INFO, logger="src.training.callbacks"):
            early_stop.on_epoch_end(trainer, epoch=1, logs={"val_loss": 0.6})
        assert "早停等待中" in caplog.text

    def test_verbose_early_stop_trigger(self, caplog):
        """verbose>0时触发早停记录日志"""
        import logging

        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=1, verbose=1)
        trainer = MockTrainer()

        early_stop.on_epoch_end(trainer, epoch=0, logs={"val_loss": 0.5})
        with caplog.at_level(logging.INFO, logger="src.training.callbacks"):
            early_stop.on_epoch_end(trainer, epoch=1, logs={"val_loss": 0.6})
        assert "早停触发于" in caplog.text


class TestTensorBoardCallback:
    """TensorBoard回调测试"""

    def test_writes_accuracy_metrics(self, tmp_path):
        """train_acc 和 val_acc 被写入 SummaryWriter"""
        callback = TensorBoardCallback(log_dir=str(tmp_path / "runs"))
        # 绕过真实 SummaryWriter 初始化，直接注入 mock writer
        callback._writer = MagicMock()

        logs = {
            "train_loss": 0.5,
            "val_loss": 0.6,
            "train_acc": 0.8,
            "val_acc": 0.75,
            "learning_rate": 1e-3,
        }
        callback.on_epoch_end(MockTrainer(), epoch=0, logs=logs)

        written_keys = {call.args[0] for call in callback._writer.add_scalar.call_args_list}
        assert "train_acc" in written_keys
        assert "val_acc" in written_keys
        assert "train_loss" in written_keys
        assert "val_loss" in written_keys
        assert "learning_rate" in written_keys

    def test_missing_val_acc_is_safe(self, tmp_path):
        """无 val_loader 时 val_acc 缺失，不写入且不报错"""
        callback = TensorBoardCallback(log_dir=str(tmp_path / "runs"))
        callback._writer = MagicMock()

        logs = {"train_loss": 0.5, "train_acc": 0.8}
        callback.on_epoch_end(MockTrainer(), epoch=0, logs=logs)

        written_keys = {call.args[0] for call in callback._writer.add_scalar.call_args_list}
        assert "train_acc" in written_keys
        assert "val_acc" not in written_keys  # 守卫: 缺失键被跳过

    def test_no_writer_is_noop(self, tmp_path):
        """writer 为 None 时安全返回"""
        callback = TensorBoardCallback(log_dir=str(tmp_path / "runs"))
        callback._writer = None
        # 不应抛出异常
        callback.on_epoch_end(MockTrainer(), epoch=0, logs={"train_acc": 0.9})


class TestCallbackIntegration:
    """回调集成测试"""

    def test_checkpoint_with_early_stopping(self, tmp_path):
        """两者协同工作"""
        filepath = str(tmp_path / "best.pt")
        checkpoint = ModelCheckpoint(filepath, monitor="val_loss", mode="min", verbose=0)
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=1, verbose=0)
        trainer = MockTrainer()

        checkpoint.on_epoch_end(trainer, epoch=0, logs={"val_loss": 0.5})
        early_stop.on_epoch_end(trainer, epoch=0, logs={"val_loss": 0.5})

        checkpoint.on_epoch_end(trainer, epoch=1, logs={"val_loss": 0.6})
        early_stop.on_epoch_end(trainer, epoch=1, logs={"val_loss": 0.6})

        assert checkpoint.epochs_since_improvement == 1
        assert early_stop.should_stop is True

    def test_training_loop_simulation(self, tmp_path):
        """模拟完整训练循环"""
        filepath = str(tmp_path / "best.pt")
        checkpoint = ModelCheckpoint(filepath, monitor="val_loss", mode="min", save_last=True, verbose=0)
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3, verbose=0)
        trainer = MockTrainer()

        early_stop.on_train_start(trainer)

        logs = [
            {"val_loss": 0.5},
            {"val_loss": 0.4},
            {"val_loss": 0.45},
            {"val_loss": 0.46},
            {"val_loss": 0.47},
        ]

        for epoch, log in enumerate(logs):
            checkpoint.on_epoch_end(trainer, epoch, log)
            early_stop.on_epoch_end(trainer, epoch, log)
            if early_stop.should_stop:
                break

        assert checkpoint.best_value == 0.4
        assert early_stop.should_stop is True
        assert (tmp_path / "best.pt").exists()
        assert (tmp_path / "best_last.pt").exists()


class TestRngState:
    """RNG 状态捕获/恢复测试（P1-01 精确续训）"""

    def test_capture_contains_all_rngs(self):
        state = _capture_rng_state()
        assert "python" in state
        assert "numpy" in state
        assert "torch_cpu" in state
        assert "torch_cuda" in state
        assert isinstance(state["torch_cpu"], torch.Tensor)

    def test_roundtrip_preserves_draws(self):
        import random

        import numpy as np

        # 推进 RNG 后再捕获，验证恢复后序列一致
        random.random()
        np.random.rand()
        torch.rand(3)
        state = _capture_rng_state()

        first_py = random.random()
        first_np = np.random.rand()
        first_t = torch.rand(2)

        _restore_rng_state(state)
        assert random.random() == first_py
        assert np.random.rand() == first_np
        assert torch.equal(torch.rand(2), first_t)

    def test_restore_ignores_missing_keys(self):
        # 旧 checkpoint 可能只有部分键，恢复不应报错
        _restore_rng_state({"python": random.getstate()})
        _restore_rng_state({})
        _restore_rng_state({"torch_cpu": torch.get_rng_state()})


class TestModelCheckpointFullState:
    """ModelCheckpoint 完整训练状态测试（P1-01）"""

    def test_build_checkpoint_includes_rng(self):
        ckpt = ModelCheckpoint("x.pt")
        payload = ckpt._build_checkpoint(MockModel(), epoch=3)
        assert "rng_state" in payload
        assert payload["epoch"] == 3
        assert "param" in payload["model_state_dict"]

    def test_build_checkpoint_includes_optimizer_and_scheduler(self):
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        ckpt = ModelCheckpoint("x.pt")
        ckpt.optimizer = optimizer
        ckpt.scheduler = scheduler

        payload = ckpt._build_checkpoint(model, epoch=1)
        assert "optimizer_state_dict" in payload
        assert "scheduler_state_dict" in payload
        assert payload["optimizer_state_dict"]["param_groups"][0]["lr"] == 0.1

    def test_build_checkpoint_includes_scaler_from_trainer(self):
        class FakeScaler:
            def state_dict(self):
                return {"scale": 1.0}

        class FakeTrainer:
            scaler = FakeScaler()

        ckpt = ModelCheckpoint("x.pt")
        payload = ckpt._build_checkpoint(MockModel(), epoch=0, trainer=FakeTrainer())
        assert payload["scaler_state_dict"] == {"scale": 1.0}

    def test_build_checkpoint_scaler_none_is_safe(self):
        class FakeTrainer:
            scaler = None

        ckpt = ModelCheckpoint("x.pt")
        payload = ckpt._build_checkpoint(MockModel(), epoch=0, trainer=FakeTrainer())
        assert "scaler_state_dict" not in payload

    def test_saved_checkpoint_roundtrip_via_epoch_end(self, tmp_path):
        import torch.nn as nn

        filepath = str(tmp_path / "best.pt")
        ckpt = ModelCheckpoint(filepath, monitor="val_loss", mode="min", verbose=0)
        model = nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        ckpt.optimizer = optimizer

        class RealTrainer:
            def __init__(self):
                self.model = model
                self.scaler = None

        ckpt.on_epoch_end(RealTrainer(), epoch=2, logs={"val_loss": 0.3})

        loaded = torch.load(filepath, weights_only=False)
        assert loaded["epoch"] == 2
        assert "optimizer_state_dict" in loaded
        assert "rng_state" in loaded
        # 恢复 optimizer 后 param_groups 结构一致
        restored_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        restored_opt.load_state_dict(loaded["optimizer_state_dict"])
        assert restored_opt.param_groups[0]["lr"] == 1e-3


class TestModelCheckpointTopK:
    """top-k 检查点管理测试（P1-03 覆盖率补充）"""

    def test_top_k_keeps_best_and_removes_worst(self, tmp_path):
        filepath = str(tmp_path / "model.pt")
        ckpt = ModelCheckpoint(
            filepath,
            monitor="val_loss",
            mode="min",
            save_best_only=False,
            save_top_k=2,
            verbose=0,
        )
        trainer = MockTrainer()
        for epoch, loss in enumerate([0.5, 0.3, 0.7, 0.4]):
            ckpt.on_epoch_end(trainer, epoch=epoch, logs={"val_loss": loss})
        files = sorted(p.name for p in tmp_path.iterdir())
        # 保留最优 2 个：loss 0.3 (epoch1) 与 0.4 (epoch3)
        assert files == ["model_epoch0001.pt", "model_epoch0003.pt"], files

    def test_top_k_max_mode_keeps_highest(self, tmp_path):
        filepath = str(tmp_path / "model.pt")
        ckpt = ModelCheckpoint(
            filepath,
            monitor="val_acc",
            mode="max",
            save_best_only=False,
            save_top_k=1,
            verbose=0,
        )
        trainer = MockTrainer()
        for epoch, acc in enumerate([0.5, 0.9, 0.7]):
            ckpt.on_epoch_end(trainer, epoch=epoch, logs={"val_acc": acc})
        files = sorted(p.name for p in tmp_path.iterdir())
        assert files == ["model_epoch0001.pt"], files

    def test_top_k_zero_keeps_all(self, tmp_path):
        filepath = str(tmp_path / "model.pt")
        ckpt = ModelCheckpoint(
            filepath,
            monitor="val_loss",
            mode="min",
            save_best_only=False,
            save_top_k=0,
            verbose=0,
        )
        trainer = MockTrainer()
        for epoch in range(3):
            ckpt.on_epoch_end(trainer, epoch=epoch, logs={"val_loss": 0.5 - epoch * 0.1})
        assert len(list(tmp_path.iterdir())) == 3

    def test_top_k_cleanup_verbose_logs(self, tmp_path, caplog):
        import logging

        filepath = str(tmp_path / "model.pt")
        ckpt = ModelCheckpoint(
            filepath,
            monitor="val_loss",
            mode="min",
            save_best_only=False,
            save_top_k=1,
            verbose=1,
        )
        trainer = MockTrainer()
        with caplog.at_level(logging.INFO, logger="src.training.callbacks"):
            ckpt.on_epoch_end(trainer, 0, {"val_loss": 0.5})
            ckpt.on_epoch_end(trainer, 1, {"val_loss": 0.4})
        assert "移除检查点" in caplog.text


class TestLearningRateMonitor:
    """学习率监控回调测试（P1-03 覆盖率补充）"""

    def test_invalid_interval_rejected(self):
        import pytest

        with pytest.raises(ValueError, match="logging_interval"):
            LearningRateMonitor(logging_interval="step")

    def test_epoch_and_batch_recording(self):
        from src.training.callbacks import LearningRateMonitor

        class Trainer:
            optimizer = type("Opt", (), {"param_groups": [{"lr": 0.001}]})()

        lrm = LearningRateMonitor(logging_interval="batch")
        trainer = Trainer()
        lrm.on_train_start(trainer)
        lrm.on_epoch_end(trainer, epoch=0, logs={})
        lrm.on_batch_end(trainer, batch_idx=0, logs={})
        lrm.on_batch_end(trainer, batch_idx=1, logs={})
        assert lrm.lrs["epoch"] == [0.001]
        assert lrm.lrs["batch"] == [0.001, 0.001]

    def test_batch_recording_skipped_for_epoch_interval(self):
        from src.training.callbacks import LearningRateMonitor

        class Trainer:
            optimizer = type("Opt", (), {"param_groups": [{"lr": 0.001}]})()

        lrm = LearningRateMonitor(logging_interval="epoch")
        lrm.on_train_start(Trainer())
        lrm.on_batch_end(Trainer(), 0, {})
        assert lrm.lrs["batch"] == []


class TestTensorBoardLifecycle:
    """TensorBoard 生命周期测试（P1-03 覆盖率补充）"""

    def test_train_start_creates_writer_and_end_closes(self, tmp_path):
        callback = TensorBoardCallback(log_dir=str(tmp_path / "runs"))
        callback.on_train_start(MockTrainer())
        assert callback._writer is not None
        callback.on_epoch_end(MockTrainer(), 0, {"train_loss": 0.5})
        callback.on_train_end(MockTrainer())
        assert callback._writer is None


class TestProgressBarCallback:
    """进度条回调测试（P1-03 覆盖率补充）"""

    def test_verbose_zero_is_noop(self, tmp_path):
        from src.training.callbacks import ProgressBarCallback

        pbar = ProgressBarCallback(verbose=0)
        trainer = MockTrainer()
        pbar.on_train_start(trainer)
        pbar.on_epoch_start(trainer, 0)
        pbar.on_batch_end(trainer, 0, {"loss": 0.5})
        pbar.on_epoch_end(trainer, 0, {"train_loss": 0.5})
        pbar.on_train_end(trainer)
        assert pbar.epoch_pbar is None
        assert pbar.batch_pbar is None

    def test_logger_fallback_path(self, tmp_path, caplog):
        """tqdm 不可用时回退到 logger 输出"""
        import logging

        import src.training.callbacks as cb
        from src.training.callbacks import ProgressBarCallback

        orig = cb._HAS_TQDM
        cb._HAS_TQDM = False
        try:
            pbar = ProgressBarCallback(verbose=1)
            trainer = MockTrainer()
            with caplog.at_level(logging.INFO, logger="src.training.callbacks"):
                pbar.on_train_start(trainer)
                pbar.on_epoch_start(trainer, 0)
                pbar.on_batch_end(trainer, 0, {"loss": 0.5})
                pbar.on_epoch_end(trainer, 0, {"train_loss": 0.5, "val_loss": 0.4})
                pbar.on_train_end(trainer)
            assert "Training started" in caplog.text
            assert "Epoch 0 started" in caplog.text
            assert "Epoch 0 ended" in caplog.text
            assert "Training ended" in caplog.text
        finally:
            cb._HAS_TQDM = orig


class TestProgressBarTqdmPath:
    """进度条 tqdm 真实路径测试（P1-03 覆盖率补充）"""

    def test_tqdm_full_lifecycle(self):
        import src.training.callbacks as cb
        from src.training.callbacks import ProgressBarCallback

        if not cb._HAS_TQDM:  # pragma: no cover - 环境无 tqdm 时跳过
            import pytest

            pytest.skip("tqdm not installed")

        pbar = ProgressBarCallback(verbose=1)
        trainer = MockTrainer()
        pbar.on_train_start(trainer)
        assert pbar.epoch_pbar is not None
        pbar.on_epoch_start(trainer, 0)
        assert pbar.batch_pbar is not None
        pbar.on_batch_end(trainer, 0, {"loss": 0.5})
        pbar.on_epoch_end(trainer, 0, {"train_loss": 0.5, "val_loss": 0.4})
        assert pbar.batch_pbar is None
        pbar.on_epoch_end(trainer, 1, {"train_loss": 0.4})
        pbar.on_train_end(trainer)
        assert pbar.epoch_pbar is None

    def test_tensorboard_missing_warning(self, caplog):
        """tensorboard 不可用时初始化应记录警告"""
        import logging

        import src.training.callbacks as cb

        orig = cb._HAS_TENSORBOARD
        cb._HAS_TENSORBOARD = False
        try:
            with caplog.at_level(logging.WARNING, logger="src.training.callbacks"):
                TensorBoardCallback(log_dir="runs")
            assert "tensorboard 不可用" in caplog.text
        finally:
            cb._HAS_TENSORBOARD = orig

    def test_tensorboard_train_start_skipped_without_tb(self):
        """tensorboard 不可用时 on_train_start 不初始化 writer"""
        import src.training.callbacks as cb

        orig = cb._HAS_TENSORBOARD
        cb._HAS_TENSORBOARD = False
        try:
            callback = TensorBoardCallback(log_dir="runs")
            callback.on_train_start(MockTrainer())
            assert callback._writer is None
        finally:
            cb._HAS_TENSORBOARD = orig


class TestCallbackStateDict:
    """TD-M01: callback state_dict/load_state_dict 精确续训契约"""

    def test_checkpoint_state_roundtrip(self, tmp_path):
        """ModelCheckpoint 状态可保存并恢复（best/top-k/未改善计数）"""
        filepath = str(tmp_path / "model.pt")
        source = ModelCheckpoint(
            filepath,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=0,
        )
        trainer = MockTrainer()
        for epoch, loss in enumerate([0.5, 0.3, 0.7]):
            source.on_epoch_end(trainer, epoch=epoch, logs={"val_loss": loss})
        assert source.best_value == 0.3
        assert source.epochs_since_improvement == 1

        target = ModelCheckpoint(
            filepath,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=0,
        )
        target.load_state_dict(source.state_dict())
        assert target.best_value == 0.3
        assert target.epochs_since_improvement == 1

    def test_checkpoint_state_top_k_roundtrip(self, tmp_path):
        """top-k 模式恢复 top_k_checkpoints 列表"""
        filepath = str(tmp_path / "model.pt")
        source = ModelCheckpoint(
            filepath,
            monitor="val_loss",
            mode="min",
            save_best_only=False,
            save_top_k=2,
            verbose=0,
        )
        trainer = MockTrainer()
        for epoch, loss in enumerate([0.5, 0.3, 0.7]):
            source.on_epoch_end(trainer, epoch=epoch, logs={"val_loss": loss})
        assert len(source._top_k_checkpoints) == 2

        target = ModelCheckpoint(
            filepath,
            monitor="val_loss",
            mode="min",
            save_best_only=False,
            save_top_k=2,
            verbose=0,
        )
        target.load_state_dict(source.state_dict())
        assert target._top_k_checkpoints == source._top_k_checkpoints

    def test_checkpoint_state_mismatch_rejected(self):
        """monitor/mode 不一致时拒绝恢复"""
        source = ModelCheckpoint("x.pt", monitor="val_loss", mode="min")
        source.on_epoch_end(MockTrainer(), 0, {"val_loss": 0.5})
        target = ModelCheckpoint("x.pt", monitor="val_acc", mode="max")
        with pytest.raises(ValueError, match="不一致"):
            target.load_state_dict(source.state_dict())

    def test_top_k_restore_resorts_by_mode(self, tmp_path):
        """恢复后 top-k 按当前 mode 重新排序（防御数据漂移）"""
        state = {
            "monitor": "val_acc",
            "mode": "max",
            "best_value": 0.9,
            "epochs_since_improvement": 2,
            "top_k_checkpoints": [[0.5, "a.pt"], [0.9, "b.pt"], [0.7, "c.pt"]],
        }
        target = ModelCheckpoint(str(tmp_path / "m.pt"), monitor="val_acc", mode="max")
        target.load_state_dict(state)
        assert target.best_value == 0.9
        assert target.epochs_since_improvement == 2
        assert target._top_k_checkpoints[0] == (0.9, "b.pt")

    def test_early_stopping_state_roundtrip(self):
        """EarlyStopping 状态可保存并恢复（wait/stopped_epoch/should_stop）"""
        source = EarlyStopping(monitor="val_loss", mode="min", patience=2, verbose=0)
        source.on_epoch_end(MockTrainer(), 0, {"val_loss": 0.5})
        source.on_epoch_end(MockTrainer(), 1, {"val_loss": 0.6})
        assert source.wait == 1

        target = EarlyStopping(monitor="val_loss", mode="min", patience=2, verbose=0)
        target.load_state_dict(source.state_dict())
        assert target.best_value == 0.5
        assert target.wait == 1
        assert target._state_restored is True

    def test_early_stopping_state_mismatch_rejected(self):
        """patience/min_delta 不一致时拒绝恢复"""
        source = EarlyStopping(monitor="val_loss", patience=2, verbose=0)
        source.on_epoch_end(MockTrainer(), 0, {"val_loss": 0.5})
        source.on_epoch_end(MockTrainer(), 1, {"val_loss": 0.6})
        target = EarlyStopping(monitor="val_loss", patience=5, verbose=0)
        with pytest.raises(ValueError, match="不一致"):
            target.load_state_dict(source.state_dict())

    def test_early_stopping_resume_preserves_wait_after_train_start(self):
        """resume 恢复后 on_train_start 不重置 wait（精确续训）"""
        source = EarlyStopping(monitor="val_loss", patience=2, verbose=0)
        source.on_epoch_end(MockTrainer(), 0, {"val_loss": 0.5})
        source.on_epoch_end(MockTrainer(), 1, {"val_loss": 0.6})
        resumed = EarlyStopping(monitor="val_loss", patience=2, verbose=0)
        resumed.load_state_dict(source.state_dict())
        resumed.on_train_start(MockTrainer())
        assert resumed.wait == 1
        assert resumed.best_value == 0.5
        # 标志被消费：下一次 fit 重新走重置路径
        resumed.on_train_start(MockTrainer())
        assert resumed.wait == 0

    def test_fresh_early_stopping_still_resets_on_train_start(self):
        """未恢复状态时 on_train_start 仍重置（回归守卫）"""
        early_stop = EarlyStopping(monitor="val_loss", patience=2, verbose=0)
        early_stop.on_epoch_end(MockTrainer(), 0, {"val_loss": 0.5})
        early_stop.on_epoch_end(MockTrainer(), 1, {"val_loss": 0.6})
        assert early_stop.wait == 1
        early_stop.on_train_start(MockTrainer())
        assert early_stop.wait == 0
        assert early_stop.should_stop is False

    def test_build_checkpoint_collects_callback_states(self, tmp_path):
        """_build_checkpoint 收集 trainer.callbacks 中 stateful 回调状态"""
        filepath = str(tmp_path / "best.pt")
        ckpt_cb = ModelCheckpoint(filepath, monitor="val_loss", mode="min", verbose=0)
        stop_cb = EarlyStopping(monitor="val_loss", patience=3, verbose=0)
        stop_cb.on_epoch_end(MockTrainer(), 0, {"val_loss": 0.5})
        stop_cb.on_epoch_end(MockTrainer(), 1, {"val_loss": 0.6})

        class FakeTrainer:
            callbacks = [ckpt_cb, stop_cb]

        payload = ckpt_cb._build_checkpoint(MockModel(), epoch=2, trainer=FakeTrainer())
        states = payload["callback_states"]
        assert "ModelCheckpoint" in states
        assert "EarlyStopping" in states
        assert states["EarlyStopping"]["wait"] == 1
        assert states["EarlyStopping"]["best_value"] == 0.5

    def test_build_checkpoint_preserves_global_step(self, tmp_path):
        """完整 checkpoint 应保存 global_step，避免 resume 重置优化步数"""
        ckpt_cb = ModelCheckpoint(str(tmp_path / "best.pt"), verbose=0)

        class FakeTrainer:
            callbacks = [ckpt_cb]
            global_step = 17

        payload = ckpt_cb._build_checkpoint(MockModel(), epoch=2, trainer=FakeTrainer())
        assert payload["global_step"] == 17

    def test_save_last_contains_updated_callback_state(self, tmp_path):
        """checkpoint_last 应记录当前 epoch 更新后的 best 状态"""
        filepath = str(tmp_path / "best.pt")
        callback = ModelCheckpoint(
            filepath,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_last=True,
            verbose=0,
        )
        trainer = MockTrainer()
        trainer.global_step = 3
        trainer.callbacks = [callback]
        callback.on_epoch_end(trainer, epoch=0, logs={"val_loss": 0.25})
        last = torch.load(tmp_path / "best_last.pt", map_location="cpu", weights_only=False)
        assert last["global_step"] == 3
        assert last["callback_states"]["ModelCheckpoint"]["best_value"] == 0.25

    def test_build_checkpoint_skips_stateless_callbacks(self, tmp_path):
        """无 state_dict 的 callback（日志/进度条）不进入 callback_states"""
        ckpt_cb = ModelCheckpoint(str(tmp_path / "best.pt"), verbose=0)
        stateless = Callback()

        class FakeTrainer:
            callbacks = [ckpt_cb, stateless]

        payload = ckpt_cb._build_checkpoint(MockModel(), epoch=0, trainer=FakeTrainer())
        assert set(payload["callback_states"].keys()) == {"ModelCheckpoint"}

    def test_duplicate_class_names_get_indexed_keys(self):
        """同类 callback 多次出现时 key 追加序号，保存/恢复一致"""
        from src.training.callbacks import collect_callback_states

        first = EarlyStopping(monitor="val_loss", verbose=0)
        second = EarlyStopping(monitor="val_acc", mode="max", verbose=0)
        states = collect_callback_states([first, second])
        assert set(states.keys()) == {"EarlyStopping", "EarlyStopping_2"}

    def test_collect_callback_states_none_is_empty(self):
        from src.training.callbacks import collect_callback_states

        assert collect_callback_states(None) == {}
        assert collect_callback_states([]) == {}
