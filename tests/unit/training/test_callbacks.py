"""
训练回调模块单元测试
使用mock隔离文件系统和模型依赖
"""

import torch
from unittest.mock import MagicMock

from src.training.callbacks import (
    Callback,
    ModelCheckpoint,
    EarlyStopping,
    TensorBoardCallback,
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
        checkpoint = ModelCheckpoint(
            filepath, monitor="val_loss", save_last=True, verbose=0
        )
        trainer = MockTrainer()

        checkpoint.on_epoch_end(trainer, 0, {"val_loss": 0.5})
        assert (tmp_path / "model_last.pt").exists()

    def test_different_monitor(self, tmp_path):
        """监控不同指标(val_acc)"""
        filepath = str(tmp_path / "best.pt")
        checkpoint = ModelCheckpoint(
            filepath, monitor="val_acc", mode="max", verbose=0
        )
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
        checkpoint = ModelCheckpoint(
            filepath, monitor="val_loss", save_last=True, verbose=1
        )
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
        early_stop = EarlyStopping(
            monitor="val_loss", mode="min", patience=2, min_delta=0.1, verbose=0
        )
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

        written_keys = {
            call.args[0] for call in callback._writer.add_scalar.call_args_list
        }
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

        written_keys = {
            call.args[0] for call in callback._writer.add_scalar.call_args_list
        }
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
        checkpoint = ModelCheckpoint(
            filepath, monitor="val_loss", mode="min", save_last=True, verbose=0
        )
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
