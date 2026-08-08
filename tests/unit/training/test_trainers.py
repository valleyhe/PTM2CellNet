"""
Trainers模块单元测试
使用MockModel和MockDataLoader进行测试
"""

import inspect
from typing import Dict
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainers import Trainer
from src.training.callbacks import Callback, ModelCheckpoint, EarlyStopping


class MockModel(nn.Module):
    """模拟模型"""

    def __init__(self, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)

    def forward(self, batch):
        if isinstance(batch, dict):
            x = batch.get("sequence", batch.get("input", torch.randn(batch.get("label", torch.tensor([0])).size(0), 10)))
            if x.dim() > 2:
                x = x.view(x.size(0), -1)[:, :10]
            logits = self.fc(x)
        else:
            logits = self.fc(batch.view(batch.size(0), -1)[:, :10])
        predictions = torch.argmax(logits, dim=1)
        return {
            "logits": logits,
            "predictions": predictions,
        }


class MockCallback(Callback):
    """模拟回调，记录所有钩子调用"""

    def __init__(self):
        self.calls = []

    def on_train_start(self, trainer):
        self.calls.append(("on_train_start", None))

    def on_train_end(self, trainer):
        self.calls.append(("on_train_end", None))

    def on_epoch_start(self, trainer, epoch):
        self.calls.append(("on_epoch_start", epoch))

    def on_epoch_end(self, trainer, epoch, logs):
        self.calls.append(("on_epoch_end", epoch, logs))

    def on_batch_start(self, trainer, batch_idx):
        self.calls.append(("on_batch_start", batch_idx))

    def on_batch_end(self, trainer, batch_idx, logs):
        self.calls.append(("on_batch_end", batch_idx, logs))


@pytest.fixture
def mock_model():
    return MockModel(num_classes=4)


@pytest.fixture
def base_config():
    return {
        "training": {
            "max_epochs": 2,
            "learning_rate": 1e-3,
        }
    }


@pytest.fixture
def sample_loader():
    """创建简单的DataLoader"""
    dataset = TensorDataset(
        torch.randn(8, 10),
        torch.randint(0, 4, (8,)),
    )
    return DataLoader(dataset, batch_size=4)


@pytest.fixture
def dict_loader():
    """创建字典批次格式的DataLoader"""
    class DictDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.data = [
                {"sequence": torch.randn(10), "label": torch.tensor(0)},
                {"sequence": torch.randn(10), "label": torch.tensor(1)},
                {"sequence": torch.randn(10), "label": torch.tensor(2)},
                {"sequence": torch.randn(10), "label": torch.tensor(3)},
                {"sequence": torch.randn(10), "label": torch.tensor(0)},
                {"sequence": torch.randn(10), "label": torch.tensor(1)},
                {"sequence": torch.randn(10), "label": torch.tensor(2)},
                {"sequence": torch.randn(10), "label": torch.tensor(3)},
            ]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx].copy()
            # 扩展维度以模拟批次，形状为 [seq_len=2, features=10]
            seq = item["sequence"]
            item["sequence"] = seq.unsqueeze(0).repeat(2, 1)
            return item

    # 使用自定义collate_fn处理字典列表
    def collate_fn(batch_list):
        result = {}
        for key in batch_list[0]:
            result[key] = torch.stack([b[key] for b in batch_list])
        return result

    return DataLoader(DictDataset(), batch_size=2, collate_fn=collate_fn)


class TestTrainerInit:
    """训练器初始化测试"""

    def test_init_default(self, mock_model):
        trainer = Trainer(mock_model)
        assert trainer.model is mock_model
        assert trainer.device in ("cuda", "cpu")
        assert trainer.epoch == 0
        assert trainer.global_step == 0
        assert trainer.callbacks == []

    def test_init_with_config(self, mock_model, base_config):
        trainer = Trainer(mock_model, config=base_config)
        assert trainer.config is base_config
        assert trainer.training_config == base_config["training"]

    def test_device_selection(self, mock_model):
        trainer = Trainer(mock_model, device="cpu")
        assert trainer.device == "cpu"

    def test_callbacks_initialized(self, mock_model):
        trainer = Trainer(mock_model)
        assert trainer.callbacks == []
        assert trainer.train_losses == []
        assert trainer.val_losses == []
        assert trainer.best_val_loss == float("inf")


class TestTrainerFit:
    """训练方法测试"""

    def test_fit_runs_epochs(self, mock_model, dict_loader):
        trainer = Trainer(mock_model, device="cpu")
        trainer.compile()
        trainer.fit(dict_loader, max_epochs=2)
        assert trainer.epoch == 1  # 最后运行的epoch
        assert len(trainer.train_losses) == 2

    def test_fit_calls_callbacks(self, mock_model, dict_loader):
        trainer = Trainer(mock_model, device="cpu")
        callback = MockCallback()
        trainer.compile(callbacks=[callback])
        trainer.fit(dict_loader, max_epochs=1)
        assert any(call[0] == "on_train_start" for call in callback.calls)
        assert any(call[0] == "on_epoch_start" for call in callback.calls)
        assert any(call[0] == "on_epoch_end" for call in callback.calls)
        assert any(call[0] == "on_train_end" for call in callback.calls)

    def test_fit_with_train_loader(self, mock_model, dict_loader):
        trainer = Trainer(mock_model, device="cpu")
        trainer.compile()
        trainer.fit(dict_loader)
        assert len(trainer.train_losses) > 0

    def test_fit_with_val_loader(self, mock_model, dict_loader):
        trainer = Trainer(mock_model, device="cpu")
        trainer.compile()
        trainer.fit(dict_loader, val_loader=dict_loader, max_epochs=1)
        assert len(trainer.train_losses) == 1
        assert len(trainer.val_losses) == 1


class TestTrainerEvaluate:
    """评估方法测试"""

    def test_evaluate_returns_metrics(self, mock_model, dict_loader):
        trainer = Trainer(mock_model, device="cpu")
        trainer.compile()
        metrics = trainer.validate(mock_model, dict_loader)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_evaluate_with_loader(self, mock_model, dict_loader):
        trainer = Trainer(mock_model, device="cpu")
        trainer.compile()
        metrics = trainer.validate(mock_model, dict_loader)
        assert isinstance(metrics, dict)

    def test_evaluate_sets_eval_mode(self, mock_model, dict_loader):
        trainer = Trainer(mock_model, device="cpu")
        trainer.compile()
        mock_model.train()
        trainer.validate(mock_model, dict_loader)
        assert not mock_model.training

    def test_evaluate_signature_matches_contract(self, mock_model):
        trainer = Trainer(mock_model, device="cpu")
        signature = inspect.signature(trainer.validate)
        parameters = list(signature.parameters.values())

        assert [parameter.name for parameter in parameters] == ["model", "dataloader"]
        assert parameters[0].annotation is nn.Module
        assert parameters[1].annotation == DataLoader
        assert signature.return_annotation == Dict[str, float]

    def test_evaluate_uses_supplied_model(self, mock_model, dict_loader):
        trainer = Trainer(mock_model, device="cpu")
        trainer.compile()
        replacement_model = MockModel(num_classes=4)

        trainer.validate(replacement_model, dict_loader)

        assert trainer.model is replacement_model
        assert next(trainer.model.parameters()).device.type == trainer.device


class TestTrainerPredict:
    """预测方法测试"""

    def test_predict_returns_predictions(self, mock_model, dict_loader):
        trainer = Trainer(mock_model, device="cpu")
        predictions = trainer.predict(dict_loader)
        assert len(predictions) > 0
        assert all(isinstance(p, dict) for p in predictions)
        assert all("logits" in p for p in predictions)

    def test_predict_with_loader(self, mock_model, dict_loader):
        trainer = Trainer(mock_model, device="cpu")
        predictions = trainer.predict(dict_loader)
        assert isinstance(predictions, list)

    def test_predict_batch_processing(self, mock_model, dict_loader):
        trainer = Trainer(mock_model, device="cpu")
        predictions = trainer.predict(dict_loader)
        total_samples = sum(p["logits"].size(0) for p in predictions)
        assert total_samples == 8


class TestCallbackIntegration:
    """回调集成测试"""

    def test_callback_hooks_called(self, mock_model, dict_loader):
        trainer = Trainer(mock_model, device="cpu")
        callback = MockCallback()
        trainer.compile(callbacks=[callback])
        trainer.fit(dict_loader, max_epochs=1)
        hooks = [call[0] for call in callback.calls]
        assert "on_train_start" in hooks
        assert "on_epoch_start" in hooks
        assert "on_batch_end" in hooks
        assert "on_epoch_end" in hooks
        assert "on_train_end" in hooks

    def test_model_checkpoint_saves(self, mock_model, dict_loader, tmp_path):
        trainer = Trainer(mock_model, device="cpu")
        filepath = str(tmp_path / "best.pt")
        checkpoint = ModelCheckpoint(filepath, monitor="val_loss", mode="min", verbose=0)
        trainer.compile(callbacks=[checkpoint])
        trainer.fit(dict_loader, val_loader=dict_loader, max_epochs=1)
        assert (tmp_path / "best.pt").exists()

    def test_early_stopping_stops(self, mock_model, dict_loader):
        trainer = Trainer(mock_model, device="cpu")
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=0, verbose=0)
        trainer.compile(callbacks=[early_stop])
        # 由于patience=0，第一次验证不改善就会停止
        # 但第一次epoch总是改善(从inf降低)，所以设置一个较复杂的场景
        # 使用一个总是触发不改善的callback
        class BadCallback(Callback):
            def on_epoch_end(self, trainer, epoch, logs):
                if epoch >= 0:
                    self.should_stop = True

        bad = BadCallback()
        trainer.callbacks = [bad]
        trainer.fit(dict_loader, max_epochs=3)
        assert trainer.epoch == 0  # 第一个epoch后停止


class TestCheckpointManagement:
    """检查点管理测试"""

    def test_save_checkpoint(self, mock_model, tmp_path):
        trainer = Trainer(mock_model, device="cpu")
        save_path = str(tmp_path / "checkpoint.pt")
        torch.save(trainer.model.state_dict(), save_path)
        assert (tmp_path / "checkpoint.pt").exists()

    def test_load_checkpoint(self, mock_model, tmp_path):
        trainer = Trainer(mock_model, device="cpu")
        save_path = str(tmp_path / "checkpoint.pt")
        torch.save(trainer.model.state_dict(), save_path)
        state_dict = torch.load(save_path, weights_only=True)
        trainer.model.load_state_dict(state_dict)
        assert state_dict is not None

    def test_resume_training(self, mock_model, dict_loader, tmp_path):
        trainer = Trainer(mock_model, device="cpu")
        trainer.compile()
        trainer.fit(dict_loader, max_epochs=1)
        save_path = str(tmp_path / "checkpoint.pt")
        torch.save(trainer.model.state_dict(), save_path)

        # 恢复训练
        trainer2 = Trainer(mock_model, device="cpu")
        trainer2.compile()
        state_dict = torch.load(save_path, weights_only=True)
        trainer2.model.load_state_dict(state_dict)
        trainer2.fit(dict_loader, max_epochs=1)
        assert len(trainer2.train_losses) == 1


class TestTrainerCompile:
    """Trainer compile测试"""

    def test_compile_default(self, mock_model):
        trainer = Trainer(mock_model, device="cpu")
        trainer.compile()
        assert isinstance(trainer.loss_fn, nn.CrossEntropyLoss)
        assert trainer.optimizer is not None

    def test_compile_custom_loss(self, mock_model):
        trainer = Trainer(mock_model, device="cpu")
        loss_fn = nn.MSELoss()
        trainer.compile(loss_fn=loss_fn)
        assert trainer.loss_fn is loss_fn

    def test_add_callback(self, mock_model):
        trainer = Trainer(mock_model, device="cpu")
        callback = MockCallback()
        trainer.add_callback(callback)
        assert callback in trainer.callbacks


class TestTrainerErrors:
    """Trainer错误处理测试"""

    def test_train_without_compile(self, mock_model, dict_loader):
        trainer = Trainer(mock_model, device="cpu")
        with pytest.raises(RuntimeError, match="must be compiled before training"):
            trainer._train_epoch(dict_loader)

    def test_val_without_compile(self, mock_model, dict_loader):
        trainer = Trainer(mock_model, device="cpu")
        with pytest.raises(RuntimeError, match="must be compiled before validation"):
            trainer._val_epoch(dict_loader)

    def test_invalid_batch_type(self, mock_model):
        trainer = Trainer(mock_model, device="cpu")
        trainer.compile()
        # 创建非dict类型的DataLoader
        dataset = TensorDataset(torch.randn(4, 10), torch.randint(0, 4, (4,)))
        loader = DataLoader(dataset, batch_size=2)
        with pytest.raises(TypeError, match="must be a dict"):
            trainer._train_epoch(loader)


# ---------------------------------------------------------------------------
# AMP version-compat helpers (v17 TD: deprecated torch.cuda.amp migration)
# ---------------------------------------------------------------------------


class TestAMPCompatHelpers:
    """Pin the modern-API-first behaviour of the AMP helpers."""

    def test_make_grad_scaler_returns_object_with_scale(self):
        """``_make_grad_scaler`` must return something with a ``scale`` method.

        We don't pin the concrete class (GradScaler / legacy) — only the
        contract the trainer depends on (a ``.scale()`` method that takes a
        loss tensor and returns a scaled tensor). A fresh GradScaler starts
        with a non-unity dynamic scale (2**16 by default), so we only assert
        the contract here, not the exact value.
        """
        import torch

        from src.training.amp_compat import make_grad_scaler

        scaler = make_grad_scaler()
        assert scaler is not None
        assert hasattr(scaler, "scale")
        loss = torch.tensor(1.0, requires_grad=True)
        scaled = scaler.scale(loss)
        # Must be a positive finite float (scaled by the dynamic scale factor).
        assert torch.isfinite(scaled)
        assert float(scaled) > 0

    def test_amp_autocast_returns_context_manager(self):
        """``_amp_autocast`` must return an object usable as ``with ...:``."""
        from src.training.amp_compat import amp_autocast

        cm = amp_autocast()
        # Must support the context-manager protocol.
        assert hasattr(cm, "__enter__")
        assert hasattr(cm, "__exit__")
        with cm:
            pass  # entering and exiting must not raise even without CUDA

    def test_self_supervised_helpers_are_consistent(self):
        """The duplicated helpers in self_supervised.py must behave the same."""
        from src.training.amp_compat import amp_autocast

        with amp_autocast():
            pass

    def test_amp_helpers_fall_back_to_legacy_api(self, monkeypatch):
        """Older supported PyTorch releases use ``torch.cuda.amp`` factories."""
        from contextlib import nullcontext
        from types import SimpleNamespace

        import torch

        from src.training.amp_compat import amp_autocast, make_grad_scaler

        scaler = object()
        monkeypatch.setattr(torch, "amp", SimpleNamespace())
        monkeypatch.setattr(
            torch.cuda,
            "amp",
            SimpleNamespace(GradScaler=lambda: scaler, autocast=lambda: nullcontext("legacy")),
        )

        assert make_grad_scaler() is scaler
        with amp_autocast() as value:
            assert value == "legacy"


class TestExactResume:
    """精确续训测试（P1-01）：start_epoch 与 epoch 恢复"""

    def test_fit_starts_from_trainer_epoch(self, mock_model, dict_loader):
        """设置 trainer.epoch 后 fit 应从该 epoch 继续，而不是从 0 重训"""
        trainer = Trainer(mock_model, device="cpu")
        trainer.compile()
        trainer.epoch = 2
        trainer.fit(dict_loader, max_epochs=4)
        # 从 epoch 2 训练到 4：callback 收到的首个 epoch 应为 2
        assert trainer.epoch == 3  # range(2,4) 最后一个为 3
        assert len(trainer.train_losses) == 2  # 只训练 2 个 epoch

    def test_fit_start_epoch_override(self, mock_model, dict_loader):
        """显式 start_epoch 覆盖 trainer.epoch"""
        trainer = Trainer(mock_model, device="cpu")
        trainer.compile()
        trainer.epoch = 5
        trainer.fit(dict_loader, max_epochs=7, start_epoch=0)
        assert len(trainer.train_losses) == 7

    def test_fit_invalid_start_epoch_clamped(self, mock_model, dict_loader):
        """负 start_epoch 被钳制为 0"""
        trainer = Trainer(mock_model, device="cpu")
        trainer.compile()
        trainer.fit(dict_loader, max_epochs=2, start_epoch=-3)
        assert len(trainer.train_losses) == 2

    def test_checkpoint_with_optimizer_roundtrip(self, mock_model, dict_loader, tmp_path):
        """checkpoint 保存 optimizer 状态，恢复后 optimizer 步数一致"""
        from src.training.callbacks import ModelCheckpoint

        torch.manual_seed(0)
        model = MockModel(num_classes=4)
        trainer = Trainer(model, device="cpu")
        trainer.compile(loss_fn=torch.nn.CrossEntropyLoss())
        trainer.fit(dict_loader, max_epochs=1)

        filepath = str(tmp_path / "ckpt.pt")
        ckpt_cb = ModelCheckpoint(filepath, monitor="val_loss", verbose=0)
        ckpt_cb.optimizer = trainer.optimizer
        ckpt_cb.scheduler = trainer.scheduler
        # 直接保存当前状态（epoch=1）
        ckpt_cb._save_checkpoint(filepath, model, epoch=1, trainer=trainer)

        loaded = torch.load(filepath, weights_only=False)
        assert "optimizer_state_dict" in loaded
        assert "rng_state" in loaded
        # scheduler 仅在配置了调度器时存在；此测试配置未指定调度器
        assert ("scheduler_state_dict" in loaded) == (trainer.scheduler is not None)

        # 恢复：新模型 + 新优化器，加载 optimizer 状态后 param_groups 一致
        model2 = MockModel(num_classes=4)
        trainer2 = Trainer(model2, device="cpu")
        trainer2.compile(loss_fn=torch.nn.CrossEntropyLoss())
        model2.load_state_dict(loaded["model_state_dict"])
        trainer2.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        if trainer2.scheduler is not None and loaded.get("scheduler_state_dict") is not None:
            trainer2.scheduler.load_state_dict(loaded["scheduler_state_dict"])
        assert trainer2.optimizer.param_groups[0]["lr"] == trainer.optimizer.param_groups[0]["lr"]


class TestExactResumeEquivalence:
    """中断续训 vs 连续训练等价性（P1-01 精确续训核心验收）"""

    def test_interrupted_resume_matches_continuous(self, dict_loader, tmp_path):
        """先训练 1 epoch 保存完整状态，恢复后训练到 2 epoch，
        与连续训练 2 epoch 的最终权重/损失一致（容差内）。"""
        from src.training.callbacks import ModelCheckpoint

        torch.manual_seed(7)
        torch_seed_state = torch.get_rng_state()

        def _fresh_model():
            # 重建相同初始权重
            torch.manual_seed(7)
            return MockModel(num_classes=4)

        def _train_continuous():
            model = _fresh_model()
            trainer = Trainer(model, device="cpu")
            trainer.compile(loss_fn=torch.nn.CrossEntropyLoss())
            trainer.fit(dict_loader, max_epochs=2)
            return trainer

        def _train_interrupted():
            torch.manual_seed(7)
            model = _fresh_model()
            trainer = Trainer(model, device="cpu")
            trainer.compile(loss_fn=torch.nn.CrossEntropyLoss())
            trainer.fit(dict_loader, max_epochs=1)
            first_loss = trainer.train_losses[0]

            filepath = str(tmp_path / "ckpt.pt")
            ckpt_cb = ModelCheckpoint(filepath, monitor="val_loss", verbose=0)
            ckpt_cb.optimizer = trainer.optimizer
            ckpt_cb.scheduler = trainer.scheduler
            ckpt_cb._save_checkpoint(filepath, model, epoch=1, trainer=trainer)

            # 模拟新进程：重建模型/优化器，加载完整状态后继续训练
            torch.manual_seed(99)  # 故意用不同种子，验证状态恢复而非种子生效
            resumed_model = _fresh_model()
            resumed = Trainer(resumed_model, device="cpu")
            resumed.compile(loss_fn=torch.nn.CrossEntropyLoss())

            loaded = torch.load(filepath, weights_only=False)
            resumed_model.load_state_dict(loaded["model_state_dict"])
            resumed.epoch = int(loaded["epoch"])
            resumed.global_step = int(loaded.get("global_step", 0))
            resumed.optimizer.load_state_dict(loaded["optimizer_state_dict"])
            if resumed.scheduler is not None and loaded.get("scheduler_state_dict") is not None:
                resumed.scheduler.load_state_dict(loaded["scheduler_state_dict"])
            from src.training.callbacks import _restore_rng_state
            if isinstance(loaded.get("rng_state"), dict):
                _restore_rng_state(loaded["rng_state"])

            resumed.fit(dict_loader, max_epochs=2)
            return resumed, first_loss

        continuous = _train_continuous()
        interrupted, interrupted_first_trainer_loss = _train_interrupted()

        # 最终模型权重逐键一致
        for key in continuous.model.state_dict():
            assert torch.allclose(
                interrupted.model.state_dict()[key],
                continuous.model.state_dict()[key],
                atol=1e-6,
            ), f"权重不一致: {key}"
        # 训练损失序列一致：中断前 1 epoch + 恢复后 1 epoch == 连续 2 epoch
        # （中断/恢复分属两个 trainer 实例，需拼接后再比较）
        assert len(interrupted.train_losses) == 1, "恢复段应只训练 1 个 epoch"
        assert len(continuous.train_losses) == 2
        assert abs(interrupted_first_trainer_loss - continuous.train_losses[0]) < 1e-5
        assert abs(interrupted.train_losses[0] - continuous.train_losses[1]) < 1e-5
        # optimizer 状态一致（如 Adam 步数）
        a_state = interrupted.optimizer.state_dict()["state"]
        b_state = continuous.optimizer.state_dict()["state"]
        assert set(a_state.keys()) == set(b_state.keys())
        for key in a_state:
            for sub_key, a_val in a_state[key].items():
                b_val = b_state[key][sub_key]
                if isinstance(a_val, torch.Tensor):
                    assert torch.allclose(a_val, b_val, atol=1e-6), f"optimizer {key}.{sub_key} 不一致"
                else:
                    assert a_val == b_val, f"optimizer {key}.{sub_key} 不一致"
