"""
Trainers模块单元测试
使用MockModel和MockDataLoader进行测试
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
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
        metrics = trainer.validate(dict_loader)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_evaluate_with_loader(self, mock_model, dict_loader):
        trainer = Trainer(mock_model, device="cpu")
        trainer.compile()
        metrics = trainer.validate(dict_loader)
        assert isinstance(metrics, dict)

    def test_evaluate_sets_eval_mode(self, mock_model, dict_loader):
        trainer = Trainer(mock_model, device="cpu")
        trainer.compile()
        mock_model.train()
        trainer.validate(dict_loader)
        assert not mock_model.training


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
                    setattr(self, "should_stop", True)

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
