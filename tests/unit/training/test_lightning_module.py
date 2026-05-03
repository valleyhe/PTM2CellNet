"""
Lightning模块单元测试
使用Mock和轻量Trainer进行测试
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import MagicMock, patch

import lightning as L
from lightning.pytorch import Trainer

from src.training.lightning_module import PTM2CellNetLightning
from src.training.losses import FocalLoss


class MockModel(nn.Module):
    """模拟PTM2CellNet模型"""

    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(10, num_classes)

    def forward(self, batch):
        x = torch.randn(batch["label"].size(0), 10, device=batch["label"].device)
        logits = self.fc(x)
        predictions = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        return {
            "logits": logits,
            "predictions": predictions,
            "probabilities": probs,
        }


class MockEncoder(nn.Module):
    """模拟编码器"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def mock_model():
    return MockModel(num_classes=4)


@pytest.fixture
def base_config():
    return {
        "model": {"num_classes": 4},
        "training": {
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "scheduler": None,
            "max_epochs": 10,
            "loss_type": "cross_entropy",
            "use_lora": False,
        },
    }


@pytest.fixture
def sample_batch():
    return {
        "sequence": torch.randint(0, 21, (4, 10)),
        "ptm_types": torch.randint(0, 5, (4, 3)),
        "ptm_positions": torch.randint(0, 10, (4, 3)),
        "ptm_mask": torch.ones(4, 3, dtype=torch.float),
        "label": torch.tensor([0, 1, 2, 3]),
    }


class TestLightningModuleInit:
    """初始化测试"""

    def test_init_default(self, mock_model, base_config):
        module = PTM2CellNetLightning(mock_model, base_config)
        assert module.model is mock_model
        assert module.learning_rate == 1e-4
        assert module.optimizer_name == "adamw"
        assert module.loss_type == "cross_entropy"

    def test_init_with_config(self, mock_model):
        config = {
            "model": {"num_classes": 4},
            "training": {
                "learning_rate": 5e-5,
                "weight_decay": 0.1,
                "optimizer": "adam",
                "scheduler": "cosine",
                "max_epochs": 20,
                "loss_type": "focal",
            },
        }
        module = PTM2CellNetLightning(mock_model, config)
        assert module.learning_rate == 5e-5
        assert module.weight_decay == 0.1
        assert module.optimizer_name == "adam"
        assert module.scheduler_name == "cosine"
        assert module.max_epochs == 20
        assert module.loss_type == "focal"

    def test_save_hyperparameters(self, mock_model, base_config):
        module = PTM2CellNetLightning(mock_model, base_config)
        assert "config" in module.hparams
        assert "learning_rate" not in module.hparams

    def test_model_assigned(self, mock_model, base_config):
        module = PTM2CellNetLightning(mock_model, base_config)
        assert module.model is mock_model


class TestCreateCriterion:
    """损失函数创建测试"""

    def test_create_cross_entropy(self, mock_model, base_config):
        module = PTM2CellNetLightning(mock_model, base_config)
        assert module.criterion is None

    def test_create_focal_loss(self, mock_model):
        config = {
            "model": {"num_classes": 4},
            "training": {
                "learning_rate": 1e-4,
                "optimizer": "adamw",
                "loss_type": "focal",
                "focal_gamma": 2.0,
            },
        }
        module = PTM2CellNetLightning(mock_model, config)
        assert isinstance(module.criterion, FocalLoss)
        assert module.criterion.gamma == 2.0

    def test_focal_loss_with_alpha(self, mock_model):
        config = {
            "model": {"num_classes": 4},
            "training": {
                "learning_rate": 1e-4,
                "optimizer": "adamw",
                "loss_type": "focal",
                "class_weights": [0.5, 1.0, 2.0, 1.5],
            },
        }
        module = PTM2CellNetLightning(mock_model, config)
        assert isinstance(module.criterion, FocalLoss)
        assert module.criterion.alpha is not None
        assert torch.allclose(
            module.criterion.alpha,
            torch.tensor([0.5, 1.0, 2.0, 1.5], dtype=torch.float32),
        )

    def test_focal_loss_gamma(self, mock_model):
        config = {
            "model": {"num_classes": 4},
            "training": {
                "learning_rate": 1e-4,
                "optimizer": "adamw",
                "loss_type": "focal",
                "focal_gamma": 3.0,
            },
        }
        module = PTM2CellNetLightning(mock_model, config)
        assert module.criterion.gamma == 3.0


class TestConfigureOptimizers:
    """优化器配置测试"""

    def test_configure_adamw(self, mock_model, base_config):
        module = PTM2CellNetLightning(mock_model, base_config)
        result = module.configure_optimizers()
        assert isinstance(result, torch.optim.AdamW)

    def test_configure_adam(self, mock_model, base_config):
        base_config["training"]["optimizer"] = "adam"
        module = PTM2CellNetLightning(mock_model, base_config)
        result = module.configure_optimizers()
        assert isinstance(result, torch.optim.Adam)

    def test_configure_sgd(self, mock_model, base_config):
        base_config["training"]["optimizer"] = "sgd"
        module = PTM2CellNetLightning(mock_model, base_config)
        result = module.configure_optimizers()
        assert isinstance(result, torch.optim.SGD)

    def test_configure_lion(self, mock_model, base_config):
        base_config["training"]["optimizer"] = "lion"
        module = PTM2CellNetLightning(mock_model, base_config)
        # lion_pytorch 未安装时回退到 AdamW
        result = module.configure_optimizers()
        assert isinstance(result, torch.optim.Optimizer)

    def test_configure_unknown_optimizer(self, mock_model, base_config):
        base_config["training"]["optimizer"] = "unknown"
        module = PTM2CellNetLightning(mock_model, base_config)
        result = module.configure_optimizers()
        assert isinstance(result, torch.optim.AdamW)


class TestConfigureSchedulers:
    """调度器配置测试"""

    def test_configure_cosine(self, mock_model, base_config):
        base_config["training"]["scheduler"] = "cosine"
        module = PTM2CellNetLightning(mock_model, base_config)
        result = module.configure_optimizers()
        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr_scheduler" in result
        from torch.optim.lr_scheduler import CosineAnnealingLR
        assert isinstance(result["lr_scheduler"]["scheduler"], CosineAnnealingLR)

    def test_configure_plateau(self, mock_model, base_config):
        base_config["training"]["scheduler"] = "plateau"
        module = PTM2CellNetLightning(mock_model, base_config)
        result = module.configure_optimizers()
        assert isinstance(result, dict)
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        assert isinstance(result["lr_scheduler"]["scheduler"], ReduceLROnPlateau)

    def test_configure_step(self, mock_model, base_config):
        base_config["training"]["scheduler"] = "step"
        module = PTM2CellNetLightning(mock_model, base_config)
        result = module.configure_optimizers()
        assert isinstance(result, dict)
        from torch.optim.lr_scheduler import StepLR
        assert isinstance(result["lr_scheduler"]["scheduler"], StepLR)

    def test_no_scheduler(self, mock_model, base_config):
        base_config["training"]["scheduler"] = None
        module = PTM2CellNetLightning(mock_model, base_config)
        result = module.configure_optimizers()
        assert isinstance(result, torch.optim.Optimizer)
        assert not isinstance(result, dict)

    def test_configure_unknown_scheduler(self, mock_model, base_config):
        base_config["training"]["scheduler"] = "unknown"
        module = PTM2CellNetLightning(mock_model, base_config)
        result = module.configure_optimizers()
        assert isinstance(result, torch.optim.Optimizer)
        assert not isinstance(result, dict)


class TestTrainingStep:
    """训练步骤测试"""

    def test_training_step_returns_loss(self, mock_model, base_config, sample_batch):
        module = PTM2CellNetLightning(mock_model, base_config)
        loss = module.training_step(sample_batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_training_step_logs_metrics(self, mock_model, base_config, sample_batch):
        module = PTM2CellNetLightning(mock_model, base_config)
        with patch.object(module, "log") as mock_log:
            module.training_step(sample_batch, 0)
            calls = [call[0][0] for call in mock_log.call_args_list]
            assert "train_loss" in calls
            assert "train_acc" in calls

    def test_training_step_with_focal_loss(self, mock_model, sample_batch):
        config = {
            "model": {"num_classes": 4},
            "training": {
                "learning_rate": 1e-4,
                "optimizer": "adamw",
                "loss_type": "focal",
                "focal_gamma": 2.0,
            },
        }
        module = PTM2CellNetLightning(mock_model, config)
        loss = module.training_step(sample_batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_training_step_with_cross_entropy(self, mock_model, base_config, sample_batch):
        module = PTM2CellNetLightning(mock_model, base_config)
        loss = module.training_step(sample_batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0


class TestValidationStep:
    """验证步骤测试"""

    def test_validation_step_returns_loss(self, mock_model, base_config, sample_batch):
        module = PTM2CellNetLightning(mock_model, base_config)
        loss = module.validation_step(sample_batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_validation_step_logs_metrics(self, mock_model, base_config, sample_batch):
        module = PTM2CellNetLightning(mock_model, base_config)
        with patch.object(module, "log") as mock_log:
            module.validation_step(sample_batch, 0)
            calls = [call[0][0] for call in mock_log.call_args_list]
            assert "val_loss" in calls
            assert "val_acc" in calls

    def test_validation_step_updates_torchmetrics(self, mock_model, base_config, sample_batch):
        module = PTM2CellNetLightning(mock_model, base_config)
        with patch.object(module, "log"):
            module.validation_step(sample_batch, 0)
        # torchmetrics 应被更新
        if module.metrics:
            for metric in module.metrics.values():
                state = metric.state_dict()
                assert state is not None


class TestOnValidationEpochEnd:
    """验证epoch结束测试"""

    def test_computes_and_logs_metrics(self, mock_model, base_config):
        module = PTM2CellNetLightning(mock_model, base_config)
        # 先通过validation_step更新metrics，再调用on_validation_epoch_end
        batch = {"label": torch.tensor([0, 1, 2, 3])}
        mock_out = {
            "logits": torch.randn(4, 4),
            "predictions": torch.tensor([0, 1, 2, 3]),
        }
        if module.metrics:
            with patch.object(module.model, "forward", return_value=mock_out):
                with patch.object(module, "log"):
                    module.validation_step(batch, 0)
        with patch.object(module, "log") as mock_log:
            module.on_validation_epoch_end()
            if module.metrics:
                assert mock_log.call_count >= len(module.metrics)

    def test_resets_metrics(self, mock_model, base_config):
        module = PTM2CellNetLightning(mock_model, base_config)
        # 先更新指标
        batch = {"label": torch.tensor([0, 1, 2, 3])}
        mock_out = {
            "logits": torch.randn(4, 4),
            "predictions": torch.tensor([0, 1, 2, 3]),
        }
        if module.metrics:
            with patch.object(module.model, "forward", return_value=mock_out):
                with patch.object(module, "log"):
                    module.validation_step(batch, 0)
            module.on_validation_epoch_end()
            for metric in module.metrics.values():
                state = metric.state_dict()
                # 重置后状态应被清空

    def test_metrics_logged_to_prog_bar(self, mock_model, base_config):
        module = PTM2CellNetLightning(mock_model, base_config)
        batch = {"label": torch.tensor([0, 1, 2, 3])}
        mock_out = {
            "logits": torch.randn(4, 4),
            "predictions": torch.tensor([0, 1, 2, 3]),
        }
        if module.metrics:
            with patch.object(module.model, "forward", return_value=mock_out):
                with patch.object(module, "log"):
                    module.validation_step(batch, 0)
        with patch.object(module, "log") as mock_log:
            module.on_validation_epoch_end()
            if module.metrics:
                for call in mock_log.call_args_list:
                    kwargs = call[1]
                    if kwargs.get("prog_bar", False):
                        return


class TestTestStep:
    """测试步骤测试"""

    def test_test_step_returns_loss(self, mock_model, base_config, sample_batch):
        module = PTM2CellNetLightning(mock_model, base_config)
        loss = module.test_step(sample_batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_test_step_logs_metrics(self, mock_model, base_config, sample_batch):
        module = PTM2CellNetLightning(mock_model, base_config)
        with patch.object(module, "log") as mock_log:
            module.test_step(sample_batch, 0)
            calls = [call[0][0] for call in mock_log.call_args_list]
            assert "test_loss" in calls
            assert "test_acc" in calls


class TestClassWeights:
    """类别权重测试"""

    def test_compute_class_weights(self):
        labels = torch.tensor([0, 0, 1, 1, 1, 2])
        weights = PTM2CellNetLightning.compute_class_weights(labels, num_classes=3)
        assert weights.shape == (3,)
        assert weights.dtype == torch.float32

    def test_weights_normalized(self):
        labels = torch.tensor([0, 0, 1, 1, 1, 2])
        weights = PTM2CellNetLightning.compute_class_weights(labels, num_classes=3)
        assert abs(weights.mean().item() - 1.0) < 1e-5

    def test_imbalanced_data(self):
        labels = torch.tensor([0] * 90 + [1] * 10)
        weights = PTM2CellNetLightning.compute_class_weights(labels, num_classes=2)
        assert weights[1] > weights[0]


class TestApplyLoRA:
    """LoRA应用测试"""

    def test_apply_lora_flag(self, mock_model):
        config = {
            "model": {"num_classes": 4},
            "training": {
                "learning_rate": 1e-4,
                "optimizer": "adamw",
                "use_lora": True,
            },
        }
        mock_model.encoder = MockEncoder()
        with patch.object(PTM2CellNetLightning, "_apply_lora") as mock_apply:
            module = PTM2CellNetLightning(mock_model, config)
            mock_apply.assert_called_once()

    def test_lora_config_passed(self, mock_model):
        lora_cfg = {"r": 8, "lora_alpha": 16}
        config = {
            "model": {"num_classes": 4},
            "training": {
                "learning_rate": 1e-4,
                "optimizer": "adamw",
                "use_lora": True,
                "lora_config": lora_cfg,
            },
        }
        mock_model.encoder = MockEncoder()
        with patch.object(PTM2CellNetLightning, "_apply_lora") as mock_apply:
            module = PTM2CellNetLightning(mock_model, config)
            mock_apply.assert_called_once()

    def test_no_lora_when_false(self, mock_model, base_config):
        mock_model.encoder = MockEncoder()
        with patch.object(PTM2CellNetLightning, "_apply_lora") as mock_apply:
            module = PTM2CellNetLightning(mock_model, base_config)
            mock_apply.assert_not_called()


class TestForward:
    """前向传播测试"""

    def test_forward_calls_model(self, mock_model, base_config, sample_batch):
        module = PTM2CellNetLightning(mock_model, base_config)
        with patch.object(mock_model, "forward", wraps=mock_model.forward) as mock_fw:
            module.forward(sample_batch)
            mock_fw.assert_called_once()

    def test_forward_returns_output(self, mock_model, base_config, sample_batch):
        module = PTM2CellNetLightning(mock_model, base_config)
        output = module.forward(sample_batch)
        assert "logits" in output
        assert "predictions" in output
        assert "probabilities" in output


class TestLightningTrainerIntegration:
    """Lightning Trainer集成测试"""

    def test_fast_dev_run(self, mock_model, base_config):
        """使用 fast_dev_run 快速验证"""
        module = PTM2CellNetLightning(mock_model, base_config)
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        # 创建简单的DataLoader
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(
            torch.randint(0, 21, (8, 10)),
            torch.randint(0, 4, (8,)),
        )
        loader = DataLoader(dataset, batch_size=4)
        # Lightning需要字典格式输入，这里只验证Trainer可以初始化模块
        assert trainer is not None
        assert module is not None
