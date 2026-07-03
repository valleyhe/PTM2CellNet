"""
优化器模块单元测试
测试configure_optimizer和各种优化器/调度器配置
"""

import pytest
import torch
from torch import nn

from src.training.optimizers import configure_optimizer, _create_optimizer, _create_scheduler


class TestConfigureOptimizer:
    """优化器配置测试"""

    @pytest.fixture
    def simple_model(self):
        """简单的测试模型"""
        return nn.Linear(10, 2)

    def test_configure_adam(self, simple_model):
        """配置Adam优化器"""
        config = {
            "training": {
                "optimizer": "adam",
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
            }
        }
        optimizer, scheduler = configure_optimizer(simple_model, config)

        assert isinstance(optimizer, torch.optim.Adam)
        assert scheduler is None
        assert optimizer.param_groups[0]["lr"] == 0.001
        assert optimizer.param_groups[0]["weight_decay"] == 0.0001

    def test_configure_adamw(self, simple_model):
        """配置AdamW优化器"""
        config = {
            "training": {
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
            }
        }
        optimizer, scheduler = configure_optimizer(simple_model, config)

        assert isinstance(optimizer, torch.optim.AdamW)

    def test_configure_sgd(self, simple_model):
        """配置SGD优化器，验证momentum=0.9"""
        config = {
            "training": {
                "optimizer": "sgd",
                "learning_rate": 0.01,
                "weight_decay": 0.0005,
            }
        }
        optimizer, scheduler = configure_optimizer(simple_model, config)

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["momentum"] == 0.9

    def test_invalid_optimizer(self, simple_model):
        """无效优化器类型抛出ValueError"""
        with pytest.raises(ValueError, match="未知的优化器类型"):
            _create_optimizer(simple_model, "rmsprop", 0.001, 0.0001)

    def test_only_trainable_params(self):
        """只优化requires_grad=True的参数"""
        model = nn.Linear(10, 2)
        # 全部冻结时，PyTorch会抛出ValueError
        for param in model.parameters():
            param.requires_grad = False

        with pytest.raises(ValueError, match="empty parameter list"):
            _create_optimizer(model, "adam", 0.001, 0.0001)

        # 恢复一个参数可训练
        model.weight.requires_grad = True
        optimizer = _create_optimizer(model, "adam", 0.001, 0.0001)
        assert len(optimizer.param_groups[0]["params"]) == 1


class TestConfigureScheduler:
    """学习率调度器测试"""

    @pytest.fixture
    def adam_optimizer(self):
        """Adam优化器fixture"""
        model = nn.Linear(10, 2)
        return torch.optim.Adam(model.parameters(), lr=0.001)

    def test_configure_cosine(self, adam_optimizer):
        """配置CosineAnnealingLR"""
        scheduler = _create_scheduler(
            adam_optimizer, "cosine", max_epochs=100, scheduler_params={}
        )

        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert scheduler.T_max == 100
        assert scheduler.eta_min == 0.0

    def test_configure_plateau(self, adam_optimizer):
        """配置ReduceLROnPlateau"""
        scheduler = _create_scheduler(
            adam_optimizer, "plateau", max_epochs=100, scheduler_params={}
        )

        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert scheduler.patience == 10
        assert scheduler.factor == 0.1

    def test_configure_step(self, adam_optimizer):
        """配置StepLR"""
        scheduler = _create_scheduler(
            adam_optimizer, "step", max_epochs=100, scheduler_params={}
        )

        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 30
        assert scheduler.gamma == 0.1

    def test_invalid_scheduler(self, adam_optimizer):
        """无效调度器类型抛出ValueError"""
        with pytest.raises(ValueError, match="未知的调度器类型"):
            _create_scheduler(adam_optimizer, "exponential", 100, {})

    def test_no_scheduler(self):
        """scheduler_type=None时configure_optimizer返回None"""
        model = nn.Linear(10, 2)
        config = {
            "training": {
                "optimizer": "adam",
                "scheduler": None,
            }
        }
        optimizer, scheduler = configure_optimizer(model, config)
        assert scheduler is None


class TestSchedulerBehavior:
    """调度器行为测试"""

    @pytest.fixture
    def simple_model(self):
        return nn.Linear(10, 2)

    def test_cosine_lr_decay(self, simple_model):
        """验证余弦退火学习率下降"""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=0.0
        )

        initial_lr = optimizer.param_groups[0]["lr"]
        for _ in range(5):
            optimizer.step()  # correct order: optimizer.step() before scheduler.step()
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        assert current_lr < initial_lr

    def test_step_lr_decay(self, simple_model):
        """验证StepLR在step_size后下降"""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        initial_lr = optimizer.param_groups[0]["lr"]

        # 前2步学习率不变
        optimizer.step()  # correct order: optimizer.step() before scheduler.step()
        scheduler.step()
        optimizer.step()
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == initial_lr

        # 第3步后下降
        optimizer.step()
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == initial_lr * 0.1

    def test_plateau_reduction(self, simple_model):
        """模拟验证损失plateau后学习率下降"""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=0
        )

        initial_lr = optimizer.param_groups[0]["lr"]

        # 模拟损失平台期（连续给出相同或更高的损失）
        for _ in range(2):
            scheduler.step(1.0)  # 损失不变

        current_lr = optimizer.param_groups[0]["lr"]
        assert current_lr < initial_lr


class TestIntegration:
    """集成测试"""

    @pytest.fixture
    def simple_model(self):
        return nn.Linear(10, 2)

    def test_full_config(self, simple_model):
        """完整配置 (optimizer + scheduler)"""
        config = {
            "training": {
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.01,
                "scheduler": "cosine",
                "scheduler_params": {"T_max": 50},
                "max_epochs": 50,
            }
        }
        optimizer, scheduler = configure_optimizer(simple_model, config)

        assert isinstance(optimizer, torch.optim.AdamW)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert scheduler.T_max == 50

    def test_config_override(self, simple_model):
        """配置参数覆盖默认值"""
        config = {
            "training": {
                "optimizer": "sgd",
                "learning_rate": 0.05,
                "weight_decay": 0.001,
                "scheduler": "step",
                "scheduler_params": {"step_size": 10, "gamma": 0.5},
                "max_epochs": 100,
            }
        }
        optimizer, scheduler = configure_optimizer(simple_model, config)

        assert optimizer.param_groups[0]["lr"] == 0.05
        assert optimizer.param_groups[0]["weight_decay"] == 0.001
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 10
        assert scheduler.gamma == 0.5

    def test_optimizer_updates_params(self, simple_model):
        """验证优化器正确更新参数"""
        config = {
            "training": {
                "optimizer": "adam",
                "learning_rate": 0.01,
            }
        }
        optimizer, _ = configure_optimizer(simple_model, config)

        # 记录初始参数
        initial_weight = simple_model.weight.data.clone()

        # 进行一次前向和反向传播
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        loss = torch.nn.functional.cross_entropy(simple_model(x), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 验证参数已更新
        assert not torch.equal(simple_model.weight.data, initial_weight)
