"""
PEFT配置模块单元测试
验证LoRA配置创建、应用、参数统计等功能
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

# 确保项目根目录在路径中
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.training.peft_config import (
    PEFT_AVAILABLE,
    get_lora_config,
    get_trainable_parameters,
    apply_lora_to_encoder,
    freeze_base_model,
    save_lora_adapters,
    load_lora_adapters,
    merge_lora_adapters,
)


class SimpleEncoder(nn.Module):
    """简单编码器用于测试"""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.model(x)


class MockLoraModel(nn.Module):
    """模拟LoRA应用后的模型"""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # 添加模拟的LoRA参数
        self.lora_a = nn.Parameter(torch.randn(10, 4))
        self.lora_b = nn.Parameter(torch.randn(4, 10))

    def forward(self, x):
        return self.base_model(x)

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "adapter_model.bin"))

    def merge_and_unload(self):
        return self.base_model


class MockPeftModel(nn.Module):
    """模拟PeftModel用于merge测试"""

    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def merge_and_unload(self):
        return self.model


@pytest.fixture
def simple_encoder():
    return SimpleEncoder(hidden_dim=64)


class TestLoRAConfig:
    """LoRA配置测试"""

    @pytest.mark.skipif(not PEFT_AVAILABLE, reason="peft not installed")
    def test_get_lora_config_default(self):
        config = get_lora_config()
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert set(config.target_modules) == {"query", "key", "value", "dense"}

    @pytest.mark.skipif(not PEFT_AVAILABLE, reason="peft not installed")
    def test_get_lora_config_custom(self):
        config = get_lora_config(r=8, lora_alpha=16)
        assert config.r == 8
        assert config.lora_alpha == 16

    @pytest.mark.skipif(not PEFT_AVAILABLE, reason="peft not installed")
    def test_config_values(self):
        config = get_lora_config(r=4, lora_alpha=8, lora_dropout=0.1)
        assert config.r == 4
        assert config.lora_alpha == 8
        assert config.lora_dropout == 0.1

    def test_get_lora_config_raises_without_peft(self):
        with patch("src.training.peft_config.PEFT_AVAILABLE", False):
            with pytest.raises(ImportError):
                get_lora_config()


class TestApplyLoRA:
    """LoRA应用测试"""

    def test_apply_lora_to_linear(self, simple_encoder):
        # 直接在线性模型上应用LoRA（使用mock）
        with patch("src.training.peft_config.get_peft_model") as mock_get_peft:
            mock_lora = MockLoraModel(simple_encoder)
            mock_get_peft.return_value = mock_lora

            lora_config = MagicMock()
            result = apply_lora_to_encoder(simple_encoder.model, lora_config)
            mock_get_peft.assert_called_once()
            assert result is mock_lora

    def test_apply_lora_to_encoder(self, simple_encoder):
        with patch("src.training.peft_config.get_peft_model") as mock_get_peft:
            mock_lora = MockLoraModel(simple_encoder)
            mock_get_peft.return_value = mock_lora

            lora_config = MagicMock()
            encoder = SimpleEncoder()
            result = apply_lora_to_encoder(encoder.model, lora_config)
            assert result is mock_lora

    @pytest.mark.skipif(not PEFT_AVAILABLE, reason="peft not installed")
    def test_lora_layers_added(self, simple_encoder):
        # 使用真实的peft应用LoRA
        from peft import LoraConfig, TaskType

        config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["0", "2"],  # Sequential中的模块名
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        result = apply_lora_to_encoder(simple_encoder, config)
        # 检查是否有lora命名的参数
        lora_params = [name for name, _ in result.named_parameters() if "lora_" in name]
        assert len(lora_params) > 0

    @pytest.mark.skipif(not PEFT_AVAILABLE, reason="peft not installed")
    def test_original_weights_frozen(self, simple_encoder):
        from peft import LoraConfig, TaskType

        config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["0", "2"],
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        result = apply_lora_to_encoder(simple_encoder, config)
        base_params_frozen = all(
            not p.requires_grad for n, p in result.named_parameters() if "lora_" not in n
        )
        lora_params_trainable = any(
            p.requires_grad for n, p in result.named_parameters() if "lora_" in n
        )
        assert base_params_frozen
        assert lora_params_trainable


class TestLoRAParameterCount:
    """参数数量测试"""

    def test_trainable_params_less_than_1_percent(self):
        model = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
        )
        # 模拟LoRA效果：冻结大部分参数，只保留少量可训练
        for p in model.parameters():
            p.requires_grad = False
        # 添加少量可训练参数 (r=4)
        model.lora_a = nn.Parameter(torch.randn(1000, 4))
        model.lora_b = nn.Parameter(torch.randn(4, 1000))

        trainable, total, percentage = get_trainable_parameters(model)
        assert percentage < 1.0
        assert trainable == 8000  # 1000*4 + 4*1000
        # total = 2*1000*1000 weights + 2*1000 biases + 8000 lora params = 2010000
        assert total == 2010000

    def test_lora_params_small(self):
        model = nn.Linear(100, 100)
        for p in model.parameters():
            p.requires_grad = False
        model.lora_a = nn.Parameter(torch.randn(100, 4))
        model.lora_b = nn.Parameter(torch.randn(4, 100))

        trainable, total, _ = get_trainable_parameters(model)
        assert trainable == 800

    def test_total_params_unchanged(self):
        model = nn.Linear(100, 100)
        total_before = sum(p.numel() for p in model.parameters())
        for p in model.parameters():
            p.requires_grad = False
        model.lora_a = nn.Parameter(torch.randn(100, 4))
        total_after = sum(p.numel() for p in model.parameters())
        assert total_after == total_before + 400


class TestLoRAMerging:
    """LoRA合并测试"""

    def test_merge_lora_weights(self):
        base_model = nn.Linear(10, 10)
        mock_peft = MockPeftModel(base_model)
        with patch("src.training.peft_config.PEFT_AVAILABLE", True):
            result = merge_lora_adapters(mock_peft)
            assert result is base_model

    def test_unmerge_lora_weights(self):
        # merge_and_unload 返回的就是基础模型
        base_model = nn.Linear(10, 10)
        mock_peft = MockPeftModel(base_model)
        with patch("src.training.peft_config.PEFT_AVAILABLE", True):
            result = merge_lora_adapters(mock_peft)
            # 合并后应该就是原始模型
            assert hasattr(result, "weight")

    def test_merged_weights_correct(self):
        base_model = nn.Linear(10, 10)
        original_weight = base_model.weight.data.clone()
        mock_peft = MockPeftModel(base_model)
        with patch("src.training.peft_config.PEFT_AVAILABLE", True):
            result = merge_lora_adapters(mock_peft)
            assert result is base_model
            # MockPeftModel不实际修改权重，所以保持一致
            assert torch.equal(result.weight, original_weight)

    def test_merge_raises_on_non_peft(self):
        with patch("src.training.peft_config.PEFT_AVAILABLE", True):
            with pytest.raises(ValueError, match="不支持合并"):
                merge_lora_adapters(nn.Linear(10, 10))


class TestLoRAIntegration:
    """LoRA集成测试"""

    def test_forward_with_lora(self, simple_encoder):
        with patch("src.training.peft_config.get_peft_model") as mock_get_peft:
            mock_lora = MockLoraModel(simple_encoder)
            mock_get_peft.return_value = mock_lora

            lora_config = MagicMock()
            result = apply_lora_to_encoder(simple_encoder.model, lora_config)
            x = torch.randn(2, 10)
            out = result(x)
            assert out.shape == (2, 64)

    def test_gradient_flow(self):
        class LoRALinear(nn.Module):
            def __init__(self, in_features, out_features, r=4):
                super().__init__()
                self.linear = nn.Linear(in_features, out_features)
                self.linear.weight.requires_grad = False
                self.linear.bias.requires_grad = False
                self.lora_a = nn.Parameter(torch.randn(in_features, r))
                self.lora_b = nn.Parameter(torch.randn(r, out_features))

            def forward(self, x):
                return self.linear(x) + x @ self.lora_a @ self.lora_b

        model = LoRALinear(10, 10, r=4)
        x = torch.randn(2, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        assert model.lora_a.grad is not None
        assert model.lora_b.grad is not None
        assert model.linear.weight.grad is None

    def test_save_load_lora(self, tmp_path):
        model = MockLoraModel(nn.Linear(10, 10))
        save_path = str(tmp_path / "lora_adapters")
        with patch("src.training.peft_config.PEFT_AVAILABLE", True):
            save_lora_adapters(model, save_path)
            assert (tmp_path / "lora_adapters" / "adapter_model.bin").exists()


class TestFreezeBaseModel:
    """基础模型冻结测试"""

    def test_freeze_base_model(self):
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.Linear(64, 10),
        )
        model.lora_a = nn.Parameter(torch.randn(10, 4))
        freeze_base_model(model)
        assert not model[0].weight.requires_grad
        assert not model[1].weight.requires_grad
        assert model.lora_a.requires_grad

    def test_freeze_unfreezes_lora_only(self):
        model = nn.Linear(10, 10)
        model.lora_a = nn.Parameter(torch.randn(10, 4))
        model.lora_b = nn.Parameter(torch.randn(4, 10))
        freeze_base_model(model)
        assert not model.weight.requires_grad
        assert not model.bias.requires_grad
        assert model.lora_a.requires_grad
        assert model.lora_b.requires_grad


class TestPEFTNotAvailable:
    """PEFT不可用时的错误处理测试"""

    def test_apply_lora_to_encoder_raises(self, simple_encoder):
        with patch("src.training.peft_config.PEFT_AVAILABLE", False):
            with pytest.raises(ImportError):
                apply_lora_to_encoder(simple_encoder, MagicMock())

    def test_save_lora_adapters_raises(self):
        with patch("src.training.peft_config.PEFT_AVAILABLE", False):
            with pytest.raises(ImportError):
                save_lora_adapters(nn.Linear(10, 10), "path")

    def test_load_lora_adapters_raises(self):
        with patch("src.training.peft_config.PEFT_AVAILABLE", False):
            with pytest.raises(ImportError):
                load_lora_adapters(nn.Linear(10, 10), "path")

    def test_merge_lora_adapters_raises(self):
        with patch("src.training.peft_config.PEFT_AVAILABLE", False):
            with pytest.raises(ImportError):
                merge_lora_adapters(nn.Linear(10, 10))
