"""
Mamba编码器单元测试
"""

import pytest
import torch

from src.models.mamba_encoder import SelectiveSSM, MambaBlock, MambaEncoder


class TestSelectiveSSM:
    """测试SelectiveSSM模块"""

    def test_ssm_initialization(self):
        """测试SSM初始化"""
        d_model = 128
        d_state = 16
        ssm = SelectiveSSM(d_model, d_state)

        # 检查参数形状
        assert ssm.A.shape == (d_model * 2, d_state)
        assert ssm.D.shape == (d_model * 2,)

        # 检查A矩阵为负值（HiPPO初始化）
        assert (ssm.A <= 0).all()

    def test_ssm_forward(self):
        """测试SSM前向传播"""
        batch_size = 2
        seq_len = 10
        d_model = 128
        d_state = 16

        ssm = SelectiveSSM(d_model, d_state)
        x = torch.randn(batch_size, seq_len, d_model)

        output = ssm(x)

        # 检查输出形状
        assert output.shape == (batch_size, seq_len, d_model)

        # 检查无NaN/Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_ssm_gradient(self):
        """测试SSM梯度"""
        batch_size = 2
        seq_len = 10
        d_model = 128

        ssm = SelectiveSSM(d_model)
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        output = ssm(x)
        loss = output.sum()
        loss.backward()

        # 检查梯度存在
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_ssm_different_lengths(self):
        """测试不同序列长度"""
        d_model = 128
        ssm = SelectiveSSM(d_model)

        for seq_len in [10, 50, 100, 500]:
            x = torch.randn(2, seq_len, d_model)
            output = ssm(x)
            assert output.shape == (2, seq_len, d_model)


class TestMambaBlock:
    """测试MambaBlock模块"""

    def test_block_initialization(self):
        """测试MambaBlock初始化"""
        d_model = 128
        block = MambaBlock(d_model)

        assert isinstance(block.norm, torch.nn.LayerNorm)
        assert isinstance(block.ssm, SelectiveSSM)

    def test_block_forward(self):
        """测试MambaBlock前向传播"""
        batch_size = 2
        seq_len = 10
        d_model = 128

        block = MambaBlock(d_model)
        x = torch.randn(batch_size, seq_len, d_model)

        output = block(x)

        # 检查输出形状
        assert output.shape == (batch_size, seq_len, d_model)

        # 检查无NaN/Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_block_residual(self):
        """测试残差连接"""
        batch_size = 2
        seq_len = 10
        d_model = 128

        block = MambaBlock(d_model)
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        # 残差连接应该保持梯度流
        output = block(x)
        loss = output.sum()
        loss.backward()

        # 输入应该有梯度
        assert x.grad is not None

    def test_block_gradient_flow(self):
        """测试梯度流"""
        d_model = 128
        num_layers = 12

        # 堆叠多个block
        blocks = torch.nn.Sequential(*[
            MambaBlock(d_model) for _ in range(num_layers)
        ])

        x = torch.randn(2, 10, d_model, requires_grad=True)
        output = blocks(x)
        loss = output.sum()
        loss.backward()

        # 检查梯度正常
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert (x.grad.abs() > 0).any()  # 梯度不全为0


class TestMambaEncoder:
    """测试MambaEncoder"""

    def test_encoder_initialization(self):
        """测试编码器初始化"""
        vocab_size = 20
        hidden_dim = 128
        num_layers = 6

        encoder = MambaEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        assert len(encoder.layers) == num_layers
        assert encoder.hidden_dim == hidden_dim

    def test_encoder_forward(self):
        """测试编码器前向传播"""
        batch_size = 2
        seq_len = 10
        vocab_size = 20
        hidden_dim = 128

        encoder = MambaEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=6,
        )

        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = encoder(x)

        # 检查输出形状
        assert output.shape == (batch_size, seq_len, hidden_dim)

        # 检查无NaN/Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_encoder_with_mask(self):
        """测试mask机制"""
        batch_size = 2
        seq_len = 10
        vocab_size = 20
        hidden_dim = 128

        encoder = MambaEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=6,
        )

        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len)
        mask[:, 5:] = 0  # 后半部分为padding

        output = encoder(x, mask=mask)

        # 检查padding位置输出为0
        assert (output[:, 5:] == 0).all()

        # 检查有效位置输出不为0
        assert (output[:, :5].abs() > 0).any()

    def test_encoder_gradient_flow(self):
        """测试梯度流"""
        batch_size = 2
        seq_len = 10
        vocab_size = 20
        hidden_dim = 128

        encoder = MambaEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=12,
        )

        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = encoder(x)
        loss = output.sum()
        loss.backward()

        # 检查embedding有梯度
        assert encoder.embedding.weight.grad is not None
        assert not torch.isnan(encoder.embedding.weight.grad).any()

    def test_encoder_different_configs(self):
        """测试不同配置"""
        configs = [
            {"hidden_dim": 256, "num_layers": 6, "state_dim": 16},  # Small
            {"hidden_dim": 768, "num_layers": 12, "state_dim": 16},  # Medium
            {"hidden_dim": 1024, "num_layers": 24, "state_dim": 32},  # Large
        ]

        for config in configs:
            encoder = MambaEncoder(vocab_size=20, **config)
            x = torch.randint(0, 20, (2, 10))
            output = encoder(x)
            assert output.shape == (2, 10, config["hidden_dim"])

    def test_encoder_long_sequence(self):
        """测试长序列"""
        encoder = MambaEncoder(
            vocab_size=20,
            hidden_dim=256,
            num_layers=6,
            max_len=2048,
        )

        # 测试不同长度
        for seq_len in [100, 500, 1000]:
            x = torch.randint(0, 20, (1, seq_len))
            output = encoder(x)
            assert output.shape == (1, seq_len, 256)
            assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
