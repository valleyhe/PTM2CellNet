"""
Mamba集成测试
测试MambaEncoder与PTM2CellNet系统的集成
"""

import pytest
import torch

from src.models.architectures import PTM2CellNet
from src.models.mamba_encoder import MambaEncoder


class TestMambaIntegration:
    """测试Mamba与系统集成"""

    def test_mamba_encoder_creation(self):
        """测试通过配置创建Mamba编码器"""
        model = PTM2CellNet(
            encoder_type="mamba",
            vocab_size=20,
            embed_dim=256,
            num_layers=6,
            max_seq_len=100,
            num_ptm_types=10,
            num_classes=4,
        )

        assert isinstance(model.encoder, MambaEncoder)
        assert model.encoder.hidden_dim == 256

    def test_mamba_end_to_end(self):
        """测试端到端前向传播"""
        batch_size = 2
        seq_len = 50
        vocab_size = 20

        model = PTM2CellNet(
            encoder_type="mamba",
            vocab_size=vocab_size,
            embed_dim=256,
            num_layers=6,
            max_seq_len=100,
            num_ptm_types=10,
            num_classes=4,
        )

        # 准备输入
        batch = {
            "sequence": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "ptm_types": torch.randint(0, 10, (batch_size, seq_len)),
            "ptm_mask": torch.ones(batch_size, seq_len),
        }

        # 前向传播
        output = model(batch)

        # 检查输出
        assert "logits" in output
        assert output["logits"].shape == (batch_size, 4)
        assert not torch.isnan(output["logits"]).any()

    def test_mamba_with_ptm_module(self):
        """测试Mamba与PTM模块集成"""
        batch_size = 2
        seq_len = 50

        model = PTM2CellNet(
            encoder_type="mamba",
            vocab_size=20,
            embed_dim=256,
            num_layers=6,
            max_seq_len=100,
            num_ptm_types=10,
            num_classes=4,
        )

        batch = {
            "sequence": torch.randint(0, 20, (batch_size, seq_len)),
            "ptm_types": torch.randint(0, 10, (batch_size, seq_len)),
        }

        output = model(batch)

        assert "logits" in output
        assert output["logits"].shape == (batch_size, 4)

    def test_mamba_gradient_flow(self):
        """测试梯度流"""
        model = PTM2CellNet(
            encoder_type="mamba",
            vocab_size=20,
            embed_dim=256,
            num_layers=6,
            max_seq_len=100,
            num_ptm_types=10,
            num_classes=4,
        )

        batch = {
            "sequence": torch.randint(0, 20, (2, 50)),
            "ptm_types": torch.randint(0, 10, (2, 50)),
        }

        output = model(batch)
        loss = output["logits"].sum()
        loss.backward()

        # 检查编码器有梯度
        assert model.encoder.embedding.weight.grad is not None
        assert not torch.isnan(model.encoder.embedding.weight.grad).any()

    def test_mamba_config_loading(self):
        """测试从配置文件加载"""
        config = {
            "model": {
                "encoder_type": "mamba",
                "hidden_dim": 256,
                "num_layers": 6,
                "num_classes": 4,
                "num_ptm_types": 10,
                "dropout": 0.1,
            },
            "data": {
                "max_sequence_length": 100,
                "valid_amino_acids": "ACDEFGHIKLMNPQRSTVWY",
                "ptm_types": ["Phosphorylation"] * 10,
            },
        }

        model = PTM2CellNet.from_config(config)

        assert isinstance(model.encoder, MambaEncoder)
        assert model.encoder.hidden_dim == 256

    def test_mamba_different_configs(self):
        """测试不同配置"""
        configs = [
            {"embed_dim": 256, "num_layers": 6},  # Small
            {"embed_dim": 768, "num_layers": 12},  # Medium
            {"embed_dim": 1024, "num_layers": 24},  # Large
        ]

        for config in configs:
            model = PTM2CellNet(
                encoder_type="mamba",
                vocab_size=20,
                max_seq_len=100,
                num_ptm_types=10,
                num_classes=4,
                **config,
            )

            batch = {
                "sequence": torch.randint(0, 20, (2, 50)),
            }

            output = model(batch)
            assert output["logits"].shape == (2, 4)

    def test_mamba_vs_transformer_interface(self):
        """测试Mamba与Transformer接口一致性"""
        batch = {
            "sequence": torch.randint(0, 20, (2, 50)),
            "ptm_types": torch.randint(0, 10, (2, 50)),
        }

        # Transformer模型
        model_transformer = PTM2CellNet(
            encoder_type="transformer",
            vocab_size=20,
            embed_dim=256,
            num_layers=6,
            num_heads=4,
            max_seq_len=100,
            num_ptm_types=10,
            num_classes=4,
        )

        # Mamba模型
        model_mamba = PTM2CellNet(
            encoder_type="mamba",
            vocab_size=20,
            embed_dim=256,
            num_layers=6,
            max_seq_len=100,
            num_ptm_types=10,
            num_classes=4,
        )

        # 两者输出形状应该一致
        output_transformer = model_transformer(batch)
        output_mamba = model_mamba(batch)

        assert output_transformer["logits"].shape == output_mamba["logits"].shape

    def test_mamba_model_save_load(self):
        """测试模型保存和加载"""
        import tempfile
        import os

        model = PTM2CellNet(
            encoder_type="mamba",
            vocab_size=20,
            embed_dim=256,
            num_layers=6,
            max_seq_len=100,
            num_ptm_types=10,
            num_classes=4,
        )

        # 保存模型
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.pt")
            torch.save(model.state_dict(), save_path)

            # 加载模型
            model2 = PTM2CellNet(
                encoder_type="mamba",
                vocab_size=20,
                embed_dim=256,
                num_layers=6,
                max_seq_len=100,
                num_ptm_types=10,
                num_classes=4,
            )
            model2.load_state_dict(torch.load(save_path, weights_only=True))

            # 设置为eval模式以消除dropout影响
            model.eval()
            model2.eval()

            # 测试输出一致
            batch = {"sequence": torch.randint(0, 20, (2, 50))}

            with torch.no_grad():
                output1 = model(batch)
                output2 = model2(batch)

            assert torch.allclose(output1["logits"], output2["logits"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
