"""
GatedPTMFusion模块单元测试
"""

import pytest
import torch
from src.models.ptm_modules import GatedPTMFusion, PTMModule


class TestGatedPTMFusion:
    """GatedPTMFusion单元测试"""

    @pytest.fixture
    def sample_data(self):
        """测试数据fixture"""
        batch_size, seq_len, embed_dim = 4, 50, 128
        sequence_emb = torch.randn(batch_size, seq_len, embed_dim)
        ptm_emb = torch.randn(batch_size, seq_len, embed_dim)
        ptm_mask = torch.randint(0, 2, (batch_size, seq_len)).float()
        return sequence_emb, ptm_emb, ptm_mask, embed_dim

    def test_gated_ptm_fusion_init(self, sample_data):
        """测试GatedPTMFusion初始化"""
        _, _, _, embed_dim = sample_data
        module = GatedPTMFusion(embed_dim=embed_dim, dropout=0.1)

        assert isinstance(module.gate_proj, torch.nn.Linear)
        assert module.gate_proj.in_features == embed_dim * 2
        assert module.gate_proj.out_features == embed_dim
        assert isinstance(module.dropout, torch.nn.Dropout)
        assert isinstance(module.layer_norm, torch.nn.LayerNorm)
        assert module.use_residual is True

    def test_gated_ptm_fusion_forward(self, sample_data):
        """测试GatedPTMFusion前向传播"""
        sequence_emb, ptm_emb, _, embed_dim = sample_data
        module = GatedPTMFusion(embed_dim=embed_dim)

        output = module(sequence_emb, ptm_emb)

        assert output.shape == sequence_emb.shape
        assert output.shape == (4, 50, 128)

    def test_gated_ptm_fusion_with_mask(self, sample_data):
        """测试带mask的前向传播"""
        sequence_emb, ptm_emb, ptm_mask, embed_dim = sample_data
        module = GatedPTMFusion(embed_dim=embed_dim)

        output = module(sequence_emb, ptm_emb, ptm_mask)

        assert output.shape == sequence_emb.shape

    def test_gated_ptm_fusion_without_residual(self, sample_data):
        """测试无残差连接模式"""
        sequence_emb, ptm_emb, _, embed_dim = sample_data
        module = GatedPTMFusion(embed_dim=embed_dim, use_residual=False)

        output = module(sequence_emb, ptm_emb)

        assert output.shape == sequence_emb.shape
        assert module.use_residual is False

    def test_gate_values_range(self, sample_data):
        """测试门控值范围在[0,1]之间"""
        sequence_emb, ptm_emb, _, embed_dim = sample_data
        module = GatedPTMFusion(embed_dim=embed_dim)

        with torch.no_grad():
            combined = torch.cat([sequence_emb, ptm_emb], dim=-1)
            gate = torch.sigmoid(module.gate_proj(combined))

        assert (gate >= 0).all()
        assert (gate <= 1).all()

    def test_ptm_module_with_gated_fusion(self, sample_data):
        """测试PTMModule使用gated融合类型"""
        sequence_emb, _, ptm_mask, embed_dim = sample_data

        ptm_module = PTMModule(num_ptm_types=10, embed_dim=embed_dim, num_layers=2, fusion_type="gated")

        # 生成有效的ptm_types (0-9，其中0表示无PTM)
        ptm_types = torch.randint(0, 10, (sequence_emb.size(0), sequence_emb.size(1)))
        ptm_positions = torch.arange(sequence_emb.size(1)).unsqueeze(0).repeat(sequence_emb.size(0), 1)
        output = ptm_module(sequence_emb, ptm_types, ptm_positions, ptm_mask)

        assert output.shape == sequence_emb.shape

    def test_backward_pass(self, sample_data):
        """测试反向传播"""
        sequence_emb, ptm_emb, ptm_mask, embed_dim = sample_data
        module = GatedPTMFusion(embed_dim=embed_dim)

        sequence_emb.requires_grad = True
        output = module(sequence_emb, ptm_emb, ptm_mask)
        loss = output.sum()
        loss.backward()

        assert sequence_emb.grad is not None
        assert not torch.isnan(sequence_emb.grad).any()

    def test_ptm2cellnet_with_gated_fusion(self):
        """测试PTM2CellNet使用gated融合"""
        from src.models.architectures import PTM2CellNet

        model = PTM2CellNet(encoder_type="cnn", num_classes=4, ptm_fusion_type="gated")

        batch = {
            "sequence": torch.randint(0, 20, (2, 50)),
            "ptm_types": torch.randint(0, 10, (2, 50)),
            "ptm_mask": torch.ones(2, 50),
        }

        output = model(batch)

        assert "logits" in output
        assert output["logits"].shape == (2, 4)
