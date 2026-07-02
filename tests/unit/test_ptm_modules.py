"""Focused PTM module tests for fusion-type exposure."""

import torch

from src.models.architectures import PTM2CellNet
from src.models.ptm_modules import GatedPTMFusion, PTMModule


def test_ptm_module_gated_fusion_layers_and_forward():
    sequence_emb = torch.randn(2, 5, 16)
    ptm_types = torch.tensor([[1, 2, 0, 0, 0], [3, 4, 5, 0, 0]], dtype=torch.long)
    ptm_positions = torch.arange(5).unsqueeze(0).repeat(2, 1)
    ptm_mask = (ptm_types > 0).float()

    module = PTMModule(
        num_ptm_types=6,
        embed_dim=16,
        num_layers=2,
        fusion_type="gated",
    )
    output = module(sequence_emb, ptm_types, ptm_positions, ptm_mask)

    assert output.shape == sequence_emb.shape
    assert all(isinstance(layer, GatedPTMFusion) for layer in module.layers)


def test_ptm2cellnet_passes_gated_fusion_type_to_ptm_module():
    model = PTM2CellNet(
        encoder_type="cnn",
        vocab_size=20,
        embed_dim=16,
        max_seq_len=16,
        num_ptm_types=6,
        num_classes=3,
        num_layers=1,
        num_heads=2,
        ptm_fusion_type="gated",
    )

    assert all(isinstance(layer, GatedPTMFusion) for layer in model.ptm_module.layers)
