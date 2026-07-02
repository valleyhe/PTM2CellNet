"""Unit tests for masked PTM self-supervised pretraining."""

import torch
from torch.utils.data import DataLoader

from src.training.self_supervised import MaskedPTMPrediction, pretrain_masked_ptm


class TestMaskedPTMPrediction:
    def test_forward_returns_logits_and_predictions(self):
        model = MaskedPTMPrediction(num_ptm_types=6, embed_dim=8, max_position=32)
        masked_ptm_types = torch.tensor([[1, 6, 2, 0], [3, 4, 6, 0]], dtype=torch.long)
        ptm_positions = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
        ptm_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.float32)

        output = model(masked_ptm_types, ptm_positions, ptm_mask)

        assert output["logits"].shape == (2, 4, 6)
        assert output["predictions"].shape == (2, 4)
        assert output["predictions"].dtype == torch.long

    def test_pretrain_masked_ptm_runs_and_returns_epoch_losses(self):
        torch.manual_seed(0)
        model = MaskedPTMPrediction(num_ptm_types=5, embed_dim=12, max_position=16, mask_probability=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        dataset = [
            {
                "ptm_types": torch.tensor([1, 2, 3, 0], dtype=torch.long),
                "ptm_positions": torch.tensor([0, 1, 2, 3], dtype=torch.long),
                "ptm_mask": torch.tensor([1, 1, 1, 0], dtype=torch.float32),
            },
            {
                "ptm_types": torch.tensor([2, 3, 4, 0], dtype=torch.long),
                "ptm_positions": torch.tensor([0, 1, 2, 3], dtype=torch.long),
                "ptm_mask": torch.tensor([1, 1, 1, 0], dtype=torch.float32),
            },
        ]
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        losses = pretrain_masked_ptm(model, dataloader, optimizer, device="cpu", epochs=2)

        assert len(losses) == 2
        assert all(loss >= 0.0 for loss in losses)
