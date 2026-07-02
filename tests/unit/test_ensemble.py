"""Unit tests for src/models/ensemble.py (PTM2CellNetEnsemble)."""

import pytest
import torch
from torch import nn

from src.models.ensemble import PTM2CellNetEnsemble


class _DummyModel(nn.Module):
    """Mock sub-model returning fixed logits for deterministic aggregation."""

    def __init__(self, logits):
        super().__init__()
        self._logits = logits  # tensor [B, C]
        # A parameter so count_parameters works
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        return {"logits": self._logits}


@pytest.fixture
def dummy_models():
    torch.manual_seed(0)
    # Two models with distinct logits for 3 samples, 4 classes
    return [
        _DummyModel(torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])),
        _DummyModel(torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])),
    ]


class TestEnsembleConstruction:
    def test_mean_aggregation_default(self, dummy_models):
        ens = PTM2CellNetEnsemble(dummy_models)
        assert ens.aggregation == "mean"
        assert ens.num_models == 2

    def test_empty_models_raises(self):
        with pytest.raises(ValueError, match="不能为空"):
            PTM2CellNetEnsemble([])

    def test_invalid_aggregation_raises(self, dummy_models):
        with pytest.raises(ValueError, match="aggregation"):
            PTM2CellNetEnsemble(dummy_models, aggregation="bogus")

    def test_weighted_requires_weights(self, dummy_models):
        with pytest.raises(ValueError, match="weights"):
            PTM2CellNetEnsemble(dummy_models, aggregation="weighted")

    def test_weights_length_mismatch_raises(self, dummy_models):
        with pytest.raises(ValueError, match="weights 长度"):
            PTM2CellNetEnsemble(dummy_models, aggregation="weighted",
                                weights=[0.5, 0.3, 0.2])


class TestEnsembleForward:
    def test_mean_aggregation(self, dummy_models):
        ens = PTM2CellNetEnsemble(dummy_models, aggregation="mean")
        out = ens({})
        # mean of the two models' logits
        expected_logits = (dummy_models[0]._logits + dummy_models[1]._logits) / 2
        assert torch.allclose(out["logits"], expected_logits)
        assert out["logits"].shape == (3, 4)
        # predictions = argmax
        assert torch.equal(out["predictions"], out["logits"].argmax(dim=-1))
        # probabilities sum to 1
        assert torch.allclose(out["probabilities"].sum(dim=-1),
                              torch.ones(3), atol=1e-5)
        assert len(out["individual_predictions"]) == 2

    def test_voting_aggregation(self, dummy_models):
        ens = PTM2CellNetEnsemble(dummy_models, aggregation="voting")
        out = ens({})
        # Both models agree on sample 0 (class 0) and sample 2 (class 3)
        assert out["predictions"][0].item() == 0
        assert out["predictions"][2].item() == 3
        # Sample 1: models disagree (class1 vs class2) -> tie broken by argmax
        assert out["predictions"][1].item() in (1, 2)

    def test_weighted_aggregation(self, dummy_models):
        ens = PTM2CellNetEnsemble(
            dummy_models, aggregation="weighted", weights=[0.9, 0.1]
        )
        out = ens({})
        expected = 0.9 * dummy_models[0]._logits + 0.1 * dummy_models[1]._logits
        assert torch.allclose(out["logits"], expected)

    def test_weights_move_with_device(self, dummy_models):
        ens = PTM2CellNetEnsemble(dummy_models, aggregation="weighted",
                                  weights=[0.7, 0.3])
        # _weights is a buffer -> accessible
        assert ens._weights.shape == (2,)
        assert torch.allclose(ens._weights, torch.tensor([0.7, 0.3]))


class TestEnsembleModelInfo:
    def test_get_model_info(self, dummy_models):
        ens = PTM2CellNetEnsemble(dummy_models, aggregation="mean")
        info = ens.get_model_info()
        assert info["model_type"] == "PTM2CellNetEnsemble"
        assert info["aggregation"] == "mean"
        assert info["num_models"] == 2
        assert len(info["models"]) == 2
        assert "total_params" in info and "trainable_params" in info
