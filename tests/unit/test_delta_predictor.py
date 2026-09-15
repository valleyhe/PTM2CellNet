"""Unit tests for src/models/delta_predictor.py (V22-01-DELTA-PREDICTOR)."""

import pytest
import torch

from src.models.delta_predictor import DeltaPredictor
from src.models.latent_davf import LatentDAVF, LatentDAVFConfig


@pytest.fixture
def config():
    return LatentDAVFConfig(latent_dim=8, num_genes=64, hidden_dim=64)


@pytest.fixture
def model(config):
    return DeltaPredictor(config)


@pytest.fixture
def inputs():
    torch.manual_seed(0)
    B, K = 3, 5
    z_0 = torch.randn(B, 8)
    gene_ids = torch.randint(0, 64, (B, K))
    directions = torch.randint(0, 3, (B, K))
    attention_mask = torch.ones(B, K)
    return z_0, gene_ids, directions, attention_mask


class TestDeltaPredictorConstruction:
    def test_is_latentdavf_subclass(self, model):
        assert isinstance(model, LatentDAVF)

    def test_has_delta_mlp(self, model):
        assert hasattr(model, "delta_mlp")
        # delta_mlp.* keys exist for checkpoint-format detection
        keys = [n for n, _ in model.named_parameters() if "delta_mlp" in n]
        assert len(keys) > 0

    def test_uses_delta_mlp_keys_for_detection(self, model):
        state_keys = set(model.state_dict().keys())
        assert any("delta_mlp" in k for k in state_keys)


class TestDeltaPredictorPredict:
    def test_predict_shape(self, model, inputs):
        z_0, gene_ids, directions, mask = inputs
        z_1 = model.predict(z_0, gene_ids=gene_ids, directions=directions, attention_mask=mask)
        assert z_1.shape == z_0.shape

    def test_predict_delta_equals_predict_minus_z0(self, model, inputs):
        z_0, gene_ids, directions, mask = inputs
        z_1 = model.predict(z_0, gene_ids=gene_ids, directions=directions, attention_mask=mask)
        delta = model.predict_delta(z_0, gene_ids=gene_ids, directions=directions, attention_mask=mask)
        assert torch.allclose(z_1, z_0 + delta, atol=1e-5)

    def test_predict_rejects_wrong_latent_dim(self, model):
        z_0 = torch.randn(2, 16)  # wrong latent_dim
        with pytest.raises(ValueError, match="latent_dim mismatch"):
            model.predict(z_0)

    def test_predict_rejects_non_2d(self, model):
        z_0 = torch.randn(2, 8, 1)
        with pytest.raises(ValueError, match="must be 2D"):
            model.predict(z_0)

    def test_predict_sets_eval_mode_then_restores(self, model, inputs):
        z_0, gene_ids, directions, mask = inputs
        model.train()
        _ = model.predict(z_0, gene_ids=gene_ids, directions=directions, attention_mask=mask)
        assert model.training  # restored


class TestDeltaPredictorForward:
    def test_forward_supervised_returns_loss(self, model, inputs):
        z_0, gene_ids, directions, mask = inputs
        z_1 = torch.randn_like(z_0)
        out = model.forward(
            z_0,
            z_1=z_1,
            gene_ids=gene_ids,
            directions=directions,
            attention_mask=mask,
        )
        assert "loss" in out
        assert "delta_pred" in out
        assert "delta_target" in out
        assert out["loss"].dim() == 0  # scalar

    def test_forward_inference_returns_z1_pred(self, model, inputs):
        z_0, gene_ids, directions, mask = inputs
        out = model.forward(z_0, gene_ids=gene_ids, directions=directions, attention_mask=mask)
        assert "z_1_pred" in out
        assert out["z_1_pred"].shape == z_0.shape

    def test_forward_loss_is_differentiable(self, model, inputs):
        z_0, gene_ids, directions, mask = inputs
        z_1 = torch.randn_like(z_0)
        out = model.forward(
            z_0,
            z_1=z_1,
            gene_ids=gene_ids,
            directions=directions,
            attention_mask=mask,
        )
        out["loss"].backward()
        # delta_mlp weights should have gradients
        grads = [p.grad for p in model.delta_mlp.parameters()]
        assert all(g is not None for g in grads)


class TestDeltaPredictorCheckpointCompat:
    def test_load_state_dict_roundtrip(self, model, inputs):
        """State dict saves/loads with strict=True (no missing keys)."""
        sd = model.state_dict()
        model2 = DeltaPredictor(model.config)
        missing, unexpected = model2.load_state_dict(sd, strict=False)
        # No DeltaPredictor-specific keys should be missing/unexpected
        assert not any("delta_mlp" in k for k in missing)
        assert not any("delta_mlp" in k for k in unexpected)
