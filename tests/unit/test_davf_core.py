"""Tests for src/models/davf.py — DAVF (Direction-Aware Velocity Field) model.

Tests DAVFConfig validation, DAVF instantiation, forward_flow_matching,
predict, DAVFLoss, and DirectionConsistencyLoss with small tensor sizes.
"""

import pytest
import torch

from src.models.davf import (
    DAVFConfig,
    DAVF,
    DAVFLoss,
    DirectionConsistencyLoss,
    TimeEncoder,
    GeneSpecificModulation,
)


# ---------------------------------------------------------------------------
# DAVFConfig validation tests
# ---------------------------------------------------------------------------


class TestDAVFConfig:
    """Tests for DAVFConfig dataclass validation."""

    def test_default_config_instantiation(self):
        """Default config is valid and creates without error."""
        config = DAVFConfig()
        assert config.gene_embed_dim == 192
        assert config.num_directions == 3
        assert config.hidden_dim == 256
        assert config.num_genes == 5000
        assert config.condition_injection == "hybrid"

    def test_minimal_config(self):
        """Small config for fast testing."""
        config = DAVFConfig(
            gene_embed_dim=8,
            hidden_dim=8,
            num_heads=1,
            num_genes=10,
            direction_embed_dim=8,
            time_embed_dim=8,
            x_encoder_hidden=16,
            velocity_hidden=16,
            modulation_dim=8,
            num_kv_heads=1,
        )
        assert config.gene_embed_dim == 8
        assert config.num_genes == 10

    def test_invalid_hidden_dim_raises(self):
        """Negative hidden_dim raises ValueError."""
        with pytest.raises(ValueError, match="hidden_dim"):
            DAVFConfig(hidden_dim=0)

    def test_invalid_num_heads_raises(self):
        """Zero num_heads raises ValueError."""
        with pytest.raises(ValueError, match="num_heads"):
            DAVFConfig(num_heads=0)

    def test_undivisible_hidden_dim_raises(self):
        """hidden_dim not divisible by num_heads raises ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            DAVFConfig(hidden_dim=7, num_heads=2)

    def test_undivisible_gene_embed_dim_raises(self):
        """gene_embed_dim not divisible by num_heads raises ValueError."""
        with pytest.raises(ValueError, match="gene_embed_dim"):
            DAVFConfig(gene_embed_dim=7, num_heads=2)

    def test_invalid_num_genes_raises(self):
        """Zero num_genes raises ValueError."""
        with pytest.raises(ValueError, match="num_genes"):
            DAVFConfig(num_genes=0)

    def test_invalid_condition_injection_raises(self):
        """Invalid condition_injection raises ValueError."""
        with pytest.raises(ValueError, match="condition_injection"):
            DAVFConfig(condition_injection="invalid")

    def test_invalid_residual_gate_init_raises(self):
        """residual_gate_init outside (0,1) raises ValueError."""
        with pytest.raises(ValueError, match="residual_gate_init"):
            DAVFConfig(residual_gate_init=0.0)

    def test_to_biperturb_config(self):
        """to_biperturb_config converts correctly."""
        config = DAVFConfig()
        bp_config = config.to_biperturb_config()
        assert bp_config.gene_embed_dim == config.gene_embed_dim
        assert bp_config.num_directions == config.num_directions
        assert bp_config.hidden_dim == config.hidden_dim


# ---------------------------------------------------------------------------
# TimeEncoder tests
# ---------------------------------------------------------------------------


class TestTimeEncoder:
    """Tests for TimeEncoder module."""

    def test_output_shape(self):
        """TimeEncoder produces [B, embed_dim] output."""
        encoder = TimeEncoder(embed_dim=16)
        t = torch.rand(4)
        out = encoder(t)
        assert out.shape == (4, 16)

    def test_single_time_step(self):
        """TimeEncoder handles batch size 1."""
        encoder = TimeEncoder(embed_dim=8)
        t = torch.tensor([0.5])
        out = encoder(t)
        assert out.shape == (1, 8)


# ---------------------------------------------------------------------------
# GeneSpecificModulation tests
# ---------------------------------------------------------------------------


class TestGeneSpecificModulation:
    """Tests for GeneSpecificModulation module."""

    def test_output_shape(self):
        """Modulation preserves [B, num_genes] shape."""
        mod = GeneSpecificModulation(
            num_genes=10,
            hidden_dim=8,
            modulation_dim=8,
            num_kv=2,
            num_heads=1,
        )
        condition = torch.randn(2, 8)
        velocity = torch.randn(2, 10)
        out = mod(condition, velocity)
        assert out.shape == (2, 10)


# ---------------------------------------------------------------------------
# DAVF model instantiation & forward tests (small config)
# ---------------------------------------------------------------------------


def _small_config(**overrides):
    """Create a minimal DAVFConfig for fast testing."""
    defaults = dict(
        gene_embed_dim=8,
        hidden_dim=8,
        num_heads=1,
        num_genes=10,
        direction_embed_dim=8,
        time_embed_dim=8,
        x_encoder_hidden=16,
        velocity_hidden=16,
        modulation_dim=8,
        num_kv_heads=1,
        num_velocity_layers=2,
        condition_projection_depth=1,
        condition_injection="concat",
    )
    defaults.update(overrides)
    return DAVFConfig(**defaults)


class TestDAVFModel:
    """Tests for DAVF model instantiation and forward pass."""

    def test_instantiation_without_embedding_loader(self):
        """DAVF can be created without a GeneformerEmbeddingLoader."""
        config = _small_config(condition_injection="concat")
        model = DAVF(config)
        assert model is not None
        assert model.gene_embed_table is not None  # learnable table

    def test_forward_flow_matching_output_shapes(self):
        """forward_flow_matching returns dict with expected keys and shapes."""
        config = _small_config(condition_injection="concat")
        model = DAVF(config)
        model.eval()

        B = 2
        x_0 = torch.randn(B, config.num_genes)
        x_1 = torch.randn(B, config.num_genes)
        gene_ids = torch.randint(0, config.num_genes, (B, 3))
        directions = torch.randint(0, 3, (B, 3))

        result = model.forward_flow_matching(x_0, x_1, gene_ids, directions)

        assert "x_t" in result
        assert "v_t" in result
        assert "u_t" in result
        assert "loss" in result
        assert result["x_t"].shape == (B, config.num_genes)
        assert result["v_t"].shape == (B, config.num_genes)
        assert result["u_t"].shape == (B, config.num_genes)
        assert result["loss"].dim() == 0  # scalar

    def test_forward_flow_matching_with_explicit_t(self):
        """forward_flow_matching works with explicit time steps."""
        config = _small_config(condition_injection="concat")
        model = DAVF(config)
        model.eval()

        B = 2
        t = torch.tensor([0.3, 0.7])
        result = model.forward_flow_matching(
            torch.randn(B, config.num_genes),
            torch.randn(B, config.num_genes),
            torch.randint(0, config.num_genes, (B, 3)),
            torch.randint(0, 3, (B, 3)),
            t=t,
        )
        assert torch.allclose(result["t"], t, atol=1e-6)

    def test_forward_flow_matching_shape_mismatch_raises(self):
        """Mismatched x_0 and x_1 shapes raise ValueError."""
        config = _small_config(condition_injection="concat")
        model = DAVF(config)
        with pytest.raises(ValueError, match="identical shape"):
            model.forward_flow_matching(
                torch.randn(2, 10),
                torch.randn(3, 10),
            )

    def test_predict_output_shape(self):
        """predict returns [B, num_genes] tensor."""
        config = _small_config(condition_injection="concat")
        model = DAVF(config)
        model.eval()

        B = 2
        x_0 = torch.randn(B, config.num_genes)
        gene_ids = torch.randint(0, config.num_genes, (B, 3))
        directions = torch.randint(0, 3, (B, 3))

        with torch.no_grad():
            x_1_pred = model.predict(x_0, gene_ids, directions, num_steps=3)
        assert x_1_pred.shape == (B, config.num_genes)

    def test_predict_with_external_condition(self):
        """predict works with external_condition embedding."""
        config = _small_config(condition_injection="concat")
        model = DAVF(config)
        model.eval()

        B = 2
        x_0 = torch.randn(B, config.num_genes)
        ext_cond = torch.randn(B, config.hidden_dim)

        with torch.no_grad():
            x_1_pred = model.predict(
                x_0,
                condition_source="external_embedding",
                external_condition=ext_cond,
                num_steps=3,
            )
        assert x_1_pred.shape == (B, config.num_genes)

    def test_negative_gene_ids_raise(self):
        """Negative gene_ids raise ValueError."""
        config = _small_config(condition_injection="concat")
        model = DAVF(config)
        gene_ids = torch.tensor([[0, -1, 2]])
        with pytest.raises(ValueError, match="non-negative"):
            model._get_gene_embeddings(gene_ids)

    def test_predict_invalid_num_steps_raises(self):
        """Non-positive num_steps raises ValueError."""
        config = _small_config(condition_injection="concat")
        model = DAVF(config)
        model.eval()
        with pytest.raises(ValueError, match="num_steps"):
            model.predict(
                torch.randn(1, config.num_genes),
                torch.randint(0, config.num_genes, (1, 3)),
                torch.randint(0, 3, (1, 3)),
                num_steps=0,
            )


# ---------------------------------------------------------------------------
# DAVFLoss tests
# ---------------------------------------------------------------------------


class TestDAVFLoss:
    """Tests for DAVFLoss module."""

    def test_loss_output_keys(self):
        """DAVFLoss returns mse, mag, total keys."""
        loss_fn = DAVFLoss()
        v_t = torch.randn(4, 10)
        u_t = torch.randn(4, 10)
        result = loss_fn(v_t, u_t)
        assert "mse" in result
        assert "mag" in result
        assert "total" in result

    def test_zero_loss_on_identical_tensors(self):
        """MSE component is zero when predicted equals target."""
        loss_fn = DAVFLoss(mag_weight=0.0)
        v = torch.randn(4, 10)
        result = loss_fn(v, v)
        assert result["mse"].item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# DirectionConsistencyLoss tests
# ---------------------------------------------------------------------------


class TestDirectionConsistencyLoss:
    """Tests for DirectionConsistencyLoss module."""

    def test_output_is_scalar(self):
        """Loss returns a scalar tensor."""
        loss_fn = DirectionConsistencyLoss()
        v_t = torch.randn(2, 10)
        u_t = torch.randn(2, 10)
        gene_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
        directions = torch.tensor([[0, 1, 2], [2, 0, 1]])
        loss = loss_fn(v_t, u_t, gene_ids, directions)
        assert loss.dim() == 0

    def test_with_attention_mask(self):
        """Loss works with attention_mask to ignore padding."""
        loss_fn = DirectionConsistencyLoss()
        v_t = torch.randn(2, 10)
        u_t = torch.randn(2, 10)
        gene_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
        directions = torch.tensor([[0, 1, 2], [2, 0, 1]])
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float)
        loss = loss_fn(v_t, u_t, gene_ids, directions, attention_mask=mask)
        assert loss.dim() == 0
