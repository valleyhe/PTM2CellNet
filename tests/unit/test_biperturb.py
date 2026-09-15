"""Tests for src/models/biperturb.py — BiPerturb model.

Tests BiPerturbConfig, DirectionEncoder, MagnitudeEncoder, BiPerturbEncoder,
PerturbationGNN, BiPerturb, and BiPerturbLoss with small tensor sizes.
"""

import pytest
import torch

from src.models.biperturb import (
    BiPerturbConfig,
    DirectionEncoder,
    MagnitudeEncoder,
    BiPerturbEncoder,
    PerturbationGNN,
    BiPerturb,
    BiPerturbLoss,
)


# ---------------------------------------------------------------------------
# BiPerturbConfig tests
# ---------------------------------------------------------------------------


class TestBiPerturbConfig:
    """Tests for BiPerturbConfig dataclass."""

    def test_default_config(self):
        """Default config creates with expected values."""
        config = BiPerturbConfig()
        assert config.gene_embed_dim == 192
        assert config.num_directions == 3
        assert config.hidden_dim == 256
        assert config.num_genes == 5000

    def test_custom_config(self):
        """Custom config values are stored correctly."""
        config = BiPerturbConfig(gene_embed_dim=8, hidden_dim=8, num_heads=1, num_genes=10)
        assert config.gene_embed_dim == 8
        assert config.num_genes == 10


# ---------------------------------------------------------------------------
# DirectionEncoder tests
# ---------------------------------------------------------------------------


class TestDirectionEncoder:
    """Tests for DirectionEncoder module."""

    def test_output_shape(self):
        """DirectionEncoder produces [B, K, embed_dim] output."""
        encoder = DirectionEncoder(num_directions=3, embed_dim=8, dropout=0.0)
        directions = torch.randint(0, 3, (2, 4))
        out = encoder(directions)
        assert out.shape == (2, 4, 8)

    def test_ko_embedding_negative_prior(self):
        """KO (index 0) embedding weight is initialized with negative values."""
        encoder = DirectionEncoder(num_directions=3, embed_dim=8)
        # KO weight should be -0.5 * ones before projection
        assert (encoder.embedding.weight[0] < 0).all()

    def test_oe_embedding_positive_prior(self):
        """OE (index 2) embedding weight is initialized with positive values."""
        encoder = DirectionEncoder(num_directions=3, embed_dim=8)
        assert (encoder.embedding.weight[2] > 0).all()


# ---------------------------------------------------------------------------
# MagnitudeEncoder tests
# ---------------------------------------------------------------------------


class TestMagnitudeEncoder:
    """Tests for MagnitudeEncoder module (backward compatibility)."""

    def test_output_shape(self):
        """MagnitudeEncoder produces [B, K, output_dim] output."""
        encoder = MagnitudeEncoder(output_dim=8, hidden_dim=4, dropout=0.0)
        magnitudes = torch.randn(2, 3)
        out = encoder(magnitudes)
        assert out.shape == (2, 3, 8)

    def test_1d_input_expansion(self):
        """2D input is automatically expanded to 3D."""
        encoder = MagnitudeEncoder(output_dim=8, hidden_dim=4, dropout=0.0)
        magnitudes = torch.randn(2, 3)  # [B, K]
        out = encoder(magnitudes)
        assert out.dim() == 3


# ---------------------------------------------------------------------------
# BiPerturbEncoder tests
# ---------------------------------------------------------------------------


def _small_encoder_config():
    """Minimal BiPerturbConfig for fast testing."""
    return BiPerturbConfig(
        gene_embed_dim=8,
        hidden_dim=8,
        num_heads=1,
        num_genes=10,
        direction_embed_dim=8,
        gnn_hidden_dim=8,
        gnn_num_layers=1,
        dropout=0.0,
        attention_dropout=0.0,
    )


class TestBiPerturbEncoder:
    """Tests for BiPerturbEncoder module."""

    def test_output_shape(self):
        """Encoder produces [B, hidden_dim] perturbation embedding."""
        config = _small_encoder_config()
        encoder = BiPerturbEncoder(config)
        B, K = 2, 3
        gene_emb = torch.randn(B, K, config.gene_embed_dim)
        directions = torch.randint(0, 3, (B, K))

        perturb_emb, attn_weights = encoder(gene_emb, directions)
        assert perturb_emb.shape == (B, config.hidden_dim)

    def test_return_attention_false(self):
        """When return_attention=False, attn_weights is None."""
        config = _small_encoder_config()
        encoder = BiPerturbEncoder(config)
        B, K = 2, 3
        gene_emb = torch.randn(B, K, config.gene_embed_dim)
        directions = torch.randint(0, 3, (B, K))

        _, attn_weights = encoder(gene_emb, directions, return_attention=False)
        assert attn_weights is None

    def test_disable_direction_ablation(self):
        """disable_direction=True skips direction encoding."""
        config = _small_encoder_config()
        encoder = BiPerturbEncoder(config)
        B, K = 2, 3
        gene_emb = torch.randn(B, K, config.gene_embed_dim)
        directions = torch.randint(0, 3, (B, K))

        perturb_emb, _ = encoder(gene_emb, directions, disable_direction=True)
        assert perturb_emb.shape == (B, config.hidden_dim)

    def test_with_attention_mask(self):
        """Encoder works with attention_mask for padded targets."""
        config = _small_encoder_config()
        encoder = BiPerturbEncoder(config)
        B, K = 2, 4
        gene_emb = torch.randn(B, K, config.gene_embed_dim)
        directions = torch.randint(0, 3, (B, K))
        mask = torch.ones(B, K)
        mask[0, -1] = 0  # mask out last target in first sample

        perturb_emb, _ = encoder(gene_emb, directions, attention_mask=mask)
        assert perturb_emb.shape == (B, config.hidden_dim)

    def test_with_magnitude_encoder(self):
        """Encoder works with magnitude encoding enabled."""
        config = BiPerturbConfig(
            gene_embed_dim=8,
            hidden_dim=8,
            num_heads=1,
            num_genes=10,
            direction_embed_dim=8,
            magnitude_embed_dim=4,
            magnitude_hidden_dim=4,
            gnn_hidden_dim=8,
            gnn_num_layers=1,
            dropout=0.0,
            attention_dropout=0.0,
        )
        encoder = BiPerturbEncoder(config)
        B, K = 2, 3
        gene_emb = torch.randn(B, K, config.gene_embed_dim)
        directions = torch.randint(0, 3, (B, K))

        perturb_emb, _ = encoder(gene_emb, directions)
        assert perturb_emb.shape == (B, config.hidden_dim)

    def test_invalid_mask_shape_raises(self):
        """Wrong attention_mask shape raises ValueError."""
        config = _small_encoder_config()
        encoder = BiPerturbEncoder(config)
        B, K = 2, 3
        gene_emb = torch.randn(B, K, config.gene_embed_dim)
        directions = torch.randint(0, 3, (B, K))
        bad_mask = torch.ones(B, K + 1)  # wrong K

        with pytest.raises(ValueError, match="attention_mask"):
            encoder(gene_emb, directions, attention_mask=bad_mask)


# ---------------------------------------------------------------------------
# PerturbationGNN tests
# ---------------------------------------------------------------------------


class TestPerturbationGNN:
    """Tests for PerturbationGNN module."""

    def test_output_shape(self):
        """GNN produces [B, num_genes] delta expression."""
        gnn = PerturbationGNN(hidden_dim=8, num_genes=10, num_layers=1)
        B = 2
        baseline = torch.randn(B, 10)
        perturb_emb = torch.randn(B, 8)
        delta = gnn(baseline, perturb_emb)
        assert delta.shape == (B, 10)


# ---------------------------------------------------------------------------
# BiPerturb full model tests
# ---------------------------------------------------------------------------


class TestBiPerturb:
    """Tests for the full BiPerturb model."""

    def test_instantiation(self):
        """BiPerturb creates with matching hidden_dim and gnn_hidden_dim."""
        config = _small_encoder_config()
        model = BiPerturb(config)
        assert model is not None

    def test_mismatched_dims_raises(self):
        """hidden_dim != gnn_hidden_dim raises ValueError."""
        config = BiPerturbConfig(
            gene_embed_dim=8,
            hidden_dim=8,
            num_heads=1,
            num_genes=10,
            direction_embed_dim=8,
            gnn_hidden_dim=16,  # mismatch
        )
        with pytest.raises(ValueError, match="hidden_dim"):
            BiPerturb(config)

    def test_forward_output_keys(self):
        """Forward returns dict with expected keys."""
        config = _small_encoder_config()
        model = BiPerturb(config)
        model.eval()
        B, K = 2, 3
        gene_emb = torch.randn(B, K, config.gene_embed_dim)
        directions = torch.randint(0, 3, (B, K))
        baseline = torch.randn(B, config.num_genes)

        result = model(gene_emb, directions, baseline)
        assert "predicted_expression" in result
        assert "delta_expression" in result
        assert "perturbation_embedding" in result
        assert "attention_weights" in result

    def test_forward_output_shapes(self):
        """Forward output tensors have correct shapes."""
        config = _small_encoder_config()
        model = BiPerturb(config)
        model.eval()
        B, K = 2, 3
        gene_emb = torch.randn(B, K, config.gene_embed_dim)
        directions = torch.randint(0, 3, (B, K))
        baseline = torch.randn(B, config.num_genes)

        result = model(gene_emb, directions, baseline)
        assert result["predicted_expression"].shape == (B, config.num_genes)
        assert result["delta_expression"].shape == (B, config.num_genes)
        assert result["perturbation_embedding"].shape == (B, config.hidden_dim)

    def test_predict_knockout(self):
        """predict_knockout convenience method works."""
        config = _small_encoder_config()
        model = BiPerturb(config)
        model.eval()
        B, K = 2, 3
        gene_emb = torch.randn(B, K, config.gene_embed_dim)
        baseline = torch.randn(B, config.num_genes)

        with torch.no_grad():
            pred = model.predict_knockout(gene_emb, baseline)
        assert pred.shape == (B, config.num_genes)

    def test_predict_overexpression(self):
        """predict_overexpression convenience method works."""
        config = _small_encoder_config()
        model = BiPerturb(config)
        model.eval()
        B, K = 2, 3
        gene_emb = torch.randn(B, K, config.gene_embed_dim)
        baseline = torch.randn(B, config.num_genes)

        with torch.no_grad():
            pred = model.predict_overexpression(gene_emb, baseline)
        assert pred.shape == (B, config.num_genes)

    def test_predict_mixed(self):
        """predict_mixed convenience method works."""
        config = _small_encoder_config()
        model = BiPerturb(config)
        model.eval()
        B, K = 2, 3
        gene_emb = torch.randn(B, K, config.gene_embed_dim)
        directions = torch.randint(0, 3, (B, K))
        baseline = torch.randn(B, config.num_genes)

        with torch.no_grad():
            pred = model.predict_mixed(gene_emb, directions, baseline)
        assert pred.shape == (B, config.num_genes)


# ---------------------------------------------------------------------------
# BiPerturbLoss tests
# ---------------------------------------------------------------------------


class TestBiPerturbLoss:
    """Tests for BiPerturbLoss module."""

    def test_basic_loss_output(self):
        """Loss returns dict with mse, pearson, total keys."""
        loss_fn = BiPerturbLoss()
        pred = torch.randn(2, 10)
        target = torch.randn(2, 10)
        result = loss_fn(pred, target)
        assert "mse" in result
        assert "pearson" in result
        assert "total" in result

    def test_zero_mse_on_identical(self):
        """MSE is zero when predicted equals target."""
        loss_fn = BiPerturbLoss(pearson_weight=0.0)
        x = torch.randn(2, 10)
        result = loss_fn(x, x)
        assert result["mse"].item() == pytest.approx(0.0, abs=1e-6)

    def test_direction_loss_with_mask(self):
        """Direction loss works with attention_mask."""
        loss_fn = BiPerturbLoss(direction_weight=1.0)
        pred = torch.randn(2, 10)
        target = torch.randn(2, 10)
        delta_pred = torch.randn(2, 10)
        directions = torch.tensor([[0, 2], [2, 0]])
        gene_ids = torch.tensor([[0, 1], [2, 3]])
        mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.float)

        result = loss_fn(
            pred,
            target,
            directions=directions,
            delta_pred=delta_pred,
            target_gene_ids=gene_ids,
            attention_mask=mask,
        )
        assert "direction" in result
        assert result["direction"].dim() == 0
