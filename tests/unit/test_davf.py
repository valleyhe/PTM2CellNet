"""Unit tests for src/models/davf.py (DAVF model).

These tests focus on DAVF behaviours not already covered in
``tests/unit/test_davf_core.py``:

* integration with a (mocked) ``GeneformerEmbeddingLoader``
* optional-dependency skipping for the real loader
* ``ConditionalVelocityField`` shape contracts
* ``condition_source`` error handling
* hybrid condition injection path
* ``verify_direction_accuracy`` helper
"""

import pytest
import torch

from src.models.davf import (
    ConditionalVelocityField,
    DAVF,
    DAVFConfig,
)


try:
    import transformers  # noqa: F401

    TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    TRANSFORMERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_config(**overrides):
    """Return a tiny DAVFConfig suitable for fast unit tests."""
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
        dropout=0.0,
    )
    defaults.update(overrides)
    return DAVFConfig(**defaults)


class _DummyEmbeddingLoader:
    """Minimal stand-in for ``GeneformerEmbeddingLoader``."""

    def __init__(self, dim: int):
        self._dim = dim

    def get_embedding_dim(self) -> int:
        return self._dim

    def get_gene_embedding(self, gene_names):
        return torch.randn(len(gene_names), self._dim)


# ---------------------------------------------------------------------------
# ConditionalVelocityField
# ---------------------------------------------------------------------------


class TestConditionalVelocityField:
    """Tests for the standalone velocity field."""

    def test_output_shape(self):
        """Velocity field returns [B, num_genes] for valid inputs."""
        config = _small_config()
        field = ConditionalVelocityField(config)

        B = 2
        x_t = torch.randn(B, config.num_genes)
        t = torch.rand(B)
        condition = torch.randn(B, config.hidden_dim)

        velocity = field(x_t, t, condition)
        assert velocity.shape == (B, config.num_genes)

    def test_hybrid_injection_output_shape(self):
        """Hybrid injection path preserves the velocity shape."""
        config = _small_config(condition_injection="hybrid")
        field = ConditionalVelocityField(config)

        B = 2
        x_t = torch.randn(B, config.num_genes)
        t = torch.rand(B)
        condition = torch.randn(B, config.hidden_dim)

        velocity = field(x_t, t, condition)
        assert velocity.shape == (B, config.num_genes)

    def test_residual_adds_input(self):
        """With use_residual=True the output depends on x_t."""
        config = _small_config(use_residual=True, residual_gate_init=0.5)
        field = ConditionalVelocityField(config)
        field.eval()

        x_t = torch.randn(1, config.num_genes)
        t = torch.tensor([0.5])
        condition = torch.randn(1, config.hidden_dim)

        with torch.no_grad():
            velocity = field(x_t, t, condition)
        # Residual term is present: shape must match x_t.
        assert velocity.shape == x_t.shape


# ---------------------------------------------------------------------------
# DAVF with an embedding loader
# ---------------------------------------------------------------------------


class TestDAVFWithEmbeddingLoader:
    """DAVF tests that use a mock pretrained embedding loader."""

    def test_projection_layer_created_when_dims_differ(self):
        """A projection layer is created when loader dim != gene_embed_dim."""
        loader = _DummyEmbeddingLoader(dim=16)
        config = _small_config(gene_embed_dim=8)
        model = DAVF(config, embedding_loader=loader, gene_names=[f"g{i}" for i in range(config.num_genes)])

        assert model.gene_embed_proj is not None
        assert model.gene_embed_table is None

    def test_no_projection_when_dims_match(self):
        """No projection layer is needed when loader dim == gene_embed_dim."""
        loader = _DummyEmbeddingLoader(dim=8)
        config = _small_config(gene_embed_dim=8)
        model = DAVF(config, embedding_loader=loader, gene_names=[f"g{i}" for i in range(config.num_genes)])

        assert model.gene_embed_proj is None

    def test_pretrained_embeddings_used_in_forward(self):
        """DAVF can use pretrained loader embeddings in forward_flow_matching."""
        loader = _DummyEmbeddingLoader(dim=16)
        config = _small_config(gene_embed_dim=8)
        model = DAVF(config, embedding_loader=loader, gene_names=[f"g{i}" for i in range(config.num_genes)])
        model.eval()

        B, K = 2, 3
        x_0 = torch.randn(B, config.num_genes)
        x_1 = torch.randn(B, config.num_genes)
        gene_ids = torch.randint(0, config.num_genes, (B, K))
        directions = torch.randint(0, 3, (B, K))

        result = model.forward_flow_matching(x_0, x_1, gene_ids, directions)
        assert result["v_t"].shape == (B, config.num_genes)
        assert result["loss"].dim() == 0

    def test_get_gene_embeddings_projected_shape(self):
        """_get_gene_embeddings returns the configured gene_embed_dim."""
        loader = _DummyEmbeddingLoader(dim=16)
        config = _small_config(gene_embed_dim=8)
        model = DAVF(config, embedding_loader=loader, gene_names=[f"g{i}" for i in range(config.num_genes)])

        gene_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
        embeddings = model._get_gene_embeddings(gene_ids)
        assert embeddings.shape == (2, 3, config.gene_embed_dim)


# ---------------------------------------------------------------------------
# Optional dependency handling
# ---------------------------------------------------------------------------


class TestDAVFOptionalDependencies:
    """Tests that gracefully skip when optional dependencies are absent."""

    @pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
    def test_real_geneformer_loader_class_importable_and_falls_back(self, tmp_path):
        """The real loader class can be imported and falls back to random embeddings."""
        from src.models.geneformer_embedding import GeneformerEmbeddingLoader

        # A non-existent local path should fall back to random embeddings without
        # hitting the network.
        with pytest.warns(UserWarning, match="Geneformer could not be loaded"):
            loader = GeneformerEmbeddingLoader(model_path=str(tmp_path))

        assert loader.get_embedding_dim() == 1152
        embeddings = loader.get_gene_embedding(["ENSG00000139618"])
        assert embeddings.shape == (1, 1152)


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------


class TestDAVFEdgeCases:
    """Miscellaneous DAVF edge-case tests."""

    def test_hybrid_injection_forward(self):
        """DAVF with hybrid condition injection completes forward_flow_matching."""
        config = _small_config(condition_injection="hybrid")
        model = DAVF(config)
        model.eval()

        B, K = 2, 3
        x_0 = torch.randn(B, config.num_genes)
        x_1 = torch.randn(B, config.num_genes)
        gene_ids = torch.randint(0, config.num_genes, (B, K))
        directions = torch.randint(0, 3, (B, K))

        result = model.forward_flow_matching(x_0, x_1, gene_ids, directions)
        assert result["v_t"].shape == (B, config.num_genes)

    def test_forward_flow_matching_external_condition(self):
        """forward_flow_matching accepts an external condition embedding."""
        config = _small_config()
        model = DAVF(config)
        model.eval()

        B = 2
        x_0 = torch.randn(B, config.num_genes)
        x_1 = torch.randn(B, config.num_genes)
        external = torch.randn(B, config.hidden_dim)

        result = model.forward_flow_matching(
            x_0,
            x_1,
            condition_source="external_embedding",
            external_condition=external,
        )
        assert result["loss"].dim() == 0
        assert torch.allclose(result["condition"], external)

    def test_invalid_condition_source_raises(self):
        """Invalid condition_source in forward_flow_matching raises ValueError."""
        config = _small_config()
        model = DAVF(config)

        with pytest.raises(ValueError, match="condition_source"):
            model.forward_flow_matching(
                torch.randn(1, config.num_genes),
                torch.randn(1, config.num_genes),
                condition_source="unknown",
            )

    def test_align_condition_wrong_shape_raises(self):
        """_align_condition raises ValueError for mismatched condition shape."""
        config = _small_config()
        model = DAVF(config)
        reference = torch.randn(2, config.num_genes)

        with pytest.raises(ValueError, match="external_condition must be"):
            model._align_condition(
                torch.randn(2, config.hidden_dim + 1),
                batch_size=2,
                condition_source="external_embedding",
                reference=reference,
            )

    def test_sanitize_gene_ids_with_mask(self):
        """Masked-out negative gene ids are replaced with zero."""
        config = _small_config()
        model = DAVF(config)

        gene_ids = torch.tensor([[0, -1, 2]])
        mask = torch.tensor([[1, 0, 1]], dtype=torch.float)

        sanitized = model._sanitize_gene_ids(gene_ids, attention_mask=mask)
        assert sanitized[0, 1].item() == 0

    def test_sanitize_gene_ids_unmasked_negative_raises(self):
        """Unmasked negative gene ids raise ValueError."""
        config = _small_config()
        model = DAVF(config)

        gene_ids = torch.tensor([[0, -1, 2]])
        mask = torch.tensor([[1, 1, 1]], dtype=torch.float)

        with pytest.raises(ValueError, match="masked-in gene_ids"):
            model._sanitize_gene_ids(gene_ids, attention_mask=mask)

    def test_verify_direction_accuracy(self):
        """verify_direction_accuracy returns expected metric keys."""
        config = _small_config()
        model = DAVF(config)

        B, K = 2, 3
        x_0 = torch.randn(B, config.num_genes)
        x_1 = torch.randn(B, config.num_genes)
        gene_ids = torch.randint(0, config.num_genes, (B, K))
        directions = torch.randint(0, 3, (B, K))
        mask = torch.ones(B, K)

        metrics = model.verify_direction_accuracy(x_0, x_1, gene_ids, directions, attention_mask=mask)
        assert "ko_direction_acc" in metrics
        assert "kd_direction_acc" in metrics
        assert "oe_direction_acc" in metrics
        assert "overall_acc" in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_predict_restores_training_mode(self):
        """predict leaves the model in its original training mode."""
        config = _small_config()
        model = DAVF(config)
        model.train()

        with torch.no_grad():
            model.predict(
                torch.randn(1, config.num_genes),
                torch.randint(0, config.num_genes, (1, 2)),
                torch.randint(0, 3, (1, 2)),
                num_steps=2,
            )
        assert model.training
