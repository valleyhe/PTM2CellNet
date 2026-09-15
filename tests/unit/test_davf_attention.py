"""Tests for src/models/davf_attention.py — Attention modules for BiPerturb.

Tests MultiTargetAttention, DirectionAwareAttention, and CrossModalAttention
with small tensor sizes for fast execution.
"""

import pytest
import torch

from src.models.davf_attention import (
    MultiTargetAttention,
    DirectionAwareAttention,
    CrossModalAttention,
)


# ---------------------------------------------------------------------------
# MultiTargetAttention tests
# ---------------------------------------------------------------------------


class TestMultiTargetAttention:
    """Tests for MultiTargetAttention module."""

    def test_instantiation(self):
        """Module creates with valid embed_dim and num_heads."""
        attn = MultiTargetAttention(embed_dim=8, num_heads=2, dropout=0.0)
        assert attn.embed_dim == 8
        assert attn.num_heads == 2

    def test_undivisible_embed_dim_raises(self):
        """embed_dim not divisible by num_heads raises ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            MultiTargetAttention(embed_dim=7, num_heads=2)

    def test_forward_output_shapes(self):
        """Forward returns (output, attention_weights) with correct shapes."""
        B, K, D = 2, 3, 8
        attn = MultiTargetAttention(embed_dim=D, num_heads=2, dropout=0.0)
        x = torch.randn(B, K, D)
        output, weights = attn(x)
        assert output.shape == (B, K, D)
        assert weights.shape == (B, K, K)  # averaged across heads

    def test_forward_with_2d_mask(self):
        """Forward works with [B, K] boolean mask."""
        B, K, D = 2, 4, 8
        attn = MultiTargetAttention(embed_dim=D, num_heads=2, dropout=0.0)
        x = torch.randn(B, K, D)
        mask = torch.ones(B, K)
        mask[0, -1] = 0  # mask out last target in first batch
        output, weights = attn(x, mask=mask)
        assert output.shape == (B, K, D)

    def test_forward_with_3d_mask(self):
        """Forward works with [B, K, K] mask."""
        B, K, D = 2, 3, 8
        attn = MultiTargetAttention(embed_dim=D, num_heads=2, dropout=0.0)
        x = torch.randn(B, K, D)
        mask = torch.ones(B, K, K)
        output, weights = attn(x, mask=mask)
        assert output.shape == (B, K, D)

    def test_forward_with_4d_mask(self):
        """Forward works with [B, H, K, K] mask."""
        B, K, D, H = 2, 3, 8, 2
        attn = MultiTargetAttention(embed_dim=D, num_heads=H, dropout=0.0)
        x = torch.randn(B, K, D)
        mask = torch.ones(B, H, K, K)
        output, weights = attn(x, mask=mask)
        assert output.shape == (B, K, D)

    def test_invalid_mask_shape_raises(self):
        """Mask with unexpected shape raises ValueError."""
        B, K, D = 2, 3, 8
        attn = MultiTargetAttention(embed_dim=D, num_heads=2, dropout=0.0)
        x = torch.randn(B, K, D)
        bad_mask = torch.ones(B, K, K, K)  # 4D but wrong dims
        with pytest.raises(ValueError, match="mask must be"):
            attn(x, mask=bad_mask)

    def test_single_target(self):
        """Works with a single target (K=1)."""
        B, K, D = 2, 1, 8
        attn = MultiTargetAttention(embed_dim=D, num_heads=2, dropout=0.0)
        x = torch.randn(B, K, D)
        output, weights = attn(x)
        assert output.shape == (B, K, D)
        assert weights.shape == (B, K, K)


# ---------------------------------------------------------------------------
# DirectionAwareAttention tests
# ---------------------------------------------------------------------------


class TestDirectionAwareAttention:
    """Tests for DirectionAwareAttention module."""

    def test_instantiation(self):
        """Module creates with valid parameters."""
        attn = DirectionAwareAttention(
            embed_dim=8,
            direction_embed_dim=8,
            num_heads=2,
            dropout=0.0,
        )
        assert attn.embed_dim == 8
        assert attn.num_heads == 2

    def test_forward_output_shapes(self):
        """Forward returns (output, attention_weights) with correct shapes."""
        B, K, D = 2, 3, 8
        attn = DirectionAwareAttention(
            embed_dim=D,
            direction_embed_dim=8,
            num_heads=2,
            dropout=0.0,
        )
        x = torch.randn(B, K, D)
        direction_emb = torch.randn(B, K, 8)
        directions = torch.randint(0, 3, (B, K))

        output, weights = attn(x, direction_emb, directions)
        assert output.shape == (B, K, D)
        assert weights.shape == (B, K, K)

    def test_forward_with_attention_mask(self):
        """Forward works with attention_mask."""
        B, K, D = 2, 4, 8
        attn = DirectionAwareAttention(
            embed_dim=D,
            direction_embed_dim=8,
            num_heads=2,
            dropout=0.0,
        )
        x = torch.randn(B, K, D)
        direction_emb = torch.randn(B, K, 8)
        directions = torch.randint(0, 3, (B, K))
        mask = torch.ones(B, K)
        mask[0, -1] = 0

        output, weights = attn(x, direction_emb, directions, attention_mask=mask)
        assert output.shape == (B, K, D)

    def test_direction_interaction_shape(self):
        """_compute_direction_interactions returns [B, H, K, K]."""
        B, K, H = 2, 3, 2
        attn = DirectionAwareAttention(
            embed_dim=8,
            direction_embed_dim=8,
            num_heads=H,
            dropout=0.0,
        )
        directions = torch.randint(0, 3, (B, K))
        interactions = attn._compute_direction_interactions(directions)
        assert interactions.shape == (B, H, K, K)

    def test_all_same_direction(self):
        """Works when all targets have the same direction."""
        B, K, D = 2, 3, 8
        attn = DirectionAwareAttention(
            embed_dim=D,
            direction_embed_dim=8,
            num_heads=2,
            dropout=0.0,
        )
        x = torch.randn(B, K, D)
        direction_emb = torch.randn(B, K, 8)
        directions = torch.zeros(B, K, dtype=torch.long)  # all KO

        output, weights = attn(x, direction_emb, directions)
        assert output.shape == (B, K, D)


# ---------------------------------------------------------------------------
# CrossModalAttention tests
# ---------------------------------------------------------------------------


class TestCrossModalAttention:
    """Tests for CrossModalAttention module."""

    def test_instantiation(self):
        """Module creates with valid parameters."""
        attn = CrossModalAttention(embed_dim=8, num_heads=2, dropout=0.0)
        assert attn is not None

    def test_forward_output_shapes(self):
        """Forward returns (output, attention_weights) with correct shapes."""
        B, Q, KV, D = 2, 3, 5, 8
        attn = CrossModalAttention(embed_dim=D, num_heads=2, dropout=0.0)
        query = torch.randn(B, Q, D)
        key_value = torch.randn(B, KV, D)

        output, weights = attn(query, key_value)
        assert output.shape == (B, Q, D)
        assert weights.shape == (B, Q, KV)

    def test_single_query(self):
        """Works with a single query token."""
        B, Q, KV, D = 2, 1, 5, 8
        attn = CrossModalAttention(embed_dim=D, num_heads=2, dropout=0.0)
        query = torch.randn(B, Q, D)
        key_value = torch.randn(B, KV, D)
        output, weights = attn(query, key_value)
        assert output.shape == (B, Q, D)
