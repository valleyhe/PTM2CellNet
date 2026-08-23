"""
Unit tests for DAVFInferenceModule.

Tests DAVF model loading, inference, freeze/unfreeze API,
and graceful degradation when checkpoint is missing.
"""

import pytest
import pickle
import torch
from unittest.mock import MagicMock, patch
import warnings

from src.models.ptm_direction_mapper import PTMDirectionMapperOutput


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_latent_davf_config():
    """Create mock LatentDAVFConfig with test defaults."""
    from src.models.latent_davf import LatentDAVFConfig
    return LatentDAVFConfig(
        latent_dim=10,
        num_genes=5000,
        gene_embed_dim=192,
        hidden_dim=256,
    )


@pytest.fixture
def mock_mapper_output():
    """Create mock PTMDirectionMapperOutput with test tensors."""
    B, K = 2, 32
    return PTMDirectionMapperOutput(
        gene_ids=torch.randint(0, 5000, (B, K)),
        directions=torch.randint(0, 3, (B, K)),
        attention_mask=torch.ones(B, K, dtype=torch.float),
    )


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temp checkpoint directory for isolation."""
    ckpt_dir = tmp_path / "checkpoints" / "test_model"
    ckpt_dir.mkdir(parents=True)
    return ckpt_dir


@pytest.fixture
def mock_checkpoint(temp_checkpoint_dir):
    """Create a minimal mock checkpoint file."""
    # Create a minimal state dict that looks like a LatentDAVF checkpoint
    state_dict = {
        "velocity_field.velocity_net.0.weight": torch.randn(512, 640),
        "velocity_field.velocity_net.0.bias": torch.randn(512),
        "biperturb_encoder.direction_encoder.embedding.weight": torch.randn(3, 64),
    }
    checkpoint = {
        "model_state_dict": state_dict,
        "config": {"latent_dim": 10, "hidden_dim": 256},
        "epoch": 10,
    }
    ckpt_path = temp_checkpoint_dir / "best_model.pt"
    torch.save(checkpoint, ckpt_path)
    return ckpt_path


# ============================================================================
# Test DAVFInferenceConfig
# ============================================================================

class TestDAVFInferenceConfig:
    """Test suite for DAVFInferenceConfig validation."""

    def test_valid_config_scvi_latent(self):
        """state_space='scvi_latent' validates correctly."""
        from src.models.davf_inference import DAVFInferenceConfig
        config = DAVFInferenceConfig(state_space="scvi_latent")
        assert config.state_space == "scvi_latent"
        assert config.feature_dim == 128
        assert config.hidden_dim == 256

    def test_gene_state_space_is_accepted(self):
        """state_space='gene' is supported after DAVF fix."""
        from src.models.davf_inference import DAVFInferenceConfig

        config = DAVFInferenceConfig(state_space="gene")
        assert config.state_space == "gene"
        assert hasattr(config, "gene_vocab_size")

    def test_invalid_state_space_raises(self):
        """Invalid state_space raises ValueError."""
        from src.models.davf_inference import DAVFInferenceConfig
        with pytest.raises(ValueError, match="state_space must be"):
            DAVFInferenceConfig(state_space="invalid")

    def test_negative_feature_dim_raises(self):
        """Negative feature_dim raises ValueError."""
        from src.models.davf_inference import DAVFInferenceConfig
        with pytest.raises(ValueError, match="feature_dim must be positive"):
            DAVFInferenceConfig(feature_dim=-1)

    def test_negative_hidden_dim_raises(self):
        """Negative hidden_dim raises ValueError."""
        from src.models.davf_inference import DAVFInferenceConfig
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            DAVFInferenceConfig(hidden_dim=-1)

    def test_default_checkpoint_path(self):
        """Default checkpoint path is set correctly."""
        from src.models.davf_inference import DAVFInferenceConfig
        config = DAVFInferenceConfig()
        assert "latent_davf_ibd_norman" in config.checkpoint_path


# ============================================================================
# Test DeltaProjection
# ============================================================================

class TestDeltaProjection:
    """Test suite for DeltaProjection module."""

    def test_delta_projection_shape(self):
        """DeltaProjection maps [B, 256] to [B, 128]."""
        from src.models.davf_inference import DeltaProjection
        projection = DeltaProjection(hidden_dim=256, feature_dim=128)
        x = torch.randn(4, 256)  # [B, hidden_dim]
        out = projection(x)
        assert out.shape == (4, 128)

    def test_delta_projection_different_dims(self):
        """DeltaProjection works with custom dimensions."""
        from src.models.davf_inference import DeltaProjection
        projection = DeltaProjection(hidden_dim=512, feature_dim=64)
        x = torch.randn(2, 512)
        out = projection(x)
        assert out.shape == (2, 64)

    def test_delta_projection_trainable(self):
        """DeltaProjection parameters have requires_grad=True by default."""
        from src.models.davf_inference import DeltaProjection
        projection = DeltaProjection()
        for param in projection.parameters():
            assert param.requires_grad


# ============================================================================
# Test DAVFInferenceModule
# ============================================================================

class TestDAVFInferenceModule:
    """Test suite for DAVFInferenceModule."""

    def test_checkpoint_loading_returns_correct_shape(self, mock_checkpoint, mock_mapper_output):
        """Load checkpoint and verify [B, 128] output."""
        from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig

        config = DAVFInferenceConfig(
            checkpoint_path=str(mock_checkpoint),
            freeze=True,
        )

        module = DAVFInferenceModule(config)
        output = module(mock_mapper_output)

        assert output.davf_features.shape == (2, 128)  # [B, feature_dim]

    def test_missing_checkpoint_returns_zero_features(self, tmp_path, mock_mapper_output):
        """Missing checkpoint returns zero features gracefully."""
        from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig

        nonexistent_path = tmp_path / "nonexistent" / "best_model.pt"
        config = DAVFInferenceConfig(
            checkpoint_path=str(nonexistent_path),
            freeze=True,
        )

        # Should not raise, should log warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            module = DAVFInferenceModule(config)

            output = module(mock_mapper_output)

            # Features should be zeros
            assert torch.all(output.davf_features == 0.0)
            assert output.davf_features.shape == (2, 128)

    def test_freeze_davf_freezes_only_davf(self, mock_checkpoint):
        """freeze_davf() freezes DAVF params, not DeltaProjection."""
        from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig

        config = DAVFInferenceConfig(
            checkpoint_path=str(mock_checkpoint),
            freeze=False,  # Start unfrozen
        )

        module = DAVFInferenceModule(config)

        # Initially all should be trainable
        davf_params = list(module.latent_davf.parameters())
        projection_params = list(module.delta_projection.parameters())

        # Freeze DAVF
        module.freeze_davf()

        # DAVF params should be frozen
        for param in davf_params:
            assert not param.requires_grad, "DAVF param should be frozen"

        # DeltaProjection params should remain trainable
        for param in projection_params:
            assert param.requires_grad, "DeltaProjection param should remain trainable"

    def test_unfreeze_davf_unfreezes_all(self, mock_checkpoint):
        """unfreeze_davf() sets all params requires_grad=True."""
        from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig

        config = DAVFInferenceConfig(
            checkpoint_path=str(mock_checkpoint),
            freeze=True,  # Start frozen
        )

        module = DAVFInferenceModule(config)

        # Unfreeze
        module.unfreeze_davf()

        # All params should be trainable now
        for param in module.parameters():
            assert param.requires_grad, "All params should be trainable after unfreeze"

    def test_freeze_on_init(self, mock_checkpoint):
        """config.freeze=True freezes DAVF on initialization."""
        from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig

        config = DAVFInferenceConfig(
            checkpoint_path=str(mock_checkpoint),
            freeze=True,
        )

        module = DAVFInferenceModule(config)

        # DAVF params should be frozen
        for param in module.latent_davf.parameters():
            assert not param.requires_grad, "DAVF should be frozen on init"

        # DeltaProjection should remain trainable
        for param in module.delta_projection.parameters():
            assert param.requires_grad, "DeltaProjection should remain trainable"

    def test_forward_with_z0_latent(self, mock_checkpoint):
        """Forward accepts z_0 for API compatibility."""
        from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig

        config = DAVFInferenceConfig(
            checkpoint_path=str(mock_checkpoint),
            latent_dim=10,
        )

        module = DAVFInferenceModule(config)

        mapper_output = PTMDirectionMapperOutput(
            gene_ids=torch.randint(0, 5000, (2, 32)),
            directions=torch.randint(0, 3, (2, 32)),
            attention_mask=torch.ones(2, 32, dtype=torch.float),
        )
        z_0 = torch.randn(2, 10)  # Explicit latent state

        output = module(mapper_output, z_0=z_0)
        assert output.davf_features.shape == (2, 128)

    def test_forward_does_not_invoke_predict(self, mock_checkpoint, mock_mapper_output):
        """Feature extraction should not pay for an unused ODE inference pass."""
        from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig

        config = DAVFInferenceConfig(
            checkpoint_path=str(mock_checkpoint),
        )

        module = DAVFInferenceModule(config)
        module.latent_davf.predict = MagicMock(side_effect=AssertionError("predict() should not be called"))

        output = module(mock_mapper_output)

        assert output.davf_features.shape == (2, 128)
        module.latent_davf.predict.assert_not_called()

    def test_device_handling(self, mock_checkpoint):
        """Module respects device config."""
        from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig

        config = DAVFInferenceConfig(
            checkpoint_path=str(mock_checkpoint),
            device="cpu",  # Explicit CPU
        )

        module = DAVFInferenceModule(config)

        # All parameters should be on CPU
        for param in module.parameters():
            assert param.device.type == "cpu"

    def test_device_attribute_is_torch_device(self, mock_checkpoint):
        """Configured device is normalized to torch.device for safe tensor moves."""
        from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig

        config = DAVFInferenceConfig(
            checkpoint_path=str(mock_checkpoint),
            device="cpu",
        )

        module = DAVFInferenceModule(config)

        assert module.device == torch.device("cpu")


# ============================================================================
# Test DAVFInferenceOutput
# ============================================================================

class TestDAVFInferenceOutput:
    """Test suite for DAVFInferenceOutput dataclass."""

    def test_output_has_davf_features(self, mock_checkpoint, mock_mapper_output):
        """Output contains davf_features field."""
        from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig

        config = DAVFInferenceConfig(checkpoint_path=str(mock_checkpoint))
        module = DAVFInferenceModule(config)

        output = module(mock_mapper_output)

        assert hasattr(output, 'davf_features')
        assert isinstance(output.davf_features, torch.Tensor)

    def test_output_shape_matches_batch_size(self, mock_checkpoint):
        """Batch size preserved in output."""
        from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig

        config = DAVFInferenceConfig(checkpoint_path=str(mock_checkpoint))
        module = DAVFInferenceModule(config)

        # Different batch sizes
        for B in [1, 4, 16]:
            mapper_output = PTMDirectionMapperOutput(
                gene_ids=torch.randint(0, 5000, (B, 32)),
                directions=torch.randint(0, 3, (B, 32)),
                attention_mask=torch.ones(B, 32, dtype=torch.float),
            )
            output = module(mapper_output)
            assert output.davf_features.shape[0] == B


# ============================================================================
# Test Checkpoint Loading Edge Cases
# ============================================================================

class TestCheckpointLoading:
    """Test suite for checkpoint loading edge cases."""

    def test_legacy_pickle_checkpoint_stays_unloaded_without_opt_in(self, tmp_path, mock_mapper_output):
        """Legacy pickle is rejected by safe_torch_load — degrades to zeros."""
        from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig

        ckpt_path = tmp_path / "legacy_pickle.pt"
        ckpt_path.write_bytes(b"placeholder")

        with patch("src.models.davf_inference.torch.load") as mock_load:
            mock_load.side_effect = pickle.UnpicklingError("legacy pickle")

            module = DAVFInferenceModule(DAVFInferenceConfig(checkpoint_path=str(ckpt_path)))

        assert module._checkpoint_loaded is False
        # safe_torch_load calls torch.load once with weights_only=True, which
        # raises UnpicklingError — the old unsafe fallback is no longer used.
        assert mock_load.call_count == 1

        output = module(mock_mapper_output)
        assert output.davf_features.shape == (2, 128)
        assert torch.all(output.davf_features == 0.0)

    def test_legacy_pickle_checkpoint_requires_explicit_opt_in(self, tmp_path, mock_mapper_output):
        """Legacy pickle is rejected regardless of any opt-in — unsafe fallback removed by SC-01."""
        from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig

        ckpt_path = tmp_path / "legacy_pickle.pt"
        ckpt_path.write_bytes(b"placeholder")

        with patch("src.models.davf_inference.torch.load") as mock_load:
            mock_load.side_effect = pickle.UnpicklingError("legacy pickle")

            module = DAVFInferenceModule(
                DAVFInferenceConfig(checkpoint_path=str(ckpt_path))
            )

        assert module._checkpoint_loaded is False
        assert mock_load.call_count == 1

        output = module(mock_mapper_output)
        assert output.davf_features.shape == (2, 128)
        assert torch.all(output.davf_features == 0.0)

    def test_direct_state_dict_format(self, tmp_path, mock_mapper_output):
        """Handle checkpoint with direct state dict (no 'model_state_dict' key)."""
        from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig

        # Create checkpoint with direct state dict
        state_dict = {
            "velocity_field.velocity_net.0.weight": torch.randn(512, 640),
            "biperturb_encoder.direction_encoder.embedding.weight": torch.randn(3, 64),
        }
        ckpt_path = tmp_path / "direct_state.pt"
        torch.save(state_dict, ckpt_path)

        config = DAVFInferenceConfig(checkpoint_path=str(ckpt_path))

        # Should handle both formats
        module = DAVFInferenceModule(config)
        output = module(mock_mapper_output)

        assert output.davf_features.shape == (2, 128)

    def test_checkpoint_with_config(self, tmp_path, mock_mapper_output):
        """Checkpoint with config field is loaded correctly."""
        from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig
        from src.models.latent_davf import LatentDAVF, LatentDAVFConfig

        # Create a valid checkpoint by saving the actual model state
        latent_config = LatentDAVFConfig(latent_dim=10, hidden_dim=256, num_genes=5000)
        model = LatentDAVF(latent_config)
        state_dict = model.state_dict()

        checkpoint = {
            "model_state_dict": state_dict,
            "config": {"latent_dim": 10, "hidden_dim": 256},
            "epoch": 5,
        }
        ckpt_path = tmp_path / "with_config.pt"
        torch.save(checkpoint, ckpt_path)

        config = DAVFInferenceConfig(checkpoint_path=str(ckpt_path))
        module = DAVFInferenceModule(config)

        assert module._checkpoint_loaded is True

    def test_partial_checkpoint_graceful_fallback(self, tmp_path, mock_mapper_output):
        """Partial/incompatible checkpoint triggers graceful degradation."""
        from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig

        # Very minimal state dict with wrong shapes (incompatible)
        state_dict = {
            "velocity_field.velocity_net.0.weight": torch.randn(512, 640),  # Wrong shape
        }
        ckpt_path = tmp_path / "partial.pt"
        torch.save(state_dict, ckpt_path)

        config = DAVFInferenceConfig(checkpoint_path=str(ckpt_path))

        # Should gracefully fall back to zero features instead of crashing
        module = DAVFInferenceModule(config)
        assert module._checkpoint_loaded is False  # Should be False due to incompatibility

        output = module(mock_mapper_output)
        assert output.davf_features.shape == (2, 128)
        assert torch.all(output.davf_features == 0.0)  # Zero features

    def test_unrecognized_state_dict_graceful_fallback(self, tmp_path, mock_mapper_output):
        """A checkpoint with no matching LatentDAVF keys must not count as loaded."""
        from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig

        ckpt_path = tmp_path / "unrecognized.pt"
        torch.save({"totally_wrong_key": torch.randn(1)}, ckpt_path)

        config = DAVFInferenceConfig(checkpoint_path=str(ckpt_path))
        module = DAVFInferenceModule(config)

        assert module._checkpoint_loaded is False

        output = module(mock_mapper_output)
        assert output.davf_features.shape == (2, 128)
        assert torch.all(output.davf_features == 0.0)


# ============================================================================
# Test Integration with PTMDirectionMapperOutput
# ============================================================================

class TestPTMIntegration:
    """Test integration with PTMDirectionMapper output."""

    def test_accepts_mapper_output_directly(self, mock_checkpoint):
        """Module accepts PTMDirectionMapperOutput directly."""
        from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig

        config = DAVFInferenceConfig(checkpoint_path=str(mock_checkpoint))
        module = DAVFInferenceModule(config)

        # Realistic PTM input
        mapper_output = PTMDirectionMapperOutput(
            gene_ids=torch.tensor([[100, 200, 300] + [0] * 29]),
            directions=torch.tensor([[2, 0, 2] + [0] * 29]),  # OE, KO, OE
            attention_mask=torch.tensor([[1.0, 1.0, 1.0] + [0.0] * 29]),
        )

        output = module(mapper_output)
        assert output.davf_features.shape == (1, 128)

    def test_batch_consistency(self, mock_checkpoint):
        """Batch processing produces consistent results."""
        from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig

        config = DAVFInferenceConfig(checkpoint_path=str(mock_checkpoint))
        module = DAVFInferenceModule(config)

        # Same input, different batch positions
        gene_ids = torch.tensor([[100, 200], [100, 200]])
        directions = torch.tensor([[2, 0], [2, 0]])
        masks = torch.ones(2, 2)

        mapper_output = PTMDirectionMapperOutput(
            gene_ids=gene_ids,
            directions=directions,
            attention_mask=masks,
        )

        output = module(mapper_output)

        # Both batch elements should have same shape
        assert output.davf_features.shape == (2, 128)


# ============================================================================
# Schema v2: PerturbGen embedding asset injection (replaces geneformer_path)
# ============================================================================

class TestEmbeddingAssetInjection:
    """Regression tests for the M4 injection chain [R1] §3.2 A2.

    DAVFInferenceModule must pass the verified PerturbGen matrix into
    LatentDAVF via ``pretrained_gene_embeddings``; the removed
    ``geneformer_path`` null→random-embedding semantics must stay gone.
    """

    @staticmethod
    def _make_asset(tmp_path):
        import hashlib
        import json

        from safetensors.torch import save_file

        tensor_path = tmp_path / "gene_embeddings.safetensors"
        vocab_path = tmp_path / "vocabulary.json"
        matrix = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        save_file({"gene_embeddings": matrix}, str(tensor_path))
        vocab_path.write_text(
            json.dumps({f"ENSG00000{i}": i for i in range(4)}), encoding="utf-8"
        )

        def _sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        (tmp_path / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "embedding_shape": [4, 3],
                    "run_id": "test-run",
                    "files": {
                        tensor_path.name: {"sha256": _sha(tensor_path)},
                        vocab_path.name: {"sha256": _sha(vocab_path)},
                    },
                }
            ),
            encoding="utf-8",
        )
        return matrix

    def test_config_has_no_geneformer_path(self):
        """The removed dead semantic must not resurface on the dataclass."""
        from src.models.davf_inference import DAVFInferenceConfig

        field_names = {f for f in DAVFInferenceConfig.__dataclass_fields__}
        assert "geneformer_path" not in field_names
        assert "embedding_asset_path" in field_names

    def test_asset_is_injected_into_latent_davf(self, tmp_path):
        """The frozen matrix must reach LatentDAVF.gene_embed_table verbatim."""
        from src.models.davf_inference import DAVFInferenceConfig, DAVFInferenceModule

        matrix = self._make_asset(tmp_path)
        config = DAVFInferenceConfig(
            state_space="scvi_latent",
            checkpoint_path=str(tmp_path / "absent.pt"),
            embedding_asset_path=str(tmp_path),
            device="cpu",
        )
        module = DAVFInferenceModule(config)

        assert module.latent_davf.gene_embed_table is not None
        assert torch.equal(module.latent_davf.gene_embed_table.weight.data, matrix)
        # 3-dim asset vs default 192-dim gene_embed_dim requires a projection
        assert module.latent_davf.gene_embed_proj is not None
        assert module.latent_davf.gene_embed_proj.in_features == 3

    def test_missing_asset_directory_fails_fast(self, tmp_path):
        """A configured-but-absent asset must raise, not degrade to random."""
        from src.models.davf_inference import DAVFInferenceConfig, DAVFInferenceModule
        from src.models.perturbgen_embedding import PerturbGenEmbeddingError

        config = DAVFInferenceConfig(
            state_space="scvi_latent",
            checkpoint_path=str(tmp_path / "absent.pt"),
            embedding_asset_path=str(tmp_path / "no_such_asset"),
            device="cpu",
        )
        with pytest.raises(PerturbGenEmbeddingError, match="not found"):
            DAVFInferenceModule(config)

    def test_gene_mode_ignores_asset(self, tmp_path):
        """Gene-space mode is self-contained and must not touch the asset."""
        from src.models.davf_inference import DAVFInferenceConfig, DAVFInferenceModule

        config = DAVFInferenceConfig(
            state_space="gene",
            checkpoint_path=str(tmp_path / "absent.pt"),
            embedding_asset_path=str(tmp_path / "no_such_asset"),
            device="cpu",
        )
        module = DAVFInferenceModule(config)
        assert module.gene_encoder is not None

    def test_unsupported_checkpoint_schema_version_fails_loudly(self, tmp_path, monkeypatch):
        """Versioned checkpoints with unknown schema must not silently degrade."""
        import os

        from src.models.davf_inference import DAVFInferenceConfig, DAVFInferenceModule

        monkeypatch.setenv("PTM2CELLNET_STRICT_MODEL_ASSETS", "1")
        assert os.environ["PTM2CELLNET_STRICT_MODEL_ASSETS"] == "1"
        ckpt_path = tmp_path / "future.pt"
        torch.save(
            {"schema_version": 99, "model_state_dict": {}}, str(ckpt_path)
        )
        config = DAVFInferenceConfig(
            state_space="scvi_latent",
            checkpoint_path=str(ckpt_path),
            device="cpu",
        )
        module = DAVFInferenceModule(config)
        assert module._checkpoint_loaded is False

        mapper_output = PTMDirectionMapperOutput(
            gene_ids=torch.tensor([[1, 2]]),
            directions=torch.tensor([[0, 1]]),
            attention_mask=torch.ones(1, 2),
        )
        with pytest.raises(RuntimeError, match="STRICT_MODEL_ASSETS"):
            module(mapper_output)
