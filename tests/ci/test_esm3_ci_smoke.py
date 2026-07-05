"""CI smoke tests for ESM-3 encoder integration.

These tests verify that the ESM-3 integration works correctly across
environments, gracefully handling missing SDK, missing weights, and
various failure modes.

Tests are organized into tiers:

- **Tier 1** (always runs): Import health checks — no esm SDK needed
- **Tier 2** (mock): Constructs with mocked model loading — esm SDK needed
- **Tier 3** (no SDK): Fallback and strict-mode behaviour — no esm SDK needed
- **Tier 4** (mock): PTM2CellNet + ESM-3 end-to-end forward pass — esm SDK needed
"""

import importlib.util
import os

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Tier 1: Import health checks (always runs, no SDK required)
# ---------------------------------------------------------------------------


def test_esm3_encoder_class_importable():
    """ESM3Encoder and ESM3TokenizerAdapter are importable without the esm SDK.

    The ESM-3-specific imports are inside method bodies, so importing the
    class does NOT require ``esm>=3.0.0`` to be installed.
    """
    from src.models.pretrained_encoders import ESM3Encoder, ESM3TokenizerAdapter  # noqa: F811

    assert ESM3Encoder is not None
    assert hasattr(ESM3Encoder, "_normalize_model_size")
    assert ESM3TokenizerAdapter is not None


def test_esm3_model_size_normalization():
    """Model size is normalized case-insensitively to "small"."""
    from src.models.pretrained_encoders import ESM3Encoder

    assert ESM3Encoder._normalize_model_size("small") == "small"
    assert ESM3Encoder._normalize_model_size("SMALL") == "small"
    assert ESM3Encoder._normalize_model_size("Small") == "small"


def test_esm3_tokenizer_adapter_importable():
    """ESM3TokenizerAdapter can be imported and inspected."""
    from src.models.pretrained_encoders import ESM3TokenizerAdapter

    assert ESM3TokenizerAdapter is not None


# ---------------------------------------------------------------------------
# Tier 2: Mock construction (esm SDK required for these to run)
# ---------------------------------------------------------------------------

_ESM_AVAILABLE = importlib.util.find_spec("esm") is not None


class _MockSequenceTokenizer:
    """Minimal mock ESM-3 sequence tokenizer."""

    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        self.mask_token_id = 128
        self.pad_token_id = 0

    def encode(self, sequence: str) -> list:
        """Return per-char token ids (mimics ESM-3 AVG token-level encoding)."""
        return [hash(ch) % self.vocab_size for ch in sequence]


class _MockTokenizerCollection:
    """Minimal mock ESM-3 TokenizerCollection."""

    def __init__(self, vocab_size: int = 4096):
        self.sequence = _MockSequenceTokenizer(vocab_size)


class _MockESM3(nn.Module):
    """Minimal mock ESM-3 model for forward-pass testing."""

    def __init__(self, d_model: int = 1536, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.tokenizers = _MockTokenizerCollection()
        self.dummy_weight = nn.Parameter(torch.randn(d_model))

    def forward(self, sequence_tokens=None, **kwargs):
        if sequence_tokens is None:
            raise ValueError("No tokens")
        batch_size, seq_len = sequence_tokens.shape
        return type("Result", (), {
            "embeddings": torch.randn(batch_size, seq_len, self.d_model),
            "structure_logits": torch.randn(batch_size, seq_len, 4096),
            "function_logits": torch.randn(batch_size, seq_len, 5),
        })()


def _mock_load_local(self, path, device):
    """Replace ESM3Encoder._load_from_local — sets attrs and returns model."""
    model = _MockESM3(d_model=1536)
    self.model = model
    self.hidden_dim = 1536
    self.tokenizer = type(
        "Adapter", (), {
            "mask_token_id": 128,
            "pad_token_id": 0,
            "vocab_size": 4096,
            "bos_token_id": 0,
            "eos_token_id": 0,
            "unk_token_id": 128,
            "__call__": lambda self, sequences, **kw: {"input_ids": torch.randint(0, 100, (len(sequences), 50))},
        }
    )()
    self._device = device
    self._model_source = "local_checkpoint"
    return model


def _mock_load_hf(self, device):
    """Replace ESM3Encoder._load_from_huggingface — sets attrs and returns model."""
    model = _MockESM3(d_model=1536)
    self.model = model
    self.hidden_dim = 1536
    self.tokenizer = type(
        "Adapter", (), {
            "mask_token_id": 128,
            "pad_token_id": 0,
            "vocab_size": 4096,
            "bos_token_id": 0,
            "eos_token_id": 0,
            "unk_token_id": 128,
            "__call__": lambda self, sequences, **kw: {"input_ids": torch.randint(0, 100, (len(sequences), 50))},
        }
    )()
    self._device = device
    self._model_source = "huggingface"
    return model


@pytest.mark.skipif(not _ESM_AVAILABLE, reason="esm>=3.0.0 not installed")
class TestESM3WithSDK:
    """Tests that require the esm>=3.0.0 SDK to be installed.

    These tests still mock model loading — no actual 2.7GB download occurs.
    """

    @pytest.fixture(autouse=True)
    def _patch_loading(self, monkeypatch):
        """Mock all model loading paths so no actual weights are loaded."""
        from src.models.pretrained_encoders import ESM3Encoder

        monkeypatch.setattr(ESM3Encoder, "_load_from_local", _mock_load_local)
        monkeypatch.setattr(ESM3Encoder, "_load_from_huggingface", _mock_load_hf)

    def test_esm3_encoder_constructs(self):
        """ESM3Encoder constructs successfully with mocked model."""
        from src.models.pretrained_encoders import ESM3Encoder

        encoder = ESM3Encoder(model_size="small", freeze=True)
        assert encoder.model_size == "small"
        assert encoder.hidden_dim == 1536
        assert encoder.model_source == "huggingface"

    def test_esm3_encoder_forward(self):
        """ESM3Encoder forward pass works with mocked model."""
        from src.models.pretrained_encoders import ESM3Encoder

        encoder = ESM3Encoder(model_size="small", freeze=True)
        encoder.eval()
        input_ids = torch.randint(0, 100, (2, 50))
        with torch.no_grad():
            emb = encoder(input_ids)
        assert emb.shape == (2, 50, 1536)

    def test_esm3_encoder_encode_sequences(self):
        """encode_sequences returns correct shape."""
        from src.models.pretrained_encoders import ESM3Encoder

        encoder = ESM3Encoder(model_size="small", freeze=True)
        encoder.eval()
        sequences = ["MALWMRLLPLLALLALWGPDPAAA"]
        emb = encoder.encode_sequences(sequences)
        assert isinstance(emb, torch.Tensor)
        # Shape: (1, seq_len, hidden_dim); check last dim is hidden_dim
        assert emb.shape[-1] == 1536

    def test_esm3_encoder_parameter_count(self):
        """get_num_parameters returns positive count."""
        from src.models.pretrained_encoders import ESM3Encoder

        encoder = ESM3Encoder(model_size="small", freeze=True)
        assert encoder.get_num_parameters() > 0


# ---------------------------------------------------------------------------
# Tier 3: Fallback and strict-mode behaviour (no SDK required)
# ---------------------------------------------------------------------------


def test_esm3_encoder_factory_fallback(monkeypatch):
    """esm3_encoder() falls back to ESM2 when ESM-3 is unavailable.

    Deleting ESM3Encoder from the module simulates an environment where
    the SDK import fails inside esm3_encoder().
    """
    import src.models.pretrained_encoders as _mod
    from src.models.pretrained_encoders import esm3_encoder

    monkeypatch.delattr(_mod, "ESM3Encoder", raising=False)

    # Mock ESM2Encoder to avoid loading HuggingFace models
    mock_esm2 = type("FakeESM2", (), {"model_size": "150M"})()
    monkeypatch.setattr(_mod, "ESM2Encoder", lambda *a, **kw: mock_esm2)

    encoder = esm3_encoder(model_size="150M", freeze=True)
    assert encoder is mock_esm2


def test_esm3_encoder_strict_mode_raises(monkeypatch):
    """strict=True raises RuntimeError when ESM-3 is unavailable."""
    import src.models.pretrained_encoders as _mod
    from src.models.pretrained_encoders import esm3_encoder

    monkeypatch.delattr(_mod, "ESM3Encoder", raising=False)
    with pytest.raises((RuntimeError, ImportError, OSError), match="strict"):
        esm3_encoder(model_size="small", strict=True)


def test_esm3_encoder_strict_mode_env_var(monkeypatch):
    """PTM2CELLNET_STRICT_MODEL_ASSETS=1 enforces strict mode."""
    import src.models.pretrained_encoders as _mod
    from src.models.pretrained_encoders import esm3_encoder

    monkeypatch.delattr(_mod, "ESM3Encoder", raising=False)
    monkeypatch.setenv("PTM2CELLNET_STRICT_MODEL_ASSETS", "1")
    try:
        with pytest.raises((RuntimeError, ImportError, OSError)):
            esm3_encoder(model_size="small")
    finally:
        os.environ.pop("PTM2CELLNET_STRICT_MODEL_ASSETS", None)


# ---------------------------------------------------------------------------
# Tier 4: PTM2CellNet + ESM-3 integration (esm SDK required, mocked)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ESM_AVAILABLE, reason="esm>=3.0.0 not installed")
class TestESM3PTM2CellNetIntegration:
    """End-to-end forward pass through PTM2CellNet with ESM-3 encoder."""

    @pytest.fixture(autouse=True)
    def _patch_loading(self, monkeypatch):
        """Mock ESM-3 model loading."""
        from src.models.pretrained_encoders import ESM3Encoder

        monkeypatch.setattr(ESM3Encoder, "_load_from_local", _mock_load_local)
        monkeypatch.setattr(ESM3Encoder, "_load_from_huggingface", _mock_load_hf)

    def test_ptm2cellnet_esm3_forward(self):
        """PTM2CellNet with esm3 encoder completes forward pass."""
        from src.models.architectures import PTM2CellNet

        model = PTM2CellNet(
            encoder_type="esm3_small",
            num_classes=2,
            embed_dim=1536,
        )
        model.eval()
        batch = {
            "input_ids": torch.randint(0, 100, (2, 50)),
            "attention_mask": torch.ones(2, 50),
            "ptm_mask": torch.zeros(2, 50),
            "ptm_types": torch.zeros(2, 50, dtype=torch.long),
        }
        with torch.no_grad():
            out = model(batch)
        assert out["logits"].shape == (2, 2)


def test_esm3_sm_open_in_pretrained_models():
    """esm3_sm_open is registered in train_pretrained.py PRETRAINED_MODELS."""
    import sys
    import os

    # Add scripts/ to path if needed
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from train_pretrained import PRETRAINED_MODELS

    assert "esm3_sm_open" in PRETRAINED_MODELS
    # Value is a config file path (string)
    assert isinstance(PRETRAINED_MODELS["esm3_sm_open"], str)
    assert len(PRETRAINED_MODELS["esm3_sm_open"]) > 0
