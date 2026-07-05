"""Real-asset acceptance for ESM-3 (F-01, TD-H2).

Gated behind ``PTM2CELLNET_RUN_REAL_ASSET_TESTS=1``. When enabled, this
test loads a real ESM-3 checkpoint and runs a forward pass on canonical
protein sequences, then records the source/cache/duration as evidence.

Configuration via env vars:

* ``PTM2CELLNET_ESM3_CHECKPOINT`` — path to a local ``esm3_sm_open_v1.pth``
  (skips the HuggingFace download).
* ``PTM2CELLNET_ESM3_DEVICE`` — ``cuda`` or ``cpu`` (default auto-detect).
* ``PTM2CELLNET_ESM3_CACHE_DIR`` — ESM data cache directory.

If the gate is unset, the test is reported as SKIPPED with a clear reason
so a green CI run is never mistaken for "ESM-3 verified against real weights".
"""

from __future__ import annotations

import os
import time

import pytest

from tests.real_assets import real_assets_enabled, record_evidence


pytestmark = pytest.mark.skipif(
    not real_assets_enabled(),
    reason=(
        "ESM-3 real-asset test is opt-in. Set PTM2CELLNET_RUN_REAL_ASSET_TESTS=1 "
        "and provide PTM2CELLNET_ESM3_CHECKPOINT (or let it download from HF) "
        "to run this acceptance test against real weights."
    ),
)


# Canonical short proteins used for the forward-pass shape check. Using
# well-known sequences makes the recorded evidence reproducible.
_INSULIN_A = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
_UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def test_esm3_real_forward_pass():
    """Load a real ESM-3 model and verify a forward pass on real sequences."""
    checkpoint = os.environ.get("PTM2CELLNET_ESM3_CHECKPOINT")
    device = os.environ.get("PTM2CELLNET_ESM3_DEVICE")
    cache_dir = os.environ.get("PTM2CELLNET_ESM3_CACHE_DIR")

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"torch not available: {exc}")

    try:
        from src.models.pretrained_encoders import ESM3Encoder
    except ImportError as exc:
        pytest.fail(f"ESM3Encoder not importable: {exc}")

    start = time.time()
    outcome = "pass"
    err: str | None = None
    shapes: tuple[tuple[int, ...], ...] = ()
    model_source = "unknown"
    try:
        encoder = ESM3Encoder(
            model_size="small",
            freeze=True,
            cache_dir=cache_dir,
            checkpoint_path=checkpoint,
            device=device,
        )
        encoder.eval()
        model_source = encoder.model_source

        import torch

        sequences = [_INSULIN_A, _UBIQUITIN]
        with torch.no_grad():
            emb = encoder.encode_sequences(sequences)
        shapes = tuple(tuple(s) for s in emb.shape)
        # F-01 acceptance: real forward must return non-empty embeddings of
        # the expected rank and trailing dimension.
        assert emb.ndim == 3, f"expected (B, L, D) embedding, got shape {emb.shape}"
        assert emb.shape[-1] == encoder.hidden_dim
        assert emb.shape[0] == len(sequences)
        assert emb.abs().sum().item() > 0, "embeddings are all-zero — model likely not loaded"
    except Exception as exc_:  # pragma: no cover - exercised only with real assets
        outcome = "fail"
        err = f"{type(exc_).__name__}: {exc_}"
        raise
    finally:
        duration = time.time() - start
        record_evidence(
            "test_real_esm3_forward",
            asset=checkpoint or "esm3_sm_open_v1 (HuggingFace)",
            outcome=outcome,
            duration_s=duration,
            extra={
                "model_source": model_source,
                "device": device or "auto",
                "cache_dir": cache_dir,
                "output_shapes": shapes,
                "error": err,
            },
        )


def test_esm3_real_cache_hit_on_second_load():
    """Second load of the same checkpoint should hit the local cache.

    This catches the regression where a refactor silently re-downloads
    multi-GB weights on every process start.
    """
    checkpoint = os.environ.get("PTM2CELLNET_ESM3_CHECKPOINT")
    cache_dir = os.environ.get("PTM2CELLNET_ESM3_CACHE_DIR")

    try:
        from src.models.pretrained_encoders import ESM3Encoder
    except ImportError as exc:
        pytest.fail(f"ESM3Encoder not importable: {exc}")

    start = time.time()
    outcome = "pass"
    err: str | None = None
    second_load_s: float | None = None
    try:
        # First load primes the cache.
        ESM3Encoder(
            model_size="small", freeze=True, cache_dir=cache_dir,
            checkpoint_path=checkpoint,
        )
        # Second load should be cheap (cache hit).
        t0 = time.time()
        ESM3Encoder(
            model_size="small", freeze=True, cache_dir=cache_dir,
            checkpoint_path=checkpoint,
        )
        second_load_s = time.time() - t0
        # Loose bound: a 2.7GB download takes minutes; a cache hit is seconds.
        assert second_load_s < 60.0, (
            f"second ESM-3 load took {second_load_s:.1f}s — cache may not be working"
        )
    except Exception as exc_:  # pragma: no cover
        outcome = "fail"
        err = f"{type(exc_).__name__}: {exc_}"
        raise
    finally:
        record_evidence(
            "test_real_esm3_cache_hit",
            asset=checkpoint or "esm3_sm_open_v1 (HuggingFace)",
            outcome=outcome,
            duration_s=time.time() - start,
            extra={"second_load_s": second_load_s, "error": err},
        )
