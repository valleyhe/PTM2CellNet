"""Smoke tests for distributed_trainer (src/training/distributed.py)."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.training.distributed import distributed_trainer


class TestDistributedTrainingSmoke:
    """Smoke tests for distributed_trainer covering DDP, strict mode, and env-var overrides."""

    # ----------------------------------------------------------------
    # Test 1: DDP strategy when Lightning available and multiple GPUs
    # ----------------------------------------------------------------
    def test_ddp_strategy_with_multiple_gpus(self):
        """DDP strategy is used when Lightning is available and multiple GPUs detected."""
        mock_lightning = MagicMock()
        with patch("src.training.distributed._LIGHTNING_MODULE", mock_lightning):
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.device_count", return_value=4):
                    model = MagicMock()
                    config = {"training": {"lr": 0.001}}
                    distributed_trainer(model, config=config)

                    mock_lightning.Trainer.assert_called_once()
                    _call_kwargs = mock_lightning.Trainer.call_args.kwargs
                    assert _call_kwargs["strategy"] == "ddp"
                    assert _call_kwargs["devices"] == 4
                    assert _call_kwargs["accelerator"] == "gpu"

    # ----------------------------------------------------------------
    # Test 2: Strict mode raises RuntimeError when Lightning unavailable
    # ----------------------------------------------------------------
    def test_strict_mode_raises_without_lightning(self):
        """Strict mode raises RuntimeError when Lightning is not available."""
        os.environ["PTM2CELLNET_STRICT_MODEL_ASSETS"] = "1"
        try:
            with patch("src.training.distributed._LIGHTNING_MODULE", None):
                with patch(
                    "src.training.distributed._LIGHTNING_IMPORT_ERROR",
                    ImportError("missing lightning"),
                ):
                    model = MagicMock()
                    config = {"training": {"lr": 0.001}}
                    with pytest.raises(RuntimeError):
                        distributed_trainer(model, config=config, strict=True)
        finally:
            os.environ.pop("PTM2CELLNET_STRICT_MODEL_ASSETS", None)

    # ----------------------------------------------------------------
    # Test 3: Strict mode raises RuntimeError when insufficient GPUs
    # ----------------------------------------------------------------
    def test_strict_mode_raises_with_insufficient_gpus(self):
        """Strict mode raises RuntimeError when fewer GPUs than requested are available."""
        with patch("src.training.distributed._LIGHTNING_MODULE", MagicMock()):
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.device_count", return_value=1):
                    model = MagicMock()
                    config = {"training": {"lr": 0.001}}
                    with pytest.raises(RuntimeError):
                        distributed_trainer(
                            model, config=config, devices=4, strict=True
                        )

    # ----------------------------------------------------------------
    # Test 4: Env var strict mode
    # ----------------------------------------------------------------
    def test_env_var_strict_mode_with_insufficient_gpus(self):
        """PTM2CELLNET_STRICT_MODEL_ASSETS env var enforces strict mode without explicit arg."""
        os.environ["PTM2CELLNET_STRICT_MODEL_ASSETS"] = "1"
        try:
            with patch("src.training.distributed._LIGHTNING_MODULE", MagicMock()):
                with patch("torch.cuda.is_available", return_value=True):
                    with patch("torch.cuda.device_count", return_value=1):
                        model = MagicMock()
                        config = {"training": {"lr": 0.001}}
                        with pytest.raises(RuntimeError):
                            distributed_trainer(
                                model, config=config, devices=4
                            )
        finally:
            os.environ.pop("PTM2CELLNET_STRICT_MODEL_ASSETS", None)
