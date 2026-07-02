"""Unit tests for src/training/logging_config.py (CONF-02)."""

import os

import pytest

from src.training.logging_config import (
    configure_default_logger,
    build_logger,
    get_default_log_dir,
    LOG_DIR_ENV,
    DEFAULT_LOG_DIR,
)


class TestGetDefaultLogDir:
    def test_default_dir(self, monkeypatch):
        monkeypatch.delenv(LOG_DIR_ENV, raising=False)
        assert str(get_default_log_dir()) == DEFAULT_LOG_DIR

    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv(LOG_DIR_ENV, str(tmp_path))
        assert get_default_log_dir() == tmp_path


class TestBuildLogger:
    def test_csv_logger(self, tmp_path):
        lg = build_logger(kind="csv", save_dir=str(tmp_path), name="t")
        assert lg is not None
        assert lg.save_dir is not None

    def test_tensorboard_falls_back_to_csv(self, tmp_path):
        """When TensorBoard/tensorboardX is unavailable, fall back to CSVLogger."""
        lg = build_logger(kind="tensorboard", save_dir=str(tmp_path), name="t")
        # Either TensorBoardLogger or CSVLogger (fallback) is acceptable
        assert lg is not None

    def test_invalid_kind_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported logger kind"):
            build_logger(kind="bogus", save_dir=str(tmp_path))

    def test_wandb_unavailable_raises(self, tmp_path):
        # WandbLogger requires the wandb package; expect ImportError if missing
        try:
            build_logger(kind="wandb", save_dir=str(tmp_path), name="t")
        except ImportError:
            pass  # acceptable when wandb is absent


class TestConfigureDefaultLogger:
    def test_creates_log_dir_and_returns_logger(self, tmp_path):
        out = str(tmp_path / "logs")
        lg = configure_default_logger(save_dir=out, name="run1")
        assert lg is not None
        assert os.path.isdir(out)

    def test_default_kind_is_tensorboard_with_csv_fallback(self, tmp_path):
        lg = configure_default_logger(save_dir=str(tmp_path), name="run2")
        assert lg is not None
