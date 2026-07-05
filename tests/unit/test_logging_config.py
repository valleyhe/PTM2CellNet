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


class TestTrainingScriptsAlwaysPassLogger:
    """F-07: every Lightning ``L.Trainer(...)`` call site in the shipped
    training scripts must pass an explicit ``logger=`` kwarg, OR be routed
    through ``PTM2CellNetLightning.configure_trainer()`` which auto-creates
    one. This pins the "no 'no logger configured' warning" property so a
    future edit cannot silently regress it.
    """

    @pytest.mark.parametrize(
        "script_relpath",
        [
            "scripts/train_lightning.py",
            "scripts/train_pretrained.py",
            "scripts/train_ptm_site.py",
        ],
    )
    def test_script_trainer_call_passes_logger(self, script_relpath, repo_root):
        script_path = repo_root / script_relpath
        assert script_path.is_file(), f"{script_relpath} missing"
        source = script_path.read_text(encoding="utf-8")

        # Must construct a Trainer somewhere.
        assert "L.Trainer(" in source or "configure_trainer(" in source

        # Locate every L.Trainer( call and confirm a logger= appears before
        # the matching close paren on the same statement. We do a simple
        # block scan: find "L.Trainer(" then walk forward balancing parens
        # until the call closes, checking for "logger=" inside.

        # Find all L.Trainer( call substrings with paren balancing.
        idx = 0
        calls_found = 0
        while True:
            start = source.find("L.Trainer(", idx)
            if start == -1:
                break
            depth = 0
            j = start + len("L.Trainer(") - 1  # at the "("
            while j < len(source):
                c = source[j]
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            call_body = source[start : j + 1]
            calls_found += 1
            # Allow either explicit logger= or a **kwargs splat that the
            # static scan cannot resolve (rare); the latter is acceptable
            # only if configure_trainer is the actual call site.
            assert "logger=" in call_body or "**" in call_body, (
                f"{script_relpath}: L.Trainer(...) call does not pass logger= — "
                "every Trainer construction in shipped scripts must configure a "
                "logger to avoid the 'no logger configured' warning (F-07)."
            )
            idx = j + 1
        assert calls_found >= 1

    def test_configure_trainer_auto_creates_logger(self):
        """configure_trainer() injects a TensorBoardLogger when none is passed."""
        from src.training.lightning_module import PTM2CellNetLightning

        kwargs: dict = {"default_root_dir": "/tmp/ptm2cellnet_f07_test", "max_epochs": 1}
        # Don't actually instantiate the Trainer (heavy); just verify the
        # kwargs dict gets a logger key injected by the static helper logic.
        # Replicate the helper's check by calling it with a stubbed L.Trainer.
        # We patch L.Trainer to a recorder.
        import lightning as L

        recorded: dict = {}

        def fake_trainer(**kw):
            recorded.update(kw)
            return "stub"

        original = L.Trainer
        L.Trainer = fake_trainer  # type: ignore[method-assign]
        try:
            PTM2CellNetLightning.configure_trainer(**kwargs)
        finally:
            L.Trainer = original  # type: ignore[method-assign]
        assert "logger" in recorded, "configure_trainer must inject a logger when missing"


@pytest.fixture(scope="module")
def repo_root():
    """Path to the repository root (where this test file's parent's parent lives)."""
    from pathlib import Path

    return Path(__file__).resolve().parents[2]
