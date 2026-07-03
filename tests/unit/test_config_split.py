"""Config-split contract tests (P1-3).

Ensures the entrypoint configs exist and are correctly tiered:
* ``configs/smoke/*`` — lightweight CPU/1-epoch quickstart.
* ``configs/research/*`` — heavier real-data training.
The default ``train.py`` config must point at a smoke entrypoint so the
naive ``python scripts/train.py`` first run is fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"


class TestSmokeConfigs:
    def test_cnn_cpu_smoke_exists(self):
        path = CONFIGS / "smoke" / "cnn_cpu.yaml"
        assert path.exists(), "configs/smoke/cnn_cpu.yaml missing (P1-3)"

    def test_lightning_cnn_cpu_smoke_exists(self):
        path = CONFIGS / "smoke" / "lightning_cnn_cpu.yaml"
        assert path.exists(), "configs/smoke/lightning_cnn_cpu.yaml missing"

    def test_smoke_config_is_lightweight(self):
        """Smoke config must be small enough for a sub-minute CPU run."""
        cfg = Config.from_yaml(str(CONFIGS / "smoke" / "cnn_cpu.yaml"))
        assert cfg.get("model.encoder_type") == "cnn"
        assert cfg.get("training.max_epochs") == 1
        # CPU-friendly hidden/embed dims.
        assert cfg.get("model.embed_dim", 32) <= 64
        # A reasonable batch size for a quickstart.
        assert cfg.get("training.batch_size") <= 32

    def test_lightning_smoke_uses_cpu_and_no_mixed_precision(self):
        cfg = Config.from_yaml(str(CONFIGS / "smoke" / "lightning_cnn_cpu.yaml"))
        assert cfg.get("training.accelerator") == "cpu"
        assert cfg.get("training.precision") == "32"


class TestResearchConfigs:
    def test_transformer_research_exists(self):
        path = CONFIGS / "research" / "transformer.yaml"
        assert path.exists(), "configs/research/transformer.yaml missing (P1-3)"

    def test_research_config_is_heavier_than_smoke(self):
        cfg = Config.from_yaml(str(CONFIGS / "research" / "transformer.yaml"))
        assert cfg.get("model.encoder_type") == "transformer"
        # Research config trains for real — not 1 epoch.
        assert cfg.get("training.max_epochs", 0) > 1


class TestDefaultEntrypoint:
    def test_train_py_default_config_is_smoke(self):
        """``python scripts/train.py`` (no --config) must use the smoke entrypoint.

        This is the contract that makes a naive first run fast instead of
        launching the heavy transformer default.
        """
        import scripts.train as train_mod

        old_argv = __import__("sys").argv
        __import__("sys").argv = ["train.py"]
        try:
            args = train_mod.parse_args()
        finally:
            __import__("sys").argv = old_argv
        assert args.config == "configs/smoke/cnn_cpu.yaml", (
            f"train.py 默认配置应为 smoke 入口，实际为 {args.config}"
        )

    def test_default_yaml_is_documented_as_research(self):
        """configs/default.yaml must warn it is not the quickstart entrypoint."""
        text = (CONFIGS / "default.yaml").read_text(encoding="utf-8")
        assert "smoke" in text.lower(), (
            "configs/default.yaml should document that the smoke config is the "
            "recommended first-run entrypoint (P1-3)."
        )
