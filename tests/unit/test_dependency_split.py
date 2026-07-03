"""Dependency-split contract tests (P1-2).

Pins down the structure that keeps a clean CPU environment able to run the
minimal E2E loop without dragging in heavy / CUDA-sensitive optional packages
(scvi-tools, mamba-ssm, transformers). These tests are static (no pip) so they
run fast in CI and catch accidental regressions to the old monolithic
requirements.txt.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

REQ_FILES = {
    "core": REPO_ROOT / "requirements-core.txt",
    "pretrained": REPO_ROOT / "requirements-pretrained.txt",
    "mamba": REPO_ROOT / "requirements-mamba.txt",
    "analysis": REPO_ROOT / "requirements-analysis.txt",
    "index": REPO_ROOT / "requirements.txt",
}

# Packages that are heavy / CUDA-sensitive and must NOT appear in core.
HEAVY_OPTIONAL = {
    "scvi-tools": "analysis",
    "sspa": "analysis",
    "mamba-ssm": "mamba",
    "transformers": "pretrained",
}


def _package_names(text: str) -> list[str]:
    """Extract lowercased bare package names (strip version specs/comments)."""
    names = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-r"):
            continue
        # Strip environment markers and extras.
        name = re.split(r"[<>=!\[]", line, maxsplit=1)[0].strip().lower()
        if name:
            names.append(name)
    return names


class TestRequirementFilesExist:
    @pytest.mark.parametrize("key", list(REQ_FILES))
    def test_file_exists_and_nonempty(self, key):
        path = REQ_FILES[key]
        assert path.exists(), f"{path.name} missing (P1-2 split)"
        assert path.read_text(encoding="utf-8").strip(), f"{path.name} is empty"


class TestCoreIsLean:
    def test_core_excludes_heavy_optional(self):
        text = REQ_FILES["core"].read_text(encoding="utf-8")
        names = set(_package_names(text))
        for pkg, extra in HEAVY_OPTIONAL.items():
            assert pkg not in names, (
                f"{pkg} must not be in requirements-core.txt — it belongs in "
                f"requirements-{extra}.txt. Putting it in core defeats the "
                "clean-environment E2E install (P1-2)."
            )

    def test_core_has_minimal_training_stack(self):
        """Core must be sufficient for the CNN/Transformer E2E loop."""
        names = set(_package_names(REQ_FILES["core"].read_text(encoding="utf-8")))
        for required in ("torch", "numpy", "pandas", "pyyaml", "fastapi", "uvicorn"):
            assert required in names, f"{required} missing from requirements-core.txt"

    def test_pretrained_includes_transformers(self):
        names = set(_package_names(REQ_FILES["pretrained"].read_text(encoding="utf-8")))
        assert "transformers" in names
        assert "lightning" in names

    def test_mamba_includes_mamba_ssm(self):
        names = set(_package_names(REQ_FILES["mamba"].read_text(encoding="utf-8")))
        assert "mamba-ssm" in names

    def test_analysis_includes_scvi(self):
        names = set(_package_names(REQ_FILES["analysis"].read_text(encoding="utf-8")))
        assert "scvi-tools" in names


class TestIndexAggregates:
    def test_index_references_core(self):
        text = REQ_FILES["index"].read_text(encoding="utf-8")
        assert "-r requirements-core.txt" in text

    def test_index_does_not_inline_heavy_optional(self):
        """The index may include core/pretrained but must not inline mamba/scvi."""
        names = set(_package_names(REQ_FILES["index"].read_text(encoding="utf-8")))
        assert "scvi-tools" not in names
        assert "mamba-ssm" not in names


class TestDockerfileUsesCore:
    """The Dockerfile installs core by default, not the old monolithic set."""

    def test_dockerfile_installs_core(self):
        text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "requirements-core.txt" in text, (
            "Dockerfile must install requirements-core.txt for a lean image (P1-2)."
        )
        # The default install must come from core, not the aggregated index.
        assert "pip install --no-cache-dir -r requirements-core.txt" in text

    def test_dockerfile_exposes_optional_build_args(self):
        text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        for arg in ("INSTALL_PRETRAINED", "INSTALL_MAMBA", "INSTALL_ANALYSIS"):
            assert arg in text, f"Dockerfile missing optional build arg {arg}"
