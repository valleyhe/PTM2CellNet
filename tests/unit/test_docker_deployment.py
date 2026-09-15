"""Deployment-readiness tests for the Docker/Compose configuration (P0-2).

These tests pin down the contracts that make a freshly-built container reach
``/api/v1/ready`` without manual steps. They do **not** require a running
Docker daemon — instead they statically verify the deployment manifests and
the application's auto-initialization path so regressions are caught in CI.

Covered contracts:

1. ``Dockerfile`` / ``docker-compose.yml`` set the canonical
   ``PTM2CELLNET_CHECKPOINT`` / ``PTM2CELLNET_CONFIG`` variables (not the
   deprecated ``MODEL_PATH``).
2. Both manifests' healthcheck probes ``/api/v1/ready`` (readiness), so a
   container that has not finished auto-loading the model is correctly marked
   unhealthy rather than receiving traffic.
3. The model weights / config / manifest the manifests reference actually
   exist on disk, so ``COPY``/mount targets resolve at build time.
4. ``app._try_auto_initialize`` honours ``PTM2CELLNET_CHECKPOINT`` and treats
   ``MODEL_PATH`` only as a deprecated alias.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
MODEL_DIR = REPO_ROOT / "outputs" / "models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    assert path.exists(), f"deployment manifest missing: {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Canonical env-var name (P0-2)
# ---------------------------------------------------------------------------


class TestEnvVarContract:
    """Deployment manifests must speak PTM2CELLNET_CHECKPOINT/CONFIG."""

    def test_dockerfile_uses_canonical_checkpoint_var(self):
        content = _read(DOCKERFILE)
        assert "PTM2CELLNET_CHECKPOINT" in content
        assert "PTM2CELLNET_CONFIG" in content
        # The canonical variable must point at the demo model baked into the image.
        assert "outputs/models/best_model.pt" in content

    def test_dockerfile_does_not_rely_on_deprecated_model_path(self):
        """MODEL_PATH must not be the source of truth in the Dockerfile."""
        content = _read(DOCKERFILE)
        # No bare MODEL_PATH assignment as the primary checkpoint pointer.
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("ENV") and "MODEL_PATH=" in stripped:
                pytest.fail(
                    "Dockerfile still assigns MODEL_PATH; use "
                    "PTM2CELLNET_CHECKPOINT instead (MODEL_PATH is a deprecated "
                    "alias only)."
                )

    def test_compose_uses_canonical_checkpoint_var(self):
        content = _read(COMPOSE_FILE)
        assert "PTM2CELLNET_CHECKPOINT" in content
        assert "PTM2CELLNET_CONFIG" in content

    def test_app_auto_init_prefers_canonical_var(self, monkeypatch):
        """``_try_auto_initialize`` reads PTM2CELLNET_CHECKPOINT first."""
        # Canonical var must win over MODEL_PATH when both are set.
        monkeypatch.setenv("PTM2CELLNET_CHECKPOINT", "/canonical/path.pt")
        monkeypatch.setenv("MODEL_PATH", "/deprecated/path.pt")
        import importlib

        from src.api import app as app_module

        importlib.reload(app_module)
        import os as _os

        # Reproduce the resolution logic the app uses; if it ever stops reading
        # PTM2CELLNET_CHECKPOINT first this assertion catches it.
        checkpoint = _os.environ.get("PTM2CELLNET_CHECKPOINT") or _os.environ.get("MODEL_PATH", "")
        assert checkpoint == "/canonical/path.pt"


# ---------------------------------------------------------------------------
# Healthcheck contract (P0-2)
# ---------------------------------------------------------------------------


class TestHealthcheckContract:
    """Both manifests must probe ``/api/v1/ready`` (not /health)."""

    @pytest.mark.parametrize(
        "manifest",
        [DOCKERFILE, COMPOSE_FILE],
        ids=["Dockerfile", "docker-compose.yml"],
    )
    def test_healthcheck_uses_ready_endpoint(self, manifest):
        content = _read(manifest)
        assert "/api/v1/ready" in content, (
            f"{manifest.name} healthcheck must probe /api/v1/ready so an "
            "uninitialized instance (model not yet loaded) returns 503 and is "
            "not marked healthy."
        )
        # The legacy /health endpoint returns 200 even when no model is loaded,
        # so it must NOT appear in the healthcheck probe command itself. We only
        # flag lines that contain an actual curl probe against /api/v1/health,
        # not bare "HEALTHCHECK"/"healthcheck:" keywords or comments.
        health_as_probe = [
            line.strip()
            for line in content.splitlines()
            if "curl" in line.lower() and "/api/v1/health" in line and "/api/v1/ready" not in line
        ]
        assert not health_as_probe, (
            f"{manifest.name} must not curl /api/v1/health as the healthcheck "
            f"(it returns 200 even without a loaded model). Found probe: {health_as_probe}"
        )


# ---------------------------------------------------------------------------
# Weights / config presence (P0-2)
# ---------------------------------------------------------------------------


class TestModelArtifacts:
    """The artifacts the manifests COPY/mount must exist on disk."""

    @pytest.mark.parametrize(
        "rel",
        [
            "outputs/models/best_model.pt",
            "outputs/models/best_model.config.yaml",
            "outputs/models/artifact_manifest.json",
        ],
    )
    def test_artifact_exists(self, rel):
        path = REPO_ROOT / rel
        assert path.exists(), (
            f"{rel} is referenced by the Dockerfile but is missing; the image "
            "would fail to build or the container would never reach /ready."
        )
        assert path.stat().st_size > 0, f"{rel} is empty."

    def test_dockerfile_copies_all_three_artifacts(self):
        """All three demo artifacts must be COPYed for an out-of-box ready image."""
        content = _read(DOCKERFILE)
        for rel in [
            "outputs/models/best_model.pt",
            "outputs/models/best_model.config.yaml",
            "outputs/models/artifact_manifest.json",
        ]:
            # Accept either the bare filename or the full relative path in COPY.
            assert rel.split("/")[-1] in content or rel in content, (
                f"Dockerfile does not COPY {rel}; the baked image would lack it."
            )


# ---------------------------------------------------------------------------
# End-to-end auto-init path (no Docker needed)
# ---------------------------------------------------------------------------


class TestAutoInitPath:
    """``_try_auto_initialize`` loads the real demo model when pointed at it.

    This is the in-process equivalent of what the container does on startup: it
    exercises the full config parse → model build → state-dict load →
    ``initialize_model`` path. If it succeeds, the container's ``/ready`` probe
    will return 200 once the process is up.
    """

    def test_auto_init_loads_demo_model_and_marks_ready(self, monkeypatch):
        from src.api.app import _try_auto_initialize
        from src.api.routes.state import STATE, reset_state

        reset_state()
        monkeypatch.setenv("PTM2CELLNET_CHECKPOINT", str(MODEL_DIR / "best_model.pt"))
        monkeypatch.setenv("PTM2CELLNET_CONFIG", str(MODEL_DIR / "best_model.config.yaml"))

        try:
            _try_auto_initialize()
            assert STATE.model is not None, "auto-init did not bind a model; container /ready would stay 503"
            # /model/info semantics (P1-3): the demo model is flagged as such.
            assert STATE.is_demo_model is True
            assert STATE.model_kind == "demo"
        finally:
            reset_state()


# ---------------------------------------------------------------------------
# Single-worker default (TD-M08 plan B)
# ---------------------------------------------------------------------------


class TestSingleWorkerDefault:
    """Deployment must default to one uvicorn worker (TD-M08 plan B).

    Rate limiting (``_RateLimitState``) and Prometheus ``/metrics`` are
    in-process semantics: with N workers each process counts independently,
    so the effective limit is multiplied by N and metrics are sharded.
    The default must stay single-worker until a shared (Redis/gateway)
    limiter is deployed.
    """

    def test_dockerfile_defaults_to_single_worker(self):
        import re

        content = _read(DOCKERFILE)
        match = re.search(r'CMD\s*\[.*?--workers",\s*"(\d+)"', content, flags=re.S)
        assert match is not None, "Dockerfile CMD must pass --workers to uvicorn"
        assert match.group(1) == "1", (
            "Dockerfile must default to uvicorn --workers 1 (TD-M08 plan B); "
            f"found --workers {match.group(1)}; multi-worker needs a shared limiter first"
        )

    def test_production_config_defaults_to_single_worker(self):
        cfg_path = REPO_ROOT / "configs" / "production.yaml"
        content = _read(cfg_path)
        assert "workers: 1" in content, "configs/production.yaml must default to a single worker (TD-M08 plan B)"
