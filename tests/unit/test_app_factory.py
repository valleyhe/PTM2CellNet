"""Tests for the FastAPI application factory in src/api/app.py.

Covers the ``create_app()`` factory: app identity, route registration, the
health/liveness/readiness endpoints, CORS configuration, request body size
limiting, and rate limiting. Follows the patterns in ``test_api.py`` and
``test_api_routes.py`` — a minimal ``_SimpleModel`` plus the
``initialize_model``/``reset_state`` STATE lifecycle helpers.
"""

import os
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.app import (
    _DEFAULT_RATE_LIMIT_BURST,
    _DEFAULT_RATE_LIMIT_RPM,
    _MAX_REQUEST_BODY_BYTES,
    create_app,
)
from src.api.routes.state import STATE, reset_state


# ---------------------------------------------------------------------------
# Minimal test model (matches the SimpleModel pattern from test_api.py)
# ---------------------------------------------------------------------------


class _SimpleModel(nn.Module):
    """Minimal nn.Module that satisfies the route handler's expectations."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(21, 64)  # 0=padding, 1-20=amino acids
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.classifier = nn.Linear(128, num_classes)
        self.encoder_type = "lstm"
        self.embed_dim = 64

    def forward(self, batch):
        sequences = batch["sequence"]
        x = self.embedding(sequences)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        probabilities = torch.softmax(logits, dim=-1)
        return {"logits": logits, "probabilities": probabilities}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_state():
    """Reset global STATE before and after each test (P0-1)."""
    reset_state()
    yield
    reset_state()


@pytest.fixture()
def app_client():
    """TestClient with an initialized model."""
    app = create_app()
    model = _SimpleModel(num_classes=4)
    cell_states = ["proliferation", "differentiation", "apoptosis", "quiescence"]
    STATE.model = model
    STATE.cell_states = cell_states
    STATE.idx_to_label = {i: s for i, s in enumerate(cell_states)}
    STATE.label_to_idx = {s: i for i, s in enumerate(cell_states)}
    STATE.device = "cpu"
    return TestClient(app)


@pytest.fixture()
def app_client_no_model():
    """TestClient without model initialization (unhealthy state)."""
    reset_state()
    STATE.model = None
    STATE.cell_states = []
    STATE.idx_to_label = {}
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Factory: identity & return type
# ---------------------------------------------------------------------------


class TestCreateAppFactory:
    """Direct tests of the ``create_app()`` factory function."""

    def test_create_app_returns_fastapi_app(self):
        """``create_app`` returns a configured ``FastAPI`` instance."""
        app = create_app()
        assert isinstance(app, FastAPI)
        assert app.title == "PTM2CellNet API"
        assert app.version == "1.0.0"

    def test_create_app_registers_routes(self):
        """The router is mounted under the configured API prefix.

        Asserts via real HTTP requests (the only reliable way to verify route
        registration across FastAPI's ``include_router`` indirection — the
        merged route table is not directly enumerable on ``app.routes``).
        """
        app = create_app()
        client = TestClient(app)
        prefix = os.environ.get("PTM2CELLNET_API_PREFIX", "/api/v1")
        # A representative slice of the prediction/health surface. We do not
        # assert the response body — only that each route resolves (i.e. is
        # registered) rather than returning 404.
        for path in (
            "/health",
            "/live",
            "/ready",
            "/model_info",
        ):
            resp = client.get(f"{prefix}{path}")
            assert resp.status_code != 404, f"{prefix}{path} not registered"
        # POST routes (404 check only — validation/model errors are fine).
        for path in ("/predict", "/batch_predict"):
            resp = client.post(f"{prefix}{path}", json={})
            assert resp.status_code != 404, f"{prefix}{path} not registered"
        # Root index is registered directly on the app.
        assert client.get("/").status_code != 404

    def test_create_app_has_health_endpoint(self):
        """The factory exposes a ``/api/v1/health`` endpoint."""
        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_loaded"] is False  # no model loaded in this fixture
        assert "version" in data

    def test_create_app_with_cors_origins(self):
        """Explicit ``cors_origins`` override env defaults."""
        origins = ["https://example.com", "https://api.example.com"]
        app = create_app(cors_origins=origins)
        client = TestClient(app)

        resp = client.get(
            "/api/v1/health",
            headers={"Origin": "https://example.com"},
        )
        assert resp.status_code == 200
        # CORSMiddleware echoes the allowed origin back when the request origin
        # is in the allowlist.
        assert resp.headers.get("access-control-allow-origin") == "https://example.com"

    def test_create_app_with_cors_origins_env_override(self):
        """``PTM2CELLNET_CORS_ORIGINS`` env var supplies origins when kwarg is None."""
        env_key = "PTM2CELLNET_CORS_ORIGINS"
        prev = os.environ.get(env_key)
        os.environ[env_key] = "https://env.example.com"
        try:
            app = create_app()
            client = TestClient(app)
            resp = client.get(
                "/api/v1/health",
                headers={"Origin": "https://env.example.com"},
            )
            assert resp.status_code == 200
            assert (
                resp.headers.get("access-control-allow-origin")
                == "https://env.example.com"
            )
        finally:
            if prev is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = prev


# ---------------------------------------------------------------------------
# Request body size limit
# ---------------------------------------------------------------------------


class TestRequestBodySizeLimit:
    """``_RequestBodySizeLimitMiddleware`` rejects oversized payloads (413)."""

    def test_create_app_request_body_size_limit(self):
        """A payload above the configured limit yields 413."""
        # Use a tiny limit so the test stays fast and deterministic.
        env_key = "PTM2CELLNET_MAX_REQUEST_SIZE"
        prev = os.environ.get(env_key)
        os.environ[env_key] = "1024"  # 1 KB
        try:
            app = create_app()
            client = TestClient(app)
            # Build a payload clearly larger than the 1 KB limit.
            big_sequence = "A" * 5000
            payload = {"sequence": big_sequence, "ptm_sites": []}
            # TestClient sets Content-Length from the json body automatically.
            resp = client.post("/api/v1/predict", json=payload)
            assert resp.status_code == 413
        finally:
            if prev is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = prev

    def test_default_max_body_is_ten_mb(self):
        """The module-level default request size limit is 10 MB."""
        assert _MAX_REQUEST_BODY_BYTES == 10 * 1024 * 1024


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """The in-process rate limiter returns 429 once the budget is exhausted."""

    def test_create_app_rate_limiting(self):
        """Requests beyond the configured burst/rpm receive 429."""
        env_rpm = "PTM2CELLNET_RATE_LIMIT_RPM"
        env_burst = "PTM2CELLNET_RATE_LIMIT_BURST"
        prev_rpm = os.environ.get(env_rpm)
        prev_burst = os.environ.get(env_burst)
        # Tight budget: 1 req/min, burst=1 — the second request must be rejected.
        os.environ[env_rpm] = "1"
        os.environ[env_burst] = "1"
        try:
            app = create_app()
            client = TestClient(app)
            # Health probes are exempt, so exercise a prediction-shape endpoint.
            # /model_info is convenient and does not require a loaded model
            # (it returns 503, which still counts as a response, not a 429).
            first = client.get("/api/v1/model_info")
            second = client.get("/api/v1/model_info")
            assert first.status_code in (200, 503)  # not throttled
            assert second.status_code == 429
            assert "Retry-After" in second.headers
        finally:
            for key, prev in ((env_rpm, prev_rpm), (env_burst, prev_burst)):
                if prev is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prev

    def test_rate_limiting_disabled_when_rpm_zero(self):
        """``PTM2CELLNET_RATE_LIMIT_RPM=0`` disables the rate-limit middleware."""
        env_rpm = "PTM2CELLNET_RATE_LIMIT_RPM"
        prev = os.environ.get(env_rpm)
        os.environ[env_rpm] = "0"
        try:
            app = create_app()
            # When disabled, no _RateLimitMiddleware is registered.
            middleware_names = [type(m).__name__ for m in app.user_middleware]
            assert "_RateLimitMiddleware" not in middleware_names
        finally:
            if prev is None:
                os.environ.pop(env_rpm, None)
            else:
                os.environ[env_rpm] = prev

    def test_default_rate_limit_policy(self):
        """Module-level default RPM/burst match the documented production default."""
        # Defaults are read from env at import time; assert against the constants
        # which encode the fallback values.
        assert _DEFAULT_RATE_LIMIT_RPM > 0
        assert _DEFAULT_RATE_LIMIT_BURST > 0


# ---------------------------------------------------------------------------
# Liveness / readiness probes
# ---------------------------------------------------------------------------


class TestProbeEndpoints:
    """Liveness (always 200) and readiness (503 without model) probes."""

    def test_app_live_endpoint(self, app_client_no_model):
        """``/live`` reports process liveness regardless of model state."""
        resp = app_client_no_model.get("/api/v1/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_app_live_endpoint_with_model(self, app_client):
        resp = app_client.get("/api/v1/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_app_ready_endpoint_without_model(self, app_client_no_model):
        """``/ready`` returns 503 until a model is loaded."""
        resp = app_client_no_model.get("/api/v1/ready")
        assert resp.status_code == 503

    def test_app_ready_endpoint_with_model(self, app_client):
        resp = app_client.get("/api/v1/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True
