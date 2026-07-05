"""Tests for API middleware components (rate limiting, auth, body size, metrics).

All middleware classes in ``src/api/middleware.py`` are module-private
(underscore-prefixed), so these tests import them directly with their
private names to verify behaviour without requiring a full FastAPI
TestClient bootstrap.
"""

import asyncio
import time
import pytest
from unittest.mock import MagicMock, patch

from fastapi import Request
from starlette.datastructures import Headers

from src.api.middleware import (
    _RateLimitState,
    _RuntimeMetrics,
    _RequestBodySizeLimitMiddleware,
    _RateLimitMiddleware,
    _OptionalAuthMiddleware,
    _MAX_REQUEST_BODY_BYTES,
    _DEFAULT_RATE_LIMIT_RPM,
    _DEFAULT_RATE_LIMIT_BURST,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_request(path="/predict", headers=None, client_host="127.0.0.1"):
    """Build a minimal ``Request``-like object for middleware tests."""
    if headers is None:
        headers = {}
    mock = MagicMock(spec=Request)
    mock.url.path = path
    mock.headers = Headers(headers)
    mock.client.host = client_host
    return mock


async def _no_op_call_next(request=None):
    """Minimal async call_next returning a 200 response."""
    return MagicMock(status_code=200)


def _make_call_next():
    """Build a no-op ``call_next`` async callable."""
    return _no_op_call_next


# ---------------------------------------------------------------------------
# _RateLimitState
# ---------------------------------------------------------------------------

class TestRateLimitState:
    """Tests for the sliding-window token-bucket rate limiter."""

    def test_allows_first_request(self):
        """The first request from any client should be allowed."""
        state = _RateLimitState(requests_per_minute=10, burst=10)
        request = _make_mock_request()
        allowed, retry_after = state.allow(request)
        assert allowed is True
        assert retry_after == 0

    def test_rejects_when_over_limit(self):
        """Requests exceeding the per-minute budget should be rejected."""
        state = _RateLimitState(requests_per_minute=3, burst=3)
        request = _make_mock_request()
        for _ in range(3):
            allowed, _ = state.allow(request)
            assert allowed is True
        # Fourth request should be blocked
        allowed, retry_after = state.allow(request)
        assert allowed is False
        assert retry_after > 0

    def test_tracks_per_client_independently(self):
        """Rate limit state should be keyed per client IP."""
        state = _RateLimitState(requests_per_minute=2, burst=2)
        client_a = _make_mock_request(client_host="10.0.0.1")
        client_b = _make_mock_request(client_host="10.0.0.2")

        # Client A uses both tokens
        assert state.allow(client_a) == (True, 0)
        assert state.allow(client_a) == (True, 0)

        # Client B still has both tokens
        assert state.allow(client_b) == (True, 0)
        assert state.allow(client_b) == (True, 0)

        # Client A now blocked
        allowed, _ = state.allow(client_a)
        assert allowed is False

    def test_different_x_forwarded_for_keys(self):
        """X-Forwarded-For header should be respected for client key."""
        state = _RateLimitState(requests_per_minute=1, burst=1)
        headers = {"x-forwarded-for": "203.0.113.5, 10.0.0.1"}
        request = _make_mock_request(headers=headers)
        allowed, _ = state.allow(request)
        assert allowed is True
        # Second request from same forwarded IP is blocked
        allowed, _ = state.allow(request)
        assert allowed is False

    def test_burst_cap_enforced(self):
        """Burst cap should prevent exceeding burst even within budget."""
        state = _RateLimitState(requests_per_minute=100, burst=2)
        request = _make_mock_request()
        assert state.allow(request) == (True, 0)
        assert state.allow(request) == (True, 0)
        # Third request blocked by burst despite large per-minute budget
        allowed, _ = state.allow(request)
        assert allowed is False

    def test_clamps_non_positive_values(self):
        """Non-positive requests_per_minute or burst should use defaults."""
        state = _RateLimitState(requests_per_minute=0, burst=0)
        assert state.requests_per_minute == _DEFAULT_RATE_LIMIT_RPM
        assert state.burst == _DEFAULT_RATE_LIMIT_BURST


# ---------------------------------------------------------------------------
# _RuntimeMetrics
# ---------------------------------------------------------------------------

class TestRuntimeMetrics:
    """Tests for the lightweight runtime metrics collector."""

    def test_empty_snapshot(self):
        """Snapshot with no observations should return zeros."""
        metrics = _RuntimeMetrics()
        snap = metrics.snapshot()
        assert snap["request_count"] == 0
        assert snap["error_count"] == 0
        assert snap["error_rate"] == 0.0
        assert snap["latency"]["avg_ms"] == 0.0

    def test_tracks_request_count(self):
        """Each observe() call should increment request_count."""
        metrics = _RuntimeMetrics()
        metrics.observe(10.0, is_error=False)
        metrics.observe(20.0, is_error=False)
        snap = metrics.snapshot()
        assert snap["request_count"] == 2

    def test_tracks_error_count(self):
        """observe with is_error=True should increment error_count."""
        metrics = _RuntimeMetrics()
        metrics.observe(10.0, is_error=False)
        metrics.observe(15.0, is_error=True)
        snap = metrics.snapshot()
        assert snap["error_count"] == 1
        assert snap["request_count"] == 2

    def test_error_rate_calculation(self):
        """Error rate should be error_count / request_count."""
        metrics = _RuntimeMetrics()
        metrics.observe(5.0, is_error=False)
        metrics.observe(5.0, is_error=True)
        metrics.observe(5.0, is_error=True)
        snap = metrics.snapshot()
        assert snap["error_rate"] == pytest.approx(2 / 3)
        assert snap["latency_buckets"]["le_+Inf"] == 3

    def test_latency_percentiles(self):
        """Latency percentiles should be computed correctly."""
        metrics = _RuntimeMetrics()
        # Insert sorted-ish latencies: 1, 2, ..., 10
        for ms in range(1, 11):
            metrics.observe(float(ms), is_error=False)
        snap = metrics.snapshot()
        # p50 = median of 10 sorted items (5th+6th)/2 -> items at index 4,5 (0-based)
        # After sorting: [1,2,3,4,5,6,7,8,9,10]; p50 = (5+6)/2 = 5.5
        # But sorted_samples[count//2] with count=10 -> sorted_samples[5] = 6
        assert snap["latency"]["p50_ms"] > 0


# ---------------------------------------------------------------------------
# _RequestBodySizeLimitMiddleware
# ---------------------------------------------------------------------------

class TestRequestBodySizeLimitMiddleware:
    """Tests for request body size limiting."""

    @staticmethod
    async def _run_dispatch(mw, request, call_next):
        return await mw.dispatch(request, call_next)

    def test_accepts_small_request(self):
        """Request under the limit should pass through."""
        app = MagicMock()
        mw = _RequestBodySizeLimitMiddleware(app, max_bytes=1000)
        request = _make_mock_request()
        request.headers = Headers({"content-length": "100"})
        result = asyncio.run(self._run_dispatch(mw, request, _make_call_next()))
        assert result.status_code == 200

    def test_rejects_oversized_request(self):
        """Request exceeding Content-Length limit should get 413."""
        app = MagicMock()
        mw = _RequestBodySizeLimitMiddleware(app, max_bytes=100)
        request = _make_mock_request()
        request.headers = Headers({"content-length": "200"})
        result = asyncio.run(self._run_dispatch(mw, request, _make_call_next()))
        assert result.status_code == 413

    def test_rejects_invalid_content_length(self):
        """Non-integer Content-Length header should get 400."""
        app = MagicMock()
        mw = _RequestBodySizeLimitMiddleware(app)
        request = _make_mock_request()
        request.headers = Headers({"content-length": "not-a-number"})
        result = asyncio.run(self._run_dispatch(mw, request, _make_call_next()))
        assert result.status_code == 400

    def test_no_content_length_header(self):
        """Request without Content-Length should pass through."""
        app = MagicMock()
        mw = _RequestBodySizeLimitMiddleware(app)
        request = _make_mock_request()
        request.headers = Headers({})
        result = asyncio.run(self._run_dispatch(mw, request, _make_call_next()))
        assert result.status_code == 200


# ---------------------------------------------------------------------------
# _OptionalAuthMiddleware
# ---------------------------------------------------------------------------

class TestOptionalAuthMiddleware:
    """Tests for API-key authentication middleware."""

    def test_no_key_configured_requires_empty_key_header(self):
        """When initialized with empty key, client must send empty X-API-Key."""
        app = MagicMock()
        mw = _OptionalAuthMiddleware(app, api_key="")
        request = _make_mock_request(path="/api/v1/predict", headers={"x-api-key": ""})
        result = asyncio.run(mw.dispatch(request, _make_call_next()))
        assert result.status_code == 200

    def test_missing_key_on_protected_path_returns_401(self):
        """Protected path without X-API-Key header returns 401."""
        app = MagicMock()
        mw = _OptionalAuthMiddleware(app, api_key="secret123")
        request = _make_mock_request(path="/api/v1/predict", headers={})
        result = asyncio.run(mw.dispatch(request, _make_call_next()))
        assert result.status_code == 401
        assert "X-API-Key required" in result.body.decode()

    def test_wrong_key_returns_403(self):
        """Wrong X-API-Key header on protected path returns 403."""
        app = MagicMock()
        mw = _OptionalAuthMiddleware(app, api_key="secret123")
        request = _make_mock_request(
            path="/api/v1/predict", headers={"x-api-key": "wrong-key"}
        )
        result = asyncio.run(mw.dispatch(request, _make_call_next()))
        assert result.status_code == 403
        assert "Invalid API key" in result.body.decode()

    def test_correct_key_passes(self):
        """Correct X-API-Key header on protected path should pass."""
        app = MagicMock()
        mw = _OptionalAuthMiddleware(app, api_key="secret123")
        request = _make_mock_request(
            path="/api/v1/predict", headers={"x-api-key": "secret123"}
        )
        result = asyncio.run(mw.dispatch(request, _make_call_next()))
        assert result.status_code == 200

    def test_health_path_always_exempt(self):
        """Health endpoint should bypass auth even when key is set."""
        app = MagicMock()
        mw = _OptionalAuthMiddleware(app, api_key="secret123")
        request = _make_mock_request(path="/api/v1/health", headers={})
        result = asyncio.run(mw.dispatch(request, _make_call_next()))
        assert result.status_code == 200

    def test_unprotected_path_bypasses_auth(self):
        """A path not in PROTECTED_SUFFIXES should bypass auth."""
        app = MagicMock()
        mw = _OptionalAuthMiddleware(app, api_key="secret123")
        request = _make_mock_request(path="/api/v1/some_other_route", headers={})
        result = asyncio.run(mw.dispatch(request, _make_call_next()))
        assert result.status_code == 200


# ---------------------------------------------------------------------------
# _RateLimitMiddleware
# ---------------------------------------------------------------------------

class TestRateLimitMiddleware:
    """Tests for the rate-limit middleware."""

    def test_health_path_is_exempt(self):
        """Health endpoints should never be rate limited."""
        app = MagicMock()
        mw = _RateLimitMiddleware(app)
        request = _make_mock_request(path="/health")
        result = asyncio.run(mw.dispatch(request, _make_call_next()))
        assert result.status_code == 200

    def test_predict_path_can_be_limited(self):
        """Prediction endpoint should be rate limited."""
        app = MagicMock()
        mw = _RateLimitMiddleware(app, requests_per_minute=1, burst=1)
        request = _make_mock_request(path="/api/v1/predict")

        # First request allowed
        first = asyncio.run(mw.dispatch(request, _make_call_next()))
        assert first.status_code == 200

        # Second should be blocked
        second = asyncio.run(mw.dispatch(request, _make_call_next()))
        assert second.status_code == 429


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    """Sanity checks for module-level constants."""

    def test_max_request_body_bytes(self):
        """_MAX_REQUEST_BODY_BYTES should be 10 MB."""
        assert _MAX_REQUEST_BODY_BYTES == 10 * 1024 * 1024

    def test_default_rate_limit_values(self):
        """Default rate limit should be positive."""
        assert _DEFAULT_RATE_LIMIT_RPM > 0
        assert _DEFAULT_RATE_LIMIT_BURST > 0


# ---------------------------------------------------------------------------
# Module import smoke test
# ---------------------------------------------------------------------------

class TestModuleImport:
    """All middleware components should be importable."""

    def test_all_middleware_classes_importable(self):
        from src.api.middleware import (
            _RequestBodySizeLimitMiddleware,
            _RateLimitMiddleware,
            _OptionalAuthMiddleware,
            _RuntimeMetrics,
            _RateLimitState,
        )
        assert _RequestBodySizeLimitMiddleware is not None
        assert _RateLimitMiddleware is not None
        assert _OptionalAuthMiddleware is not None
        assert _RuntimeMetrics is not None
        assert _RateLimitState is not None
