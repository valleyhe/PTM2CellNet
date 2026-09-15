"""
Middleware classes extracted from app.py.

Includes request body size limiting, rate limiting, metrics collection,
and optional API-key authentication.
"""

import hmac
import os
import threading
from typing import List, Tuple, TypedDict, cast

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
from starlette.types import ASGIApp

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# TypedDict definitions used by middleware classes
# ---------------------------------------------------------------------------


class _LatencyStats(TypedDict):
    """Percentile latency statistics."""

    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


class _MetricsSnapshot(TypedDict):
    """Return type of _RuntimeMetrics.snapshot()."""

    request_count: int
    error_count: int
    error_rate: float
    latency: _LatencyStats
    latency_buckets: dict[str, int]


class _GpuMetrics(TypedDict, total=False):
    """GPU utilization metrics collected at /metrics."""

    gpu_utilization_pct: float
    gpu_memory_allocated_mb: float
    gpu_memory_reserved_mb: float


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024  # 10 MB

# Default rate-limit policy: a generous production default that still protects
# the inference endpoint from accidental DoS. Tunable via env vars so operators
# can tighten per deployment.
_DEFAULT_RATE_LIMIT_RPM = int(os.environ.get("PTM2CELLNET_RATE_LIMIT_RPM", "600"))
_DEFAULT_RATE_LIMIT_BURST = int(os.environ.get("PTM2CELLNET_RATE_LIMIT_BURST", "60"))


# ---------------------------------------------------------------------------
# Runtime metrics collector
# ---------------------------------------------------------------------------


class _RuntimeMetrics:
    """Lightweight, dependency-free runtime metrics collector.

    Tracks request count, error count, and a rolling latency histogram via
    fixed buckets. All mutable state is protected by an explicit
    ``threading.Lock``, making this safe for threaded deployments and
    ``run_in_executor`` usage. Intentionally avoids prometheus_client to
    keep dependencies minimal while still exposing actionable runtime signals.
    """

    def __init__(self) -> None:
        self.request_count: int = 0
        self.error_count: int = 0
        # Latency buckets in milliseconds (Prometheus-style histogram bounds).
        self._latency_buckets_ms: List[float] = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
        self._latency_samples_ms: List[float] = []
        self._max_samples: int = 1000
        self._lock = threading.Lock()

    def observe(self, duration_ms: float, is_error: bool) -> None:
        with self._lock:
            self.request_count += 1
            if is_error:
                self.error_count += 1
            if len(self._latency_samples_ms) >= self._max_samples:
                # Drop oldest to bound memory (ring-buffer-ish behavior).
                self._latency_samples_ms.pop(0)
            self._latency_samples_ms.append(duration_ms)

    def snapshot(self) -> _MetricsSnapshot:
        # Grab a consistent view of all mutable state under the lock, then
        # compute statistics outside the critical section.
        with self._lock:
            request_count = self.request_count
            error_count = self.error_count
            samples = list(self._latency_samples_ms)
            buckets_ms = list(self._latency_buckets_ms)

        count = len(samples)
        if count:
            sorted_samples = sorted(samples)
            avg = sum(samples) / count
            p50 = sorted_samples[count // 2]
            p95 = sorted_samples[min(int(count * 0.95), count - 1)]
            p99 = sorted_samples[min(int(count * 0.99), count - 1)]
            latency_stats: _LatencyStats = {
                "avg_ms": round(avg, 3),
                "p50_ms": round(p50, 3),
                "p95_ms": round(p95, 3),
                "p99_ms": round(p99, 3),
            }
        else:
            latency_stats = cast(_LatencyStats, {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0})

        bucket_counts: dict[str, int] = {}
        for bound in buckets_ms:
            bucket_counts[f"le_{bound}ms"] = sum(1 for s in samples if s <= bound)
        bucket_counts["le_+Inf"] = count

        error_rate = (error_count / request_count) if request_count else 0.0
        return cast(
            _MetricsSnapshot,
            {
                "request_count": request_count,
                "error_count": error_count,
                "error_rate": round(error_rate, 6),
                "latency": latency_stats,
                "latency_buckets": bucket_counts,
            },
        )


_RUNTIME_METRICS = _RuntimeMetrics()


# ---------------------------------------------------------------------------
# Request body size limit middleware
# ---------------------------------------------------------------------------


class _RequestBodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Content-Length exceeds the configured limit.

    Prevents denial-of-service via oversized payloads. The limit defaults to
    10 MB and can be overridden via the ``PTM2CELLNET_MAX_REQUEST_SIZE``
    environment variable (in bytes).
    """

    def __init__(self, app: ASGIApp, max_bytes: int = _MAX_REQUEST_BODY_BYTES) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                length = int(content_length)
            except ValueError:
                return StarletteResponse(
                    status_code=400,
                    content="Invalid Content-Length header",
                )
            if length > self._max_bytes:
                return StarletteResponse(
                    status_code=413,
                    content=f"Request body too large (limit: {self._max_bytes} bytes)",
                )
        response = await call_next(request)
        return response


# ---------------------------------------------------------------------------
# Metrics collection middleware
# ---------------------------------------------------------------------------


class _MetricsMiddleware(BaseHTTPMiddleware):
    """Record per-request runtime metrics (count, latency, errors).

    Excludes the ``/metrics`` endpoint itself so scraping does not skew the
    latency distribution. Errors are counted as any response with status >= 400.
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path.endswith("/metrics"):
            return await call_next(request)
        import time as _time

        start = _time.time()
        is_error = False
        try:
            response = await call_next(request)
            is_error = response.status_code >= 400
            return response
        except Exception as e:
            logger.warning("Unhandled exception in metrics middleware: %s", e)
            is_error = True
            raise
        finally:
            duration_ms = (_time.time() - start) * 1000
            _RUNTIME_METRICS.observe(duration_ms, is_error)


# ---------------------------------------------------------------------------
# Rate limit state (token-bucket / sliding window)
# ---------------------------------------------------------------------------


class _RateLimitState:
    """In-process sliding-window token bucket for rate limiting.

    Tracks per-client request counts in a sliding 60-second window. Uses the
    client IP (``X-Forwarded-For`` first hop, else ``request.client.host``) as
    the bucket key. Buckets older than the window are evicted lazily on each
    request to bound memory.

    This is intentionally a single-process limiter suitable for single-worker
    uvicorn / Docker deployments. For multi-worker deployments an external
    limiter (e.g. nginx, envoy, or a Redis-backed limiter) should front the
    API; this provides defence-in-depth inside the application itself.
    """

    def __init__(self, requests_per_minute: int, burst: int) -> None:
        if requests_per_minute <= 0:
            requests_per_minute = _DEFAULT_RATE_LIMIT_RPM
        if burst <= 0:
            burst = _DEFAULT_RATE_LIMIT_BURST
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        # key -> list of request timestamps (epoch seconds).
        self._hits: dict[str, list[float]] = {}
        import threading

        self._lock = threading.Lock()

    def _client_key(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # First hop is the original client.
            return forwarded.split(",")[0].strip()
        client = request.client
        return client.host if client else "unknown"

    def _evict(self, key: str, now: float) -> None:
        bucket = self._hits.get(key)
        if not bucket:
            return
        cutoff = now - 60.0
        # Drop entries older than the 60s window (in-place filter).
        i = 0
        while i < len(bucket) and bucket[i] < cutoff:
            i += 1
        if i:
            del bucket[:i]
        if not bucket:
            self._hits.pop(key, None)

    def allow(self, request: Request) -> Tuple[bool, int]:
        """Return ``(allowed, retry_after_seconds)``.

        Allowed when the request count in the last 60s is below
        ``requests_per_minute`` *and* an instantaneous burst cap is respected.
        ``retry_after_seconds`` is non-zero only when the request is rejected
        and tells the caller how long until enough tokens free up.
        """
        import time as _time

        now = _time.time()
        key = self._client_key(request)
        with self._lock:
            self._evict(key, now)
            bucket = self._hits.setdefault(key, [])
            count = len(bucket)
            # Burst cap: never allow more than ``burst`` concurrent in-window
            # requests from one client. Without this a client that fires the
            # entire per-minute budget in one tick would starve other callers
            # sharing a NAT/egress IP.
            if count >= self.requests_per_minute or count >= self.burst:
                oldest = bucket[0] if bucket else now
                retry_after = max(1, int(60 - (now - oldest)))
                return False, retry_after
            bucket.append(now)
            return True, 0


# ---------------------------------------------------------------------------
# Rate limit middleware
# ---------------------------------------------------------------------------


class _RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-client rate limiting middleware (token-bucket / sliding window).

    Returns HTTP 429 with a ``Retry-After`` header when a client exceeds the
    configured request budget. Skipped for health endpoints so liveness probes
    are never throttled.
    """

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = _DEFAULT_RATE_LIMIT_RPM,
        burst: int = _DEFAULT_RATE_LIMIT_BURST,
    ) -> None:
        super().__init__(app)
        self._state = _RateLimitState(requests_per_minute, burst)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Never throttle health/live/ready probes — they must stay responsive
        # for orchestrators (k8s, Docker healthcheck).
        if path.endswith(("/health", "/live", "/ready", "/")):
            return await call_next(request)
        allowed, retry_after = self._state.allow(request)
        if not allowed:
            return StarletteResponse(
                status_code=429,
                content=(f"Rate limit exceeded. Try again in {retry_after} second(s)."),
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)


# ---------------------------------------------------------------------------
# Optional API-key authentication middleware
# ---------------------------------------------------------------------------


class _OptionalAuthMiddleware(BaseHTTPMiddleware):
    """Optional API-key authentication for sensitive endpoints.

    Reads the expected key from the ``PTM2CELLNET_API_KEY`` environment
    variable. When unset, authentication is skipped entirely — this keeps
    the API backward-compatible with existing local/Docker deployments that
    don't configure a key (SEC-01).

    When set, requests to the prediction and model-info endpoints must
    carry a matching ``X-API-Key`` header. Health/liveness/readiness probes
    and the root index are always exempt so orchestrators can reach them.

    Comparison uses :func:`hmac.compare_digest` (constant-time) to avoid
    timing side-channels. A missing key yields 401; a wrong key yields 403.
    """

    # Paths (suffix-matched, agnostic of the API prefix) that require auth
    # when a key is configured. Mirrors the prediction surface plus the
    # model-info introspection endpoint.
    _PROTECTED_SUFFIXES: Tuple[str, ...] = (
        "/predict",
        "/batch_predict",
        "/predict/variant",
        "/model/info",
    )
    # Health/orchestration probes — never gated, so k8s/Docker probes stay
    # responsive even when auth is enabled.
    _EXEMPT_SUFFIXES: Tuple[str, ...] = ("/health", "/live", "/ready", "/")

    def __init__(self, app: ASGIApp, api_key: str) -> None:
        super().__init__(app)
        # Normalise to str; compare_digest requires equal-typed operands.
        self._api_key: str = api_key

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Health/liveness/readiness probes bypass auth (requirement 4).
        if path.endswith(self._EXEMPT_SUFFIXES):
            return await call_next(request)
        # Only the protected surface is gated; everything else passes through.
        if not path.endswith(self._PROTECTED_SUFFIXES):
            return await call_next(request)

        provided = request.headers.get("x-api-key")
        if provided is None:
            return StarletteResponse(
                status_code=401,
                content="X-API-Key required",
            )
        # Constant-time comparison to mitigate timing attacks (requirement 6).
        if not hmac.compare_digest(provided, self._api_key):
            return StarletteResponse(
                status_code=403,
                content="Invalid API key",
            )
        return await call_next(request)
