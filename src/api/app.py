"""
FastAPI应用模块
功能概述: 创建和配置FastAPI应用
设计思路: 提供应用工厂函数，支持CORS配置
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from .routes import router
from .routes.state import (
    STATE,
    initialize_model,
    initialize_pathway_mapper,
    initialize_variant_workflow,
    record_model_provenance,
)
from ..utils.io import safe_torch_load
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def _resolve_autoinit_cell_states(
    config: Dict[str, Any],
    manifest: Dict[str, Any],
    env_cell_states: Optional[str],
) -> Tuple[Optional[List[str]], str]:
    """Resolve cell-state labels for auto-init following a strict priority.

    Priority (P0-1):
        1. ``config["data"]["cell_states"]`` (written by the training script).
        2. ``manifest["cell_states"]`` from the sibling artifact_manifest.json.
        3. Explicit ``PTM2CELLNET_CELL_STATES`` env var (with a warning).
           A hard-coded default order is **never** used implicitly: if no source
           provides labels we return ``None`` so the caller can refuse to load.

    Returns ``(labels, source)`` where ``source`` describes which resolver won,
    for logging. ``labels`` is ``None`` when no authoritative source exists.
    """
    cfg_states = (config.get("data") or {}).get("cell_states")
    if isinstance(cfg_states, list) and cfg_states:
        return [str(s) for s in cfg_states], "config data.cell_states"

    manifest_states = manifest.get("cell_states") if isinstance(manifest, dict) else None
    if isinstance(manifest_states, list) and manifest_states:
        return [str(s) for s in manifest_states], "artifact_manifest cell_states"

    if env_cell_states:
        labels = [s.strip() for s in env_cell_states.split(",") if s.strip()]
        if labels:
            logger.warning(
                "PTM2CELLNET_CELL_STATES 被用作标签来源（config/manifest 均未声明 "
                "data.cell_states）。请确认该顺序与训练时一致，否则预测标签语义会错位。"
            )
            return labels, "PTM2CELLNET_CELL_STATES (explicit env)"

    return None, "none"


def _infer_logits_dim(state_dict: Dict[str, Any]) -> Optional[int]:
    """Best-effort: infer the classifier output dimension from checkpoint weights.

    Looks for the final classifier weight/bias by name patterns common across
    encoders (predictor head, classifier head). Returns ``None`` when nothing
    conclusive can be inferred, so callers treat it as "skip the check".
    """
    import torch as _torch

    candidates = []
    # Heuristic names ordered by specificity. The final linear layer of the
    # predictor head carries the per-class logits dim as its ``weight.shape[0]``.
    head_patterns = (
        "predictor.head.weight", "predictor.classifier.weight",
        "predictor.output.weight", "predictor.linear.weight",
        "classifier.weight", "head.weight", "output.weight",
    )
    for name in head_patterns:
        tensor = state_dict.get(name)
        if isinstance(tensor, _torch.Tensor) and tensor.ndim == 2:
            candidates.append((name, int(tensor.shape[0])))
            break

    if candidates:
        return candidates[0][1]

    # Fallback: the last *.weight whose shape[0] plausibly maps to num_classes.
    weight_keys = [k for k in state_dict if k.endswith(".weight")]
    # Bias-only classifiers are rare; also inspect the largest bias tensor.
    bias_keys = [k for k in state_dict if k.endswith(".bias")]
    best: Optional[Tuple[str, int]] = None
    for name in reversed(weight_keys):
        tensor = state_dict.get(name)
        if isinstance(tensor, _torch.Tensor) and tensor.ndim == 2:
            best = (name, int(tensor.shape[0]))
            break
    if best is None and bias_keys:
        # If no weight tensor, the bias of the final layer still encodes the
        # number of classes (one bias per logit).
        for name in reversed(bias_keys):
            tensor = state_dict.get(name)
            if isinstance(tensor, _torch.Tensor) and tensor.ndim == 1:
                best = (name, int(tensor.shape[0]))
                break
    if best is None:
        return None
    # Only trust the heuristic when the inferred dim is a plausible class count
    # (>1, and small enough to be a classifier head rather than a hidden layer).
    name, dim = best
    if 2 <= dim <= 256:
        logger.debug("推断 logits 维度: %s -> %d", name, dim)
        return dim
    return None


def _resolve_autoinit_config(checkpoint_path: str, config_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load config dict + sibling manifest dict for auto-init.

    Returns ``(config, manifest)``; either may be empty when the file is absent
    or unreadable. Never raises — auto-init is best-effort and the caller will
    surface label-resolution failures.
    """
    import yaml

    config: Dict[str, Any] = {}
    if Path(config_path).exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError) as exc:
            logger.warning("读取 config %s 失败: %s", config_path, exc)

    manifest: Dict[str, Any] = {}
    manifest_path = Path(checkpoint_path).with_name("artifact_manifest.json")
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f) or {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("读取 manifest %s 失败: %s", manifest_path, exc)

    # P1-3: merge provenance from the sibling artifact_manifest.json so the
    # authoritative model_kind (written by the training script) is honoured
    # even when the config file is regenerated without model_card fields.
    if isinstance(manifest, dict):
        for key in ("model_kind", "model_card", "data_provenance"):
            if key in manifest and key not in config:
                config[key] = manifest[key]

    return config, manifest


def _try_auto_initialize() -> None:
    """Attempt to load model from environment-configured paths on startup.

    Label semantics follow P0-1: cell_states come from the training artifact
    (config/manifest) — never from a hard-coded default. A model whose labels
    cannot be resolved is **not** loaded, because a successfully-loaded model
    with wrong label semantics is more dangerous than a model-less API.
    """
    # PTM2CELLNET_CHECKPOINT/CONFIG are the canonical variable names. MODEL_PATH
    # is accepted as a deprecated alias so existing Docker/K8s deployments keep
    # working, but we warn so operators migrate (P0-2).
    checkpoint_path = os.environ.get("PTM2CELLNET_CHECKPOINT")
    deprecated_checkpoint = os.environ.get("MODEL_PATH")
    if not checkpoint_path and deprecated_checkpoint:
        logger.warning(
            "MODEL_PATH is deprecated; use PTM2CELLNET_CHECKPOINT. "
            "Falling back to MODEL_PATH=%s for this startup.",
            deprecated_checkpoint,
        )
        checkpoint_path = deprecated_checkpoint
    if not checkpoint_path:
        checkpoint_path = ""

    config_path = os.environ.get("PTM2CELLNET_CONFIG", "configs/production.yaml")
    env_cell_states = os.environ.get("PTM2CELLNET_CELL_STATES")

    if not checkpoint_path:
        logger.info("PTM2CELLNET_CHECKPOINT not set — API starts without model. Load via /api/v1/initialize.")
        return

    if not Path(checkpoint_path).exists():
        logger.warning("Checkpoint not found at %s — API starts without model.", checkpoint_path)
        return

    try:
        from ..models.architectures import PTM2CellNet
        from ..utils.checkpoint_utils import extract_model_state_dict

        config, manifest = _resolve_autoinit_config(checkpoint_path, config_path)

        # P0-1: resolve labels from the artifact first; refuse to load when no
        # authoritative source exists, instead of silently using a hard-coded
        # default that would produce semantically-wrong predictions.
        cell_states, label_source = _resolve_autoinit_cell_states(
            config, manifest, env_cell_states
        )
        if not cell_states:
            logger.error(
                "自动初始化中止：无法从 config/manifest/env 解析 cell_states "
                "(config=%s, manifest=%s, env=%s)。请提供训练产物，或显式设置 "
                "PTM2CELLNET_CELL_STATES 并确认其与训练时一致。API 将以无模型状态启动。",
                config_path,
                str(Path(checkpoint_path).with_name("artifact_manifest.json")),
                "PTM2CELLNET_CELL_STATES" if env_cell_states else "<unset>",
            )
            return

        # P0-1: 一致性校验——标签数必须与 config 声明的 num_classes 一致。
        config_num_classes = (config.get("model") or {}).get("num_classes")
        if isinstance(config_num_classes, int) and config_num_classes != len(cell_states):
            raise RuntimeError(
                f"标签数不一致：data.cell_states 有 {len(cell_states)} 个 "
                f"({cell_states})，但 config model.num_classes={config_num_classes}。"
                f"请使用与该 checkpoint 匹配的训练 config。"
            )

        # 推断 checkpoint 中 predictor 的 logits 维度，与标签数做最终一致性校验。
        raw_checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
        state_dict = extract_model_state_dict(raw_checkpoint)
        inferred = _infer_logits_dim(state_dict)
        if inferred is not None and inferred != len(cell_states):
            raise RuntimeError(
                f"checkpoint logits 维度 ({inferred}) 与 cell_states 数量 "
                f"({len(cell_states)}) 不一致。标签来源={label_source}。"
                f"请使用与该 checkpoint 配套的训练 config。"
            )

        # 标签解析通过后再用 num_classes 构建模型，避免结构/标签错配。
        config.setdefault("model", {})
        if isinstance(config["model"], dict):
            config["model"]["num_classes"] = len(cell_states)

        model = PTM2CellNet.from_config(config)

        # P0-2: 启动自动初始化同样使用严格加载。权重不匹配时不进入 healthy 状态，
        # 避免后续预测来自随机/部分初始化模型（这比启动失败更危险）。
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint 权重与 config 不匹配：缺失 {len(missing)} 个键，"
                f"多余 {len(unexpected)} 个键。"
                f"missing[:5]={missing[:5]}, unexpected[:5]={unexpected[:5]}"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "自动初始化标签来源: %s, cell_states=%s (num_classes=%d)",
            label_source, cell_states, len(cell_states),
        )
        initialize_model(model, cell_states, model_device=device, config=config)
        record_model_provenance(checkpoint_path, config_path, config)

        # P1-3: warn loudly when the auto-loaded model is only a demo, so
        # operators don't ship biological predictions from a smoke model.
        if STATE.is_demo_model:
            logger.warning(
                "Loaded model '%s' is a DEMO/smoke model (model_kind=demo). "
                "It only validates the engineering pipeline and must NOT be "
                "used for real biological interpretation.",
                checkpoint_path,
            )

        # Initialize optional components
        initialize_pathway_mapper()
        try:
            initialize_variant_workflow(checkpoint_path)
        except Exception as e:
            # Variant workflow is optional; continue starting the API without it
            # so that standard prediction endpoints remain available.
            logger.warning("Variant workflow initialization failed: %s", e)

        logger.info("Auto-initialized model from %s on %s", checkpoint_path, device)
    except Exception as e:
        logger.error("Auto-initialization failed: %s. API starts without model.", e)


_MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024  # 10 MB


class _RuntimeMetrics:
    """Lightweight, dependency-free runtime metrics collector.

    Tracks request count, error count, and a rolling latency histogram via
    fixed buckets. Thread-safe enough for the GIL-bound FastAPI request loop;
    counters use plain int increments and latencies use a list append guarded
    by the GIL. Intentionally avoids prometheus_client to keep dependencies
    minimal while still exposing actionable runtime signals.
    """

    def __init__(self) -> None:
        self.request_count: int = 0
        self.error_count: int = 0
        # Latency buckets in milliseconds (Prometheus-style histogram bounds).
        self._latency_buckets_ms: List[float] = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
        self._latency_samples_ms: List[float] = []
        self._max_samples: int = 1000

    def observe(self, duration_ms: float, is_error: bool) -> None:
        self.request_count += 1
        if is_error:
            self.error_count += 1
        if len(self._latency_samples_ms) >= self._max_samples:
            # Drop oldest to bound memory (ring-buffer-ish behavior).
            self._latency_samples_ms.pop(0)
        self._latency_samples_ms.append(duration_ms)

    def snapshot(self) -> Dict[str, Any]:
        samples = self._latency_samples_ms
        count = len(samples)
        if count:
            sorted_samples = sorted(samples)
            avg = sum(samples) / count
            p50 = sorted_samples[count // 2]
            p95 = sorted_samples[min(int(count * 0.95), count - 1)]
            p99 = sorted_samples[min(int(count * 0.99), count - 1)]
            latency_stats = {
                "avg_ms": round(avg, 3),
                "p50_ms": round(p50, 3),
                "p95_ms": round(p95, 3),
                "p99_ms": round(p99, 3),
            }
        else:
            latency_stats = {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}

        bucket_counts: Dict[str, int] = {}
        for bound in self._latency_buckets_ms:
            bucket_counts[f"le_{bound}ms"] = sum(1 for s in samples if s <= bound)
        bucket_counts["le_+Inf"] = count

        error_rate = (self.error_count / self.request_count) if self.request_count else 0.0
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": round(error_rate, 6),
            "latency": latency_stats,
            "latency_buckets": bucket_counts,
        }


_RUNTIME_METRICS = _RuntimeMetrics()


class _RequestBodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Content-Length exceeds the configured limit.

    Prevents denial-of-service via oversized payloads. The limit defaults to
    10 MB and can be overridden via the ``PTM2CELLNET_MAX_REQUEST_SIZE``
    environment variable (in bytes).
    """

    def __init__(self, app: FastAPI, max_bytes: int = _MAX_REQUEST_BODY_BYTES) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
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


class _MetricsMiddleware(BaseHTTPMiddleware):
    """Record per-request runtime metrics (count, latency, errors).

    Excludes the ``/metrics`` endpoint itself so scraping does not skew the
    latency distribution. Errors are counted as any response with status >= 400.
    """

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if request.url.path.endswith("/metrics"):
            return await call_next(request)
        import time as _time

        start = _time.time()
        is_error = False
        try:
            response = await call_next(request)
            is_error = response.status_code >= 400
            return response
        except Exception:
            is_error = True
            raise
        finally:
            duration_ms = (_time.time() - start) * 1000
            _RUNTIME_METRICS.observe(duration_ms, is_error)


# Default rate-limit policy: a generous production default that still protects
# the inference endpoint from accidental DoS. Tunable via env vars so operators
# can tighten per deployment.
_DEFAULT_RATE_LIMIT_RPM = int(os.environ.get("PTM2CELLNET_RATE_LIMIT_RPM", "600"))
_DEFAULT_RATE_LIMIT_BURST = int(os.environ.get("PTM2CELLNET_RATE_LIMIT_BURST", "60"))


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
        self._hits: Dict[str, List[float]] = {}
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


class _RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-client rate limiting middleware (token-bucket / sliding window).

    Returns HTTP 429 with a ``Retry-After`` header when a client exceeds the
    configured request budget. Skipped for health endpoints so liveness probes
    are never throttled.
    """

    def __init__(
        self,
        app: FastAPI,
        requests_per_minute: int = _DEFAULT_RATE_LIMIT_RPM,
        burst: int = _DEFAULT_RATE_LIMIT_BURST,
    ) -> None:
        super().__init__(app)
        self._state = _RateLimitState(requests_per_minute, burst)

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        path = request.url.path
        # Never throttle health/live/ready probes — they must stay responsive
        # for orchestrators (k8s, Docker healthcheck).
        if path.endswith(("/health", "/live", "/ready", "/")):
            return await call_next(request)
        allowed, retry_after = self._state.allow(request)
        if not allowed:
            return StarletteResponse(
                status_code=429,
                content=(
                    "Rate limit exceeded. Try again in "
                    f"{retry_after} second(s)."
                ),
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)


def create_app(
    title: str = "PTM2CellNet API",
    description: str = "蛋白质PTM与细胞状态预测API",
    version: str = "1.0.0",
    cors_origins: Optional[list[str]] = None,
) -> FastAPI:
    """
    创建FastAPI应用

    参数:
        title: 应用标题
        description: 应用描述
        version: 应用版本
        cors_origins: 允许的CORS源列表

    返回:
        配置好的FastAPI应用
    """
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        logger.info("PTM2CellNet API 启动")
        _try_auto_initialize()
        yield
        logger.info("PTM2CellNet API 关闭")

    app = FastAPI(
        title=title,
        description=description,
        version=version,
        lifespan=lifespan,
    )

    # Request body size limit — prevents DoS via oversized payloads.
    max_body = int(os.environ.get("PTM2CELLNET_MAX_REQUEST_SIZE", _MAX_REQUEST_BODY_BYTES))
    app.add_middleware(_RequestBodySizeLimitMiddleware, max_bytes=max_body)
    # Per-client rate limiting — protects inference endpoints from accidental
    # DoS. Disabled when PTM2CELLNET_RATE_LIMIT_RPM=0.
    rpm = int(os.environ.get("PTM2CELLNET_RATE_LIMIT_RPM", _DEFAULT_RATE_LIMIT_RPM))
    burst = int(os.environ.get("PTM2CELLNET_RATE_LIMIT_BURST", _DEFAULT_RATE_LIMIT_BURST))
    if rpm > 0:
        app.add_middleware(_RateLimitMiddleware, requests_per_minute=rpm, burst=burst)
        logger.info(
            "Rate limiting enabled: %s req/min, burst=%s (set PTM2CELLNET_RATE_LIMIT_RPM=0 to disable)",
            rpm,
            burst,
        )
    else:
        logger.info("Rate limiting disabled (PTM2CELLNET_RATE_LIMIT_RPM=0)")
    # Runtime metrics collection (request count, latency, error rate).
    app.add_middleware(_MetricsMiddleware)

    if cors_origins is None:
        env_origins = os.environ.get("PTM2CELLNET_CORS_ORIGINS", "")
        if env_origins:
            cors_origins = [origin.strip() for origin in env_origins.split(",") if origin.strip()]
        elif os.environ.get("PTM2CELLNET_ENV") == "development":
            # Localhost-only defaults for local development only.
            cors_origins = [
                "http://localhost",
                "http://localhost:3000",
                "http://localhost:5173",
                "http://localhost:8080",
                "http://127.0.0.1",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173",
                "http://127.0.0.1:8080",
            ]
            logger.info(
                "CORS: development mode — allowing localhost origins. "
                "Set PTM2CELLNET_CORS_ORIGINS to override."
            )
        else:
            # Production or unspecified environment: deny all cross-origin by default.
            cors_origins = []
            logger.warning(
                "⚠️  SECURITY: No CORS origins configured in a non-development "
                "environment (PTM2CELLNET_ENV=%s). Cross-origin requests will be "
                "denied. Set PTM2CELLNET_CORS_ORIGINS to allow specific origins.",
                os.environ.get("PTM2CELLNET_ENV", "<unset>"),
            )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_prefix = os.environ.get("PTM2CELLNET_API_PREFIX", "/api/v1")
    app.include_router(router, prefix=api_prefix)

    # Expose optional monitoring metrics based on configs/production.yaml monitoring.*
    _setup_monitoring(app)

    @app.get("/")
    async def root():
        """根路径"""
        return {
            "message": "Welcome to PTM2CellNet API",
            "version": version,
            "docs": "/docs",
        }

    logger.info("FastAPI应用已创建")
    return app


def _setup_monitoring(app: FastAPI) -> None:
    """Optionally register a /metrics endpoint from production.yaml monitoring config."""
    config_path = os.environ.get("PTM2CELLNET_CONFIG", "configs/production.yaml")
    monitoring_config: dict = {}
    try:
        import yaml

        if Path(config_path).exists():
            cfg = yaml.safe_load(open(config_path, encoding="utf-8")) or {}
            monitoring_config = cfg.get("monitoring", {})
    except Exception as exc:  # pragma: no cover - config loading is best-effort
        logger.debug("Could not load monitoring config: %s", exc)

    metrics_enabled = monitoring_config.get("metrics_enabled", False)
    health_check_interval = monitoring_config.get("health_check_interval", 30)

    if metrics_enabled:
        logger.info(
            "Monitoring enabled (metrics_port=%s, health_check_interval=%s)",
            monitoring_config.get("metrics_port", 9090),
            health_check_interval,
        )

        def _format_prometheus() -> str:
            """Render runtime metrics in Prometheus text exposition format.

            Prometheus scrapes ``/metrics`` and expects a flat text body with
            ``# HELP`` / ``# TYPE`` headers followed by ``name{labels} value``
            lines. Returning JSON there silently breaks scrape pipelines, so
            this emits real Prometheus text by default.
            """
            snap = _RUNTIME_METRICS.snapshot()
            lines: List[str] = []
            lines.append("# HELP ptm2cellnet_requests_total Total HTTP requests processed.")
            lines.append("# TYPE ptm2cellnet_requests_total counter")
            lines.append(f"ptm2cellnet_requests_total {snap['request_count']}")
            lines.append("# HELP ptm2cellnet_errors_total Total HTTP responses with status>=400.")
            lines.append("# TYPE ptm2cellnet_errors_total counter")
            lines.append(f"ptm2cellnet_errors_total {snap['error_count']}")
            lines.append("# HELP ptm2cellnet_error_rate Rolling error rate [0,1].")
            lines.append("# TYPE ptm2cellnet_error_rate gauge")
            lines.append(f"ptm2cellnet_error_rate {snap['error_rate']}")
            latency = snap.get("latency", {})
            lines.append("# HELP ptm2cellnet_latency_ms Request latency in milliseconds.")
            lines.append("# TYPE ptm2cellnet_latency_ms summary")
            for quantile in ("avg_ms", "p50_ms", "p95_ms", "p99_ms"):
                if quantile in latency:
                    label = quantile.replace("_ms", "")
                    lines.append(
                        f'ptm2cellnet_latency_ms{{quantile="{label}"}} {latency[quantile]}'
                    )
            # Histogram buckets as cumulative counters (Prometheus convention).
            lines.append("# HELP ptm2cellnet_latency_bucket_ms Latency histogram buckets (cumulative).")
            lines.append("# TYPE ptm2cellnet_latency_bucket_ms counter")
            buckets = snap.get("latency_buckets", {})
            for name, value in buckets.items():
                # name looks like "le_50ms" or "le_+Inf".
                le = name.split("_", 1)[1] if "_" in name else name
                lines.append(f'ptm2cellnet_latency_bucket_ms{{le="{le}"}} {value}')
            # GPU metrics when available.
            try:
                if torch.cuda.is_available():
                    lines.append("# HELP ptm2cellnet_gpu_utilization_pct GPU utilization percent.")
                    lines.append("# TYPE ptm2cellnet_gpu_utilization_pct gauge")
                    lines.append(
                        f"ptm2cellnet_gpu_utilization_pct {round(float(torch.cuda.utilization()), 2)}"
                    )
                    lines.append("# HELP ptm2cellnet_gpu_memory_allocated_mb GPU memory allocated (MB).")
                    lines.append("# TYPE ptm2cellnet_gpu_memory_allocated_mb gauge")
                    allocated = torch.cuda.memory_allocated()
                    lines.append(
                        f"ptm2cellnet_gpu_memory_allocated_mb {round(allocated / (1024 * 1024), 2)}"
                    )
                    lines.append("# HELP ptm2cellnet_gpu_memory_reserved_mb GPU memory reserved (MB).")
                    lines.append("# TYPE ptm2cellnet_gpu_memory_reserved_mb gauge")
                    reserved = torch.cuda.memory_reserved()
                    lines.append(
                        f"ptm2cellnet_gpu_memory_reserved_mb {round(reserved / (1024 * 1024), 2)}"
                    )
            except Exception as exc:  # GPU metrics are best-effort.
                logger.debug("GPU metrics unavailable: %s", exc)
            return "\n".join(lines) + "\n"

        @app.get("/metrics")
        async def metrics(request: Request):
            """Service metrics endpoint.

            By default returns Prometheus text exposition format
            (``text/plain; version=0.0.4``) so a Prometheus scraper can ingest
            it directly. Pass ``Accept: application/json`` (or
            ``?format=json``) to receive the legacy JSON payload — useful for
            ad-hoc inspection or dashboards that already consume the JSON.
            """
            accept = request.headers.get("accept", "")
            query_format = request.query_params.get("format", "").lower()
            want_json = query_format == "json" or "application/json" in accept

            runtime = _RUNTIME_METRICS.snapshot()

            gpu_metrics: Dict[str, Any] = {}
            try:
                if torch.cuda.is_available():
                    gpu_metrics["gpu_utilization_pct"] = round(
                        float(torch.cuda.utilization()), 2
                    )
                    allocated = torch.cuda.memory_allocated()
                    reserved = torch.cuda.memory_reserved()
                    gpu_metrics["gpu_memory_allocated_mb"] = round(
                        allocated / (1024 * 1024), 2
                    )
                    gpu_metrics["gpu_memory_reserved_mb"] = round(
                        reserved / (1024 * 1024), 2
                    )
            except Exception as exc:  # GPU metrics are best-effort.
                logger.debug("GPU metrics unavailable: %s", exc)

            if not want_json:
                # Prometheus exposition format.
                return StarletteResponse(
                    content=_format_prometheus(),
                    media_type="text/plain; version=0.0.4; charset=utf-8",
                )

            return {
                "service": "PTM2CellNet",
                "version": app.version,
                "health_check_interval_seconds": health_check_interval,
                "runtime": runtime,
                "gpu": gpu_metrics if gpu_metrics else None,
            }
    else:
        logger.info("Monitoring metrics endpoint disabled (metrics_enabled=false)")


# 模块级 app 实例：便于 ``uvicorn src.api.app:app`` 直接启动，也便于
# Dockerfile CMD 使用单一入口（见 B3）。工厂函数 create_app 仍保留供
# 测试与需要自定义参数的场景使用。
app = create_app()
