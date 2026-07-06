"""
FastAPI应用模块
功能概述: 创建和配置FastAPI应用
设计思路: 提供应用工厂函数，支持CORS配置
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..utils.logging import setup_logger
from .autoinit import _try_auto_initialize
from .production_security import configure_production_security
from .middleware import (
    _DEFAULT_RATE_LIMIT_BURST,
    _DEFAULT_RATE_LIMIT_RPM,
    _MAX_REQUEST_BODY_BYTES,
    _MetricsMiddleware,
    _OptionalAuthMiddleware,
    _RateLimitMiddleware,
    _RequestBodySizeLimitMiddleware,
)
from .monitoring import _setup_monitoring
from .routes import router

logger = setup_logger(__name__)


def _build_lifespan():
    """Return a FastAPI ``lifespan`` async context manager.

    F-08 migration note (v17): the previous implementation used the now
    deprecated ``@app.on_event("startup")`` / ``@app.on_event("shutdown")``
    decorators. FastAPI has supported the Starlette ``lifespan`` ASGI
    keyword since 0.54 (well below our declared minimum of 0.68), and
    ``on_event`` was officially deprecated in FastAPI 0.93 in favour of
    the lifespan context manager. We switch to ``lifespan`` now because:

    * the installed FastAPI is ≥0.93 everywhere we test,
    * ``on_event`` emits ``DeprecationWarning`` noise on modern FastAPI,
    * a single lifespan keeps startup *and* shutdown logic together,
      which is easier to reason about for the auto-init side effect.

    The lifespan is intentionally minimal: it calls the same
    ``_try_auto_initialize`` side effect as before on entry and logs on
    exit. Anything more elaborate belongs in dedicated middleware so it
    stays testable in isolation.
    """
    @asynccontextmanager
    async def _lifespan(app: FastAPI):  # noqa: ARG001 - FastAPI contract
        logger.info("PTM2CellNet API 启动")
        _try_auto_initialize()
        try:
            yield
        finally:
            logger.info("PTM2CellNet API 关闭")

    return _lifespan


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
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        lifespan=_build_lifespan(),
    )

    # Production security guards (TD-M3 / TD-M4). In production this refuses
    # to start when PTM2CELLNET_API_KEY is unset or the download allowlist is
    # blank, unless the operator explicitly opts out. No-op outside prod.
    configure_production_security()

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

    # Optional API-key authentication (SEC-01). Backward-compatible: when
    # PTM2CELLNET_API_KEY is unset, auth is skipped entirely. When set, the
    # prediction and model-info endpoints require a matching X-API-Key
    # header; health/liveness/readiness probes stay exempt.
    configured_api_key = os.environ.get("PTM2CELLNET_API_KEY")
    if configured_api_key:
        app.add_middleware(_OptionalAuthMiddleware, api_key=configured_api_key)
        logger.info(
            "API-key authentication enabled: protected endpoints require X-API-Key "
            "(unset PTM2CELLNET_API_KEY to disable)."
        )
    else:
        logger.info(
            "API-key authentication disabled (PTM2CELLNET_API_KEY unset). "
            "Set it to protect prediction endpoints."
        )

    # Strict model assets mode — propagates PTM2CELLNET_STRICT_MODEL_ASSETS
    # to all sub-modules so fallbacks raise errors instead of silently degrading.
    _strict_assets = os.environ.get("PTM2CELLNET_STRICT_MODEL_ASSETS", "0")
    os.environ.setdefault("PTM2CELLNET_STRICT_MODEL_ASSETS", _strict_assets)
    if _strict_assets == "1":
        logger.info("Strict model assets mode enabled — fallbacks will raise errors")

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


# 模块级 app 实例：便于 ``uvicorn src.api.app:app`` 直接启动，也便于
# Dockerfile CMD 使用单一入口（见 B3）。工厂函数 create_app 仍保留供
# 测试与需要自定义参数的场景使用。
app = create_app()
