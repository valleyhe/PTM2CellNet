"""
FastAPI应用模块
功能概述: 创建和配置FastAPI应用
设计思路: 提供应用工厂函数，支持CORS配置
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


def _try_auto_initialize() -> None:
    """Attempt to load model from environment-configured paths on startup."""
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
    cell_states_str = os.environ.get(
        "PTM2CELLNET_CELL_STATES",
        "proliferation,differentiation,apoptosis,quiescence",
    )

    if not checkpoint_path:
        logger.info("PTM2CELLNET_CHECKPOINT not set — API starts without model. Load via /api/v1/initialize.")
        return

    if not Path(checkpoint_path).exists():
        logger.warning("Checkpoint not found at %s — API starts without model.", checkpoint_path)
        return

    try:
        import json
        import yaml
        from ..models.architectures import PTM2CellNet
        from ..utils.checkpoint_utils import extract_model_state_dict

        config = {}
        if Path(config_path).exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

        # P1-3: merge provenance from the sibling artifact_manifest.json so the
        # authoritative model_kind (written by the training script) is honoured
        # even when the config file is regenerated without model_card fields.
        manifest = {}
        manifest_path = Path(checkpoint_path).with_name("artifact_manifest.json")
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f) or {}
            except (json.JSONDecodeError, OSError):
                manifest = {}
        # Promote manifest-level provenance into the config dict so
        # ``record_model_provenance`` / ``_read_model_kind`` can see it.
        if isinstance(manifest, dict):
            for key in ("model_kind", "model_card", "data_provenance"):
                if key in manifest and key not in config:
                    config[key] = manifest[key]

        model = PTM2CellNet.from_config(config)
        raw_checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
        state_dict = extract_model_state_dict(raw_checkpoint)

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
        cell_states = [s.strip() for s in cell_states_str.split(",") if s.strip()]
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

    if cors_origins is None:
        env_origins = os.environ.get("PTM2CELLNET_CORS_ORIGINS", "")
        if env_origins:
            cors_origins = [origin.strip() for origin in env_origins.split(",") if origin.strip()]
        else:
            # M3: 默认放行本地开发来源，避免浏览器端调用 API 被全量拦截。
            # 生产环境可通过 PTM2CELLNET_CORS_ORIGINS 环境变量或显式参数收紧。
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
                "CORS 未显式配置，启用本地开发默认来源: %s。"
                "生产环境请通过 PTM2CELLNET_CORS_ORIGINS 收紧。",
                cors_origins,
            )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

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

        @app.get("/metrics")
        async def metrics() -> dict:
            """Basic service metrics endpoint."""
            return {
                "service": "PTM2CellNet",
                "version": app.version,
                "health_check_interval_seconds": health_check_interval,
            }
    else:
        logger.info("Monitoring metrics endpoint disabled (metrics_enabled=false)")


# 模块级 app 实例：便于 ``uvicorn src.api.app:app`` 直接启动，也便于
# Dockerfile CMD 使用单一入口（见 B3）。工厂函数 create_app 仍保留供
# 测试与需要自定义参数的场景使用。
app = create_app()
