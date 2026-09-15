"""Model info and health endpoints."""

import os
from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from ..schemas import (
    HealthResponse,
    LiveResponse,
    ModelInfoResponse,
    ReadyResponse,
    VariantReadyResponse,
)
from ...data.features import DEFAULT_PTM_TYPES
from .state import STATE

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="综合健康检查（遗留/兼容端点）",
    description=(
        "返回服务综合健康状态（含模型加载情况）。\n\n"
        "**职责边界**：\n"
        "- `/live`：进程存活探针（liveness），始终返回 200，用于判断进程是否存活。\n"
        "- `/ready`：服务就绪探针（readiness），模型未加载时返回 **503**，"
        "用于 Kubernetes `readinessProbe` 与 Docker `HEALTHCHECK` 的流量接入判断。\n"
        "- `/health`（本端点）：**遗留/综合**健康检查，始终返回 200，"
        "通过 `status` 字段（`healthy` / `unhealthy`）与 `model_loaded` 反映模型加载情况。\n\n"
        "⚠️ **不要用 `/health` 作为 readiness 判断**：它在模型未加载时也返回 200，"
        "会把未就绪实例判定为健康。流量接入判断请使用 `/ready`，存活判断请使用 `/live`。"
    ),
    tags=["health"],
)
async def health_check():
    """
    健康检查（遗留/综合端点）

    返回:
        服务健康状态（初始化失败时 HTTP 503，否则 HTTP 200，通过 status 字段区分 healthy/unhealthy）
    """
    if STATE.initialization_failed:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="启动自动初始化失败，服务不可用。请通过 /api/v1/initialize 手动加载模型。",
        )

    return HealthResponse(
        status="healthy" if STATE.model is not None else "unhealthy",
        version="1.0.0",
        model_loaded=STATE.model is not None,
        timestamp=datetime.now().isoformat(),
    )


@router.get(
    "/live",
    response_model=LiveResponse,
    summary="存活探针（liveness）",
    description=(
        '进程存活探针。只要 FastAPI 进程在运行就返回 200 `{"status": "alive"}`，'
        "与模型是否加载无关。\n\n"
        "**用途**：Kubernetes `livenessProbe` / 负载均衡存活判断。"
        "失败（非 200）意味着进程需要被重启。"
    ),
    tags=["health"],
)
async def live_check():
    """
    存活检查

    返回:
        服务存活状态
    """
    return LiveResponse(status="alive")


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="就绪探针（readiness）",
    description=(
        "服务就绪探针。模型未加载（未调用 `/initialize`）时返回 **HTTP 503**，"
        '模型加载后返回 200 `{"status": "ready", "model_loaded": true}`。\n\n'
        "**用途**：Kubernetes `readinessProbe`、Docker `HEALTHCHECK`、"
        "负载均衡流量接入判断。**不要用 `/health` 替代本端点**做流量判断。"
    ),
    tags=["health"],
)
async def ready_check():
    """
    就绪检查

    返回:
        服务就绪状态（初始化失败、生产安全降级或模型未加载时 HTTP 503）
    """
    if STATE.initialization_failed:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="启动自动初始化失败，服务不可用",
        )

    # Production security degradation (TD-M3 / TD-M4). When the operator opted
    # out of fail-fast but a security check still failed (e.g. no API key in
    # production), mark the instance not-ready so orchestrators drain it.
    from ..production_security import get_last_security_report

    sec_report = get_last_security_report()
    if sec_report is not None and sec_report.degraded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=("生产安全配置降级: " + "; ".join(sec_report.reasons) if sec_report.reasons else "生产安全配置降级"),
        )

    if STATE.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型未初始化",
        )

    return ReadyResponse(status="ready", model_loaded=True)


@router.get(
    "/ready/variant",
    response_model=VariantReadyResponse,
    summary="变体预测就绪探针（P1-1）",
    description=(
        "单独报告变体预测能力的就绪状态。\n\n"
        "核心模型加载成功但 variant workflow 初始化失败时，"
        "`/ready` 返回 200 而 `/ready/variant` 返回 **503**，"
        "从而可以分别对核心预测与变体预测做流量接入判断。\n\n"
        "**用途**：当部署只暴露 `/predict/variant` 时，用它作为 readinessProbe。"
    ),
    tags=["health"],
)
async def variant_ready_check():
    """Report readiness of the variant prediction workflow (P1-1)."""
    if STATE.model is None:
        # Without the core model the variant workflow cannot be meaningful.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="核心模型未初始化，变体能力不可用",
        )
    if STATE.variant_workflow is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variant prediction workflow not initialized",
        )
    return VariantReadyResponse(
        status="ready",
        variant_workflow_loaded=True,
        model_loaded=True,
    )


@router.get("/model/info", response_model=ModelInfoResponse)
@router.get("/model_info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    获取模型信息

    返回:
        模型信息
    """
    if STATE.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型未初始化",
        )

    return ModelInfoResponse(
        model_name="PTM2CellNet",
        model_version="1.0.0",
        encoder_type=getattr(STATE.model, "encoder_type", "unknown"),
        embed_dim=getattr(STATE.model, "embed_dim", 256),
        num_classes=len(STATE.cell_states),
        cell_states=STATE.cell_states,
        supported_ptm_types=list(STATE.ptm_type_to_idx.keys()) if STATE.ptm_type_to_idx else DEFAULT_PTM_TYPES,
        variant_workflow_loaded=STATE.variant_workflow is not None,
        pathway_mapper_loaded=STATE.pathway_mapper is not None,
        model_kind=STATE.model_kind,
        is_demo_model=STATE.is_demo_model,
        # SEC-03: 只返回文件名，避免向客户端泄露服务器绝对路径。
        # 仍返回文件名以便运维/前端识别当前加载的 checkpoint/config 来源。
        checkpoint_path=os.path.basename(STATE.checkpoint_path) if STATE.checkpoint_path else None,
        config_path=os.path.basename(STATE.config_path) if STATE.config_path else None,
    )


__all__ = [
    "router",
    "health_check",
    "live_check",
    "ready_check",
    "variant_ready_check",
    "get_model_info",
]
