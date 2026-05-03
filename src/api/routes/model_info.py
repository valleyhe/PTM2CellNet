"""Model info and health endpoints."""

from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from ..schemas import HealthResponse, ModelInfoResponse
from ...data.features import DEFAULT_PTM_TYPES
from .state import STATE

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查

    返回:
        服务健康状态
    """
    return HealthResponse(
        status="healthy" if STATE.model is not None else "unhealthy",
        version="1.0.0",
        model_loaded=STATE.model is not None,
        timestamp=datetime.now().isoformat(),
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
        supported_ptm_types=list(STATE.ptm_type_to_idx.keys())
        if STATE.ptm_type_to_idx
        else DEFAULT_PTM_TYPES,
    )


__all__ = ["router", "health_check", "get_model_info"]
