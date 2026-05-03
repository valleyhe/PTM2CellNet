"""API模块 - FastAPI应用、路由和数据模型"""

from .schemas import (
    PTMSite,
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from .app import create_app

__all__ = [
    "PTMSite",
    "PredictionRequest",
    "BatchPredictionRequest",
    "PredictionResponse",
    "BatchPredictionResponse",
    "HealthResponse",
    "ModelInfoResponse",
    "create_app",
]
