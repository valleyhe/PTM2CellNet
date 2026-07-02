"""API route package with composed sub-routers."""

from fastapi import APIRouter

from .initialize import router as initialize_router
from .model_info import router as model_info_router
from .predictions import preprocess_request, router as predictions_router
from .state import (
    STATE,
    _ModelState,
    initialize_model,
    initialize_variant_workflow,
)

router = APIRouter()
router.include_router(predictions_router)
router.include_router(model_info_router)
router.include_router(initialize_router)

# Direct endpoint definitions (mirroring ``src/api/routes.py``).
# The ``src/api/routes/`` package shadows ``src/api/routes.py`` at import time,
# so the live endpoints are exposed here as well.
from ..schemas import (  # noqa: E402
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    VariantPredictionRequest,
    VariantPredictionResponse,
)
from .model_info import get_model_info, health_check as _health_check_impl
from .predictions import (
    batch_predict as _batch_predict_impl,
    predict as _predict_impl,
    predict_variant as _predict_variant_impl,
)


async def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict cell state from a sequence and PTM sites."""
    return await _predict_impl(request)


async def batch_predict(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Batch-predict cell states (legacy ``/batch_predict`` path)."""
    return await _batch_predict_impl(request)


async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Batch-predict cell states (``/predict/batch`` path)."""
    return await _batch_predict_impl(request)


async def predict_variant(request: VariantPredictionRequest) -> VariantPredictionResponse:
    """Predict cell-state changes from a protein variant."""
    return await _predict_variant_impl(request)


async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return await _health_check_impl()


async def model_info() -> ModelInfoResponse:
    """Return model information (legacy ``/model_info`` path)."""
    return await get_model_info()


async def model_info_v2() -> ModelInfoResponse:
    """Return model information (``/model/info`` path)."""
    return await get_model_info()


__all__ = [
    "router",
    "STATE",
    "_ModelState",
    "initialize_model",
    "preprocess_request",
    "initialize_variant_workflow",
    "predict",
    "batch_predict",
    "predict_batch",
    "predict_variant",
    "health_check",
    "model_info",
    "model_info_v2",
]
