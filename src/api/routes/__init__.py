"""API route package with composed sub-routers."""

from fastapi import APIRouter

from .cross_scale import router as cross_scale_router
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
router.include_router(cross_scale_router)


__all__ = [
    "router",
    "STATE",
    "_ModelState",
    "initialize_model",
    "preprocess_request",
    "initialize_variant_workflow",
]
