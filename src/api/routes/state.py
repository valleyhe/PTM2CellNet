# mypy: ignore-errors
"""Shared API router state and initialization helpers."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from torch import nn

from ...data.features import DEFAULT_PTM_TYPES, FeatureExtractor
from ...utils.logging import setup_logger

try:
    from ...analysis.variant_workflow import VariantEffectWorkflow

    VARIANT_WORKFLOW_AVAILABLE = True
except ImportError:
    VariantEffectWorkflow = None
    VARIANT_WORKFLOW_AVAILABLE = False

logger = setup_logger(__name__)


@dataclass
class _ModelState:
    model: Optional[nn.Module] = None
    cell_states: List[str] = field(default_factory=list)
    label_to_idx: Dict[str, int] = field(default_factory=dict)
    idx_to_label: Dict[int, str] = field(default_factory=dict)
    device: str = "cpu"
    feature_extractor: Optional[FeatureExtractor] = None
    max_sequence_length: int = 1000
    ptm_type_to_idx: Dict[str, int] = field(default_factory=dict)
    variant_workflow: Optional[Any] = None


STATE = _ModelState()


def initialize_model(
    model_instance: nn.Module,
    cell_state_labels: List[str],
    model_device: str = "cpu",
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """初始化模型。"""
    STATE.model = model_instance
    STATE.cell_states = cell_state_labels
    STATE.label_to_idx = {label: i for i, label in enumerate(cell_state_labels)}
    STATE.idx_to_label = dict(enumerate(cell_state_labels))
    STATE.device = model_device
    STATE.feature_extractor = FeatureExtractor(config or {})
    STATE.max_sequence_length = (config or {}).get("data", {}).get("max_sequence_length", 1000)
    ptm_types = (config or {}).get("data", {}).get("ptm_types", DEFAULT_PTM_TYPES)
    STATE.ptm_type_to_idx = {ptm: i + 1 for i, ptm in enumerate(ptm_types)}
    STATE.model.to(STATE.device)
    STATE.model.eval()
    logger.info("模型已初始化，细胞状态: %s", STATE.cell_states)


def initialize_variant_workflow(model_path: str) -> None:
    """Initialize variant effect prediction workflow."""
    if not VARIANT_WORKFLOW_AVAILABLE:
        logger.warning("Variant workflow not available - variant_workflow module not found")
        STATE.variant_workflow = None
        return

    try:
        STATE.variant_workflow = VariantEffectWorkflow(model_path=model_path)
        logger.info("Variant workflow initialized")
    except Exception as e:
        logger.error("Failed to initialize variant workflow: %s", e)
        STATE.variant_workflow = None


__all__ = [
    "_ModelState",
    "STATE",
    "initialize_model",
    "initialize_variant_workflow",
    "VARIANT_WORKFLOW_AVAILABLE",
    "VariantEffectWorkflow",
]
