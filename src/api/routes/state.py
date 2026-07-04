"""Shared API router state and initialization helpers."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from torch import nn

from ...data.features import DEFAULT_PTM_TYPES, FeatureExtractor
from ...utils.logging import setup_logger

try:
    from ...analysis.variant_workflow import VariantEffectWorkflow

    VARIANT_WORKFLOW_AVAILABLE = True
    VARIANT_WORKFLOW_IMPORT_ERROR = None
except ImportError as exc:
    VariantEffectWorkflow: Any = None
    VARIANT_WORKFLOW_AVAILABLE = False
    VARIANT_WORKFLOW_IMPORT_ERROR = exc

try:
    from ...models.signaling_network import SignalingNetworkMapper

    SIGNALING_NETWORK_AVAILABLE = True
except ImportError:
    SignalingNetworkMapper: Any = None
    SIGNALING_NETWORK_AVAILABLE = False

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
    pathway_mapper: Optional[Any] = None
    # P1-3: Provenance metadata describing what kind of model is loaded.
    # Populated by initialize_model / app auto-init / initialize endpoint, so
    # /model/info and CLI tooling can surface "demo vs real" and avoid misuse.
    checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None
    model_kind: Optional[str] = None  # "demo" | "real" | None when unknown
    is_demo_model: bool = False
    # Set when startup auto-initialization was *attempted* (a checkpoint path
    # was configured) but failed. Lets /health and /ready return 503 so a
    # broken instance is never mistaken for a deliberately model-less one.
    initialization_failed: bool = False


STATE = _ModelState()

# Fields considered part of the "model + variant workflow" lifecycle. When a new
# model is initialized these are reset together so a stale variant workflow can
# never outlive the model that produced it (P0-1 / P1-1).
_LIFECYCLE_FIELDS = (
    "model",
    "cell_states",
    "label_to_idx",
    "idx_to_label",
    "device",
    "feature_extractor",
    "max_sequence_length",
    "ptm_type_to_idx",
    "variant_workflow",
    "pathway_mapper",
    "checkpoint_path",
    "config_path",
    "model_kind",
    "is_demo_model",
    "initialization_failed",
)


def reset_state() -> None:
    """Reset the shared API STATE to its uninitialized defaults.

    Exposed so tests, the ``/initialize`` endpoint, and operational tooling can
    deterministically clear every field — including the optional variant
    workflow and pathway mapper — rather than relying on each caller to remember
    which fields exist. This is the single source of truth for "no model
    loaded" and eliminates cross-test / cross-request STATE leakage (P0-1).
    """
    fresh = _ModelState()
    for field_name in _LIFECYCLE_FIELDS:
        setattr(STATE, field_name, getattr(fresh, field_name))


def initialize_model(
    model_instance: nn.Module,
    cell_state_labels: List[str],
    model_device: str = "cpu",
    config: Optional[Dict[str, Union[str, int, float, bool, List[Any], Dict[str, Any]]]] = None,
) -> None:
    """初始化模型。

    A new model load is treated as a fresh lifecycle: any stale variant
    workflow / pathway mapper from a previous model is cleared first, so the
    variant workflow can never silently outlive the model it was built for
    (P0-1 / P1-1). Callers that still want a variant workflow must re-run
    ``initialize_variant_workflow`` against the new checkpoint.
    """
    # Clear optional components tied to the previous model lifecycle before
    # binding the new one. ``model``/feature_extractor/etc. are overwritten below.
    STATE.variant_workflow = None
    STATE.pathway_mapper = None

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
        logger.warning(
            "Variant workflow unavailable; initialization skipped: %s",
            VARIANT_WORKFLOW_IMPORT_ERROR or "variant_workflow module not found",
        )
        STATE.variant_workflow = None
        return

    try:
        STATE.variant_workflow = VariantEffectWorkflow(model_path=model_path)
        logger.info("Variant workflow initialized")
    except (ImportError, ValueError, RuntimeError) as e:
        logger.warning("Variant workflow initialization failed: %s", e)
        STATE.variant_workflow = None


def initialize_pathway_mapper() -> None:
    """Initialize the signaling network pathway mapper."""
    if not SIGNALING_NETWORK_AVAILABLE:
        logger.warning("SignalingNetworkMapper not available - signaling_network module not found")
        STATE.pathway_mapper = None
        return

    try:
        STATE.pathway_mapper = SignalingNetworkMapper()
        logger.info("Signaling network pathway mapper initialized")
    except (ImportError, ValueError, RuntimeError) as e:
        logger.error("Failed to initialize pathway mapper: %s", e)
        STATE.pathway_mapper = None


def _read_model_kind(config: Optional[Dict[str, Union[str, int, float, bool, List[Any], Dict[str, Any]]]]) -> Optional[str]:
    """Resolve ``model_kind`` from a config dict (P1-3).

    Looks for an explicit ``model.model_kind`` / ``model_card.model_kind`` field.
    Falls back to inspecting ``data_provenance.training_data`` — synthetic
    training data marks the model as a demo.
    Returns ``None`` when it cannot be determined.
    """
    if not config:
        return None

    model_cfg = config.get("model", {}) or {}
    if isinstance(model_cfg, dict):
        explicit = model_cfg.get("model_kind")
        if isinstance(explicit, str) and explicit:
            return explicit.lower()

    card = config.get("model_card") or config.get("data_provenance") or {}
    if isinstance(card, dict):
        explicit = card.get("model_kind")
        if isinstance(explicit, str) and explicit:
            return explicit.lower()
        training_data = card.get("training_data")
        if isinstance(training_data, str) and "synthetic" in training_data.lower():
            return "demo"
    return None


def record_model_provenance(
    checkpoint_path: Optional[str],
    config_path: Optional[str],
    config: Optional[Dict[str, Union[str, int, float, bool, List[Any], Dict[str, Any]]]] = None,
) -> None:
    """Record where the loaded model came from and whether it is a demo (P1-3).

    Called by the auto-init path in ``app.py`` and the ``/initialize`` endpoint
    so ``/model/info`` and CLI tooling can surface demo vs real and avoid
    accidental biological interpretation of a smoke model.
    """
    STATE.checkpoint_path = checkpoint_path
    STATE.config_path = config_path
    model_kind = _read_model_kind(config)
    STATE.model_kind = model_kind
    STATE.is_demo_model = model_kind == "demo"


__all__ = [
    "_ModelState",
    "STATE",
    "reset_state",
    "initialize_model",
    "initialize_variant_workflow",
    "initialize_pathway_mapper",
    "record_model_provenance",
    "VARIANT_WORKFLOW_AVAILABLE",
    "SIGNALING_NETWORK_AVAILABLE",
    "VariantEffectWorkflow",
    "SignalingNetworkMapper",
]
