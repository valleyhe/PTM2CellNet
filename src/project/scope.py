"""Scope registry for deferred and cancelled roadmap features."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeferredFeature:
    """Metadata for a deferred roadmap feature."""

    requirement_id: str
    name: str
    summary: str
    rationale: str
    recommended_path: str


@dataclass(frozen=True)
class CancelledFeature:
    """Metadata for a roadmap feature that is explicitly out of scope."""

    requirement_id: str
    name: str
    rationale: str


DEFERRED_FEATURES: list[DeferredFeature] = [
    DeferredFeature(
        requirement_id="V2-01",
        name="ESM-3 integration",
        summary="Integrate ESM-3 (Evolutionary Scale Model 3) protein language model.",
        rationale=(
            "Implemented in v1.0 via ``ESM3Encoder`` in "
            "``src/models/pretrained_encoders.py``. Supports multi-modal inputs "
            "(sequence, structure, function tokens), LoRA fine-tuning, and "
            "sliding-window long-sequence encoding. Falls back to ESM-2 if "
            "ESM-3 weights are unavailable."
        ),
        recommended_path=(
            "Already implemented; extend with additional ESM-3 model sizes "
            "or specialized structure/function tokenizers as they become available."
        ),
    ),
    DeferredFeature(
        requirement_id="V2-04",
        name="Distributed training support",
        summary="Multi-GPU / distributed (DDP) training.",
        rationale=(
            "Deferred at v1.0. Lightning (already a dependency) provides DDP "
            "via ``L.Trainer(devices=N, strategy='ddp')``; no dedicated module "
            "is required until distributed training is exercised in practice."
        ),
        recommended_path=(
            "Pass ``strategy='ddp'`` and ``devices>1`` to the Lightning "
            "Trainer in training scripts (e.g. ``scripts/finetune_davf.py``); "
            "optionally add a helper in ``src/training/`` to centralize it."
        ),
    ),
]

CANCELLED_FEATURES: list[CancelledFeature] = [
    CancelledFeature(
        requirement_id="V2-02",
        name="Real-time mass-spec streaming",
        rationale=(
            "Cancelled by project scope update on 2026-07-05. The project "
            "uses offline/batch data preparation and prediction paths; no "
            "streaming ingestion, queue consumer, or online mass-spec pipeline "
            "should be planned."
        ),
    ),
    CancelledFeature(
        requirement_id="V2-03",
        name="Custom PTM database support",
        rationale=(
            "Cancelled by project scope update on 2026-07-05. Standard public "
            "PTM data sources and table/file imports remain valid, but a "
            "user-managed custom PTM database/catalog/API is no longer a "
            "project requirement."
        ),
    ),
    CancelledFeature(
        requirement_id="V2-05",
        name="GUI interface",
        rationale=(
            "Cancelled by project scope update on 2026-07-05. The supported "
            "interfaces are CLI scripts, Python APIs, and FastAPI endpoints; "
            "no Streamlit/Gradio/desktop GUI should be implemented."
        ),
    ),
    CancelledFeature(
        requirement_id="SEC-AUTH-APIKEY",
        name="API key feature expansion",
        rationale=(
            "Cancelled by project scope update on 2026-07-05. Existing "
            "compatibility middleware may remain, but roadmap or plan documents "
            "must not add new API-key authentication work."
        ),
    ),
]


def get_deferred_features() -> list[DeferredFeature]:
    """Return metadata for all deferred v1.0 roadmap features."""
    return list(DEFERRED_FEATURES)


def get_cancelled_features() -> list[CancelledFeature]:
    """Return metadata for features explicitly removed from project scope."""
    return list(CANCELLED_FEATURES)


__all__ = [
    "CancelledFeature",
    "DeferredFeature",
    "CANCELLED_FEATURES",
    "DEFERRED_FEATURES",
    "get_cancelled_features",
    "get_deferred_features",
]
