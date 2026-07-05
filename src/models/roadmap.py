"""Project roadmap helpers and scope registry."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ..training.trainers import Trainer

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import lightning as _LIGHTNING_MODULE
    _LIGHTNING_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - optional dependency
    _LIGHTNING_MODULE = None  # type: ignore[assignment]  # optional dep absent
    _LIGHTNING_IMPORT_ERROR = exc


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

_MASS_SPEC_COLUMNS = ["position", "ptm_type", "intensity", "confidence"]
_STANDARD_PTM_COLUMNS = [
    "protein_accession",
    "position",
    "ptm_type",
    "amino_acid",
    "source",
    "confidence",
]
_COLUMN_ALIASES = {
    "accession": "protein_accession",
    "protein_id": "protein_accession",
    "uniprot": "protein_accession",
    "uniprot_accession": "protein_accession",
    "site": "position",
    "ptm_position": "position",
    "modification": "ptm_type",
    "residue": "amino_acid",
    "residue_aa": "amino_acid",
    "database": "source",
    "origin": "source",
    "score": "confidence",
    "probability": "confidence",
}


def _ensure_dataframe(records: Any, *, columns: list[str] | None = None) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, (str, Path)):
        return pd.read_csv(records, sep="\t")
    if hasattr(records, "read"):
        return pd.read_csv(records, sep="\t")

    rows = list(records)
    if not rows:
        return pd.DataFrame(columns=columns)
    if isinstance(rows[0], Mapping):
        return pd.DataFrame(rows)
    return pd.DataFrame.from_records(rows, columns=columns)


def esm3_encoder(*args: Any, **kwargs: Any):
    """V2-01: return an ESM-3 encoder. Falls back to ESM-2 if ESM3Encoder is unavailable."""
    from . import pretrained_encoders

    esm3_cls = getattr(pretrained_encoders, "ESM3Encoder", None)
    if esm3_cls is not None:
        return esm3_cls(*args, **kwargs)

    # ESM-3不可用时安全回退到ESM-2
    esm2_cls = pretrained_encoders.ESM2Encoder
    logger.info("ESM3Encoder unavailable; using ESM2Encoder as fallback.")
    return esm2_cls(*args, **kwargs)


def mass_spec_stream(file_or_stream: Any, **kwargs: Any) -> pd.DataFrame:
    """V2-02: parse tab-delimited mass-spec PTM records into a DataFrame."""
    frame = _ensure_dataframe(file_or_stream, columns=_MASS_SPEC_COLUMNS)
    missing = [column for column in _MASS_SPEC_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required mass-spec columns: {missing}")

    frame = frame.loc[:, _MASS_SPEC_COLUMNS].copy()
    frame["position"] = pd.to_numeric(frame["position"], errors="raise").astype(int)
    frame["intensity"] = pd.to_numeric(frame["intensity"], errors="coerce")
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    if kwargs.get("dropna", False):
        frame = frame.dropna(subset=_MASS_SPEC_COLUMNS)
    return frame.reset_index(drop=True)


def load_custom_ptm_database(path_or_df: Any, **kwargs: Any) -> pd.DataFrame:
    """V2-03: load a user-supplied PTM table into the standard schema."""
    if isinstance(path_or_df, pd.DataFrame):
        frame = path_or_df.copy()
    else:
        path = Path(path_or_df)
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv", ".txt"}:
            separator = kwargs.get("sep", "\t" if suffix == ".tsv" else ",")
            frame = pd.read_csv(path, sep=separator)
        elif suffix in {".json", ".jsonl"}:
            frame = pd.read_json(path)
        elif suffix in {".xls", ".xlsx"}:
            frame = pd.read_excel(path, sheet_name=kwargs.get("sheet_name", 0))
        else:
            raise ValueError(f"Unsupported PTM database format: {suffix or '<no suffix>'}")

    frame = frame.rename(columns=_COLUMN_ALIASES)
    if "source" not in frame.columns:
        frame["source"] = kwargs.get("source", "custom")
    if "confidence" not in frame.columns:
        frame["confidence"] = kwargs.get("confidence", pd.NA)

    required_input_columns = {"protein_accession", "position", "ptm_type", "amino_acid"}
    missing = sorted(required_input_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns after standardization: {missing}")

    standardized = frame.reindex(columns=_STANDARD_PTM_COLUMNS).copy()
    missing_values = standardized.isna().sum()
    missing_columns = [name for name, count in missing_values.items() if int(count) > 0]
    if missing_columns:
        logger.warning(
            "Custom PTM database contains missing values in columns: %s",
            ", ".join(missing_columns),
        )

    standardized["position"] = pd.to_numeric(standardized["position"], errors="raise").astype(int)
    standardized["confidence"] = pd.to_numeric(standardized["confidence"], errors="coerce")
    return standardized.reset_index(drop=True)


def distributed_trainer(model: Any, datamodule: Any, **kwargs: Any):
    """V2-04: build a Lightning DDP trainer when possible, else fall back safely."""
    del datamodule

    if _LIGHTNING_MODULE is not None:
        trainer_kwargs = dict(kwargs)
        devices = int(trainer_kwargs.pop("devices", 1))
        if torch.cuda.is_available():
            available_devices = torch.cuda.device_count()
            if available_devices > 1:
                trainer_kwargs["accelerator"] = trainer_kwargs.get("accelerator", "gpu")
                trainer_kwargs["devices"] = max(devices, available_devices)
                trainer_kwargs["strategy"] = "ddp"
            elif devices > 1:
                logger.warning(
                    "Requested devices=%s but only %s GPU is available; using a normal Lightning Trainer.",
                    devices,
                    available_devices,
                )
                trainer_kwargs["devices"] = 1
        return _LIGHTNING_MODULE.Trainer(**trainer_kwargs)

    logger.warning(
        "lightning is unavailable (%s); falling back to plain Trainer.",
        _LIGHTNING_IMPORT_ERROR,
    )
    return Trainer(
        model,
        config=kwargs.get("config"),
        device=kwargs.get("device"),
    )


def launch_gui(**kwargs: Any):
    """V2-05: retained compatibility shim for a cancelled GUI requirement."""
    del kwargs
    logger.warning("GUI launch skipped: GUI is no longer in PTM2CellNet scope.")
    return False


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
    "esm3_encoder",
    "mass_spec_stream",
    "load_custom_ptm_database",
    "distributed_trainer",
    "launch_gui",
]
