"""Project roadmap helpers and scope registry.

This module provides backward-compatible re-exports from the refactored
:mod:`src.project.scope` and :mod:`src.training.distributed` modules,
along with retained deprecated shims for cancelled features.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.project.scope import (
    CancelledFeature,
    DeferredFeature,
    CANCELLED_FEATURES,
    DEFERRED_FEATURES,
    get_cancelled_features,
    get_deferred_features,
)
from src.training.distributed import distributed_trainer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backward-compat private helpers used by the deprecated shims below
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Backward-compat shims for cancelled / implemented features
# ---------------------------------------------------------------------------


def esm3_encoder(*args: Any, strict: bool = False, **kwargs: Any):
    """V2-01: return an ESM-3 encoder. Falls back to ESM-2 if ESM3Encoder is unavailable.

    Args:
        *args: Positional arguments forwarded to the encoder constructor.
        strict: If True, raise RuntimeError when ESM-3 is unavailable instead
            of falling back to ESM-2. Use in production to prevent silent
            model substitution.  May also be enabled globally via the
            ``PTM2CELLNET_STRICT_MODEL_ASSETS`` environment variable.
        **kwargs: Keyword arguments forwarded to the encoder constructor.

    Note: ESM-3 model weights may not be publicly available. When unavailable,
    this function transparently falls back to ESM2Encoder with the same arguments
    (unless ``strict=True`` or the environment variable is set).
    Check ``src.models.pretrained_encoders.ESM3Encoder`` for the actual implementation.
    """
    # Also check global strict mode env var
    _global_strict = os.environ.get("PTM2CELLNET_STRICT_MODEL_ASSETS", "").lower() in ("1", "true", "yes")
    if _global_strict:
        strict = True

    from . import pretrained_encoders

    esm3_cls = getattr(pretrained_encoders, "ESM3Encoder", None)
    if esm3_cls is not None:
        try:
            return esm3_cls(*args, **kwargs)
        except (OSError, ImportError, ValueError, RuntimeError) as exc:
            # ESM3Encoder class exists but instantiation failed (e.g. HuggingFace
            # weights unavailable or download error). Log and fall back.
            logger.warning(
                "ESM3Encoder instantiation failed (%s: %s). %s",
                type(exc).__name__, exc,
                "Strict mode forbids fallback." if strict else "Falling back to ESM2Encoder.",
            )
            if strict:
                raise RuntimeError(
                    "ESM3Encoder instantiation failed and strict=True forbids "
                    f"fallback to ESM-2. Original error: {type(exc).__name__}: {exc}"
                ) from exc
            # Fall through to ESM-2 below

    # ESM-3 class not available or instantiation failed (non-strict)
    if strict:
        raise RuntimeError(
            "ESM3Encoder is not available and strict=True forbids fallback. "
            "Ensure ESM-3 model weights and the required dependencies are installed, "
            "or set strict=False to allow ESM-2 fallback."
        )

    # ESM-3不可用时安全回退到ESM-2
    esm2_cls = pretrained_encoders.ESM2Encoder
    logger.info("ESM3Encoder unavailable; using ESM2Encoder as fallback.")
    return esm2_cls(*args, **kwargs)


def mass_spec_stream(file_or_stream: Any, **kwargs: Any) -> pd.DataFrame:
    """V2-02: parse tab-delimited mass-spec PTM records into a DataFrame.

    .. deprecated:: V2-02
        This feature was cancelled on 2026-07-05. The function remains
        functional for backward compatibility but should not be used in
        new code. Use standard batch data loading instead.
    """
    import warnings

    warnings.warn(
        "mass_spec_stream is deprecated (V2-02 cancelled on 2026-07-05). "
        "Use standard batch data loading instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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
    """V2-03: load a user-supplied PTM table into the standard schema.

    .. deprecated:: V2-03
        This feature was cancelled on 2026-07-05. The function remains
        functional for backward compatibility but should not be used in
        new code. Use standard data import paths instead.
    """
    import warnings

    warnings.warn(
        "load_custom_ptm_database is deprecated (V2-03 cancelled on 2026-07-05). "
        "Use standard data import paths instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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


def launch_gui(**kwargs: Any):
    """V2-05: retained compatibility shim for a cancelled GUI requirement.

    .. deprecated:: V2-05
        GUI is no longer in PTM2CellNet scope (cancelled 2026-07-05).
        This function always returns False.
    """
    import warnings

    warnings.warn(
        "launch_gui is deprecated (V2-05 cancelled on 2026-07-05). "
        "GUI is no longer in PTM2CellNet scope.",
        DeprecationWarning,
        stacklevel=2,
    )
    del kwargs
    logger.warning("GUI launch skipped: GUI is no longer in PTM2CellNet scope.")
    return False


# ---------------------------------------------------------------------------
# Public API — full backward compatibility with pre-refactor imports
# ---------------------------------------------------------------------------

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
