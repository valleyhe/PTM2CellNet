"""Legacy loaders for cancelled roadmap features (V2-02, V2-03).

This module contains backward-compatible implementations of data loading
functions for features that have been removed from the project scope.
They remain functional but emit deprecation warnings.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd

logger = logging.getLogger(__name__)

# Column schemas for legacy loaders
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
        return cast(pd.DataFrame, records.copy())
    if isinstance(records, (str, Path)):
        return cast(pd.DataFrame, pd.read_csv(records, sep="\t"))
    if hasattr(records, "read"):
        return cast(pd.DataFrame, pd.read_csv(records, sep="\t"))

    rows = list(records)
    if not rows:
        return pd.DataFrame(columns=columns)
    if isinstance(rows[0], Mapping):
        return pd.DataFrame(rows)
    return cast(pd.DataFrame, pd.DataFrame.from_records(rows, columns=columns))


def mass_spec_stream(file_or_stream: Any, **kwargs: Any) -> pd.DataFrame:
    """V2-02: parse tab-delimited mass-spec PTM records into a DataFrame.

    .. deprecated:: V2-02
        This feature was cancelled on 2026-07-05. The function remains
        functional for backward compatibility but should not be used in
        new code. Use standard batch data loading instead.
    """
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
    return cast(pd.DataFrame, frame.reset_index(drop=True))


def load_custom_ptm_database(path_or_df: Any, **kwargs: Any) -> pd.DataFrame:
    """V2-03: load a user-supplied PTM table into the standard schema.

    .. deprecated:: V2-03
        This feature was cancelled on 2026-07-05. The function remains
        functional for backward compatibility but should not be used in
        new code. Use standard data import paths instead.
    """
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
    return cast(pd.DataFrame, standardized.reset_index(drop=True))
