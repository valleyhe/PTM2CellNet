"""Real-data schema contract & profiling (P1-1).

Defines the contract a *real* (non-synthetic) training dataset must satisfy so
that a produced model can be responsibly released, and reports a profile that
goes into ``artifact_manifest.json`` for provenance / release gating.

Required columns (case-insensitive not enforced — exact names):

* ``sequence``        — amino-acid sequence (uppercase ``ACDEFGHIKLMNPQRSTVWY``).
* ``ptm_sites``       — JSON list of ``{"position": int, "type": str}``; the
  position is **1-based** and must be ``1 <= position <= len(sequence)``.
* ``cell_state``      — the classification label.

Recommended (optional) columns that strengthen provenance:

* ``protein_accession`` — UniProt / NCBI accession for dedup & leakage checks.
* ``gene_symbol``       — gene name.
* ``source_db``         — where the annotation came from.
* ``evidence_level``    — confidence/evidence of the annotation.
* ``split_group``       — explicit train/val/test assignment (homology-safe).

The contract checker is intentionally separate from ``DataPreprocessor``'s
quality check: the latter is a best-effort warning surface; this one defines
the stricter bar a *release-eligible* dataset must clear.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
from typing_extensions import TypedDict

from ..utils.logging import setup_logger

logger = setup_logger(__name__)

REQUIRED_COLUMNS = ("sequence", "ptm_sites", "cell_state")
RECOMMENDED_COLUMNS = (
    "protein_accession",
    "gene_symbol",
    "source_db",
    "evidence_level",
    "split_group",
)
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ---------------------------------------------------------------------------
# TypedDict definitions for structured dicts used in this module
# ---------------------------------------------------------------------------

class PTMSiteRaw(TypedDict, total=False):
    """Shape of a single parsed PTM site dict from data contract parsing."""

    position: int
    type: str


class ContractReport(TypedDict):
    """Shape of the dict returned by ``validate_data_contract``."""

    ok: bool
    hard_failures: List[str]
    warnings: List[str]
    missing_required: List[str]
    missing_recommended: List[str]
    invalid_ptm_rows: int
    row_count: int


class SequenceLengthProfile(TypedDict):
    """Shape of the sequence_length sub-dict in dataset profiles."""

    min: int
    max: int
    mean: float


class DatasetProfile(TypedDict, total=False):
    """Shape of the dict returned by ``profile_dataset``."""

    row_count: int
    label_distribution: Dict[str, int]
    num_classes: int
    sequence_length: SequenceLengthProfile
    duplicate_rate: float
    ptm_type_distribution: Dict[str, int]
    homology_leakage_accessions: int
    homology_leakage_warning: str


class DataContractError(ValueError):
    """Raised when a dataset violates the real-data release contract."""


def _parse_ptm_sites(raw: Any) -> List[PTMSiteRaw]:
    """Tolerantly parse a ptm_sites cell into a list of dicts.

    Accepts an already-parsed list, a JSON string, or empty/NaN. Returns ``[]``
    for missing data so callers can treat "no PTMs" uniformly.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, float) and pd.isna(raw):
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if not s or s.lower() in ("nan", "none"):
            return []
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError as exc:
            raise DataContractError(f"ptm_sites 不是合法 JSON: {s!r} ({exc})") from exc
        return parsed if isinstance(parsed, list) else []
    return []


def validate_data_contract(
    df: pd.DataFrame,
    *,
    require_all_rows_valid_ptm: bool = False,
    require_recommended: bool = False,
) -> ContractReport:
    """Validate ``df`` against the real-data release contract.

    Args:
        df: Raw dataset.
        require_all_rows_valid_ptm: When True, any PTM site whose 1-based
            ``position`` falls outside ``[1, len(sequence)]`` is a hard failure.
            When False (default) such rows are reported as warnings so a mostly
            correct dataset can still be used after cleanup.
        require_recommended: When True (real-data release gate, P2-1), missing
            recommended provenance columns (protein_accession / gene_symbol /
            source_db / evidence_level / split_group) are hard failures — the
            dataset cannot be traced or leakage-audited without them. When
            False (default) they are warnings, so synthetic/demo data or
            legacy pipelines keep working.

    Returns:
        A report dict with ``ok``, ``hard_failures``, ``warnings``,
        ``missing_required``, ``missing_recommended``, and ``invalid_ptm_rows``.

    Raises:
        DataContractError: Only when invoked in strict mode by the caller (the
            function itself collects failures into the report; raising is the
            caller's policy).
    """
    hard_failures: List[str] = []
    warnings: List[str] = []

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        hard_failures.append(
            "缺少必需列: " + ", ".join(missing_required)
            + "。真实数据契约要求: " + ", ".join(REQUIRED_COLUMNS) + "。"
        )
    missing_recommended = [c for c in RECOMMENDED_COLUMNS if c not in df.columns]
    if missing_recommended:
        msg = (
            "缺少推荐 provenance 列: " + ", ".join(missing_recommended)
            + "（影响 dataset_hash/leakage 审计与发布门禁）。"
        )
        if require_recommended:
            hard_failures.append(
                msg + " 真实数据必须可追溯：请重新运行 scripts/integrate_data_v2.py"
                " 生成含 provenance 列的数据，或手工补齐后重试。"
            )
        else:
            warnings.append(msg)

    invalid_ptm_rows = 0
    if not missing_required and len(df) > 0:
        # Sequence characters.
        bad_chars = set()
        for seq in df["sequence"].astype(str):
            for ch in seq:
                if ch not in VALID_AA:
                    bad_chars.add(ch)
        if bad_chars:
            warnings.append(
                f"序列含非标准氨基酸字符: {sorted(bad_chars)}。"
                f"标准集合为 {''.join(sorted(VALID_AA))}。"
            )

        # PTM site position semantics (1-based, within sequence length).
        for _idx, row in df.iterrows():
            sites = _parse_ptm_sites(row["ptm_sites"])
            seq_len = len(str(row["sequence"]))
            for site in sites:
                if not isinstance(site, dict):
                    continue
                pos = site.get("position")
                if pos is None:
                    continue
                try:
                    pos_int = int(pos)
                except (TypeError, ValueError):
                    invalid_ptm_rows += 1
                    continue
                if pos_int < 1 or pos_int > seq_len:
                    invalid_ptm_rows += 1
        if invalid_ptm_rows:
            msg = (
                f"{invalid_ptm_rows} 个 PTM 位点的 position 超出 [1, 序列长度] "
                f"（position 为 1-based）。"
            )
            if require_all_rows_valid_ptm:
                hard_failures.append(msg)
            else:
                warnings.append(msg)

    return {
        "ok": not hard_failures,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "missing_required": missing_required,
        "missing_recommended": missing_recommended,
        "invalid_ptm_rows": invalid_ptm_rows,
        "row_count": int(len(df)),
    }


def profile_dataset(df: pd.DataFrame, label_col: str = "cell_state") -> DatasetProfile:
    """Produce a dataset profile for ``artifact_manifest.json`` provenance (P1-1).

    Covers: sample count, label distribution, PTM-type distribution, sequence
    length stats, duplicate rate, and a homology-leakage heuristic when an
    accession column is present.
    """
    profile: DatasetProfile = {"row_count": int(len(df))}

    if label_col in df.columns:
        dist = df[label_col].dropna().astype(str).value_counts().to_dict()
        profile["label_distribution"] = {str(k): int(v) for k, v in dist.items()}
        profile["num_classes"] = len(dist)

    if "sequence" in df.columns:
        seqs = df["sequence"].astype(str)
        lengths = seqs.str.len()
        profile["sequence_length"] = {
            "min": int(lengths.min()) if len(lengths) else 0,
            "max": int(lengths.max()) if len(lengths) else 0,
            "mean": float(round(lengths.mean(), 2)) if len(lengths) else 0.0,
        }
        profile["duplicate_rate"] = float(round(seqs.duplicated().mean(), 4))

    # PTM-type distribution.
    if "ptm_sites" in df.columns:
        type_counts: Dict[str, int] = {}
        for raw in df["ptm_sites"]:
            for site in _parse_ptm_sites(raw):
                if isinstance(site, dict) and "type" in site:
                    type_counts[str(site["type"])] = (
                        type_counts.get(str(site["type"]), 0) + 1
                    )
        profile["ptm_type_distribution"] = type_counts

    # Homology-leakage heuristic: an accession must not appear in >1 split_group.
    if "protein_accession" in df.columns and "split_group" in df.columns:
        acc_groups = (
            df.dropna(subset=["protein_accession", "split_group"])
            .groupby("protein_accession")["split_group"].nunique()
        )
        leaking = int((acc_groups > 1).sum())
        profile["homology_leakage_accessions"] = leaking
        if leaking:
            profile["homology_leakage_warning"] = (
                f"{leaking} 个 accession 同时出现在多个 split_group，存在同源泄漏风险。"
            )

    return profile


__all__ = [
    "REQUIRED_COLUMNS",
    "RECOMMENDED_COLUMNS",
    "DataContractError",
    "validate_data_contract",
    "profile_dataset",
]
