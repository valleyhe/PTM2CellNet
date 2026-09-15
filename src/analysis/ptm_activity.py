"""PTM site quantification and activity-table contracts (方案 §4.1/§4.2/§5.1).

This module owns the *standard-table* boundary of the PTM-activity mainline:

* ``load_ptm_site_quantification`` validates the ``ptm_site_quantification.tsv``
  contract (方案 §4.1) — every row is one sample × donor × site × PTM-type
  quantification with traceable protein/site/PTM identifiers;
* ``standardize_ptm_input`` applies the registered preprocessing (方案 §5.1):
  explicit replicate policy, site/total-protein normalization on a single
  declared value scale, protein → canonical Ensembl mapping via an explicit
  gene map (unmapped rows are kept and counted, never guessed);
* ``write_ptm_input_manifest`` records provenance, parameters, species,
  samples and donor semantics;
* ``load_ptm_activity_table`` parses the ``ptm_activity.tsv`` contract
  (方案 §4.2) produced by the external activity tools (KSTAR/PhosR run in
  their own environment; this module only consumes standard tables).

KSTAR/PhosR themselves are *not* invoked here — the plan (§6.2) keeps heavy
dependencies out of the core environment and hands over via tables and
manifests.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from src.analysis.ptm_research_config import PTMResearchConfig
from src.models.gene_vocabulary import normalize_ensembl_id

PTM_INPUT_REQUIRED_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "donor_id",
    "condition",
    "protein_id",
    "gene_symbol",
    "residue",
    "ptm_type",
    "ptm_value",
    "value_scale",
    "ptm_qvalue",
    "total_protein_value",
    "species",
    "source_dataset",
)
#: Value scales with a defined site/total-protein normalization (方案 §4.1).
#: ``linear`` → ptm_value / total_protein_value; ``log2`` → difference
#: (log2 ratio). Any other declared scale fails fast instead of being
#: silently divided.
SUPPORTED_VALUE_SCALES = ("linear", "log2")

ACTIVITY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "activity_unit",
    "regulator_id",
    "regulator_type",
    "condition_or_contrast",
    "activity_score",
    "activity_direction",
    "activity_pvalue",
    "activity_qvalue",
    "n_substrates",
    "network_coverage",
    "method",
    "method_version",
    "input_manifest",
)
_VALID_ACTIVITY_DIRECTIONS = ("up", "down")

PTM_INPUT_SCHEMA_VERSION = "ptm2cellnet.ptm-input/v1"
PTM_INPUT_MANIFEST_SCHEMA_VERSION = "ptm2cellnet.ptm-input-manifest/v1"


class PTMActivityContractError(ValueError):
    """Raised when a PTM input or activity table violates the plan contract."""


@dataclass(frozen=True)
class PTMStandardizationAudit:
    """Audit record of the §5.1 preprocessing decisions."""

    n_rows_input: int
    n_replicate_groups: int
    n_rows_after_replicates: int
    n_rows_with_total_protein: int
    value_scale: str
    replicate_policy: str
    n_unmapped_proteins: int
    unmapped_proteins: tuple[str, ...]
    n_rows_without_donor: int
    n_samples: int
    n_donors: int
    conditions: tuple[str, ...]


def load_ptm_site_quantification(path: str | Path) -> pd.DataFrame:
    """Load and validate a ``ptm_site_quantification.tsv`` (方案 §4.1)."""

    resolved = Path(path).expanduser().resolve(strict=True)
    frame = pd.read_csv(resolved, sep="\t", dtype={"donor_id": "string", "total_protein_value": "string"})
    missing = [column for column in PTM_INPUT_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise PTMActivityContractError(f"PTM input is missing required columns: {', '.join(missing)}")
    for column in (
        "sample_id",
        "condition",
        "protein_id",
        "gene_symbol",
        "residue",
        "ptm_type",
        "value_scale",
        "species",
        "source_dataset",
    ):
        if frame[column].isna().any() or (frame[column].astype(str).str.strip() == "").any():
            raise PTMActivityContractError(f"PTM input column {column!r} must not contain empty values")
    frame["ptm_value"] = pd.to_numeric(frame["ptm_value"], errors="raise").astype(float)
    if not frame["ptm_value"].apply(math.isfinite).all():
        raise PTMActivityContractError("ptm_value must be finite")
    frame["ptm_qvalue"] = pd.to_numeric(frame["ptm_qvalue"], errors="raise").astype(float)
    if not frame["ptm_qvalue"].between(0.0, 1.0).all():
        raise PTMActivityContractError("ptm_qvalue must be within [0, 1]")
    scales = set(frame["value_scale"].astype(str).str.strip())
    unknown = scales - set(SUPPORTED_VALUE_SCALES)
    if unknown:
        raise PTMActivityContractError(
            f"value_scale must be one of {', '.join(SUPPORTED_VALUE_SCALES)}; got {sorted(unknown)}"
        )
    if len(scales) > 1:
        raise PTMActivityContractError(
            f"PTM input mixes value scales {sorted(scales)}; split the table per scale, "
            "normalization cannot pick one silently"
        )
    total = pd.to_numeric(frame["total_protein_value"], errors="coerce")
    if total.notna().any() and not total.dropna().apply(math.isfinite).all():
        raise PTMActivityContractError("total_protein_value must be finite where present")
    if (total.dropna() <= 0.0).any():
        raise PTMActivityContractError("total_protein_value must be > 0 (site/total normalization would divide by it)")
    frame["total_protein_value"] = total
    frame["donor_id"] = frame["donor_id"].fillna("").astype(str).str.strip()
    return frame


def standardize_ptm_input(
    frame: pd.DataFrame,
    *,
    config: PTMResearchConfig,
    gene_map: Mapping[str, tuple[str, str]],
) -> tuple[pd.DataFrame, PTMStandardizationAudit]:
    """Apply the registered §5.1 preprocessing to a validated input table.

    Replicates are collapsed by the frozen ``replicate_policy``; site/total
    normalization uses the single declared ``value_scale``; protein → Ensembl
    mapping comes from the explicit gene map and unmapped proteins are kept
    verbatim with a NULL ``ensembl_id`` and counted (they cannot enter
    signed propagation, which is keyed on canonical Ensembl).
    """

    if not gene_map:
        raise PTMActivityContractError("gene map must contain at least one protein entry")
    key_columns = ["sample_id", "protein_id", "residue", "ptm_type"]
    n_input = len(frame)
    n_replicate_groups = 0
    if frame.duplicated(subset=key_columns, keep=False).any():
        counts = frame.groupby(key_columns, dropna=False).size()
        n_replicate_groups = int((counts > 1).sum())
        if config.replicate_policy == "fail":
            raise PTMActivityContractError(
                f"{n_replicate_groups} replicate (sample, protein, residue, ptm_type) groups found; "
                "replicate_policy=fail refuses to collapse them (方案 §5.1 登记规则)"
            )
        value_columns = ["ptm_value", "ptm_qvalue", "total_protein_value"]
        frame = (
            frame.groupby(key_columns, dropna=False, sort=True)
            .agg(
                {column: "mean" for column in value_columns}
                | {column: "first" for column in frame.columns if column not in key_columns + value_columns}
            )
            .reset_index()
        )
    scale = str(frame["value_scale"].iloc[0]).strip()
    has_total = frame["total_protein_value"].notna()
    n_with_total = int(has_total.sum())
    # Rows without total protein stay NaN — an un-normalized raw value must
    # never masquerade as a normalized one (方案 §4.1).
    normalized = pd.Series(float("nan"), index=frame.index)
    if scale == "linear":
        normalized[has_total] = frame.loc[has_total, "ptm_value"] / frame.loc[has_total, "total_protein_value"]
    else:  # log2
        normalized[has_total] = frame.loc[has_total, "ptm_value"] - frame.loc[has_total, "total_protein_value"]
    frame = frame.copy()
    frame["ptm_value_normalized"] = normalized
    ensembl = frame["protein_id"].astype(str).str.strip().map(lambda protein: gene_map.get(protein, (None, None))[1])
    unmapped_mask = ensembl.isna()
    unmapped_proteins = tuple(sorted(set(frame.loc[unmapped_mask, "protein_id"].astype(str))))
    # Validate canonical form up front so downstream joins never see a bad id.
    for value in ensembl.dropna().astype(str):
        normalize_ensembl_id(value)
    frame["ensembl_id"] = ensembl
    frame["has_donor"] = frame["donor_id"].astype(str).str.len() > 0
    audit = PTMStandardizationAudit(
        n_rows_input=n_input,
        n_replicate_groups=n_replicate_groups,
        n_rows_after_replicates=len(frame),
        n_rows_with_total_protein=n_with_total,
        value_scale=scale,
        replicate_policy=config.replicate_policy,
        n_unmapped_proteins=len(unmapped_proteins),
        unmapped_proteins=unmapped_proteins,
        n_rows_without_donor=int((~frame["has_donor"]).sum()),
        n_samples=int(frame["sample_id"].nunique()),
        n_donors=int(frame.loc[frame["has_donor"], "donor_id"].nunique()),
        conditions=tuple(sorted(set(frame["condition"].astype(str)))),
    )
    return frame, audit


def write_ptm_input_manifest(
    path: str | Path,
    *,
    input_path: Path,
    audit: PTMStandardizationAudit,
    config: PTMResearchConfig,
    gene_map_path: Path,
    output_path: Path,
) -> Path:
    """Write ``ptm_input_manifest.json`` (方案 §5.1 step 6)."""

    manifest = {
        "schema_version": PTM_INPUT_MANIFEST_SCHEMA_VERSION,
        "source": {
            "file": str(input_path),
            "sha256": _sha256_file(input_path),
        },
        "gene_map": {
            "file": str(gene_map_path),
            "sha256": _sha256_file(gene_map_path),
        },
        "config": {
            "research_objective": config.research_objective,
            "reference_axis": config.reference_axis,
            "contrast": config.contrast,
            "primary_activity_method": config.primary_activity_method,
            "sensitivity_activity_method": config.sensitivity_activity_method,
            "species": config.species,
            "ptm_cohort": config.ptm_cohort,
            "replicate_policy": config.replicate_policy,
        },
        "standardization": {
            "n_rows_input": audit.n_rows_input,
            "n_replicate_groups": audit.n_replicate_groups,
            "n_rows_after_replicates": audit.n_rows_after_replicates,
            "value_scale": audit.value_scale,
            "n_rows_with_total_protein": audit.n_rows_with_total_protein,
            "n_unmapped_proteins": audit.n_unmapped_proteins,
            "unmapped_proteins": list(audit.unmapped_proteins),
            "n_rows_without_donor": audit.n_rows_without_donor,
            "n_samples": audit.n_samples,
            "n_donors": audit.n_donors,
            "conditions": list(audit.conditions),
        },
        "notes": {
            "donor_semantics": (
                "rows without donor_id are cohort-level exploratory only (方案 §4.1); "
                "formal donor-level comparison requires explicit donors"
            ),
            "normalization": (
                "site/total-protein normalization applied only where total_protein_value is present; "
                "absence is explicit in ptm_value_normalized being NaN, never filled"
            ),
        },
        "output": {"file": str(output_path)},
    }
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return resolved


def load_ptm_activity_table(path: str | Path) -> pd.DataFrame:
    """Load and validate a ``ptm_activity.tsv`` (方案 §4.2, external tool output)."""

    resolved = Path(path).expanduser().resolve(strict=True)
    frame = pd.read_csv(resolved, sep="\t")
    missing = [column for column in ACTIVITY_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise PTMActivityContractError(f"activity table is missing required columns: {', '.join(missing)}")
    for column in (
        "activity_unit",
        "regulator_id",
        "regulator_type",
        "condition_or_contrast",
        "method",
        "method_version",
        "input_manifest",
    ):
        if frame[column].isna().any() or (frame[column].astype(str).str.strip() == "").any():
            raise PTMActivityContractError(f"activity table column {column!r} must not contain empty values")
    frame["activity_score"] = pd.to_numeric(frame["activity_score"], errors="raise").astype(float)
    if not frame["activity_score"].apply(math.isfinite).all():
        raise PTMActivityContractError("activity_score must be finite")
    if (frame["activity_score"] == 0.0).any():
        raise PTMActivityContractError(
            "activity_score must be non-zero; a zero activity has no direction and must not be propagated"
        )
    frame["activity_pvalue"] = pd.to_numeric(frame["activity_pvalue"], errors="raise").astype(float)
    frame["activity_qvalue"] = pd.to_numeric(frame["activity_qvalue"], errors="raise").astype(float)
    for column in ("activity_pvalue", "activity_qvalue"):
        if not frame[column].between(0.0, 1.0).all():
            raise PTMActivityContractError(f"{column} must be within [0, 1]")
    frame["n_substrates"] = pd.to_numeric(frame["n_substrates"], errors="raise").astype(int)
    if (frame["n_substrates"] < 0).any():
        raise PTMActivityContractError("n_substrates must be >= 0")
    frame["network_coverage"] = pd.to_numeric(frame["network_coverage"], errors="raise").astype(float)
    if not frame["network_coverage"].between(0.0, 1.0).all():
        raise PTMActivityContractError("network_coverage must be within [0, 1]")
    direction = frame["activity_direction"].astype(str).str.strip().str.lower()
    invalid = sorted(set(direction) - set(_VALID_ACTIVITY_DIRECTIONS))
    if invalid:
        raise PTMActivityContractError(f"activity_direction must be 'up' or 'down'; got {invalid}")
    frame["activity_direction"] = direction
    sign_mismatch = ((direction == "up") & (frame["activity_score"] < 0)) | (
        (direction == "down") & (frame["activity_score"] > 0)
    )
    if sign_mismatch.any():
        raise PTMActivityContractError("activity rows disagree between activity_direction and activity_score sign")
    duplicated = frame.duplicated(subset=["regulator_id", "condition_or_contrast", "method"], keep=False)
    if duplicated.any():
        keys = frame.loc[duplicated, ["regulator_id", "condition_or_contrast", "method"]].drop_duplicates()
        raise PTMActivityContractError(
            f"activity table must have one row per (regulator_id, condition_or_contrast, method); duplicates: {keys.to_dict('records')}"
        )
    return frame


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gene_map(path: str | Path) -> dict[str, tuple[str, str]]:
    """Load the explicit protein → (gene_symbol, ensembl_id) mapping TSV.

    Same on-disk contract as the candidate-spec gene map
    (``protein_id, gene_symbol, ensembl_id``): unmapped proteins must be
    omitted by the author, never blanked; conflicts fail fast.
    """

    resolved = Path(path).expanduser().resolve(strict=True)
    mapping: dict[str, tuple[str, str]] = {}
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        import csv

        reader = csv.DictReader(handle, delimiter="\t")
        required = ("protein_id", "gene_symbol", "ensembl_id")
        missing = [column for column in required if column not in (reader.fieldnames or [])]
        if missing:
            raise PTMActivityContractError(f"gene map is missing required columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            protein_id = (row.get("protein_id") or "").strip()
            symbol = (row.get("gene_symbol") or "").strip()
            ensembl_id = (row.get("ensembl_id") or "").strip()
            if not protein_id:
                raise PTMActivityContractError(f"gene map row {row_number} has an empty protein_id")
            if not symbol or not ensembl_id:
                raise PTMActivityContractError(
                    f"gene map row {row_number} ({protein_id}) must provide both gene_symbol and ensembl_id"
                )
            ensembl_id = normalize_ensembl_id(ensembl_id)
            previous = mapping.get(protein_id)
            if previous is not None and previous != (symbol, ensembl_id):
                raise PTMActivityContractError(
                    f"gene map contains conflicting entries for protein {protein_id}: {previous} vs {(symbol, ensembl_id)}"
                )
            mapping[protein_id] = (symbol, ensembl_id)
    if not mapping:
        raise PTMActivityContractError("gene map must contain at least one row")
    return mapping


def activities_for_propagation(
    activity_frame: pd.DataFrame,
    *,
    method: str,
    condition_or_contrast: str | None = None,
) -> dict[str, float]:
    """Extract signed regulator activities for one method/contrast.

    The propagation input is the *primary* method's scores; a sensitivity
    method (方案 §4.2) must be propagated separately and compared, never
    averaged into one truth.
    """

    selected = activity_frame[activity_frame["method"].astype(str).str.strip() == method]
    if selected.empty:
        raise PTMActivityContractError(
            f"activity table has no rows for primary method {method!r}; available: "
            f"{sorted(set(activity_frame['method'].astype(str)))}"
        )
    if condition_or_contrast is not None:
        selected = selected[selected["condition_or_contrast"].astype(str).str.strip() == condition_or_contrast]
        if selected.empty:
            raise PTMActivityContractError(
                f"activity table has no rows for contrast {condition_or_contrast!r} under method {method!r}"
            )
    activities: dict[str, float] = {}
    for record in selected.to_dict("records"):
        regulator_id = str(record["regulator_id"]).strip()
        if not regulator_id:
            raise PTMActivityContractError("regulator_id must not be empty")
        if regulator_id in activities and activities[regulator_id] != float(record["activity_score"]):
            raise PTMActivityContractError(
                f"regulator {regulator_id} has conflicting activity scores for method {method!r}"
            )
        activities[regulator_id] = float(record["activity_score"])
    if not activities:
        raise PTMActivityContractError("no activities selected for propagation")
    return activities


__all__ = [
    "ACTIVITY_REQUIRED_COLUMNS",
    "PTMActivityContractError",
    "PTM_INPUT_MANIFEST_SCHEMA_VERSION",
    "PTM_INPUT_REQUIRED_COLUMNS",
    "PTM_INPUT_SCHEMA_VERSION",
    "PTMStandardizationAudit",
    "SUPPORTED_VALUE_SCALES",
    "activities_for_propagation",
    "load_gene_map",
    "load_ptm_activity_table",
    "load_ptm_site_quantification",
    "standardize_ptm_input",
    "write_ptm_input_manifest",
]
