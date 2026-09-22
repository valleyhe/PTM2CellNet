"""Build the DAVF candidate specification from upstream PTM and GSE artifacts.

This closes the two manual hand-off gaps of the mainline (analysis
``project_analysis_20260910.md`` §4.1 N-1/N-2):

* N-1 — ``predict_ptm_sites.py`` emits a plain site CSV; nothing turned it
  into ``PTMSiteDirectionProposal`` records for ``run_davf_perturbgen_e2e.py``
  and candidates had to be hand-written as JSON.
* N-2 — ``summarize_gse_directions.py`` emits ``direction_evidence.csv``;
  nothing joined its ``observed_*`` columns onto the candidates, so the
  independent expression side of the direction gate was transcribed by hand.

Both steps are now explicit, provenance-bearing, and fail-fast: unknown
protein IDs are never silently mapped and evidence rows are never guessed.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

CANDIDATE_SPEC_SCHEMA_VERSION = "ptm2cellnet.candidate-spec/v1"

PREDICTION_REQUIRED_COLUMNS: tuple[str, ...] = (
    "protein_id",
    "position",
    "ptm_type",
    "probability",
)
GENE_MAP_REQUIRED_COLUMNS: tuple[str, ...] = (
    "protein_id",
    "gene_symbol",
    "ensembl_id",
)
DIRECTION_MAP_REQUIRED_COLUMNS: tuple[str, ...] = (
    "protein_id",
    "position",
    "ptm_type",
    "proposed_direction",
)
EVIDENCE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "cell_type",
    "ensembl_id",
    "log2fc",
    "fdr",
    "observed_direction",
)
_VALID_DIRECTIONS = ("up", "down")


class CandidateSpecError(ValueError):
    """Raised when candidate-spec inputs violate the mainline contract."""


@dataclass(frozen=True)
class CandidateSpecSummary:
    """Audit record of how many upstream rows survived each build filter."""

    n_predictions: int
    n_below_probability: int
    n_unmapped_proteins: int
    n_neutral_direction: int
    n_without_evidence: int
    n_low_donor_support: int
    n_candidates: int
    skipped_proteins: tuple[str, ...] = ()

    @property
    def n_expected_candidates(self) -> int:
        return (
            self.n_predictions
            - self.n_below_probability
            - self.n_unmapped_proteins
            - self.n_neutral_direction
            - self.n_without_evidence
        )


def load_ptm_site_predictions(path: str | Path) -> pd.DataFrame:
    """Load and validate a ``predict_ptm_sites.py`` output table."""

    resolved = Path(path).expanduser().resolve(strict=True)
    frame = pd.read_csv(resolved)
    missing = [column for column in PREDICTION_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise CandidateSpecError(f"PTM site predictions are missing required columns: {', '.join(missing)}")
    frame["position"] = pd.to_numeric(frame["position"], errors="raise").astype(int)
    frame["probability"] = pd.to_numeric(frame["probability"], errors="raise").astype(float)
    if (frame["position"] < 1).any():
        raise CandidateSpecError("PTM site positions must be >= 1")
    if not frame["probability"].between(0.0, 1.0).all():
        raise CandidateSpecError("site probabilities must be within [0, 1]")
    if frame[list(PREDICTION_REQUIRED_COLUMNS)].isna().any().any():
        raise CandidateSpecError("PTM site predictions contain missing values")
    return frame


def load_gene_map(path: str | Path) -> dict[str, tuple[str, str]]:
    """Load the explicit protein → (gene_symbol, ensembl_id) mapping."""

    resolved = Path(path).expanduser().resolve(strict=True)
    mapping: dict[str, tuple[str, str]] = {}
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in GENE_MAP_REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise CandidateSpecError(f"gene map is missing required columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            protein_id = (row.get("protein_id") or "").strip()
            symbol = (row.get("gene_symbol") or "").strip()
            ensembl_id = (row.get("ensembl_id") or "").strip()
            if not protein_id:
                raise CandidateSpecError(f"gene map row {row_number} has an empty protein_id")
            if not symbol or not ensembl_id:
                raise CandidateSpecError(
                    f"gene map row {row_number} ({protein_id}) must provide both "
                    "gene_symbol and ensembl_id; unmapped proteins must be omitted, not blanked"
                )
            previous = mapping.get(protein_id)
            if previous is not None and previous != (symbol, ensembl_id):
                raise CandidateSpecError(
                    f"gene map contains conflicting entries for protein {protein_id}: "
                    f"{previous} vs {(symbol, ensembl_id)}"
                )
            mapping[protein_id] = (symbol, ensembl_id)
    if not mapping:
        raise CandidateSpecError("gene map must contain at least one row")
    return mapping


def load_direction_map(path: str | Path) -> dict[tuple[str, int, str], str]:
    """Load site-level PTM direction claims keyed by (protein, position, ptm_type)."""

    resolved = Path(path).expanduser().resolve(strict=True)
    mapping: dict[tuple[str, int, str], str] = {}
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in DIRECTION_MAP_REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise CandidateSpecError(f"direction map is missing required columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            protein_id = (row.get("protein_id") or "").strip()
            ptm_type = (row.get("ptm_type") or "").strip()
            direction = (row.get("proposed_direction") or "").strip().lower()
            raw_position = (row.get("position") or "").strip()
            if not protein_id or not ptm_type or not raw_position:
                raise CandidateSpecError(
                    f"direction map row {row_number} must provide protein_id, position and ptm_type"
                )
            if direction not in _VALID_DIRECTIONS:
                raise CandidateSpecError(f"direction map row {row_number} proposed_direction must be 'up' or 'down'")
            key = (protein_id, int(raw_position), ptm_type)
            if key in mapping and mapping[key] != direction:
                raise CandidateSpecError(f"direction map contains conflicting rows for {key}")
            mapping[key] = direction
    return mapping


def load_direction_evidence(path: str | Path) -> pd.DataFrame:
    """Load and validate a ``summarize_gse_directions.py`` evidence table."""

    resolved = Path(path).expanduser().resolve(strict=True)
    frame = pd.read_csv(resolved)
    missing = [column for column in EVIDENCE_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise CandidateSpecError(f"direction evidence is missing required columns: {', '.join(missing)}")
    frame["log2fc"] = pd.to_numeric(frame["log2fc"], errors="raise").astype(float)
    frame["fdr"] = pd.to_numeric(frame["fdr"], errors="raise").astype(float)
    if not frame["fdr"].between(0.0, 1.0).all():
        raise CandidateSpecError("direction evidence FDR values must be within [0, 1]")
    if not frame["log2fc"].apply(math.isfinite).all():
        raise CandidateSpecError("direction evidence log2fc values must be finite")
    direction = frame["observed_direction"].astype(str).str.strip()
    invalid = sorted(set(direction) - {"up", "down", "neutral"})
    if invalid:
        raise CandidateSpecError(f"observed_direction must be 'up', 'down' or 'neutral'; got {invalid}")
    frame["observed_direction"] = direction
    sign_mismatch = ((frame["observed_direction"] == "up") & (frame["log2fc"] <= 0)) | (
        (frame["observed_direction"] == "down") & (frame["log2fc"] >= 0)
    )
    if sign_mismatch.any():
        raise CandidateSpecError("direction evidence rows disagree between observed_direction and log2fc sign")
    duplicated = frame.duplicated(subset=["cell_type", "ensembl_id"], keep=False)
    if duplicated.any():
        keys = frame.loc[duplicated, ["cell_type", "ensembl_id"]].drop_duplicates().to_dict("records")
        raise CandidateSpecError(
            f"direction evidence must have one row per (cell_type, ensembl_id); duplicates: {keys}"
        )
    return frame


def build_spec_candidates(
    predictions: pd.DataFrame,
    gene_map: Mapping[str, tuple[str, str]],
    evidence: pd.DataFrame,
    *,
    provenance: str,
    proposed_direction: str,
    direction_overrides: Mapping[tuple[str, int, str], str] | None = None,
    site_probability_threshold: float = 0.5,
    context_cell_index: int = 0,
    cell_types: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], CandidateSpecSummary]:
    """Join PTM site predictions with GSE direction evidence.

    Returns the candidate rows consumed by ``run_davf_perturbgen_e2e.py``
    plus an audit summary.  Rows are dropped with an explicit reason
    (never silently re-mapped) when they fail the probability threshold,
    lack a gene-map entry, have a neutral observed direction, or have no
    matching direction evidence.
    """

    if not isinstance(provenance, str) or not provenance.strip():
        raise CandidateSpecError("provenance must be a non-empty string")
    if proposed_direction not in _VALID_DIRECTIONS:
        raise CandidateSpecError("proposed_direction must be 'up' or 'down'")
    if not 0.0 <= site_probability_threshold <= 1.0:
        raise CandidateSpecError("site_probability_threshold must be within [0, 1]")
    if context_cell_index < 0:
        raise CandidateSpecError("context_cell_index must be >= 0")

    evidence_index = _index_evidence(evidence, cell_types)
    overrides = dict(direction_overrides or {})

    n_below_probability = 0
    n_unmapped_proteins = 0
    n_neutral_direction = 0
    n_without_evidence = 0
    n_low_donor_support = 0
    skipped_proteins: list[str] = []
    candidates: list[dict[str, Any]] = []

    for record in predictions.to_dict("records"):
        protein_id = str(record["protein_id"]).strip()
        position = int(record["position"])
        ptm_type = str(record["ptm_type"]).strip()
        probability = float(record["probability"])
        aa = str(record.get("aa") or "").strip()

        if probability < site_probability_threshold:
            n_below_probability += 1
            continue
        mapped = gene_map.get(protein_id)
        if mapped is None:
            n_unmapped_proteins += 1
            skipped_proteins.append(protein_id)
            continue
        gene_symbol, ensembl_id = mapped
        # The evidence join is keyed by (ensembl_id, cell_type); the prediction
        # CSV does not carry a cell type, so the requested filter set resolves it.
        row = _match_evidence(evidence_index, ensembl_id, cell_types)
        if row is None:
            n_without_evidence += 1
            continue
        if row["observed_direction"] == "neutral":
            n_neutral_direction += 1
            continue

        site_direction = overrides.get((protein_id, position, ptm_type), proposed_direction)
        if site_direction not in _VALID_DIRECTIONS:
            raise CandidateSpecError(
                f"direction override for {(protein_id, position, ptm_type)} must be 'up' or 'down'"
            )
        n_normal = row.get("n_normal_donors")
        n_disease = row.get("n_disease_donors")
        n_normal_value = int(n_normal) if n_normal is not None and pd.notna(n_normal) else None
        n_disease_value = int(n_disease) if n_disease is not None and pd.notna(n_disease) else None
        donor_warning = bool(
            n_normal_value is not None and n_disease_value is not None and (n_normal_value < 3 or n_disease_value < 3)
        )
        if donor_warning:
            n_low_donor_support += 1

        candidates.append(
            {
                "context_cell_index": context_cell_index,
                "gene_symbol": gene_symbol,
                "ensembl_id": ensembl_id,
                "position": position,
                "ptm_type": ptm_type,
                "proposed_direction": site_direction,
                "site_probability": probability,
                "provenance": provenance,
                "cell_type": row["cell_type"],
                "ptm_context": f"{gene_symbol}:{aa}{position}" if aa else f"{gene_symbol}:{position}",
                "observed_log2fc": float(row["log2fc"]),
                "observed_fdr": float(row["fdr"]),
                "observed_direction": row["observed_direction"],
                "direction_evidence_donor_counts": {
                    "normal": n_normal_value,
                    "disease": n_disease_value,
                },
                "low_donor_support": donor_warning,
            }
        )

    summary = CandidateSpecSummary(
        n_predictions=len(predictions),
        n_below_probability=n_below_probability,
        n_unmapped_proteins=n_unmapped_proteins,
        n_neutral_direction=n_neutral_direction,
        n_without_evidence=n_without_evidence,
        n_low_donor_support=n_low_donor_support,
        n_candidates=len(candidates),
        skipped_proteins=tuple(sorted(set(skipped_proteins))),
    )
    if summary.n_candidates != summary.n_expected_candidates:
        raise CandidateSpecError(
            f"candidate accounting mismatch: {summary.n_candidates} built vs {summary.n_expected_candidates} expected"
        )
    return candidates, summary


def build_candidate_spec_payload(
    *,
    context_h5ad: str | Path,
    candidates: Sequence[Mapping[str, Any]],
    summary: CandidateSpecSummary,
    sources: Mapping[str, str],
) -> dict[str, Any]:
    """Assemble the JSON payload consumed by ``run_davf_perturbgen_e2e.py``."""

    if not candidates:
        raise CandidateSpecError("no candidates survived the join; refusing to emit an empty spec")
    for index, candidate in enumerate(candidates):
        required = (
            "context_cell_index",
            "gene_symbol",
            "ensembl_id",
            "position",
            "ptm_type",
            "proposed_direction",
            "site_probability",
            "provenance",
            "cell_type",
            "ptm_context",
            "observed_log2fc",
            "observed_fdr",
            "observed_direction",
        )
        missing = [key for key in required if key not in candidate]
        if missing:
            raise CandidateSpecError(f"candidate {index} is missing fields: {', '.join(missing)}")
    return {
        "schema_version": CANDIDATE_SPEC_SCHEMA_VERSION,
        "context_h5ad": str(Path(context_h5ad).expanduser().resolve()),
        "candidates": [dict(candidate) for candidate in candidates],
        "build_summary": {
            **_summary_to_dict(summary),
            "sources": {key: str(value) for key, value in sources.items()},
        },
    }


def write_candidate_spec(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write the candidate spec JSON and return the resolved path."""

    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved


def _index_evidence(
    evidence: pd.DataFrame,
    cell_types: Sequence[str] | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    if cell_types is not None and not cell_types:
        raise CandidateSpecError("cell_types filter must not be empty when provided")
    wanted = {str(value).strip() for value in cell_types} if cell_types is not None else None
    index: dict[tuple[str, str], dict[Any, Any]] = {}
    for record in evidence.to_dict("records"):
        cell_type = str(record["cell_type"]).strip()
        if wanted is not None and cell_type not in wanted:
            continue
        ensembl_id = str(record["ensembl_id"]).strip()
        index[(ensembl_id, cell_type)] = record
    return index


def _match_evidence(
    evidence_index: Mapping[tuple[str, str], Mapping[str, Any]],
    ensembl_id: str,
    cell_types: Sequence[str] | None,
) -> Mapping[str, Any] | None:
    if cell_types:
        matches = [
            evidence_index[(ensembl_id, str(cell_type).strip())]
            for cell_type in cell_types
            if (ensembl_id, str(cell_type).strip()) in evidence_index
        ]
    else:
        matches = [row for (gene, _ct), row in evidence_index.items() if gene == ensembl_id]
    unique = {id(row): row for row in matches}
    if not unique:
        return None
    if len(unique) > 1:
        keys = sorted({row["cell_type"] for row in unique.values()})
        raise CandidateSpecError(
            f"direction evidence for {ensembl_id} matches multiple cell types {keys}; pass --cell-type to disambiguate"
        )
    return next(iter(unique.values()))


def _summary_to_dict(summary: CandidateSpecSummary) -> dict[str, Any]:
    return {
        "n_predictions": summary.n_predictions,
        "n_below_probability": summary.n_below_probability,
        "n_unmapped_proteins": summary.n_unmapped_proteins,
        "n_neutral_direction": summary.n_neutral_direction,
        "n_without_evidence": summary.n_without_evidence,
        "n_low_donor_support": summary.n_low_donor_support,
        "n_candidates": summary.n_candidates,
        "skipped_proteins": list(summary.skipped_proteins),
    }
