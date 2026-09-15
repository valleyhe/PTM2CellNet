"""Global PTM gene-score table and PTM–AD cell-type intersections (方案 §4.4/§4.6/§5.4).

Two responsibilities:

1. Turn a :class:`~src.analysis.signed_network.PropagationResult` plus an
   explicit network-id → Ensembl map into the ``ptm_global_gene_scores.tsv``
   contract (§4.4). Rows are (source_activity, target_gene) pairs — the
   one-to-many source→target relation is never collapsed into a per-gene
   aggregate. ``gene_score`` lives on the propagation-internal scale only;
   without an independent null or external benchmark no q-value is emitted
   and ``prediction_status`` stays ``direction_only`` (§4.4 明确禁止写成 PTM 侧显著).
2. Join the score table with donor-level AD DEG per frozen cell type
   (§4.6/§5.4): canonical Ensembl keys, per-cell-type independent joins,
   direction-match flag, explicit PTM_only/AD_only/concordant/discordant
   membership and a formal/exploratory evidence tier driven by the frozen
   AD FDR and donor-support thresholds.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.analysis.ptm_research_config import PTMResearchConfig
from src.analysis.signed_network import PropagationResult
from src.models.gene_vocabulary import normalize_ensembl_id

GENE_SCORE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "source_activity_id",
    "target_ensembl_id",
    "target_gene_symbol",
    "gene_score",
    "predicted_gene_direction",
    "n_paths",
    "path_length_min",
    "network_coverage",
    "degree_normalized",
    "ptm_cohort",
    "prediction_status",
    "provenance",
)
GENE_SCORE_SCHEMA_VERSION = "ptm2cellnet.ptm-gene-scores/v1"

DEG_REQUIRED_COLUMNS: tuple[str, ...] = (
    "cell_type",
    "ensembl_id",
    "gene_symbol",
    "log2fc",
    "fdr",
    "observed_direction",
    "n_normal_donors",
    "n_disease_donors",
)

INTERSECTION_SCHEMA_VERSION = "ptm2cellnet.ptm-ad-intersection/v1"
TARGET_SET_SCHEMA_VERSION = "ptm2cellnet.downstream-target-set/v1"

_VALID_DIRECTIONS = ("up", "down")


class PTMGeneScoreError(ValueError):
    """Raised when gene-score or intersection inputs violate the plan contract."""


@dataclass(frozen=True)
class GeneScoreBuildAudit:
    n_propagation_pairs: int
    n_targets_unmapped: int
    unmapped_targets: tuple[str, ...]
    n_direction_indeterminate: int


def load_network_id_map(path: str | Path) -> dict[str, tuple[str, str]]:
    """Load the explicit network-id → (gene_symbol, ensembl_id) map.

    The signed network carries its own identifier space (e.g. OmniPath gene
    symbols); this map is authored from the frozen network release so no
    identifier is ever guessed. Three required columns: ``network_id``,
    ``gene_symbol``, ``ensembl_id``.
    """

    resolved = Path(path).expanduser().resolve(strict=True)
    mapping: dict[str, tuple[str, str]] = {}
    frame = pd.read_csv(resolved, sep="\t", dtype=str)
    missing = [column for column in ("network_id", "gene_symbol", "ensembl_id") if column not in frame.columns]
    if missing:
        raise PTMGeneScoreError(f"network id map is missing required columns: {', '.join(missing)}")
    for record in frame.to_dict("records"):
        network_id = (record["network_id"] or "").strip()
        symbol = (record["gene_symbol"] or "").strip()
        ensembl_id = (record["ensembl_id"] or "").strip()
        if not network_id:
            raise PTMGeneScoreError("network id map has an empty network_id")
        if not symbol or not ensembl_id:
            raise PTMGeneScoreError(f"network id map row for {network_id} must provide both gene_symbol and ensembl_id")
        ensembl_id = normalize_ensembl_id(ensembl_id)
        previous = mapping.get(network_id)
        if previous is not None and previous != (symbol, ensembl_id):
            raise PTMGeneScoreError(
                f"network id map contains conflicting entries for {network_id}: {previous} vs {(symbol, ensembl_id)}"
            )
        mapping[network_id] = (symbol, ensembl_id)
    if not mapping:
        raise PTMGeneScoreError("network id map must contain at least one row")
    return mapping


def build_gene_score_table(
    propagation: PropagationResult,
    *,
    network_id_map: Mapping[str, tuple[str, str]],
    config: PTMResearchConfig,
    provenance: str,
) -> tuple[pd.DataFrame, GeneScoreBuildAudit]:
    """Assemble the §4.4 score table from propagation results.

    Targets without an explicit Ensembl mapping are excluded from the formal
    table and reported in the audit — the downstream intersection is keyed
    on canonical Ensembl and must not guess identifiers.
    """

    if not provenance.strip():
        raise PTMGeneScoreError("provenance must be a non-empty string")
    rows: list[dict[str, Any]] = []
    unmapped: set[str] = set()
    n_indeterminate = 0
    for score in propagation.scores:
        mapped = network_id_map.get(score.target_id)
        if mapped is None:
            unmapped.add(score.target_id)
            continue
        symbol, ensembl_id = mapped
        if score.gene_score == 0.0:
            direction = ""
            status = "direction_indeterminate"
            n_indeterminate += 1
        else:
            direction = "up" if score.gene_score > 0 else "down"
            status = "direction_only"
        rows.append(
            {
                "source_activity_id": score.source_id,
                "target_ensembl_id": ensembl_id,
                "target_gene_symbol": symbol,
                "gene_score": score.gene_score,
                "predicted_gene_direction": direction,
                "n_paths": score.n_paths,
                "path_length_min": score.path_length_min,
                "network_coverage": score.network_coverage,
                "degree_normalized": score.degree_normalized,
                "ptm_cohort": config.ptm_cohort,
                "prediction_status": status,
                "provenance": provenance,
            }
        )
    audit = GeneScoreBuildAudit(
        n_propagation_pairs=len(propagation.scores),
        n_targets_unmapped=len(unmapped),
        unmapped_targets=tuple(sorted(unmapped)),
        n_direction_indeterminate=n_indeterminate,
    )
    return pd.DataFrame(rows, columns=list(GENE_SCORE_REQUIRED_COLUMNS)), audit


def write_gene_score_table(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write the score table TSV; refuses empty tables."""

    if frame.empty:
        raise PTMGeneScoreError("refusing to write an empty gene-score table")
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(resolved, sep="\t", index=False)
    return resolved


def load_gene_score_table(path: str | Path) -> pd.DataFrame:
    """Load and validate a §4.4 score table."""

    resolved = Path(path).expanduser().resolve(strict=True)
    frame = pd.read_csv(resolved, sep="\t", dtype={"predicted_gene_direction": "string"})
    missing = [column for column in GENE_SCORE_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise PTMGeneScoreError(f"gene score table is missing required columns: {', '.join(missing)}")
    for column in ("source_activity_id", "target_ensembl_id", "target_gene_symbol", "ptm_cohort", "provenance"):
        if frame[column].isna().any() or (frame[column].astype(str).str.strip() == "").any():
            raise PTMGeneScoreError(f"gene score column {column!r} must not contain empty values")
    frame["target_ensembl_id"] = frame["target_ensembl_id"].map(normalize_ensembl_id)
    frame["gene_score"] = pd.to_numeric(frame["gene_score"], errors="raise").astype(float)
    if not frame["gene_score"].apply(math.isfinite).all():
        raise PTMGeneScoreError("gene_score must be finite")
    direction = frame["predicted_gene_direction"].fillna("").astype(str).str.strip()
    invalid = sorted(set(direction) - {"", *_VALID_DIRECTIONS})
    if invalid:
        raise PTMGeneScoreError(f"predicted_gene_direction must be 'up', 'down' or empty; got {invalid}")
    sign_mismatch = ((direction == "up") & (frame["gene_score"] <= 0)) | (
        (direction == "down") & (frame["gene_score"] >= 0)
    )
    if sign_mismatch.any():
        raise PTMGeneScoreError("gene score rows disagree between predicted_gene_direction and gene_score sign")
    frame["predicted_gene_direction"] = direction
    frame["n_paths"] = pd.to_numeric(frame["n_paths"], errors="raise").astype(int)
    if (frame["n_paths"] < 1).any():
        raise PTMGeneScoreError("n_paths must be >= 1")
    frame["path_length_min"] = pd.to_numeric(frame["path_length_min"], errors="raise").astype(int)
    if (frame["path_length_min"] < 1).any():
        raise PTMGeneScoreError("path_length_min must be >= 1")
    for column in ("network_coverage", "degree_normalized"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        if not frame[column].apply(math.isfinite).all():
            raise PTMGeneScoreError(f"{column} must be finite")
    duplicated = frame.duplicated(subset=["source_activity_id", "target_ensembl_id"], keep=False)
    if duplicated.any():
        keys = frame.loc[duplicated, ["source_activity_id", "target_ensembl_id"]].drop_duplicates().to_dict("records")
        raise PTMGeneScoreError(
            f"gene score table must have one row per (source_activity_id, target_ensembl_id); duplicates: {keys}"
        )
    return frame


def load_deg_table(path: str | Path) -> pd.DataFrame:
    """Load and validate the donor-level AD DEG table (方案 §4.5)."""

    resolved = Path(path).expanduser().resolve(strict=True)
    frame = pd.read_csv(resolved, sep="\t")
    missing = [column for column in DEG_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise PTMGeneScoreError(f"AD DEG table is missing required columns: {', '.join(missing)}")
    for column in ("cell_type", "gene_symbol"):
        if frame[column].isna().any() or (frame[column].astype(str).str.strip() == "").any():
            raise PTMGeneScoreError(f"AD DEG column {column!r} must not contain empty values")
    frame["ensembl_id"] = frame["ensembl_id"].map(normalize_ensembl_id)
    frame["log2fc"] = pd.to_numeric(frame["log2fc"], errors="raise").astype(float)
    if not frame["log2fc"].apply(math.isfinite).all():
        raise PTMGeneScoreError("AD DEG log2fc must be finite")
    frame["fdr"] = pd.to_numeric(frame["fdr"], errors="raise").astype(float)
    if not frame["fdr"].between(0.0, 1.0).all():
        raise PTMGeneScoreError("AD DEG fdr must be within [0, 1]")
    direction = frame["observed_direction"].astype(str).str.strip().str.lower()
    invalid = sorted(set(direction) - {"up", "down", "neutral"})
    if invalid:
        raise PTMGeneScoreError(f"observed_direction must be 'up', 'down' or 'neutral'; got {invalid}")
    frame["observed_direction"] = direction
    sign_mismatch = ((direction == "up") & (frame["log2fc"] <= 0)) | ((direction == "down") & (frame["log2fc"] >= 0))
    if sign_mismatch.any():
        raise PTMGeneScoreError("AD DEG rows disagree between observed_direction and log2fc sign")
    for column in ("n_normal_donors", "n_disease_donors"):
        raw = frame[column]
        if raw.isna().any():
            raise PTMGeneScoreError(
                f"AD DEG {column} must be present on every row; donor-less comparisons must be "
                "declared as a different cohort contract, not left blank (方案 §4.5)"
            )
        frame[column] = pd.to_numeric(raw, errors="raise").astype(int)
        if (frame[column] < 0).any():
            raise PTMGeneScoreError(f"AD DEG {column} must be >= 0")
    duplicated = frame.duplicated(subset=["cell_type", "ensembl_id"], keep=False)
    if duplicated.any():
        keys = frame.loc[duplicated, ["cell_type", "ensembl_id"]].drop_duplicates().to_dict("records")
        raise PTMGeneScoreError(f"AD DEG table must have one row per (cell_type, ensembl_id); duplicates: {keys}")
    return frame


@dataclass(frozen=True)
class IntersectionSummary:
    cell_type: str
    n_concordant: int
    n_discordant: int
    n_ptm_only: int
    n_ad_only: int
    n_formal_concordant: int


def intersect_gene_scores_with_deg(
    score_frame: pd.DataFrame,
    deg_frame: pd.DataFrame,
    *,
    config: PTMResearchConfig,
) -> dict[str, tuple[pd.DataFrame, IntersectionSummary]]:
    """Per frozen cell type, join scores with DEG (方案 §5.4).

    Returns one row-set per cell type covering the full outer join with a
    ``membership`` column (concordant / discordant / PTM_only / AD_only) and
    an ``evidence_tier`` column (formal / exploratory). The formal
    intersection set ``I_c`` is ``membership=concordant AND
    evidence_tier=formal``; without a calibrated PTM-side null no joint
    significance is claimed (方案 §4.6).
    """

    summaries: dict[str, tuple[pd.DataFrame, IntersectionSummary]] = {}
    for cell_type in config.cell_types:
        deg_cell = deg_frame[deg_frame["cell_type"].astype(str).str.strip() == cell_type]
        if deg_cell.empty:
            raise PTMGeneScoreError(
                f"frozen cell type {cell_type!r} has no DEG rows; the cell-type list must be frozen "
                "against the DEG table before intersection (方案 §5.4 步骤 1)"
            )
        merged = score_frame.merge(
            deg_cell,
            left_on="target_ensembl_id",
            right_on="ensembl_id",
            how="outer",
            indicator=True,
            suffixes=("", "_deg"),
        )
        rows: list[dict[str, Any]] = []
        counts = {"concordant": 0, "discordant": 0, "PTM_only": 0, "AD_only": 0}
        n_formal = 0
        for record in merged.to_dict("records"):
            side = record["_merge"]
            if side == "left_only":
                membership = "PTM_only"
            elif side == "right_only":
                membership = "AD_only"
            else:
                predicted = record["predicted_gene_direction"]
                observed = record["observed_direction"]
                membership = "concordant" if predicted != "" and predicted == observed else "discordant"
            counts[membership] += 1
            fdr = record.get("fdr")
            if membership in ("PTM_only", "AD_only"):
                tier = "ptm_only" if membership == "PTM_only" else "ad_only"
            else:
                donor_ok = (
                    int(record["n_normal_donors"]) >= config.min_donors_per_state
                    and int(record["n_disease_donors"]) >= config.min_donors_per_state
                )
                fdr_ok = fdr is not None and not pd.isna(fdr) and float(fdr) <= config.deg_max_fdr
                tier = "formal" if donor_ok and fdr_ok else "exploratory"
            direction_match = membership == "concordant"
            if direction_match and tier == "formal":
                n_formal += 1
            predicted = record.get("predicted_gene_direction")
            observed = record.get("observed_direction")
            rows.append(
                {
                    "cell_type": cell_type,
                    "source_activity_id": record.get("source_activity_id", ""),
                    "target_ensembl_id": record.get("target_ensembl_id") or record.get("ensembl_id"),
                    "target_gene_symbol": record.get("target_gene_symbol") or record.get("gene_symbol"),
                    "gene_score": record.get("gene_score"),
                    "predicted_gene_direction": "" if predicted is None or pd.isna(predicted) else predicted,
                    "observed_log2fc": record.get("log2fc"),
                    "observed_fdr": fdr,
                    "observed_direction": "" if observed is None or pd.isna(observed) else observed,
                    "n_normal_donors": record.get("n_normal_donors"),
                    "n_disease_donors": record.get("n_disease_donors"),
                    "direction_match": direction_match,
                    "gene_score_network_coverage": record.get("network_coverage"),
                    "gene_score_n_paths": record.get("n_paths"),
                    "membership": membership,
                    "evidence_tier": tier,
                }
            )
        frame = pd.DataFrame(rows)
        summary = IntersectionSummary(
            cell_type=cell_type,
            n_concordant=counts["concordant"],
            n_discordant=counts["discordant"],
            n_ptm_only=counts["PTM_only"],
            n_ad_only=counts["AD_only"],
            n_formal_concordant=n_formal,
        )
        summaries[cell_type] = (frame, summary)
    return summaries


def write_intersection_outputs(
    intersections: Mapping[str, tuple[pd.DataFrame, IntersectionSummary]],
    *,
    output_dir: Path,
    config: PTMResearchConfig,
    score_table_path: Path,
    deg_table_path: Path,
) -> dict[str, Any]:
    """Write per-cell-type intersection TSVs, summary and target-set manifests (方案 §4.6)."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_payload: dict[str, Any] = {
        "schema_version": INTERSECTION_SCHEMA_VERSION,
        "sources": {
            "gene_scores": str(score_table_path),
            "deg_table": str(deg_table_path),
        },
        "thresholds": {
            "deg_max_fdr": config.deg_max_fdr,
            "min_donors_per_state": config.min_donors_per_state,
        },
        "cell_types": {},
        "note": (
            "n_cell_types_supported counts cell-type repetition and is NOT independent donor "
            "evidence (方案 §5.4 步骤 7); no joint significance is claimed without a calibrated PTM-side null"
        ),
    }
    target_sets: dict[str, Any] = {
        "schema_version": TARGET_SET_SCHEMA_VERSION,
        "cell_types": {},
    }
    gene_support: dict[tuple[str, str], list[str]] = {}
    for cell_type, (frame, summary) in sorted(intersections.items()):
        tsv_path = output_dir / f"ptm_ad_intersection_{_safe_name(cell_type)}.tsv"
        frame.to_csv(tsv_path, sep="\t", index=False)
        summary_payload["cell_types"][cell_type] = {
            "n_rows": len(frame),
            "n_concordant": summary.n_concordant,
            "n_discordant": summary.n_discordant,
            "n_ptm_only": summary.n_ptm_only,
            "n_ad_only": summary.n_ad_only,
            "n_formal_concordant": summary.n_formal_concordant,
        }
        formal = frame[(frame["membership"] == "concordant") & (frame["evidence_tier"] == "formal")]
        per_source: dict[str, list[dict[str, Any]]] = {}
        for record in formal.to_dict("records"):
            gene_support.setdefault((str(record["target_ensembl_id"]), str(record["target_gene_symbol"])), []).append(
                cell_type
            )
            per_source.setdefault(str(record["source_activity_id"]), []).append(
                {
                    "target_ensembl_id": record["target_ensembl_id"],
                    "target_gene_symbol": record["target_gene_symbol"],
                    "gene_score": record["gene_score"],
                    "predicted_gene_direction": record["predicted_gene_direction"],
                    "observed_log2fc": record["observed_log2fc"],
                    "observed_fdr": record["observed_fdr"],
                    "observed_direction": record["observed_direction"],
                    "n_paths": record["gene_score_n_paths"],
                    "network_coverage": record["gene_score_network_coverage"],
                }
            )
        target_sets["cell_types"][cell_type] = {
            "intersection_table": str(tsv_path),
            "sources": {source_id: {"targets": targets} for source_id, targets in sorted(per_source.items())},
        }
    summary_payload["n_cell_types_supported"] = {
        f"{ensembl}|{symbol}": {"n_cell_types": len(cell_list), "cell_types": sorted(set(cell_list))}
        for (ensembl, symbol), cell_list in sorted(gene_support.items())
    }
    summary_path = output_dir / "intersection_summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    target_set_path = output_dir / "target_set_manifest.json"
    target_set_path.write_text(json.dumps(target_sets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"summary": summary_path, "target_set_manifest": target_set_path}


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


__all__ = [
    "DEG_REQUIRED_COLUMNS",
    "GENE_SCORE_REQUIRED_COLUMNS",
    "GENE_SCORE_SCHEMA_VERSION",
    "GeneScoreBuildAudit",
    "INTERSECTION_SCHEMA_VERSION",
    "IntersectionSummary",
    "PTMGeneScoreError",
    "TARGET_SET_SCHEMA_VERSION",
    "build_gene_score_table",
    "intersect_gene_scores_with_deg",
    "load_deg_table",
    "load_gene_score_table",
    "load_network_id_map",
    "write_gene_score_table",
    "write_intersection_outputs",
]
