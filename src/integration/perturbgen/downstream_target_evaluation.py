"""Downstream target-set sidecar contract and delta evaluation (方案 §5.5/§6.2).

The PTM mainline keeps *source intervention* and *downstream target
evaluation* roles separate (方案 §3.3): the candidate spec carries the source
gene into the existing DAVF/PerturbGen path unchanged, and this module
carries the one-to-many source → target relation in a sidecar artifact plus
the per-target evaluation of downstream expression deltas.

The delta matrix is produced upstream (DAVF decode delta or PerturbGen
predicted perturbation difference); this module never computes deltas
itself, it only extracts target rows by canonical Ensembl and reports
per-target directions and set-level concordance. Missing targets are
reported explicitly, never zero-filled.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.models.gene_vocabulary import normalize_ensembl_id

DOWNSTREAM_TARGET_SIDECAR_SCHEMA_VERSION = "ptm2cellnet.downstream-target-sidecar/v1"
TARGET_SET_EVALUATION_SCHEMA_VERSION = "ptm2cellnet.target-set-evaluation/v1"


class DownstreamTargetEvaluationError(ValueError):
    """Raised when a sidecar or delta matrix violates the contract."""


@dataclass(frozen=True)
class TargetSetSource:
    """One source intervention and its frozen downstream target set."""

    source_activity_id: str
    gene_symbol: str
    ensembl_id: str
    position: int
    ptm_type: str
    proposed_direction: str
    site_probability: float
    provenance: str
    targets: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class DownstreamTargetSidecar:
    cell_type: str
    sources: tuple[TargetSetSource, ...]
    lineage: Mapping[str, Any]

    def source_by_ensembl(self, ensembl_id: str) -> TargetSetSource:
        canonical = normalize_ensembl_id(ensembl_id)
        for source in self.sources:
            if source.ensembl_id == canonical:
                return source
        raise DownstreamTargetEvaluationError(f"sidecar for cell type {self.cell_type!r} has no source {canonical}")


def load_downstream_target_sidecar(path: str | Path) -> DownstreamTargetSidecar:
    """Load and validate a ``downstream_targets_<cell_type>.json`` sidecar."""

    resolved = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DownstreamTargetEvaluationError("sidecar root must be a JSON object")
    if payload.get("schema_version") != DOWNSTREAM_TARGET_SIDECAR_SCHEMA_VERSION:
        raise DownstreamTargetEvaluationError(
            f"sidecar schema_version must be {DOWNSTREAM_TARGET_SIDECAR_SCHEMA_VERSION!r}"
        )
    cell_type = str(payload.get("cell_type") or "").strip()
    if not cell_type:
        raise DownstreamTargetEvaluationError("sidecar requires a non-empty cell_type")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, dict) or not raw_sources:
        raise DownstreamTargetEvaluationError("sidecar requires a non-empty sources mapping")
    sources: list[TargetSetSource] = []
    seen_sources: set[str] = set()
    for ensembl_id, entry in sorted(raw_sources.items()):
        if not isinstance(entry, dict):
            raise DownstreamTargetEvaluationError(f"sidecar source {ensembl_id} must be a JSON object")
        canonical = normalize_ensembl_id(ensembl_id)
        if canonical in seen_sources:
            raise DownstreamTargetEvaluationError(f"sidecar contains duplicate canonical source {canonical}")
        seen_sources.add(canonical)
        targets = entry.get("targets")
        if not isinstance(targets, list) or not targets:
            raise DownstreamTargetEvaluationError(
                f"sidecar source {canonical} must carry a non-empty frozen target list (方案 §5.5)"
            )
        target_ids: set[str] = set()
        for index, target in enumerate(targets):
            if not isinstance(target, dict) or not str(target.get("target_ensembl_id") or "").strip():
                raise DownstreamTargetEvaluationError(
                    f"sidecar source {canonical} target {index} requires target_ensembl_id"
                )
            target_id = normalize_ensembl_id(str(target["target_ensembl_id"]))
            if target_id in target_ids:
                raise DownstreamTargetEvaluationError(
                    f"sidecar source {canonical} contains duplicate target {target_id}"
                )
            target_ids.add(target_id)
        required = (
            "source_activity_id",
            "gene_symbol",
            "position",
            "ptm_type",
            "proposed_direction",
            "site_probability",
            "provenance",
        )
        missing = [key for key in required if key not in entry]
        if missing:
            raise DownstreamTargetEvaluationError(f"sidecar source {canonical} is missing fields: {', '.join(missing)}")
        string_fields = ("source_activity_id", "gene_symbol", "ptm_type", "provenance")
        empty_fields = [key for key in string_fields if not str(entry[key]).strip()]
        if empty_fields:
            raise DownstreamTargetEvaluationError(
                f"sidecar source {canonical} fields must not be empty: {', '.join(empty_fields)}"
            )
        if isinstance(entry["position"], bool) or not isinstance(entry["position"], int):
            raise DownstreamTargetEvaluationError(f"sidecar source {canonical} position must be an integer")
        position = entry["position"]
        if position < 1:
            raise DownstreamTargetEvaluationError(f"sidecar source {canonical} position must be >= 1")
        if isinstance(entry["site_probability"], bool):
            raise DownstreamTargetEvaluationError(f"sidecar source {canonical} site_probability must be numeric")
        probability = float(entry["site_probability"])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise DownstreamTargetEvaluationError(f"sidecar source {canonical} site_probability must be within [0, 1]")
        direction = str(entry["proposed_direction"]).strip().lower()
        if direction not in ("up", "down"):
            raise DownstreamTargetEvaluationError(
                f"sidecar source {canonical} proposed_direction must be 'up' or 'down'"
            )
        sources.append(
            TargetSetSource(
                source_activity_id=str(entry["source_activity_id"]).strip(),
                gene_symbol=str(entry["gene_symbol"]).strip(),
                ensembl_id=canonical,
                position=position,
                ptm_type=str(entry["ptm_type"]).strip(),
                proposed_direction=direction,
                site_probability=probability,
                provenance=str(entry["provenance"]).strip(),
                targets=tuple(targets),
            )
        )
    lineage = payload.get("lineage")
    if not isinstance(lineage, dict) or not lineage:
        raise DownstreamTargetEvaluationError("sidecar requires a lineage mapping")
    return DownstreamTargetSidecar(cell_type=cell_type, sources=tuple(sources), lineage=lineage)


@dataclass(frozen=True)
class TargetDeltaEvaluation:
    target_ensembl_id: str
    target_gene_symbol: str
    mean_delta: float
    delta_direction: str  # 'up' | 'down' | 'indeterminate'
    predicted_gene_direction: str
    observed_direction: str
    matches_predicted: bool
    matches_observed: bool


@dataclass(frozen=True)
class SourceTargetSetEvaluation:
    source_ensembl_id: str
    source_gene_symbol: str
    n_targets: int
    n_missing_targets: int
    missing_targets: tuple[str, ...]
    deltas: tuple[TargetDeltaEvaluation, ...]
    n_matching_predicted: int
    n_matching_observed: int
    n_indeterminate: int


def evaluate_target_set_deltas(
    delta_frame: pd.DataFrame,
    sidecar: DownstreamTargetSidecar,
    *,
    gene_id_column: str | None = None,
) -> dict[str, SourceTargetSetEvaluation]:
    """Evaluate per-source target-set deltas from a gene × unit delta matrix.

    ``delta_frame`` rows are genes (indexed or labelled by ``gene_id_column``
    with canonical Ensembl ids) and columns are evaluation units (e.g. donors,
    seeds, modes); values are signed expression deltas produced upstream.
    Per target the row mean determines the delta direction; concordance with
    the frozen ``predicted_gene_direction`` and ``observed_direction`` is
    reported per field, never merged (方案 §3.1).
    """

    if delta_frame.empty:
        raise DownstreamTargetEvaluationError("delta matrix must not be empty")
    if gene_id_column is not None:
        if gene_id_column not in delta_frame.columns:
            raise DownstreamTargetEvaluationError(f"gene_id_column {gene_id_column!r} not present in the delta matrix")
        raw_ids: list[object] = delta_frame[gene_id_column].tolist()
        value_frame = delta_frame.drop(columns=[gene_id_column])
    else:
        raw_ids = list(delta_frame.index)
        value_frame = delta_frame
    if value_frame.shape[1] == 0:
        raise DownstreamTargetEvaluationError("delta matrix must contain at least one evaluation-unit column")
    index = pd.Index([_normalize_or_none(value) for value in raw_ids])
    try:
        values = value_frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise DownstreamTargetEvaluationError("delta matrix evaluation-unit values must be numeric") from exc
    normalized = pd.DataFrame(values, index=index, columns=value_frame.columns)
    duplicated = normalized.index[normalized.index.duplicated()].tolist()
    if duplicated:
        raise DownstreamTargetEvaluationError(
            f"delta matrix contains duplicated canonical gene ids: {sorted(set(duplicated))}"
        )
    evaluations: dict[str, SourceTargetSetEvaluation] = {}
    for source in sidecar.sources:
        per_target: list[TargetDeltaEvaluation] = []
        missing: list[str] = []
        n_match_predicted = 0
        n_match_observed = 0
        n_indeterminate = 0
        for target in source.targets:
            ensembl_id = normalize_ensembl_id(str(target["target_ensembl_id"]))
            if ensembl_id not in normalized.index:
                missing.append(ensembl_id)
                continue
            values = normalized.loc[ensembl_id].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise DownstreamTargetEvaluationError(f"delta matrix for {ensembl_id} contains non-finite values")
            mean_delta = float(np.mean(values))
            if mean_delta > 0:
                delta_direction = "up"
            elif mean_delta < 0:
                delta_direction = "down"
            else:
                delta_direction = "indeterminate"
            predicted = str(target.get("predicted_gene_direction") or "").strip()
            observed = str(target.get("observed_direction") or "").strip()
            matches_predicted = bool(predicted) and delta_direction == predicted
            matches_observed = bool(observed) and delta_direction == observed
            if matches_predicted:
                n_match_predicted += 1
            if matches_observed:
                n_match_observed += 1
            if delta_direction == "indeterminate":
                n_indeterminate += 1
            per_target.append(
                TargetDeltaEvaluation(
                    target_ensembl_id=ensembl_id,
                    target_gene_symbol=str(target.get("target_gene_symbol") or ""),
                    mean_delta=mean_delta,
                    delta_direction=delta_direction,
                    predicted_gene_direction=predicted,
                    observed_direction=observed,
                    matches_predicted=matches_predicted,
                    matches_observed=matches_observed,
                )
            )
        evaluations[source.ensembl_id] = SourceTargetSetEvaluation(
            source_ensembl_id=source.ensembl_id,
            source_gene_symbol=source.gene_symbol,
            n_targets=len(source.targets),
            n_missing_targets=len(missing),
            missing_targets=tuple(sorted(missing)),
            deltas=tuple(per_target),
            n_matching_predicted=n_match_predicted,
            n_matching_observed=n_match_observed,
            n_indeterminate=n_indeterminate,
        )
    return evaluations


def evaluation_to_payload(
    evaluations: Mapping[str, SourceTargetSetEvaluation],
    *,
    cell_type: str,
    delta_matrix_source: str,
) -> dict[str, Any]:
    """Serialize evaluations with explicit lineage and no verdict language."""

    return {
        "schema_version": TARGET_SET_EVALUATION_SCHEMA_VERSION,
        "cell_type": cell_type,
        "delta_matrix_source": delta_matrix_source,
        "sources": {
            source_id: {
                "source_gene_symbol": evaluation.source_gene_symbol,
                "n_targets": evaluation.n_targets,
                "n_missing_targets": evaluation.n_missing_targets,
                "missing_targets": list(evaluation.missing_targets),
                "n_matching_predicted": evaluation.n_matching_predicted,
                "n_matching_observed": evaluation.n_matching_observed,
                "n_indeterminate": evaluation.n_indeterminate,
                "targets": [
                    {
                        "target_ensembl_id": delta.target_ensembl_id,
                        "target_gene_symbol": delta.target_gene_symbol,
                        "mean_delta": delta.mean_delta,
                        "delta_direction": delta.delta_direction,
                        "predicted_gene_direction": delta.predicted_gene_direction,
                        "observed_direction": delta.observed_direction,
                        "matches_predicted": delta.matches_predicted,
                        "matches_observed": delta.matches_observed,
                    }
                    for delta in evaluation.deltas
                ],
            }
            for source_id, evaluation in sorted(evaluations.items())
        },
        "note": (
            "concordance counts describe direction agreement only; they are not causal "
            "validation and not biological PASS (方案 §8.6)"
        ),
    }


def _normalize_or_none(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise DownstreamTargetEvaluationError("delta matrix contains an empty gene id")
    try:
        return normalize_ensembl_id(text)
    except ValueError as exc:
        raise DownstreamTargetEvaluationError(f"delta matrix contains invalid gene id: {text!r}") from exc


__all__ = [
    "DOWNSTREAM_TARGET_SIDECAR_SCHEMA_VERSION",
    "DownstreamTargetEvaluationError",
    "DownstreamTargetSidecar",
    "SourceTargetSetEvaluation",
    "TARGET_SET_EVALUATION_SCHEMA_VERSION",
    "TargetDeltaEvaluation",
    "TargetSetSource",
    "evaluate_target_set_deltas",
    "evaluation_to_payload",
    "load_downstream_target_sidecar",
]
