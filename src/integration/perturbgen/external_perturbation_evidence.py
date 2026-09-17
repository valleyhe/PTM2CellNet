"""External perturbation-model evidence contract (2026-09-17).

The mainline may consume direction evidence from an external model as a
*supplementary, field-separated* source. Current strategy B accepts GEARS
GO-graph extrapolation and Geneformer network counterfactual assets; the older
LPM/STATE path remains a historical, disabled producer. All producers share
the same manifest and payload contract, and an asset must pass the independent
APOE anchor review before any future lineage consumer may use it.

Boundaries frozen by this module (方案 §5.5 lineage rules apply):

- The asset records predictions in a *training* context (e.g. K562). It is
  baseline-context extrapolation, never a disease-context intervention.
- ``crispri_kd`` approximates, but is not, the mainline ``token_mask_ko``
  intervention; the mapping is recorded explicitly per prediction and never
  merged with DAVF/PerturbGen evidence fields.
- The evidence is model output, not causal validation and not biology PASS.
- Missing self-readout rows or missing candidates are explicit; nothing is
  zero-filled or silently dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.models.gene_vocabulary import normalize_ensembl_id

EXTERNAL_PERTURBATION_EVIDENCE_SCHEMA_VERSION = "ptm2cellnet.external-perturbation-evidence/v1"
EXTERNAL_PREDICTION_SCHEMA_VERSION = "ptm2cellnet.external-perturbation-prediction/v1"

#: Perturbation semantics an external asset may declare. ``token_mask_ko`` is
#: reserved for mainline PerturbGen runs and rejected here so the KO
#: approximation stays explicit. The trailing entries cover the 2026-09-17
#: source pivot: GEARS predicts unseen genes through the GO graph (training
#: modality mixed -> extrapolation is declared as such) and Geneformer reports
#: in-silico network counterfactuals rather than trained perturbation
#: responses.
VALID_EXTERNAL_PERTURBATION_SEMANTICS = (
    "crispri_kd",
    "shrna_kd",
    "crispra_oe",
    "unseen_perturbation_extrapolation",
    "in_silico_perturbation",
)

#: Evidence kinds driving the boundary wording; every source is supplementary
#: and field-separated, never a pass/fail decision.
VALID_EVIDENCE_KINDS = ("trained_response", "go_extrapolation", "network_counterfactual")

_REQUIRED_OBS_COLUMNS = ("ensembl_id", "gene_symbol", "context", "perturbation_semantics")

_BOUNDARY_NOTE = (
    "external perturbation-model evidence: baseline training-context extrapolation "
    "(not a disease-context intervention); crispri_kd approximates but does not equal "
    "the mainline token_mask_ko intervention; supplementary direction evidence only, "
    "field-separated from DAVF/PerturbGen sources and never a pass/fail decision, "
    "causal validation or biology PASS (方案 §5.5/§8.6)"
)


class ExternalPerturbationEvidenceError(ValueError):
    """Raised when an external prediction asset violates the contract."""


@dataclass(frozen=True)
class ExternalCandidatePrediction:
    """One candidate's self-readout direction from the external model."""

    ensembl_id: str
    gene_symbol: str
    context: str
    perturbation_semantics: str
    self_delta: float | None
    predicted_direction: str | None  # 'up' | 'down' | None when self row absent


@dataclass(frozen=True)
class ExternalPerturbationEvidence:
    """Loaded, validated external prediction asset."""

    model_source: str
    evidence_kind: str
    perturbation_semantics: str
    context: str
    seeds: tuple[int, ...]
    aggregation: str
    manifest_path: str
    predictions_h5ad: str
    predictions_sha256: str
    perturblib_commit: str
    training_config: str
    license: str
    predictions: tuple[ExternalCandidatePrediction, ...]
    n_readout_genes: int


def _load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    resolved = Path(manifest_path).expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExternalPerturbationEvidenceError(f"external prediction manifest root must be an object: {resolved}")
    if payload.get("schema_version") != EXTERNAL_PREDICTION_SCHEMA_VERSION:
        raise ExternalPerturbationEvidenceError(
            f"external prediction manifest schema_version must be {EXTERNAL_PREDICTION_SCHEMA_VERSION!r}; "
            f"got {payload.get('schema_version')!r}"
        )
    payload["_manifest_dir"] = str(resolved.parent)
    payload["_manifest_path"] = str(resolved)
    return payload


def _resolve_predictions_path(payload: Mapping[str, Any]) -> Path:
    raw = str(payload.get("predictions_h5ad") or "").strip()
    if not raw:
        raise ExternalPerturbationEvidenceError("external prediction manifest requires predictions_h5ad")
    path = Path(raw)
    if not path.is_absolute():
        path = Path(payload["_manifest_dir"]) / path
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise ExternalPerturbationEvidenceError(f"external predictions h5ad is not readable: {path}") from exc


def _verify_sha256(path: Path, expected: str) -> None:
    import hashlib

    if not expected:
        raise ExternalPerturbationEvidenceError("external prediction manifest requires predictions_sha256")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ExternalPerturbationEvidenceError(
            f"external predictions h5ad hash drifted from the manifest: {path} is {digest}, manifest records {expected}"
        )


def load_external_prediction_asset(manifest_path: str | Path) -> ExternalPerturbationEvidence:
    """Load and validate a frozen external prediction asset (manifest + h5ad)."""

    import anndata as ad

    payload = _load_manifest(manifest_path)
    predictions_path = _resolve_predictions_path(payload)
    _verify_sha256(predictions_path, str(payload.get("predictions_sha256") or ""))

    semantics = str(payload.get("perturbation_semantics") or "").strip()
    if semantics not in VALID_EXTERNAL_PERTURBATION_SEMANTICS:
        raise ExternalPerturbationEvidenceError(
            "perturbation_semantics must be one of "
            f"{', '.join(VALID_EXTERNAL_PERTURBATION_SEMANTICS)}; got {semantics!r}"
        )
    raw_seeds = payload.get("seeds")
    if not isinstance(raw_seeds, (list, tuple)) or not raw_seeds:
        raise ExternalPerturbationEvidenceError("external prediction manifest requires a non-empty seeds list")
    seeds = tuple(int(seed) for seed in raw_seeds)
    context = str(payload.get("context") or "").strip()
    if not context:
        raise ExternalPerturbationEvidenceError("external prediction manifest requires a non-empty context")
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ExternalPerturbationEvidenceError("external prediction manifest requires a model mapping")
    model_source = str(model.get("source") or "").strip()
    if not model_source:
        raise ExternalPerturbationEvidenceError("external prediction manifest requires model.source")
    evidence_kind = str(payload.get("evidence_kind") or "").strip()
    if evidence_kind not in VALID_EVIDENCE_KINDS:
        raise ExternalPerturbationEvidenceError(
            f"evidence_kind must be one of {', '.join(VALID_EVIDENCE_KINDS)}; got {evidence_kind!r}"
        )

    adata = ad.read_h5ad(predictions_path, backed="r")
    missing_columns = [column for column in _REQUIRED_OBS_COLUMNS if column not in adata.obs.columns]
    if missing_columns:
        raise ExternalPerturbationEvidenceError(f"external predictions obs is missing columns: {missing_columns}")
    asset_semantics = adata.obs["perturbation_semantics"].astype(str).str.strip().unique().tolist()
    if asset_semantics != [semantics]:
        raise ExternalPerturbationEvidenceError(
            f"external predictions perturbation_semantics {asset_semantics} does not match manifest {semantics!r}"
        )
    asset_contexts = adata.obs["context"].astype(str).str.strip().unique().tolist()
    if asset_contexts != [context]:
        raise ExternalPerturbationEvidenceError(
            f"external predictions contexts {asset_contexts} do not match manifest context {context!r}"
        )

    canonical_rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for ensembl_id, gene_symbol in zip(
        adata.obs["ensembl_id"].astype(str), adata.obs["gene_symbol"].astype(str), strict=True
    ):
        canonical = normalize_ensembl_id(ensembl_id.strip())
        if canonical in seen:
            raise ExternalPerturbationEvidenceError(f"external predictions contain duplicate candidate {canonical}")
        seen.add(canonical)
        canonical_rows.append((canonical, gene_symbol.strip()))

    readout_ids = [str(value).strip() for value in adata.var_names]
    for readout_id in readout_ids:
        try:
            normalize_ensembl_id(readout_id)
        except ValueError as exc:
            raise ExternalPerturbationEvidenceError(
                f"external predictions readout vocabulary must be canonical Ensembl ids; got {readout_id!r}"
            ) from exc
    readout_index = {value: position for position, value in enumerate(readout_ids)}

    delta_matrix = np.asarray(adata.X[:, :])
    if delta_matrix.ndim != 2 or delta_matrix.shape[0] != len(canonical_rows):
        raise ExternalPerturbationEvidenceError("external predictions X must be a candidates × readouts delta matrix")
    if not np.isfinite(delta_matrix).all():
        raise ExternalPerturbationEvidenceError("external predictions delta matrix contains non-finite values")

    predictions: list[ExternalCandidatePrediction] = []
    for row_index, (canonical, gene_symbol) in enumerate(canonical_rows):
        if canonical in readout_index:
            self_delta = float(delta_matrix[row_index, readout_index[canonical]])
            if self_delta > 0:
                direction = "up"
            elif self_delta < 0:
                direction = "down"
            else:
                direction = "indeterminate"
        else:
            self_delta = None
            direction = None
        predictions.append(
            ExternalCandidatePrediction(
                ensembl_id=canonical,
                gene_symbol=gene_symbol,
                context=context,
                perturbation_semantics=semantics,
                self_delta=self_delta,
                predicted_direction=direction,
            )
        )

    return ExternalPerturbationEvidence(
        model_source=model_source,
        evidence_kind=evidence_kind,
        perturbation_semantics=semantics,
        context=context,
        seeds=seeds,
        aggregation=str(payload.get("aggregation") or ""),
        manifest_path=str(payload["_manifest_path"]),
        predictions_h5ad=str(predictions_path),
        predictions_sha256=str(payload.get("predictions_sha256") or ""),
        perturblib_commit=str(model.get("perturblib_commit") or ""),
        training_config=str(model.get("training_config") or ""),
        license=str(model.get("license") or ""),
        predictions=tuple(predictions),
        n_readout_genes=len(readout_ids),
    )


def evidence_missing_candidates(
    evidence: ExternalPerturbationEvidence,
    requested: Mapping[str, str],
) -> tuple[str, ...]:
    """Return canonical ids from ``requested`` that the asset does not cover."""

    covered = {prediction.ensembl_id for prediction in evidence.predictions}
    missing = []
    for ensembl_id in requested:
        canonical = normalize_ensembl_id(ensembl_id)
        if canonical not in covered:
            missing.append(canonical)
    return tuple(sorted(set(missing)))


def _boundary_for_kind(evidence_kind: str) -> str:
    if evidence_kind == "go_extrapolation":
        kind_note = (
            "GO-graph extrapolation to perturbations never observed in training (training modality mixed); "
            "baseline training-context extrapolation"
        )
    elif evidence_kind == "network_counterfactual":
        kind_note = (
            "in-silico network counterfactual from a masked-gene foundation model, not a trained "
            "perturbation response; deltas are model-internal shifts"
        )
    else:
        kind_note = (
            "trained perturbation response read out in a baseline training context (not a disease-context intervention)"
        )
    return (
        f"external perturbation-model evidence: {kind_note}; perturbation semantics approximate but "
        "never equal the mainline token_mask_ko intervention; supplementary direction evidence only, "
        "field-separated from DAVF/PerturbGen sources and never a pass/fail decision, causal "
        "validation or biology PASS (方案 §5.5/§8.6)"
    )


def external_evidence_to_payload(evidence: ExternalPerturbationEvidence) -> dict[str, Any]:
    """Serialize the evidence with provenance and the boundary note explicit."""

    return {
        "schema_version": EXTERNAL_PERTURBATION_EVIDENCE_SCHEMA_VERSION,
        "source": {
            "model_source": evidence.model_source,
            "perturblib_commit": evidence.perturblib_commit,
            "training_config": evidence.training_config,
            "license": evidence.license,
            "manifest_path": evidence.manifest_path,
            "predictions_h5ad": evidence.predictions_h5ad,
            "predictions_sha256": evidence.predictions_sha256,
        },
        "design": {
            "context": evidence.context,
            "evidence_kind": evidence.evidence_kind,
            "perturbation_semantics": evidence.perturbation_semantics,
            "seeds": list(evidence.seeds),
            "aggregation": evidence.aggregation,
            "n_readout_genes": evidence.n_readout_genes,
            "n_candidates": len(evidence.predictions),
        },
        "candidates": {
            prediction.ensembl_id: {
                "gene_symbol": prediction.gene_symbol,
                "context": prediction.context,
                "perturbation_semantics": prediction.perturbation_semantics,
                "self_delta": prediction.self_delta,
                "predicted_direction": prediction.predicted_direction,
            }
            for prediction in sorted(evidence.predictions, key=lambda item: item.ensembl_id)
        },
        "boundary": _boundary_for_kind(evidence.evidence_kind),
    }


__all__ = [
    "EXTERNAL_PERTURBATION_EVIDENCE_SCHEMA_VERSION",
    "EXTERNAL_PREDICTION_SCHEMA_VERSION",
    "ExternalCandidatePrediction",
    "ExternalPerturbationEvidence",
    "ExternalPerturbationEvidenceError",
    "VALID_EVIDENCE_KINDS",
    "VALID_EXTERNAL_PERTURBATION_SEMANTICS",
    "evidence_missing_candidates",
    "external_evidence_to_payload",
    "load_external_prediction_asset",
]
