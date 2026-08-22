"""PerturbGen AnnData preflight and candidate screening."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from src.integration.perturbgen.contracts import (
    CandidateEvidence,
    CandidateScreeningResult,
    PerturbGenDataSpec,
    PreparedPerturbationData,
    PreparedPerturbationReport,
)
from src.models.gene_vocabulary import GeneVocabularyResolver, normalize_ensembl_id


def _matrix_values(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "data"):
        values = np.asarray(matrix.data, dtype=float)
        if values.size:
            return values
    dense = np.asarray(matrix, dtype=float)
    return dense.ravel()


def _row_sums(matrix: Any) -> np.ndarray:
    return np.asarray(matrix.sum(axis=1), dtype=float).ravel()


def _col_as_strings(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype(str).str.strip()


def _validate_counts_layer(adata: Any, spec: PerturbGenDataSpec) -> Any:
    if spec.counts_layer not in adata.layers:
        raise ValueError(
            f"adata.layers[{spec.counts_layer!r}] is required; "
            "do not silently treat adata.X as raw counts"
        )
    counts = adata.layers[spec.counts_layer]
    if getattr(counts, "shape", None) != adata.shape:
        raise ValueError(
            f"counts layer shape {getattr(counts, 'shape', None)!r} does not match adata shape {adata.shape!r}"
        )
    values = _matrix_values(counts)
    if values.size == 0:
        raise ValueError("counts layer must not be empty")
    if not np.isfinite(values).all():
        raise ValueError("counts layer contains non-finite values")
    if (values < 0).any():
        raise ValueError("counts layer must contain non-negative raw counts")
    if not np.allclose(values, np.round(values), atol=1e-6):
        raise ValueError(
            "counts layer must contain integer-like raw counts; "
            "log-normalized values are not allowed"
        )
    return counts


def _validate_obs_contract(
    adata: Any,
    *,
    cell_type: str,
    spec: PerturbGenDataSpec,
) -> tuple[np.ndarray, pd.Series, pd.Series, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if not getattr(adata.obs_names, "is_unique", False):
        raise ValueError("adata.obs_names must be unique")

    required_columns = (spec.cell_type_col, spec.state_col, spec.donor_col)
    missing = [column for column in required_columns if column not in adata.obs.columns]
    if missing:
        raise ValueError(f"adata.obs is missing required columns: {missing}")

    obs = adata.obs.copy()
    obs[spec.cell_type_col] = _col_as_strings(obs, spec.cell_type_col)
    obs[spec.state_col] = _col_as_strings(obs, spec.state_col)
    obs[spec.donor_col] = _col_as_strings(obs, spec.donor_col)

    empty_mask = (
        obs[spec.cell_type_col].eq("")
        | obs[spec.state_col].eq("")
        | obs[spec.donor_col].eq("")
    )
    if bool(empty_mask.any()):
        raise ValueError("adata.obs contains empty cell_type/state/donor values")

    target_mask = obs[spec.cell_type_col].eq(cell_type)
    if int(target_mask.sum()) == 0:
        raise ValueError(f"cell_type {cell_type!r} is not present in adata.obs[{spec.cell_type_col!r}]")

    states = obs.loc[target_mask, spec.state_col]
    unexpected_states = sorted(set(states) - {spec.normal_state, spec.disease_state})
    if unexpected_states:
        raise ValueError(
            f"target cell_type contains unexpected states for PerturbGen pairing: {unexpected_states}"
        )

    normal_mask = target_mask & obs[spec.state_col].eq(spec.normal_state)
    disease_mask = target_mask & obs[spec.state_col].eq(spec.disease_state)
    if int(normal_mask.sum()) == 0 or int(disease_mask.sum()) == 0:
        raise ValueError("target cell_type must contain both normal and disease cells")

    normal_donors = tuple(sorted(set(obs.loc[normal_mask, spec.donor_col])))
    disease_donors = tuple(sorted(set(obs.loc[disease_mask, spec.donor_col])))
    shared_donors = tuple(sorted(set(normal_donors) & set(disease_donors)))
    normal_only_donors = tuple(sorted(set(normal_donors) - set(shared_donors)))
    disease_only_donors = tuple(sorted(set(disease_donors) - set(shared_donors)))
    if len(shared_donors) < spec.min_donors:
        raise ValueError(
            f"target cell_type {cell_type!r} requires at least {spec.min_donors} shared donors "
            f"across {spec.normal_state!r}/{spec.disease_state!r}; got {len(shared_donors)}"
        )

    return (
        np.asarray(target_mask, dtype=bool),
        obs[spec.state_col],
        obs[spec.donor_col],
        shared_donors,
        normal_only_donors,
        disease_only_donors,
    )


def _prepare_var_contract(adata: Any, spec: PerturbGenDataSpec) -> None:
    if spec.ensembl_id_col not in adata.var.columns:
        raise ValueError(f"adata.var is missing required column {spec.ensembl_id_col!r}")
    normalized_ids = [normalize_ensembl_id(value) for value in adata.var[spec.ensembl_id_col]]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("adata.var contains duplicate ensembl ids after version stripping")
    adata.var = adata.var.copy()
    adata.var[spec.ensembl_id_col] = normalized_ids


def prepare_perturbgen_anndata(
    adata: Any,
    *,
    cell_type: str,
    spec: PerturbGenDataSpec | None = None,
) -> PreparedPerturbationData:
    """Validate raw counts/ENSG/obs contract for one target cell type.

    The function is intentionally strict:

    - ``adata.layers["counts"]`` must exist;
    - raw counts must be non-negative and integer-like;
    - ``obs`` must provide non-empty ``cell_type/state/donor``;
    - the target cell type must have at least 3 shared donors across
      normal/disease states;
    - ENSG ids must be canonical and unique after version stripping.
    """

    spec = spec or PerturbGenDataSpec()
    validated = adata.copy()
    counts = _validate_counts_layer(validated, spec)
    _prepare_var_contract(validated, spec)
    (
        target_mask,
        state_values,
        _donor_values,
        shared_donors,
        normal_only_donors,
        disease_only_donors,
    ) = _validate_obs_contract(validated, cell_type=cell_type, spec=spec)

    n_counts = _row_sums(counts)
    if (n_counts <= 0).any():
        raise ValueError("every cell must have positive raw library size")
    existing_n_counts = validated.obs.get(spec.n_counts_col)
    if existing_n_counts is not None:
        existing = np.asarray(existing_n_counts, dtype=float).ravel()
        if existing.shape != n_counts.shape or not np.allclose(existing, n_counts, atol=1e-6):
            raise ValueError(
                f"adata.obs[{spec.n_counts_col!r}] does not match raw counts row sums"
            )
    else:
        validated.obs = validated.obs.copy()
        validated.obs[spec.n_counts_col] = n_counts

    target_state_values = pd.Series(state_values[target_mask], copy=False)
    if not set(target_state_values.unique()) == {spec.normal_state, spec.disease_state}:
        raise ValueError("target cell_type must contain exactly normal and disease states")

    report = PreparedPerturbationReport(
        cell_type=cell_type,
        normal_state=spec.normal_state,
        disease_state=spec.disease_state,
        evaluable_donors=shared_donors,
        normal_only_donors=normal_only_donors,
        disease_only_donors=disease_only_donors,
        n_cells=int(target_mask.sum()),
        n_genes=int(validated.n_vars),
    )
    return PreparedPerturbationData(adata=validated, spec=spec, report=report)


def _screening_mode_from_direction(direction: str) -> str:
    return "mask" if direction == "up" else "overexpress"


def screen_candidate_for_perturbation(
    prepared: PreparedPerturbationData,
    candidate: CandidateEvidence,
    resolver: GeneVocabularyResolver,
    *,
    max_observed_fdr: float = 0.05,
) -> CandidateScreeningResult:
    """Screen one candidate against validated AnnData and vocabulary.

    ``recommended_mode`` is derived from the observed disease-vs-normal
    direction only. ``candidate.davf_action`` is retained as separate evidence
    and is never converted into ``mask``/``overexpress`` here.
    """

    if not 0.0 <= max_observed_fdr <= 1.0:
        raise ValueError("max_observed_fdr must be within [0, 1]")
    if candidate.observed_fdr > max_observed_fdr:
        return CandidateScreeningResult(
            candidate=candidate,
            status="inconclusive",
            reason_codes=("observed_expression_not_significant",),
        )
    if candidate.cell_type != prepared.report.cell_type:
        return CandidateScreeningResult(
            candidate=candidate,
            status="failed",
            reason_codes=("candidate_cell_type_mismatch",),
        )

    try:
        resolved_from_symbol = resolver.resolve_symbol(candidate.gene_symbol)
        resolved_symbol = resolver.resolve_ensembl(candidate.ensembl_id)
    except KeyError as exc:
        return CandidateScreeningResult(
            candidate=candidate,
            status="inconclusive",
            reason_codes=(str(exc),),
        )

    if resolved_from_symbol != candidate.ensembl_id or resolved_symbol != candidate.gene_symbol:
        return CandidateScreeningResult(
            candidate=candidate,
            status="failed",
            reason_codes=("candidate_symbol_ensembl_mismatch",),
        )

    adata = prepared.adata
    spec = prepared.spec
    var_ensembl = pd.Series(adata.var[spec.ensembl_id_col], copy=False)
    gene_hits = np.flatnonzero(var_ensembl.to_numpy() == candidate.ensembl_id)
    if gene_hits.size == 0:
        return CandidateScreeningResult(
            candidate=candidate,
            status="inconclusive",
            reason_codes=("gene_not_in_expression_matrix",),
        )

    gene_index = int(gene_hits[0])
    counts = adata.layers[spec.counts_layer]
    obs = adata.obs
    cell_type_mask = obs[spec.cell_type_col].astype(str).str.strip().eq(candidate.cell_type)
    normal_mask = cell_type_mask & obs[spec.state_col].astype(str).str.strip().eq(spec.normal_state)
    disease_mask = cell_type_mask & obs[spec.state_col].astype(str).str.strip().eq(spec.disease_state)

    gene_vector = np.asarray(counts[:, gene_index].toarray() if hasattr(counts[:, gene_index], "toarray") else counts[:, gene_index]).ravel()
    normal_counts = gene_vector[np.asarray(normal_mask, dtype=bool)]
    disease_counts = gene_vector[np.asarray(disease_mask, dtype=bool)]

    normal_detected = int((normal_counts > 0).sum())
    disease_detected = int((disease_counts > 0).sum())
    reasons: list[str] = []

    if normal_detected == 0 and disease_detected == 0:
        reasons.append("gene_not_detected_in_target_cell_type")
    elif candidate.observed_direction == "up":
        if disease_detected == 0:
            reasons.append("gene_not_detected_in_disease_state")
        # Source KO requires the target token in normal/source cells.  A
        # disease-induced gene is inconclusive for the strict dual-path AND,
        # not a biological failure.
        if normal_detected == 0:
            reasons.append("source_gene_not_detected_in_normal_state")
    # Overexpression does not require the target token to be present.  Do not
    # apply KO-style expression filtering to observed-down/OE candidates.

    status = "evaluable" if not reasons else "inconclusive"
    return CandidateScreeningResult(
        candidate=replace(candidate),
        status=status,
        reason_codes=tuple(reasons),
        recommended_mode=_screening_mode_from_direction(candidate.observed_direction) if not reasons else None,
        evaluable_donors=prepared.report.evaluable_donors,
        matched_gene_symbol=resolved_symbol if not reasons else None,
        matched_ensembl_id=candidate.ensembl_id if not reasons else None,
        normal_detected_cells=normal_detected,
        disease_detected_cells=disease_detected,
        normal_mean_counts=float(normal_counts.mean()) if normal_counts.size else 0.0,
        disease_mean_counts=float(disease_counts.mean()) if disease_counts.size else 0.0,
    )
