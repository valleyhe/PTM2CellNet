"""Build aggregate and donor-level AD DEG tables from an AnnData cohort."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from src.data.gse_normal_disease import (
    _bh_adjust,
    _donor_log2_means,
    _validate_direction_input,
)


class ADDEGError(ValueError):
    """Raised when an AD DEG input or output contract is invalid."""


_AGGREGATE_COLUMNS = (
    "cell_type",
    "ensembl_id",
    "gene_symbol",
    "log2fc",
    "fdr",
    "observed_direction",
    "n_normal_donors",
    "n_disease_donors",
)
_DONOR_COLUMNS = ("cell_type", "donor", "ensembl_id", "gene_symbol", "log2fc", "fdr")


def _direction_adata(adata: Any, *, cell_type_column: str, state_column: str, donor_column: str) -> Any:
    if (cell_type_column, state_column, donor_column) == ("cell_type", "state", "donor"):
        return adata
    missing = [column for column in (cell_type_column, state_column, donor_column) if column not in adata.obs.columns]
    if missing:
        raise ADDEGError(f"AnnData is missing required obs columns: {missing}")
    obs = adata.obs[[cell_type_column, state_column, donor_column]].copy()
    obs.columns = ["cell_type", "state", "donor"]
    return SimpleNamespace(layers=adata.layers, shape=adata.shape, var=adata.var, obs=obs)


def _clean_obs(obs: pd.DataFrame) -> pd.DataFrame:
    cleaned: pd.DataFrame = obs.copy()
    for column in ("cell_type", "state", "donor"):
        values = cleaned[column]
        if values.isna().any():
            raise ADDEGError(f"obs column {column!r} must not contain missing values")
        values = values.astype(str).str.strip()
        if values.eq("").any():
            raise ADDEGError(f"obs column {column!r} must not contain empty values")
        cleaned[column] = values
    invalid_states = sorted(set(cleaned["state"]) - {"normal", "disease"})
    if invalid_states:
        raise ADDEGError(f"state must contain only 'normal' or 'disease'; got {invalid_states}")
    return cleaned


def _requested_cell_types(cell_types: Sequence[str]) -> tuple[str, ...]:
    if isinstance(cell_types, (str, bytes)):
        raise ADDEGError("cell_types must be a non-empty sequence of strings")
    try:
        requested = tuple(value.strip() if isinstance(value, str) else "" for value in cell_types)
    except TypeError as exc:
        raise ADDEGError("cell_types must be a non-empty sequence of strings") from exc
    if not requested or any(not value for value in requested):
        raise ADDEGError("cell_types must contain non-empty strings")
    if len(set(requested)) != len(requested):
        raise ADDEGError("cell_types must not contain duplicates")
    return requested


def build_ad_deg_tables(
    adata: Any,
    *,
    cell_types: Sequence[str],
    cohort_pairing: str,
    min_donors_per_state: int,
    counts_layer: str = "counts",
    cell_type_column: str = "cell_type",
    state_column: str = "state",
    donor_column: str = "donor",
    normal_state: str = "normal",
    disease_state: str = "disease",
    direction_epsilon: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build aggregate and disease-donor AD DEG tables."""

    try:
        if cohort_pairing not in {"within_donor", "between_donor"}:
            raise ADDEGError("cohort_pairing must be 'within_donor' or 'between_donor'")
        if cohort_pairing != "between_donor":
            raise ADDEGError(
                "AD DEG table requires between_donor pairing; within_donor is not a valid independent "
                "normal/disease DEG contract"
            )
        if (
            isinstance(min_donors_per_state, bool)
            or not isinstance(min_donors_per_state, int)
            or min_donors_per_state < 1
        ):
            raise ADDEGError("min_donors_per_state must be a positive integer")
        if normal_state not in {"normal", "disease"} or disease_state not in {"normal", "disease"}:
            raise ADDEGError("normal_state and disease_state must be 'normal' or 'disease'")
        if normal_state == disease_state:
            raise ADDEGError("normal_state and disease_state must differ")

        direction_adata = _direction_adata(
            adata,
            cell_type_column=cell_type_column,
            state_column=state_column,
            donor_column=donor_column,
        )
        counts, obs, gene_names = _validate_direction_input(direction_adata, counts_layer=counts_layer)
        if "gene_symbol" not in adata.var.columns:
            raise ADDEGError("AnnData is missing var['gene_symbol']")
        raw_symbols = adata.var["gene_symbol"]
        if raw_symbols.isna().any():
            raise ADDEGError("var['gene_symbol'] must not contain missing values")
        gene_symbols = raw_symbols.astype(str).str.strip()
        if gene_symbols.eq("").any():
            raise ADDEGError("var['gene_symbol'] must not contain empty values")

        obs = _clean_obs(obs)
        requested = _requested_cell_types(cell_types)
        available = set(obs["cell_type"])
        missing_cell_types = sorted(set(requested) - available)
        if missing_cell_types:
            raise ADDEGError(f"requested cell types are absent from AnnData: {missing_cell_types}")

        normal_donors_all = set(obs.loc[obs["state"] == normal_state, "donor"])
        disease_donors_all = set(obs.loc[obs["state"] == disease_state, "donor"])
        shared_donors_all = sorted(normal_donors_all & disease_donors_all)
        if shared_donors_all:
            raise ADDEGError(
                "between_donor pairing requires globally disjoint normal/disease donors; "
                f"donor cannot occur in both states: {shared_donors_all}"
            )

        aggregate_rows: list[dict[str, Any]] = []
        donor_rows: list[dict[str, Any]] = []
        donor_counts: dict[str, dict[str, int]] = {}
        state_values = obs["state"].to_numpy()
        cell_type_values = obs["cell_type"].to_numpy()
        donor_values = obs["donor"].to_numpy()

        for cell_type in requested:
            cell_mask = cell_type_values == cell_type
            donor_states: dict[str, set[str]] = {}
            for donor, state in zip(donor_values[cell_mask], state_values[cell_mask], strict=True):
                donor_states.setdefault(str(donor), set()).add(str(state))
            leaking_donors = sorted(donor for donor, states in donor_states.items() if len(states) > 1)
            if leaking_donors:
                raise ADDEGError(f"donor cannot occur in both states for cell type {cell_type!r}: {leaking_donors}")

            normal_donors = tuple(sorted(set(donor_values[cell_mask & (state_values == normal_state)])))
            disease_donors = tuple(sorted(set(donor_values[cell_mask & (state_values == disease_state)])))
            if len(normal_donors) < min_donors_per_state or len(disease_donors) < min_donors_per_state:
                raise ADDEGError(
                    f"cell type {cell_type!r} requires at least {min_donors_per_state} donors in each state; "
                    f"got {len(normal_donors)} and {len(disease_donors)}"
                )
            donor_counts[cell_type] = {
                "normal": len(normal_donors),
                "disease": len(disease_donors),
            }

            donor_means: dict[tuple[str, str], np.ndarray] = {}
            for state, donors in ((normal_state, normal_donors), (disease_state, disease_donors)):
                for donor in donors:
                    indices = np.flatnonzero(cell_mask & (state_values == state) & (donor_values == donor))
                    if len(indices) == 0:
                        raise ADDEGError(f"donor {donor!r} has no cells for cell type {cell_type!r}")
                    donor_means[(state, str(donor))] = _donor_log2_means(counts, indices)

            normal_matrix = np.vstack([donor_means[(normal_state, str(donor))] for donor in normal_donors])
            disease_matrix = np.vstack([donor_means[(disease_state, str(donor))] for donor in disease_donors])
            delta = disease_matrix.mean(axis=0) - normal_matrix.mean(axis=0)
            test = ttest_ind(disease_matrix, normal_matrix, axis=0, equal_var=False, nan_policy="omit")
            p_values = np.asarray(test.pvalue, dtype=np.float64)
            invalid = ~np.isfinite(p_values)
            p_values[invalid] = np.where(np.isclose(delta[invalid], 0.0), 1.0, 0.0)
            fdr = _bh_adjust(p_values)
            direction = np.where(
                delta > direction_epsilon, "up", np.where(delta < -direction_epsilon, "down", "neutral")
            )
            normal_reference = normal_matrix.mean(axis=0)

            for gene_index, (ensembl_id, gene_symbol) in enumerate(zip(gene_names, gene_symbols, strict=True)):
                aggregate_rows.append(
                    {
                        "cell_type": cell_type,
                        "ensembl_id": ensembl_id,
                        "gene_symbol": gene_symbol,
                        "log2fc": float(delta[gene_index]),
                        "fdr": float(fdr[gene_index]),
                        "observed_direction": str(direction[gene_index]),
                        "n_normal_donors": len(normal_donors),
                        "n_disease_donors": len(disease_donors),
                    }
                )
                for donor in disease_donors:
                    donor_rows.append(
                        {
                            "cell_type": cell_type,
                            "donor": str(donor),
                            "ensembl_id": ensembl_id,
                            "gene_symbol": gene_symbol,
                            "log2fc": float(
                                donor_means[(disease_state, str(donor))][gene_index] - normal_reference[gene_index]
                            ),
                            "fdr": float(fdr[gene_index]),
                        }
                    )

        aggregate = pd.DataFrame(aggregate_rows, columns=_AGGREGATE_COLUMNS)
        donor = pd.DataFrame(donor_rows, columns=_DONOR_COLUMNS)
        audit = {
            "cell_types": list(requested),
            "counts_layer": counts_layer,
            "normal_state": normal_state,
            "disease_state": disease_state,
            "normal_reference": f"per cell type {normal_state} donor-level log2(normalized counts) mean",
            "effect_scale": f"{disease_state} donor mean minus {normal_state} donor mean",
            "state_counts": {
                "normal": int(np.count_nonzero(state_values == "normal")),
                "disease": int(np.count_nonzero(state_values == "disease")),
            },
            "donor_counts": donor_counts,
        }
        return aggregate, donor, audit
    except ADDEGError:
        raise
    except Exception as exc:
        raise ADDEGError(str(exc)) from exc


def write_ad_deg_tables(
    aggregate_frame: pd.DataFrame,
    donor_frame: pd.DataFrame,
    aggregate_output: str | Path,
    donor_output: str | Path,
) -> None:
    """Write the aggregate and donor tables using suffix-selected delimiters."""

    for frame, output in ((aggregate_frame, aggregate_output), (donor_frame, donor_output)):
        path = Path(output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        separator = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
        frame.to_csv(path, sep=separator, index=False)


def manifest_payload(
    audit: Mapping[str, Any],
    *,
    aggregate_output: str | Path | None = None,
    donor_output: str | Path | None = None,
) -> dict[str, Any]:
    """Build a small manifest payload with normalized output paths."""

    payload: dict[str, Any] = {"audit": dict(audit)}
    if aggregate_output is not None:
        payload["aggregate_output"] = str(Path(aggregate_output).expanduser().resolve())
    if donor_output is not None:
        payload["donor_output"] = str(Path(donor_output).expanduser().resolve())
    return payload


def write_manifest(payload: Mapping[str, Any], output: str | Path) -> None:
    """Write a JSON manifest and create its parent directory."""

    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "ADDEGError",
    "build_ad_deg_tables",
    "manifest_payload",
    "write_ad_deg_tables",
    "write_manifest",
]
