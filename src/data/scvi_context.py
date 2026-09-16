"""Prepare a route-aligned scVI context AnnData for DAVF E2E inference.

The DAVF inference contract requires the context AnnData to share the exact
4018-gene axis of the route's frozen scVI checkpoint and to carry the training
``davf_batch`` column (E2E guide: context must be scVI-gene-axis identical).
This module is the single preparation entry (方案 §7.3-U2 方案 A): the missing-
gene policy and the batch binding are explicit, required inputs whose values
are recorded verbatim in the emitted manifest so the preparation is replayable
and the research decision is auditable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.models.gene_vocabulary import normalize_ensembl_id

SCVI_CONTEXT_SCHEMA_VERSION = "ptm2cellnet.scvi-context/v1"
MISSING_GENE_POLICIES = ("zero_fill", "fail")
VALID_DAVF_ROUTES = ("ko", "kd")


class ScviContextError(ValueError):
    """Raised when a scVI context preparation input violates the contract."""


def _batch_registry(adapter: Any) -> tuple[str, list[str]]:
    """Return (batch_key, categorical_mapping) from the loaded scVI registry."""

    model = getattr(adapter, "model", None)
    registry = getattr(model, "registry_", None)
    if not isinstance(registry, dict):
        raise ScviContextError(
            "scVI adapter does not expose a saved AnnData registry; cannot verify the davf_batch binding"
        )
    field_registries = registry.get("field_registries")
    if not isinstance(field_registries, dict) or "batch" not in field_registries:
        raise ScviContextError("scVI registry has no batch field; the checkpoint was trained without batches")
    state_registry = field_registries["batch"].get("state_registry")
    if not isinstance(state_registry, dict) or "categorical_mapping" not in state_registry:
        raise ScviContextError("scVI batch field registry has no categorical_mapping")
    mapping = state_registry["categorical_mapping"]
    mapping_values = [str(value) for value in np.asarray(mapping).ravel().tolist()]
    if not mapping_values:
        raise ScviContextError("scVI batch categorical_mapping is empty; the davf_batch binding cannot be verified")
    setup_args = registry.get("setup_args") or {}
    batch_key = setup_args.get("batch_key")
    batch_key_text = str(batch_key) if batch_key else ""
    if not batch_key_text or batch_key_text == "None":
        raise ScviContextError("scVI setup_args does not record a batch_key; the davf_batch column name is unknown")
    return batch_key_text, mapping_values


def _adapter_gene_axis(adapter: Any) -> tuple[str, ...]:
    gene_names = getattr(adapter, "gene_names", None)
    if gene_names is None:
        raise ScviContextError("scVI adapter does not expose ordered gene_names; cannot align the context axis")
    axis = tuple(str(value) for value in gene_names)
    if len(axis) != len(set(axis)):
        raise ScviContextError("scVI adapter gene_names contain duplicates")
    if not axis:
        raise ScviContextError("scVI adapter gene_names is empty")
    return axis


def _cohort_ensembl_ids(cohort: Any) -> list[str]:
    if "ensembl_id" not in cohort.var.columns:
        raise ScviContextError("cohort AnnData is missing var['ensembl_id']")
    raw_values = cohort.var["ensembl_id"].tolist()
    ensembl_ids: list[str] = []
    for index, value in enumerate(raw_values):
        text = "" if pd.isna(value) else str(value).strip()
        try:
            ensembl_ids.append(normalize_ensembl_id(text))
        except ValueError as exc:
            raise ScviContextError(f"cohort var row {index} is not a valid Ensembl gene ID: {value!r}") from exc
    if len(set(ensembl_ids)) != len(ensembl_ids):
        raise ScviContextError("cohort var contains duplicate canonical Ensembl gene IDs")
    return ensembl_ids


def _axis_gene_symbols(axis: tuple[str, ...], gene_aliases_path: Path) -> dict[str, str]:
    alias_table = pd.read_csv(gene_aliases_path, sep="\t")
    missing_columns = [column for column in ("ensembl_id", "gene_symbol") if column not in alias_table.columns]
    if missing_columns:
        raise ScviContextError(f"gene alias table is missing columns: {missing_columns}")
    symbols: dict[str, str] = {}
    for _, row in alias_table.iterrows():
        try:
            ensembl_id = normalize_ensembl_id(str(row["ensembl_id"]))
        except ValueError as exc:
            raise ScviContextError(f"gene alias table has an invalid ensembl_id: {row['ensembl_id']!r}") from exc
        symbol = str(row["gene_symbol"]).strip()
        if not symbol:
            continue
        previous = symbols.get(ensembl_id)
        if previous is not None and previous != symbol:
            raise ScviContextError(f"gene alias table has conflicting symbols for {ensembl_id}: {previous}, {symbol}")
        symbols[ensembl_id] = symbol
    empty_axis_genes = [gene_id for gene_id in axis if not symbols.get(gene_id)]
    if empty_axis_genes:
        raise ScviContextError(
            f"gene alias table does not cover {len(empty_axis_genes)} axis genes (first: {empty_axis_genes[:5]})"
        )
    return symbols


def prepare_scvi_context(
    cohort: Any,
    *,
    adapter: Any,
    davf_route: str,
    davf_batch: str,
    missing_gene_policy: str,
    batch_binding_rationale: str,
    gene_aliases_path: str | Path,
    counts_layer: str = "counts",
) -> tuple[Any, dict[str, Any]]:
    """Align a cohort AnnData onto the route's frozen scVI gene axis.

    Returns ``(context_adata, manifest_payload)``. The context keeps the
    cohort's obs (plus the bound ``davf_batch`` column) and replaces the gene
    axis with the adapter's ordered ``gene_names``; cohort genes outside the
    axis are dropped and axis genes missing from the cohort follow the
    explicit ``missing_gene_policy``.
    """

    route = str(davf_route).strip().lower()
    if route not in VALID_DAVF_ROUTES:
        raise ScviContextError(f"davf_route must be one of {', '.join(VALID_DAVF_ROUTES)}; got {davf_route!r}")
    policy = str(missing_gene_policy).strip()
    if policy not in MISSING_GENE_POLICIES:
        raise ScviContextError(
            f"missing_gene_policy must be one of {', '.join(MISSING_GENE_POLICIES)}; got {missing_gene_policy!r}"
        )
    batch_value = str(davf_batch).strip()
    if not batch_value:
        raise ScviContextError("davf_batch must be a non-empty string from the route's scVI batch registry")
    rationale = str(batch_binding_rationale).strip()
    if not rationale:
        raise ScviContextError(
            "batch_binding_rationale is required: the cross-dataset batch binding is a frozen research decision "
            "and must be recorded verbatim in the manifest"
        )

    batch_key, batch_mapping = _batch_registry(adapter)
    if batch_value not in batch_mapping:
        raise ScviContextError(
            f"davf_batch {batch_value!r} is not in the scVI batch registry of this route; "
            f"valid batches: {batch_mapping}"
        )

    axis = _adapter_gene_axis(adapter)
    axis_symbols = _axis_gene_symbols(axis, Path(gene_aliases_path))
    cohort_ensembl = _cohort_ensembl_ids(cohort)
    cohort_index = {gene_id: position for position, gene_id in enumerate(cohort_ensembl)}

    missing_axis_genes = [gene_id for gene_id in axis if gene_id not in cohort_index]
    if missing_axis_genes and policy == "fail":
        raise ScviContextError(
            f"{len(missing_axis_genes)} axis genes are absent from the cohort and missing_gene_policy is 'fail' "
            f"(first: {missing_axis_genes[:5]})"
        )

    if counts_layer not in getattr(cohort, "layers", {}):
        raise ScviContextError(f"cohort AnnData is missing layers[{counts_layer!r}]")
    counts = cohort.layers[counts_layer]
    n_cells = int(cohort.n_obs)

    present_new_positions = [position for position, gene_id in enumerate(axis) if gene_id in cohort_index]
    present_old_positions = [cohort_index[axis[position]] for position in present_new_positions]
    aligned = sp.csc_matrix((n_cells, len(axis)), dtype=counts.dtype)
    aligned[:, present_new_positions] = counts[:, present_old_positions]
    aligned = aligned.tocsr()

    import anndata

    obs = cohort.obs.copy()
    obs[batch_key] = batch_value
    var = pd.DataFrame(
        {
            "ensembl_id": list(axis),
            "gene_symbol": [axis_symbols[gene_id] for gene_id in axis],
        },
        index=list(axis),
    )
    context = anndata.AnnData(X=aligned.copy(), obs=obs, var=var)
    context.layers[counts_layer] = aligned

    manifest = {
        "schema_version": SCVI_CONTEXT_SCHEMA_VERSION,
        "davf_route": route,
        "scvi_model_path": str(getattr(getattr(adapter, "config", None), "model_path", "")),
        "batch_key": batch_key,
        "davf_batch": batch_value,
        "batch_registry_values": batch_mapping,
        "batch_binding_rationale": rationale,
        "missing_gene_policy": policy,
        "n_missing_axis_genes": len(missing_axis_genes),
        "missing_axis_genes": missing_axis_genes,
        "n_dropped_cohort_genes": len(cohort_ensembl) - len(present_new_positions),
        "gene_axis_size": len(axis),
        "n_cells": n_cells,
        "counts_layer": counts_layer,
        "gene_aliases_path": str(Path(gene_aliases_path)),
    }
    return context, manifest


def write_scvi_context(
    context: Any,
    manifest: dict[str, Any],
    output_h5ad: str | Path,
    manifest_output: str | Path,
) -> None:
    """Write the context AnnData and its manifest next to it."""

    output_path = Path(output_h5ad).expanduser().resolve()
    manifest_path = Path(manifest_output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    context.write_h5ad(output_path, compression="gzip")
    payload = dict(manifest)
    payload["output_h5ad"] = str(output_path)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "MISSING_GENE_POLICIES",
    "ScviContextError",
    "SCVI_CONTEXT_SCHEMA_VERSION",
    "VALID_DAVF_ROUTES",
    "prepare_scvi_context",
    "write_scvi_context",
]
