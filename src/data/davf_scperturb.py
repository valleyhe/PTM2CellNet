"""Prepare scPerturb AnnData and build strict latent DAVF pairs.

The two operations in this module deliberately share one data contract:

* intervention labels are resolved to Ensembl IDs and then to PerturbGen
  token rows;
* the prepared AnnData contains exactly the 4018 genes that the new scVI
  model will decode, in one immutable order;
* target labels, rather than individual cells, are split into train/val/test;
* each split receives a disjoint subset of control cells and uses the
  within-batch control latent mean as its baseline.

No gene index is inferred from the PerturbGen vocabulary. PerturbGen indices
are only written to ``gene_ids``; scVI decoder indices remain
``adapter.gene_names`` at inference time.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.models.davf_checkpoint_contract import FORMAL_DAVF_NUM_GENES  # noqa: F401 - re-exported contract constant
from src.models.gene_vocabulary import normalize_ensembl_id, normalize_gene_symbol
from src.utils.dependency_check import require_extras


FORMAL_DAVF_LATENT_DIM = 64
DIRECTION_CODES = {"KO": 0, "KD": 1, "OE": 2}
CONTROL_LABELS = frozenset({"control", "control_", "non-targeting", "non_targeting", "nt"})


class DAVFScPerturbError(ValueError):
    """Raised when a supported scPerturb input cannot satisfy the DAVF contract."""


def _require_anndata():
    require_extras(["anndata"], feature="DAVF scPerturb data preparation")
    import anndata as ad

    return ad


@dataclass
class _PreparedSource:
    """One filtered source before all sources are concatenated."""

    label: str
    adata: Any
    gene_ids: tuple[str, ...]
    target_genes: tuple[str, ...]
    feature_scores: np.ndarray
    report: dict[str, Any]


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _is_control_label(value: Any) -> bool:
    value_text = _text(value).lower()
    return bool(value_text) and value_text in CONTROL_LABELS


def _canonical_gene_ids(adata: Any) -> tuple[str, ...]:
    for column in ("ensembl_id", "gene_id", "gene_ids"):
        if column in adata.var.columns:
            raw_values = adata.var[column].tolist()
            break
    else:
        raw_values = list(adata.var_names)

    gene_ids: list[str] = []
    for index, value in enumerate(raw_values):
        try:
            gene_ids.append(normalize_ensembl_id(_text(value)))
        except ValueError as exc:
            raise DAVFScPerturbError(f"AnnData var row {index} is not a valid Ensembl gene ID: {value!r}") from exc
    if len(set(gene_ids)) != len(gene_ids):
        raise DAVFScPerturbError("AnnData contains duplicate Ensembl gene IDs")
    return tuple(gene_ids)


def _symbol_to_gene_id(adata: Any, gene_ids: Sequence[str]) -> dict[str, str]:
    if "gene_symbol" in adata.var.columns:
        raw_symbols = adata.var["gene_symbol"].tolist()
    else:
        raw_symbols = list(adata.var_names)

    mapping: dict[str, str] = {}
    ambiguous: set[str] = set()
    for raw_symbol, gene_id in zip(raw_symbols, gene_ids, strict=True):
        symbol = _text(raw_symbol)
        if not symbol:
            continue
        try:
            normalized = normalize_gene_symbol(symbol)
        except ValueError:
            continue
        previous = mapping.get(normalized)
        if previous is not None and previous != gene_id:
            ambiguous.add(normalized)
        else:
            mapping[normalized] = gene_id
    for symbol in ambiguous:
        mapping.pop(symbol, None)
    return mapping


def _candidate_columns(obs: pd.DataFrame, modality: str) -> tuple[str, ...]:
    preferred = {
        "KO": ("target", "gene_symbol", "gene", "perturbation"),
        "KD": ("gene_id", "gene", "target", "gene_symbol", "perturbation"),
        "OE": ("target", "gene_symbol", "gene", "perturbation"),
    }[modality]
    columns = tuple(column for column in preferred if column in obs.columns)
    if not columns:
        raise DAVFScPerturbError(
            f"{modality} AnnData must contain one perturbation identity column; available columns={list(obs.columns)!r}"
        )
    return columns


def _resolve_cell_target(
    values: Sequence[Any],
    *,
    source_gene_ids: set[str],
    symbol_to_id: Mapping[str, str],
    asset_gene_to_token: Mapping[str, int],
) -> str | None:
    for value in values:
        raw = _text(value)
        if not raw or _is_control_label(raw):
            continue
        candidate_id: str | None
        try:
            candidate_id = normalize_ensembl_id(raw)
        except ValueError:
            try:
                candidate_id = symbol_to_id.get(normalize_gene_symbol(raw))
            except ValueError:
                candidate_id = None
        if candidate_id is None or candidate_id not in source_gene_ids:
            continue
        if candidate_id not in asset_gene_to_token:
            continue
        return candidate_id
    return None


def _feature_scores(adata: Any) -> np.ndarray:
    """Return deterministic feature scores without duplicating a dense matrix."""

    for column in ("ncells", "ncounts"):
        if column in adata.var.columns:
            values = np.asarray(pd.to_numeric(adata.var[column], errors="coerce").to_numpy(dtype=np.float64))
            if np.isfinite(values).all() and (values >= 0).all():
                return values

    matrix = adata.X
    if sp.issparse(matrix):
        return np.asarray(np.asarray((matrix > 0).sum(axis=0)).ravel().astype(np.float64))
    dense = np.asarray(matrix)
    return np.asarray((dense > 0).sum(axis=0), dtype=np.float64)


def _validate_counts(matrix: Any, *, source: str) -> None:
    if sp.issparse(matrix):
        values = np.asarray(matrix.data)
    else:
        values = np.asarray(matrix)
    if values.size and (not np.isfinite(values).all() or (values < 0).any()):
        raise DAVFScPerturbError(f"{source} expression matrix must contain finite non-negative counts")


def _prepare_source(
    path: Path,
    *,
    modality: str,
    asset_gene_to_token: Mapping[str, int],
) -> _PreparedSource:
    ad = _require_anndata()

    if not path.is_file():
        raise FileNotFoundError(f"scPerturb AnnData not found: {path}")
    try:
        source = ad.read_h5ad(path)
    except (OSError, ValueError, KeyError) as exc:
        raise DAVFScPerturbError(f"failed to read {path}: {exc}") from exc
    if source.n_obs < 2 or source.n_vars < FORMAL_DAVF_NUM_GENES:
        raise DAVFScPerturbError(
            f"{path.name} has shape {source.shape}; at least two cells and {FORMAL_DAVF_NUM_GENES} genes are required"
        )

    _validate_counts(source.X, source=path.name)
    gene_ids = _canonical_gene_ids(source)
    source_gene_set = set(gene_ids)
    symbol_to_id = _symbol_to_gene_id(source, gene_ids)
    identity_columns = _candidate_columns(source.obs, modality)
    control_columns = tuple(
        column
        for column in ("perturbation", "target", "gene", "gene_id", "gene_symbol")
        if column in source.obs.columns
    )
    identity_values = {column: source.obs[column].to_numpy() for column in identity_columns}
    control_values = {column: source.obs[column].to_numpy() for column in control_columns}

    targets: list[str] = []
    keep = np.zeros(source.n_obs, dtype=bool)
    reasons: Counter[str] = Counter()
    raw_target_values: list[str] = []
    for row_index in range(source.n_obs):
        control = any(_is_control_label(control_values[column][row_index]) for column in control_columns)
        values = [identity_values[column][row_index] for column in identity_columns]
        target = (
            None
            if control
            else _resolve_cell_target(
                values,
                source_gene_ids=source_gene_set,
                symbol_to_id=symbol_to_id,
                asset_gene_to_token=asset_gene_to_token,
            )
        )
        if control:
            keep[row_index] = True
            raw_target_values.append("")
        elif target is not None:
            keep[row_index] = True
            raw_target_values.append(target)
            targets.append(target)
        else:
            raw_target_values.append("")
            reasons["unmapped_or_unobserved_target"] += 1

    if not keep.any():
        raise DAVFScPerturbError(f"{path.name} contains no explicit controls or asset-resolvable target cells")
    if not targets:
        raise DAVFScPerturbError(f"{path.name} contains no asset-resolvable target cells")

    filtered = source[keep].copy()
    filtered.obs_names_make_unique()
    filtered.var_names = list(gene_ids)
    filtered.var = pd.DataFrame(
        {
            "gene_id": np.asarray(gene_ids, dtype="U"),
            "gene_symbol": np.asarray(
                [
                    _text(source.var.iloc[index]["gene_symbol"])
                    if "gene_symbol" in source.var.columns
                    else _text(source.var_names[index])
                    for index in range(source.n_vars)
                ],
                dtype="U",
            ),
        },
        index=list(gene_ids),
    )

    target_series = np.asarray(raw_target_values, dtype="U")[keep]
    # Use ordinary object-backed strings here. anndata 0.11 refuses to write
    # pandas nullable StringArray unless a global opt-in is enabled; the
    # prepared artifact must be portable without changing process settings.
    filtered.obs["davf_target_ensembl"] = pd.Series(target_series, index=filtered.obs_names)
    filtered.obs["davf_modality"] = modality
    raw_batches: Sequence[Any]
    if "batch" in filtered.obs.columns:
        raw_batches = filtered.obs["batch"].tolist()
    elif "time" in filtered.obs.columns:
        raw_batches = filtered.obs["time"].tolist()
    else:
        raw_batches = ["batch_0"] * filtered.n_obs
    source_label = path.stem
    filtered.obs["davf_batch"] = pd.Series(
        [f"{source_label}:{_text(value) or 'batch_0'}" for value in raw_batches],
        index=filtered.obs_names,
    )
    filtered.obs["davf_source"] = source_label

    scores = _feature_scores(filtered)
    target_genes = tuple(sorted(set(targets)))
    report = {
        "input": str(path.resolve()),
        "source": source_label,
        "input_shape": [int(source.n_obs), int(source.n_vars)],
        "kept_cells": int(filtered.n_obs),
        "control_cells": int((filtered.obs["davf_target_ensembl"] == "").sum()),
        "target_cells": int((filtered.obs["davf_target_ensembl"] != "").sum()),
        "target_genes": list(target_genes),
        "excluded_cells": int((~keep).sum()),
        "exclusion_reasons": dict(reasons),
    }
    return _PreparedSource(
        label=source_label,
        adata=filtered,
        gene_ids=gene_ids,
        target_genes=target_genes,
        feature_scores=scores,
        report=report,
    )


def prepare_scperturb_anndata(
    input_paths: Sequence[str | Path],
    *,
    modality: str,
    asset_gene_to_token: Mapping[str, int],
    output_path: str | Path,
    embedding_asset_path: str | Path,
    embedding_manifest: Mapping[str, Any],
    n_genes: int = FORMAL_DAVF_NUM_GENES,
) -> dict[str, Any]:
    """Filter one or more real scPerturb H5AD files to a formal DAVF input."""

    if modality not in DIRECTION_CODES:
        raise DAVFScPerturbError(f"unsupported DAVF modality {modality!r}; expected KO, KD or OE")
    if n_genes != FORMAL_DAVF_NUM_GENES:
        raise DAVFScPerturbError("formal DAVF preparation is fixed at exactly 4018 genes")
    paths = tuple(Path(path).expanduser().resolve() for path in input_paths)
    if not paths:
        raise DAVFScPerturbError("at least one input AnnData path is required")

    sources = [_prepare_source(path, modality=modality, asset_gene_to_token=asset_gene_to_token) for path in paths]
    common = set(sources[0].gene_ids)
    for source in sources[1:]:
        common.intersection_update(source.gene_ids)
    ordered_common = [gene_id for gene_id in sources[0].gene_ids if gene_id in common]
    if len(ordered_common) < n_genes:
        raise DAVFScPerturbError(
            f"sources share only {len(ordered_common)} genes; {n_genes} are required for formal scVI"
        )

    target_union = set().union(*(set(source.target_genes) for source in sources))
    target_in_common = [gene_id for gene_id in ordered_common if gene_id in target_union]
    if not target_in_common:
        raise DAVFScPerturbError("no perturbation target remains in the common gene vocabulary")

    score_by_gene = {gene_id: 0.0 for gene_id in ordered_common}
    for source in sources:
        source_index = {gene_id: index for index, gene_id in enumerate(source.gene_ids)}
        for gene_id in ordered_common:
            score_by_gene[gene_id] += float(source.feature_scores[source_index[gene_id]])
    remaining = [gene_id for gene_id in ordered_common if gene_id not in set(target_in_common)]
    position = {gene_id: index for index, gene_id in enumerate(ordered_common)}
    remaining.sort(key=lambda gene_id: (-score_by_gene[gene_id], position[gene_id]))
    selected_gene_ids = tuple(target_in_common + remaining[: n_genes - len(target_in_common)])
    if len(selected_gene_ids) != n_genes or len(set(selected_gene_ids)) != n_genes:
        raise DAVFScPerturbError("failed to build a unique 4018-gene scVI vocabulary")

    selected_sources = []
    for source in sources:
        source_index = {gene_id: index for index, gene_id in enumerate(source.gene_ids)}
        indices = [source_index[gene_id] for gene_id in selected_gene_ids]
        selected = source.adata[:, indices].copy()
        selected.var_names = list(selected_gene_ids)
        selected.var = pd.DataFrame(
            {
                "gene_id": list(selected_gene_ids),
                "gene_symbol": [
                    _text(source.adata.var.iloc[source_index[gene_id]].get("gene_symbol", gene_id))
                    for gene_id in selected_gene_ids
                ],
            },
            index=list(selected_gene_ids),
        )
        selected_sources.append(selected)

    ad = _require_anndata()
    combined = ad.concat(
        selected_sources,
        axis=0,
        join="inner",
        merge="first",
        label="davf_input_source",
        keys=[source.label for source in sources],
        index_unique="-",
    )
    combined.var_names = list(selected_gene_ids)
    combined.var["gene_id"] = list(selected_gene_ids)
    combined.uns = {
        "davf_preparation": {
            "schema_version": "ptm2cellnet.davf-scperturb-prepared.v1",
            "modality": modality,
            "direction_code": DIRECTION_CODES[modality],
            "num_genes": n_genes,
            "gene_names": list(selected_gene_ids),
            "input_paths": [str(path) for path in paths],
            "embedding_asset_path": str(Path(embedding_asset_path).expanduser().resolve()),
            "embedding_manifest": dict(embedding_manifest),
            "target_genes": list(target_in_common),
        }
    }

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(destination, compression=None)
    alias_path = destination.with_suffix(".gene_aliases.tsv")
    alias_rows = ["ensembl_id\tgene_symbol"]
    for gene_id in selected_gene_ids:
        symbol = _text(combined.var.loc[gene_id, "gene_symbol"]) or gene_id
        alias_rows.append(f"{gene_id}\t{symbol}")
    alias_path.write_text("\n".join(alias_rows) + "\n", encoding="utf-8")
    report = {
        "schema_version": "ptm2cellnet.davf-scperturb-prepared.v1",
        "modality": modality,
        "direction_code": DIRECTION_CODES[modality],
        "output": str(destination),
        "gene_aliases": str(alias_path),
        "shape": [int(combined.n_obs), int(combined.n_vars)],
        "gene_names": list(selected_gene_ids),
        "target_genes": list(target_in_common),
        "target_gene_tokens": {gene_id: int(asset_gene_to_token[gene_id]) for gene_id in target_in_common},
        "embedding_asset": {
            "path": str(Path(embedding_asset_path).expanduser().resolve()),
            "manifest": dict(embedding_manifest),
            "vocab_size": len(asset_gene_to_token),
        },
        "sources": [source.report for source in sources],
    }
    manifest_path = destination.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def split_target_labels(
    labels: Iterable[str],
    *,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, tuple[str, ...]]:
    """Split unique target genes; all cells of one target stay in one split."""

    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=float)
    if np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0):
        raise DAVFScPerturbError("train/val/test ratios must be positive and sum to 1")
    unique = np.asarray(sorted(set(str(label) for label in labels if str(label))), dtype="U")
    if len(unique) < 3:
        raise DAVFScPerturbError("at least three target genes are required for train/val/test")
    shuffled = unique[np.random.default_rng(seed).permutation(len(unique))]
    counts = np.floor(ratios * len(unique)).astype(int)
    counts = np.maximum(counts, 1)
    while int(counts.sum()) > len(unique):
        candidates = np.flatnonzero(counts > 1)
        index = int(candidates[np.argmin(ratios[candidates])])
        counts[index] -= 1
    while int(counts.sum()) < len(unique):
        deficits = ratios - counts / len(unique)
        counts[int(np.argmax(deficits))] += 1
    first = int(counts[0])
    second = first + int(counts[1])
    return {
        "train": tuple(str(value) for value in shuffled[:first]),
        "val": tuple(str(value) for value in shuffled[first:second]),
        "test": tuple(str(value) for value in shuffled[second:]),
    }


def split_target_cells(
    target_values: Sequence[Any] | np.ndarray,
    *,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, tuple[int, ...]]:
    """Split target cells while keeping every sufficiently sampled target in all splits.

    This is the production split for evaluating known perturbation targets on
    held-out cells. Targets with fewer than three cells cannot contribute one
    cell to each split without inventing observations, so all of their real
    cells are assigned to train and the omission is explicit in the result.
    """

    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=float)
    if np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0):
        raise DAVFScPerturbError("train/val/test ratios must be positive and sum to 1")
    rows_by_label: dict[str, list[int]] = {}
    for index, value in enumerate(target_values):
        label = str(value)
        if label:
            rows_by_label.setdefault(label, []).append(index)
    labels = tuple(sorted(rows_by_label))
    if not labels:
        raise DAVFScPerturbError("at least one target gene is required for a cell split")

    split_order = ("train", "val", "test")
    rows_by_split: dict[str, list[int]] = {split: [] for split in split_order}
    for label_index, label in enumerate(labels):
        rows = np.asarray(rows_by_label[label], dtype=np.int64)
        shuffled = rows[np.random.default_rng(seed + 7919 * (label_index + 1)).permutation(len(rows))]
        if len(rows) < len(split_order):
            rows_by_split["train"].extend(int(index) for index in shuffled)
            continue

        counts = np.floor(ratios * len(rows)).astype(int)
        counts = np.maximum(counts, 1)
        while int(counts.sum()) > len(rows):
            candidates = np.flatnonzero(counts > 1)
            counts[int(candidates[np.argmin(ratios[candidates])])] -= 1
        while int(counts.sum()) < len(rows):
            deficits = ratios - counts / len(rows)
            counts[int(np.argmax(deficits))] += 1

        start = 0
        for split, count in zip(split_order, counts, strict=True):
            stop = start + int(count)
            rows_by_split[split].extend(int(index) for index in shuffled[start:stop])
            start = stop

    if any(not rows_by_split[split] for split in split_order):
        raise DAVFScPerturbError("cell split produced an empty train, val, or test split")
    return {split: tuple(rows_by_split[split]) for split in split_order}


def _validate_pair_build_options(
    modality: str,
    split_strategy: str,
    control_baseline: str,
    max_cells_per_target: int,
    encoder_batch_size: int,
) -> None:
    if modality not in DIRECTION_CODES:
        raise DAVFScPerturbError(f"unsupported DAVF modality {modality!r}")
    if split_strategy not in {"target", "cell"}:
        raise DAVFScPerturbError("split_strategy must be 'target' or 'cell'")
    if control_baseline not in {"mean", "cell"}:
        raise DAVFScPerturbError("control_baseline must be 'mean' or 'cell'")
    if max_cells_per_target <= 0 or encoder_batch_size <= 0:
        raise DAVFScPerturbError("max_cells_per_target and encoder_batch_size must be positive")


def _load_prepared_anndata(prepared_path: str | Path, modality: str) -> tuple[Any, Path]:
    """Load the prepared AnnData and enforce the shared data contract."""

    ad = _require_anndata()
    prepared = Path(prepared_path).expanduser().resolve()
    adata = ad.read_h5ad(prepared)
    metadata = adata.uns.get("davf_preparation")
    if not isinstance(metadata, Mapping):
        raise DAVFScPerturbError("prepared AnnData is missing uns['davf_preparation']")
    if metadata.get("modality") != modality:
        raise DAVFScPerturbError(
            f"prepared AnnData modality={metadata.get('modality')!r} does not match requested {modality!r}"
        )
    if adata.n_vars != FORMAL_DAVF_NUM_GENES:
        raise DAVFScPerturbError(f"prepared AnnData must have {FORMAL_DAVF_NUM_GENES} genes, got {adata.n_vars}")
    if "davf_target_ensembl" not in adata.obs or "davf_batch" not in adata.obs:
        raise DAVFScPerturbError("prepared AnnData must contain davf_target_ensembl and davf_batch columns")
    return adata, prepared


def _resolve_donor_labels(
    adata: Any,
    donor_obs_column: str | None,
    donor_split: Mapping[str, Any] | None,
    require_state_coverage: bool,
) -> tuple[np.ndarray | None, frozenset[str] | None, frozenset[str] | None]:
    """Validate the donor split payload and derive per-cell donor labels."""

    if require_state_coverage and donor_split is None:
        raise DAVFScPerturbError("require_state_coverage only applies to donor-bound splits; provide donor_split")
    if donor_split is None:
        return None, None, None
    if donor_obs_column is None:
        raise DAVFScPerturbError("donor_split requires donor_obs_column")
    if donor_obs_column not in adata.obs.columns:
        raise DAVFScPerturbError(f"prepared AnnData has no donor column {donor_obs_column!r} for donor-bound splitting")
    train_list = donor_split.get("train_donors")
    held_list = donor_split.get("held_out_donors")
    if not isinstance(train_list, list) or not train_list or not isinstance(held_list, list) or not held_list:
        raise DAVFScPerturbError("donor_split payload requires non-empty train_donors and held_out_donors lists")
    if not all(isinstance(item, str) and item for item in (*train_list, *held_list)):
        raise DAVFScPerturbError("donor_split donor labels must be non-empty strings")
    train_donors = frozenset(train_list)
    held_out_donors = frozenset(held_list)
    if train_donors & held_out_donors:
        overlap = sorted(train_donors & held_out_donors)
        raise DAVFScPerturbError(f"donor_split leaks donors into both pools: {overlap}")
    raw_donors = adata.obs[donor_obs_column]
    donor_labels = np.asarray(
        raw_donors.astype(object).where(raw_donors.notna(), ""),
        dtype="U",
    )
    unknown = sorted(set(donor_labels.tolist()) - train_donors - held_out_donors)
    if unknown:
        raise DAVFScPerturbError(
            "every prepared cell must carry a donor from the donor_split pools; "
            f"unassigned/missing donors: {unknown[:10]}"
        )
    if not (set(donor_labels.tolist()) & train_donors) or not (set(donor_labels.tolist()) & held_out_donors):
        raise DAVFScPerturbError("both the train_donor pool and the held-out donor pool must contain cells")
    return donor_labels, train_donors, held_out_donors


def _encode_latent_with_scvi(
    adata: Any,
    scvi_model_path: str | Path,
    device: str,
    encoder_batch_size: int,
) -> tuple[tuple[str, ...], Path, np.ndarray]:
    """Load the scVI adapter, verify gene order and encode the latent matrix."""

    from src.models.scvi_adapter import ScVIAdapter, ScVIAdapterConfig

    require_extras(["scvi"], feature="DAVF latent-pair construction")
    scvi_path = Path(scvi_model_path).expanduser().resolve()
    adapter = ScVIAdapter.from_trained_model(
        scvi_path,
        config=ScVIAdapterConfig(
            model_path=str(scvi_path),
            n_latent=FORMAL_DAVF_LATENT_DIM,
            batch_key="davf_batch",
            device=device,
        ),
        adata=adata,
    )
    gene_names = tuple(str(name) for name in adapter.gene_names)
    adapter.validate_compatibility(
        expected_latent_dim=FORMAL_DAVF_LATENT_DIM,
        expected_num_genes=FORMAL_DAVF_NUM_GENES,
        expected_gene_names=gene_names,
    )
    if gene_names != tuple(str(name) for name in adata.var_names):
        raise DAVFScPerturbError("prepared AnnData gene order does not match the scVI decoder vocabulary")
    latent = np.asarray(adapter.encode(adata, batch_size=encoder_batch_size), dtype=np.float32)
    if latent.shape != (adata.n_obs, FORMAL_DAVF_LATENT_DIM) or not np.isfinite(latent).all():
        raise DAVFScPerturbError(f"scVI encoder returned invalid latent array with shape {latent.shape}")
    return gene_names, scvi_path, latent


def _check_held_out_state_coverage(
    adata: Any,
    state_obs_column: str | None,
    held_mask: np.ndarray,
) -> dict[str, int]:
    """Demand that the held-out donor pool covers at least two states."""

    if state_obs_column is None:
        raise DAVFScPerturbError("require_state_coverage needs an explicit state_obs_column")
    if state_obs_column not in adata.obs.columns:
        raise DAVFScPerturbError(
            f"prepared AnnData has no state column {state_obs_column!r} for held-out state coverage"
        )
    held_states = adata.obs[state_obs_column].astype(str).str.strip().to_numpy()[held_mask]
    coverage = {str(state): int((held_states == state).sum()) for state in sorted(set(held_states) - {""})}
    if len(coverage) < 2:
        raise DAVFScPerturbError(
            "held-out donor pool covers fewer than two states of "
            f"{state_obs_column!r}: {coverage}; rescue judgements on the "
            "test split would lose their state contrast"
        )
    return coverage


def _donor_pool_label_split(
    unique: np.ndarray,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Shuffle donor-pool target labels and apportion them into train/val."""

    shuffled_labels = unique[np.random.default_rng(seed).permutation(len(unique))]
    ratios = np.asarray([train_ratio, val_ratio], dtype=float)
    if np.any(ratios <= 0):
        raise DAVFScPerturbError("train/val ratios must be positive for donor-bound splits")
    counts = np.maximum(np.floor(ratios / ratios.sum() * len(unique)).astype(int), 1)
    while int(counts.sum()) > len(unique):
        index = int(np.argmax(counts))
        counts[index] -= 1
    while int(counts.sum()) < len(unique):
        counts[int(np.argmax(ratios - counts / len(unique)))] += 1
    return (
        tuple(str(value) for value in shuffled_labels[: int(counts[0])]),
        tuple(str(value) for value in shuffled_labels[int(counts[0]) :]),
    )


def _donor_bound_target_rows(
    target_values: np.ndarray,
    target_labels: tuple[str, ...],
    train_mask: np.ndarray,
    held_mask: np.ndarray,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, dict[str, np.ndarray]]:
    """Split donor-bound rows by target gene inside the train donor pool."""

    train_pool_labels = tuple(sorted(set(target_values[train_mask]) - {""}))
    if len(train_pool_labels) < 2:
        raise DAVFScPerturbError("donor-bound target split needs at least two target genes inside the train donor pool")
    unique = np.asarray(train_pool_labels, dtype="U")
    train_labels, val_labels = _donor_pool_label_split(unique, seed, train_ratio, val_ratio)
    pool_label_splits = {"train": train_labels, "val": val_labels}
    target_rows_by_split = {
        split: {
            label: np.flatnonzero((target_values == label) & train_mask)
            for label in labels
            if np.any((target_values == label) & train_mask)
        }
        for split, labels in pool_label_splits.items()
    }
    target_rows_by_split["test"] = {
        label: np.flatnonzero((target_values == label) & held_mask)
        for label in target_labels
        if np.any((target_values == label) & held_mask)
    }
    return target_rows_by_split


def _donor_bound_cell_rows(
    target_values: np.ndarray,
    target_labels: tuple[str, ...],
    train_mask: np.ndarray,
    held_mask: np.ndarray,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, dict[str, np.ndarray]]:
    """Split donor-bound rows at cell level within the train donor pool."""

    pool_ratio = train_ratio / (train_ratio + val_ratio)
    pool_rows_by_split: dict[str, list[int]] = {"train": [], "val": []}
    for label_index, label in enumerate(target_labels):
        label_rows = np.flatnonzero((target_values == label) & train_mask)
        if len(label_rows) == 0:
            continue
        if len(label_rows) == 1:
            pool_rows_by_split["train"].extend(int(row) for row in label_rows)
            continue
        shuffled_rows = label_rows[np.random.default_rng(seed + 7919 * (label_index + 1)).permutation(len(label_rows))]
        train_count = max(1, int(np.floor(pool_ratio * len(label_rows))))
        train_count = min(train_count, len(label_rows) - 1)
        pool_rows_by_split["train"].extend(int(row) for row in shuffled_rows[:train_count])
        pool_rows_by_split["val"].extend(int(row) for row in shuffled_rows[train_count:])
    target_rows_by_split: dict[str, dict[str, np.ndarray]] = {}
    donor_split_row_sets = (
        ("train", pool_rows_by_split["train"]),
        ("val", pool_rows_by_split["val"]),
        ("test", np.flatnonzero(held_mask).tolist()),
    )
    for split, rows in donor_split_row_sets:
        donor_grouped_rows: dict[str, list[int]] = {}
        for index in rows:
            if str(target_values[index]) == "":
                continue
            donor_grouped_rows.setdefault(str(target_values[index]), []).append(int(index))
        target_rows_by_split[split] = {
            label: np.asarray(indices, dtype=np.int64) for label, indices in sorted(donor_grouped_rows.items())
        }
    return target_rows_by_split


def _plain_target_rows(
    target_values: np.ndarray,
    target_labels: tuple[str, ...],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, dict[str, np.ndarray]]:
    label_splits = split_target_labels(
        target_labels,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    return {
        split: {label: np.flatnonzero(target_values == label) for label in labels}
        for split, labels in label_splits.items()
    }


def _plain_cell_rows(
    target_values: np.ndarray,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, dict[str, np.ndarray]]:
    row_splits = split_target_cells(
        target_values,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    target_rows_by_split: dict[str, dict[str, np.ndarray]] = {}
    for split, rows in row_splits.items():
        grouped_rows: dict[str, list[int]] = {}
        for index in rows:
            grouped_rows.setdefault(str(target_values[index]), []).append(int(index))
        target_rows_by_split[split] = {
            label: np.asarray(indices, dtype=np.int64) for label, indices in sorted(grouped_rows.items())
        }
    return target_rows_by_split


def _build_target_rows_by_split(
    adata: Any,
    target_values: np.ndarray,
    target_labels: tuple[str, ...],
    donor_labels: np.ndarray | None,
    train_donors: frozenset[str] | None,
    held_out_donors: frozenset[str] | None,
    split_strategy: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    require_state_coverage: bool,
    state_obs_column: str | None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int] | None]:
    """Assign every row to a split under the active split strategy."""

    held_out_state_coverage: dict[str, int] | None = None
    if donor_labels is not None:
        assert train_donors is not None and held_out_donors is not None
        train_mask = np.isin(donor_labels, sorted(train_donors))
        held_mask = np.isin(donor_labels, sorted(held_out_donors))
        if require_state_coverage:
            held_out_state_coverage = _check_held_out_state_coverage(adata, state_obs_column, held_mask)
        if split_strategy == "target":
            target_rows_by_split = _donor_bound_target_rows(
                target_values, target_labels, train_mask, held_mask, seed, train_ratio, val_ratio
            )
        else:
            target_rows_by_split = _donor_bound_cell_rows(
                target_values, target_labels, train_mask, held_mask, seed, train_ratio, val_ratio
            )
        return target_rows_by_split, held_out_state_coverage
    if split_strategy == "target":
        return _plain_target_rows(target_values, target_labels, seed, train_ratio, val_ratio, test_ratio), None
    return _plain_cell_rows(target_values, seed, train_ratio, val_ratio, test_ratio), None


def _allocate_pool_counts(pool: np.ndarray, pool_ratios: np.ndarray) -> np.ndarray:
    """Apportion shuffled control cells across ratios without dropping any cell."""

    counts = np.maximum(np.floor(pool_ratios / pool_ratios.sum() * len(pool)).astype(int), 1)
    while int(counts.sum()) > len(pool):
        index = int(np.argmax(counts))
        counts[index] -= 1
    while int(counts.sum()) < len(pool):
        deficits = pool_ratios - counts / len(pool)
        counts[int(np.argmax(deficits))] += 1
    return np.asarray(counts)


def _allocate_donor_bound_controls(
    target_values: np.ndarray,
    batches: np.ndarray,
    donor_labels: np.ndarray,
    train_donors: frozenset[str],
    held_out_donors: frozenset[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    control_indices: dict[str, dict[str, np.ndarray]],
) -> None:
    train_mask = np.isin(donor_labels, sorted(train_donors))
    held_mask = np.isin(donor_labels, sorted(held_out_donors))
    for batch in sorted(set(batches)):
        for pool_mask, pool_splits, pool_ratios in (
            (train_mask, ("train", "val"), np.asarray([train_ratio, val_ratio])),
            (held_mask, ("test",), np.asarray([1.0])),
        ):
            pool = np.flatnonzero((target_values == "") & (batches == batch) & pool_mask)
            if len(pool) < len(pool_splits):
                raise DAVFScPerturbError(
                    f"batch {batch!r} has fewer than {len(pool_splits)} control cells in a donor pool"
                )
            shuffled = pool[np.random.default_rng(seed + sum(map(ord, batch))).permutation(len(pool))]
            counts = _allocate_pool_counts(pool, pool_ratios)
            start = 0
            for split, count in zip(pool_splits, counts, strict=True):
                stop = start + int(count)
                control_indices[split][batch] = shuffled[start:stop]
                start = stop


def _allocate_plain_controls(
    target_values: np.ndarray,
    batches: np.ndarray,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    control_indices: dict[str, dict[str, np.ndarray]],
) -> None:
    split_order = ("train", "val", "test")
    for batch in sorted(set(batches)):
        pool = np.flatnonzero((target_values == "") & (batches == batch))
        if len(pool) < 3:
            raise DAVFScPerturbError(f"batch {batch!r} has fewer than three control cells")
        shuffled = pool[np.random.default_rng(seed + sum(map(ord, batch))).permutation(len(pool))]
        counts = np.floor(np.asarray([train_ratio, val_ratio, test_ratio]) * len(pool)).astype(int)
        counts = np.maximum(counts, 1)
        while int(counts.sum()) > len(pool):
            index = int(np.argmax(counts))
            counts[index] -= 1
        while int(counts.sum()) < len(pool):
            index = int(np.argmax(np.asarray([train_ratio, val_ratio, test_ratio]) - counts / len(pool)))
            counts[index] += 1
        start = 0
        for split, count in zip(split_order, counts, strict=True):
            stop = start + int(count)
            control_indices[split][batch] = shuffled[start:stop]
            start = stop


@dataclass
class _SplitExportContext:
    """Shared state for exporting one leakage-free split NPZ."""

    adata: Any
    latent: np.ndarray
    batches: np.ndarray
    donor_labels: np.ndarray | None
    train_donors: frozenset[str] | None
    held_out_donors: frozenset[str] | None
    embedding_asset: Any
    embedding_asset_path: str | Path
    scvi_path: Path
    prepared_path: Path
    gene_names: tuple[str, ...]
    modality: str
    direction_code: int
    split_strategy: str
    control_baseline: str
    seed: int
    max_cells_per_target: int
    donor_obs_column: str | None
    donor_split: Mapping[str, Any] | None


def _export_split_npz(
    ctx: _SplitExportContext,
    split: str,
    split_target_rows: Mapping[str, np.ndarray],
    split_controls: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    output_path: Path,
) -> dict[str, Any]:
    """Assemble one split's pairs, validate donor binding and write the NPZ."""

    rows_z0: list[np.ndarray] = []
    rows_z1: list[np.ndarray] = []
    rows_gene: list[int] = []
    rows_direction: list[int] = []
    rows_cell_ids: list[str] = []
    rows_batches: list[str] = []
    rows_donors: list[str] = []
    control_cell_ids: list[str] = []
    control_batches: list[str] = []
    control_donors: list[str] = []
    pair_control_cell_ids: list[str] = []
    pair_control_batches: list[str] = []
    labels = tuple(sorted(split_target_rows))
    split_control_count = sum(len(values) for values in split_controls.values())
    for batch in sorted(split_controls):
        indices = split_controls[batch]
        control_cell_ids.extend(str(ctx.adata.obs_names[index]) for index in indices)
        control_batches.extend([batch] * len(indices))
        if ctx.donor_labels is not None:
            control_donors.extend(str(ctx.donor_labels[index]) for index in indices)
    if len(control_cell_ids) != split_control_count:
        raise DAVFScPerturbError(f"split {split} control barcode count does not match its allocation")
    control_means = {
        batch: ctx.latent[indices].mean(axis=0).astype(np.float32) for batch, indices in split_controls.items()
    }
    for label in labels:
        target_rows = split_target_rows[label]
        if len(target_rows) > ctx.max_cells_per_target:
            target_rows = np.sort(rng.choice(target_rows, size=ctx.max_cells_per_target, replace=False))
        token = ctx.embedding_asset.gene_to_token.get(label)
        if token is None:
            raise DAVFScPerturbError(f"target {label!r} is absent from the PerturbGen asset")
        for row in target_rows:
            batch = ctx.batches[row]
            if batch not in control_means:
                raise DAVFScPerturbError(f"target row {row} has no control baseline in batch {batch!r}")
            control_pool = split_controls[batch]
            if ctx.control_baseline == "cell":
                control_index = int(rng.choice(control_pool))
                baseline = ctx.latent[control_index]
                pair_control_cell_ids.append(str(ctx.adata.obs_names[control_index]))
                pair_control_batches.append(batch)
            else:
                baseline = control_means[batch]
                pair_control_cell_ids.append("")
                pair_control_batches.append(batch)
            rows_z0.append(baseline)
            rows_z1.append(ctx.latent[row])
            rows_gene.append(int(token))
            rows_direction.append(ctx.direction_code)
            rows_cell_ids.append(str(ctx.adata.obs_names[row]))
            rows_batches.append(batch)
            if ctx.donor_labels is not None:
                rows_donors.append(str(ctx.donor_labels[row]))
    if not rows_z1:
        raise DAVFScPerturbError(f"split {split} contains no target cells")
    if ctx.donor_labels is not None and ctx.train_donors is not None and ctx.held_out_donors is not None:
        expected_pool = sorted(ctx.train_donors) if split in ("train", "val") else sorted(ctx.held_out_donors)
        leaked = sorted(set(rows_donors) - set(expected_pool))
        if leaked:
            raise DAVFScPerturbError(f"split {split} target rows carry donors outside their donor pool: {leaked}")

    split_metadata: dict[str, Any] = {
        "schema_version": "ptm2cellnet.latent-davf-pairs.v1",
        "scvi": {
            "model_path": str(ctx.scvi_path),
            "latent_dim": FORMAL_DAVF_LATENT_DIM,
            "num_genes": FORMAL_DAVF_NUM_GENES,
            "gene_names": list(ctx.gene_names),
        },
        "embedding_asset": {
            "path": str(Path(ctx.embedding_asset_path).expanduser().resolve()),
            "vocab_size": int(ctx.embedding_asset.vocab_size),
            "embedding_dim": int(ctx.embedding_asset.embedding_dim),
            "manifest": dict(ctx.embedding_asset.manifest),
        },
        "dataset": {
            "name": "scPerturb prepared AnnData",
            "modality": ctx.modality,
            "intervention_type": ctx.modality,
            "direction_code": ctx.direction_code,
            "split": split,
            "split_strategy": ctx.split_strategy,
            "seed": int(ctx.seed),
            "pair_direction": (
                "within_batch_control_cell_to_perturbation_cell"
                if ctx.control_baseline == "cell"
                else "within_batch_control_mean_to_perturbation_cell"
            ),
            "control_baseline": (
                "same_batch_split_disjoint_control_cell"
                if ctx.control_baseline == "cell"
                else "latent_mean_of_split_disjoint_control_cells"
            ),
            "target_labels": list(labels),
            "target_cells": len(rows_z1),
            "control_cells": int(split_control_count),
            "target_gene_tokens": {label: int(ctx.embedding_asset.gene_to_token[label]) for label in labels},
            "prepared_anndata": str(ctx.prepared_path),
        },
    }
    if ctx.donor_labels is not None:
        split_metadata["dataset"]["donor_obs_column"] = ctx.donor_obs_column
        split_metadata["dataset"]["donor_rows"] = rows_donors
        split_metadata["dataset"]["donor_pool"] = "train_donors" if split in ("train", "val") else "held_out_donors"
        split_metadata["donor_split"] = dict(ctx.donor_split) if ctx.donor_split is not None else None
    npz_arrays: dict[str, Any] = {
        "metadata_json": np.asarray(
            json.dumps(split_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ),
        "z_0": np.asarray(rows_z0, dtype=np.float32),
        "z_1": np.asarray(rows_z1, dtype=np.float32),
        "gene_ids": np.asarray(rows_gene, dtype=np.int64)[:, None],
        "directions": np.asarray(rows_direction, dtype=np.int64)[:, None],
        "attention_mask": np.ones((len(rows_z1), 1), dtype=np.float32),
        "target_cell_ids": np.asarray(rows_cell_ids, dtype="U"),
        "target_batches": np.asarray(rows_batches, dtype="U"),
        "control_cell_ids": np.asarray(control_cell_ids, dtype="U"),
        "control_batches": np.asarray(control_batches, dtype="U"),
        "pair_control_cell_ids": np.asarray(pair_control_cell_ids, dtype="U"),
        "pair_control_batches": np.asarray(pair_control_batches, dtype="U"),
    }
    if ctx.donor_labels is not None:
        npz_arrays["target_donors"] = np.asarray(rows_donors, dtype="U")
        npz_arrays["control_donors"] = np.asarray(control_donors, dtype="U")
    np.savez_compressed(output_path, **npz_arrays)
    return {
        "path": str(output_path),
        "samples": len(rows_z1),
        "target_labels": list(labels),
        "control_cells": int(split_control_count),
    }


def build_scperturb_latent_pairs(
    prepared_path: str | Path,
    *,
    scvi_model_path: str | Path,
    embedding_asset: Any,
    embedding_asset_path: str | Path,
    output_dir: str | Path,
    modality: str,
    seed: int = 42,
    max_cells_per_target: int = 256,
    encoder_batch_size: int = 512,
    device: str = "cpu",
    split_strategy: str = "target",
    control_baseline: str = "mean",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    donor_obs_column: str | None = None,
    donor_split: Mapping[str, Any] | None = None,
    state_obs_column: str | None = None,
    require_state_coverage: bool = False,
) -> dict[str, dict[str, Any]]:
    """Encode a prepared AnnData and export leakage-free latent-pair NPZ files.

    ``donor_split`` (a canonical ``ptm2cellnet.donor_split/v1`` payload)
    together with ``donor_obs_column`` switches the split to donor binding:
    rows whose donor is in ``train_donors`` may only enter the train/val NPZ,
    rows whose donor is in ``held_out_donors`` may only enter the test NPZ,
    and every exported row records its donor in ``dataset.donor_rows`` plus a
    ``target_donors`` array so training can prove no held-out leakage.

    ``require_state_coverage`` additionally demands that the held-out donor
    pool covers at least two values of ``state_obs_column``; rescue verdicts
    computed on the test split lose their disease-state contrast when every
    held-out donor comes from a single state (a realistic between_donor
    split mistake).  The observed held-out state distribution is recorded in
    the pair manifest so the declared design stays auditable.
    """

    _validate_pair_build_options(modality, split_strategy, control_baseline, max_cells_per_target, encoder_batch_size)
    adata, prepared = _load_prepared_anndata(prepared_path, modality)
    donor_labels, train_donors, held_out_donors = _resolve_donor_labels(
        adata, donor_obs_column, donor_split, require_state_coverage
    )
    gene_names, scvi_path, latent = _encode_latent_with_scvi(adata, scvi_model_path, device, encoder_batch_size)
    target_values = np.asarray(
        adata.obs["davf_target_ensembl"].astype(object).where(adata.obs["davf_target_ensembl"].notna(), ""),
        dtype="U",
    )
    batches = np.asarray(
        adata.obs["davf_batch"].astype(object).where(adata.obs["davf_batch"].notna(), ""),
        dtype="U",
    )
    target_labels = tuple(sorted(set(target_values) - {""}))
    target_rows_by_split, held_out_state_coverage = _build_target_rows_by_split(
        adata,
        target_values,
        target_labels,
        donor_labels,
        train_donors,
        held_out_donors,
        split_strategy,
        seed,
        train_ratio,
        val_ratio,
        test_ratio,
        require_state_coverage,
        state_obs_column,
    )

    # Allocate control cells once, independently per batch and split. This
    # makes the control barcode sets disjoint across the three exported files.
    split_order = ("train", "val", "test")
    control_indices: dict[str, dict[str, np.ndarray]] = {split: {} for split in split_order}
    if donor_labels is not None:
        assert train_donors is not None and held_out_donors is not None
        _allocate_donor_bound_controls(
            target_values,
            batches,
            donor_labels,
            train_donors,
            held_out_donors,
            seed,
            train_ratio,
            val_ratio,
            control_indices,
        )
    else:
        _allocate_plain_controls(target_values, batches, seed, train_ratio, val_ratio, test_ratio, control_indices)

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    direction_code = DIRECTION_CODES[modality]
    ctx = _SplitExportContext(
        adata=adata,
        latent=latent,
        batches=batches,
        donor_labels=donor_labels,
        train_donors=train_donors,
        held_out_donors=held_out_donors,
        embedding_asset=embedding_asset,
        embedding_asset_path=embedding_asset_path,
        scvi_path=scvi_path,
        prepared_path=prepared,
        gene_names=gene_names,
        modality=modality,
        direction_code=direction_code,
        split_strategy=split_strategy,
        control_baseline=control_baseline,
        seed=seed,
        max_cells_per_target=max_cells_per_target,
        donor_obs_column=donor_obs_column,
        donor_split=donor_split,
    )
    report: dict[str, dict[str, Any]] = {}
    for split_index, split in enumerate(split_order):
        rng = np.random.default_rng(seed + 1009 * (split_index + 1))
        report[split] = _export_split_npz(
            ctx,
            split,
            target_rows_by_split[split],
            control_indices[split],
            rng,
            output_root / f"{split}.npz",
        )

    report_path = output_root / "pair_manifest.json"
    pair_manifest: dict[str, Any] = {
        "schema_version": "ptm2cellnet.latent-davf-pairs.v1",
        "modality": modality,
        "direction_code": direction_code,
        "prepared_anndata": str(prepared),
        "scvi_model": str(scvi_path),
        "embedding_asset": str(Path(embedding_asset_path).expanduser().resolve()),
        "splits": report,
        "split_strategy": split_strategy,
        "control_baseline": control_baseline,
    }
    if donor_labels is not None:
        pair_manifest["donor_obs_column"] = donor_obs_column
        pair_manifest["donor_split"] = dict(donor_split) if donor_split is not None else None
    if held_out_state_coverage is not None:
        pair_manifest["state_obs_column"] = state_obs_column
        pair_manifest["held_out_state_coverage"] = held_out_state_coverage
    report_path.write_text(
        json.dumps(pair_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "DAVFScPerturbError",
    "DIRECTION_CODES",
    "FORMAL_DAVF_LATENT_DIM",
    "FORMAL_DAVF_NUM_GENES",
    "build_scperturb_latent_pairs",
    "prepare_scperturb_anndata",
    "split_target_cells",
    "split_target_labels",
]
