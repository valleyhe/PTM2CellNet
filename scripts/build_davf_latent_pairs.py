#!/usr/bin/env python3
"""Build formal latent-pair NPZ files from the Norman GSE133344 release.

The generated pair has ``z_0`` from a randomly selected non-targeting cell
and ``z_1`` from the annotated perturbation cell.  The control sampling and
the perturbation-label split are deterministic for a given seed.

This script intentionally has no synthetic-data path and no index fallback:
the 10x matrix is aligned to the exact ``ScVIAdapter.gene_names`` order, and
perturbation ``gene_ids`` are resolved only through the verified PerturbGen
vocabulary.  Any missing input, annotation column, gene mapping, dependency,
or model/asset contract fails immediately.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.io import mmread

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.latent_davf_dataset import LATENT_DAVF_DATA_SCHEMA_VERSION
from src.models.perturbgen_embedding import load_perturbgen_embedding_asset
from src.models.scvi_adapter import ScVIAdapter, ScVIAdapterConfig


EXPECTED_LATENT_DIM = 64
EXPECTED_SCVI_GENES = 4018
MAX_PERTURBATION_GENES = 2
CONTROL_LABEL = "control"


class BuildDAVFLatentPairsError(RuntimeError):
    """Raised when the Norman-to-latent-pair conversion is not valid."""


@dataclass(frozen=True)
class Norman10xData:
    """Validated cell-by-gene Norman matrix and its perturbation labels."""

    matrix: sp.csr_matrix
    raw_gene_ids: tuple[str, ...]
    raw_gene_symbols: tuple[str, ...]
    barcodes: tuple[str, ...]
    labels: tuple[str, ...]
    pairing_groups: tuple[str, ...]


@dataclass(frozen=True)
class PairArrays:
    """One NPZ split's arrays before metadata is added."""

    z_0: np.ndarray
    z_1: np.ndarray
    gene_ids: np.ndarray
    directions: np.ndarray
    attention_mask: np.ndarray
    labels: tuple[str, ...]
    target_barcodes: tuple[str, ...]
    control_barcodes: tuple[str, ...]


_FILE_CANDIDATES = {
    "matrix": ("GSE133344_filtered_matrix.mtx.gz", "GSE133344_filtered_matrix.mtx"),
    "genes": ("GSE133344_filtered_genes.tsv.gz", "GSE133344_filtered_genes.tsv"),
    "barcodes": ("GSE133344_filtered_barcodes.tsv.gz", "GSE133344_filtered_barcodes.tsv"),
    "identities": (
        "GSE133344_filtered_cell_identities.csv.gz",
        "GSE133344_filtered_cell_identities.csv",
    ),
}


def _resolve_input_file(dataset_dir: Path, kind: str) -> Path:
    candidates = [dataset_dir / name for name in _FILE_CANDIDATES[kind]]
    present = [path for path in candidates if path.is_file()]
    if not present:
        names = ", ".join(path.name for path in candidates)
        raise BuildDAVFLatentPairsError(f"missing Norman {kind} input; expected one of: {names}")
    if len(present) > 1:
        raise BuildDAVFLatentPairsError(f"ambiguous Norman {kind} input: {present}")
    return present[0]


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _read_rows(path: Path, delimiter: str) -> list[list[str]]:
    try:
        with _open_text(path) as handle:
            rows = [[cell.strip() for cell in row] for row in csv.reader(handle, delimiter=delimiter)]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise BuildDAVFLatentPairsError(f"cannot read {path}: {exc}") from exc
    return [row for row in rows if any(cell for cell in row)]


def _read_genes(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    rows = _read_rows(path, "\t")
    if not rows:
        raise BuildDAVFLatentPairsError(f"Norman genes table is empty: {path}")
    gene_ids: list[str] = []
    symbols: list[str] = []
    for line_number, row in enumerate(rows, start=1):
        if len(row) < 2 or not row[0] or not row[1]:
            raise BuildDAVFLatentPairsError(f"Norman genes row {line_number} must contain gene_id and gene_symbol")
        gene_ids.append(row[0])
        symbols.append(row[1])
    if len(set(gene_ids)) != len(gene_ids):
        raise BuildDAVFLatentPairsError("Norman genes table contains duplicate gene IDs")
    return tuple(gene_ids), tuple(symbols)


def _read_barcodes(path: Path) -> tuple[str, ...]:
    rows = _read_rows(path, "\t")
    barcodes = tuple(row[0] for row in rows if row and row[0])
    if not barcodes:
        raise BuildDAVFLatentPairsError(f"Norman barcode table is empty: {path}")
    if len(set(barcodes)) != len(barcodes):
        raise BuildDAVFLatentPairsError("Norman barcode table contains duplicates")
    if any(len(row) != 1 for row in rows):
        raise BuildDAVFLatentPairsError("Norman barcode table must contain exactly one column")
    return barcodes


def _read_matrix(path: Path, *, expected_rows: int, expected_columns: int) -> sp.csr_matrix:
    try:
        matrix = mmread(str(path))
    except (OSError, ValueError, TypeError) as exc:
        raise BuildDAVFLatentPairsError(f"cannot parse Norman 10x matrix {path}: {exc}") from exc
    matrix = sp.csr_matrix(matrix, dtype=np.float32)
    if matrix.shape != (expected_rows, expected_columns):
        raise BuildDAVFLatentPairsError(
            "Norman matrix shape does not match genes/barcodes: "
            f"matrix={matrix.shape}, expected=({expected_rows}, {expected_columns})"
        )
    if matrix.nnz and (not np.isfinite(matrix.data).all() or (matrix.data < 0).any()):
        raise BuildDAVFLatentPairsError("Norman count matrix must contain finite non-negative values")
    return matrix.T.tocsr()


def _normalise_header(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _identity_columns(header: Sequence[str]) -> tuple[int, int, int]:
    names = [_normalise_header(value) for value in header]
    barcode_names = {"barcode", "cell_barcode", "cell_id"}
    try:
        barcode_index = next(index for index, name in enumerate(names) if name in barcode_names)
    except StopIteration as exc:
        raise BuildDAVFLatentPairsError("cell identities is missing the cell_barcode column") from exc
    try:
        guide_index = names.index("guide_identity")
    except ValueError as exc:
        raise BuildDAVFLatentPairsError("cell identities is missing the guide_identity column") from exc
    try:
        group_index = next(index for index, name in enumerate(names) if name in {"gemgroup", "gem_group"})
    except StopIteration as exc:
        raise BuildDAVFLatentPairsError(
            "cell identities is missing the gemgroup column required for batch-matched controls"
        ) from exc
    return barcode_index, guide_index, group_index


_GUIDE_CONTROL_RE = re.compile(r"_(?:negctrl|posctrl)\d+$", re.IGNORECASE)
_GUIDE_COPY_SUFFIX_RE = re.compile(r"_\d+$")


def _canonical_guide_component(value: str) -> str:
    return _GUIDE_COPY_SUFFIX_RE.sub("", value.strip())


def parse_guide_identity(value: str) -> tuple[str, bool]:
    """Return ``(perturbation_label, is_control)`` for a Norman guide label.

    Norman stores the two guide assignments as ``left__right``.  The right
    side may carry a row-copy suffix (for example ``...__..._1``); that suffix
    is metadata, not a gene.  Both sides must otherwise agree exactly.
    """

    raw = value.strip()
    parts = raw.split("__")
    if len(parts) != 2 or not all(parts):
        raise BuildDAVFLatentPairsError(f"invalid Norman guide_identity {value!r}; expected left__right")
    left = _canonical_guide_component(parts[0])
    right = _canonical_guide_component(parts[1])
    if not left or left != right:
        raise BuildDAVFLatentPairsError(f"guide_identity sides disagree: {value!r}")
    if _GUIDE_CONTROL_RE.search(left):
        target = _GUIDE_CONTROL_RE.sub("", left).strip("_")
        if not target:
            raise BuildDAVFLatentPairsError(f"control guide_identity has no gene prefix: {value!r}")
        return CONTROL_LABEL, True
    target_parts = [
        part for part in left.split("_") if not re.fullmatch(r"(?:negctrl|posctrl)\d+", part, flags=re.IGNORECASE)
    ]
    if not 1 <= len(target_parts) <= MAX_PERTURBATION_GENES:
        raise BuildDAVFLatentPairsError(f"guide_identity has more than two target genes: {value!r}")
    return "_".join(target_parts), False


def _read_labels(
    path: Path,
    barcodes: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    rows = _read_rows(path, ",")
    if not rows:
        raise BuildDAVFLatentPairsError(f"Norman cell identities table is empty: {path}")
    barcode_index, guide_index, group_index = _identity_columns(rows[0])
    annotations: dict[str, str] = {}
    groups: dict[str, str] = {}
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) <= max(barcode_index, guide_index, group_index):
            raise BuildDAVFLatentPairsError(f"cell identities row {line_number} is missing a required value")
        barcode = row[barcode_index]
        group = row[group_index]
        if not barcode or barcode in annotations or not group:
            raise BuildDAVFLatentPairsError(f"cell identities has an empty or duplicate barcode at row {line_number}")
        label, is_control = parse_guide_identity(row[guide_index])
        annotations[barcode] = CONTROL_LABEL if is_control else label
        groups[barcode] = group

    extra = sorted(set(annotations) - set(barcodes))
    if extra:
        raise BuildDAVFLatentPairsError(
            f"cell identities contains barcodes absent from the 10x matrix (extra={extra[:3]})"
        )
    # The GEO identity table can omit a small number of filtered cells.  Those
    # cells have no valid perturbation label and are excluded by the caller;
    # they are never assigned a guessed control or perturbation.
    return (
        tuple(annotations.get(barcode, "") for barcode in barcodes),
        tuple(groups.get(barcode, "") for barcode in barcodes),
    )


def load_norman_10x(dataset_dir: str | Path) -> Norman10xData:
    """Read and validate the four raw GSE133344 files."""

    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise BuildDAVFLatentPairsError(f"Norman dataset directory not found: {root}")
    matrix_path = _resolve_input_file(root, "matrix")
    genes_path = _resolve_input_file(root, "genes")
    barcodes_path = _resolve_input_file(root, "barcodes")
    identities_path = _resolve_input_file(root, "identities")
    raw_gene_ids, raw_gene_symbols = _read_genes(genes_path)
    barcodes = _read_barcodes(barcodes_path)
    matrix = _read_matrix(path=matrix_path, expected_rows=len(raw_gene_ids), expected_columns=len(barcodes))
    labels, pairing_groups = _read_labels(identities_path, barcodes)
    keep = np.asarray([bool(label) for label in labels], dtype=bool)
    if not keep.any():
        raise BuildDAVFLatentPairsError("Norman cell identities contains no annotated cells")
    return Norman10xData(
        matrix[keep].tocsr(),
        raw_gene_ids,
        raw_gene_symbols,
        tuple(barcode for barcode, include in zip(barcodes, keep, strict=True) if include),
        tuple(label for label, include in zip(labels, keep, strict=True) if include),
        tuple(group for group, include in zip(pairing_groups, keep, strict=True) if include),
    )


def align_to_scvi_gene_order(
    data: Norman10xData,
    scvi_gene_names: Sequence[str],
) -> sp.csr_matrix:
    """Return the cell-by-4018 matrix in the exact scVI decoder order."""

    names = tuple(str(name) for name in scvi_gene_names)
    if len(names) != EXPECTED_SCVI_GENES:
        raise BuildDAVFLatentPairsError(
            f"ScVIAdapter must expose exactly {EXPECTED_SCVI_GENES} genes, got {len(names)}"
        )
    if len(set(names)) != len(names) or any(not name for name in names):
        raise BuildDAVFLatentPairsError("ScVIAdapter gene_names must be unique non-empty strings")
    id_index = {gene_id: index for index, gene_id in enumerate(data.raw_gene_ids)}
    symbol_indices: dict[str, list[int]] = {}
    for index, symbol in enumerate(data.raw_gene_symbols):
        symbol_indices.setdefault(symbol, []).append(index)
    id_complete = all(name in id_index for name in names)
    symbol_complete = all(name in symbol_indices for name in names)
    if id_complete and symbol_complete:
        raise BuildDAVFLatentPairsError("adapter.gene_names ambiguously match both raw gene IDs and symbols")
    if id_complete:
        raw_indices = [id_index[name] for name in names]
        return data.matrix[:, raw_indices].tocsr()
    elif symbol_complete:
        # The Norman 10x release contains a small number of duplicate
        # symbol annotations.  Summing their raw count columns is the
        # deterministic gene-level aggregation used for a symbol-vocabulary
        # scVI input; selecting one transcript row would discard counts.
        first_indices = [symbol_indices[name][0] for name in names]
        aligned = data.matrix[:, first_indices].tocsr()
        blocks = []
        start = 0
        for column_index, name in enumerate(names):
            raw_indices = symbol_indices[name]
            if len(raw_indices) == 1:
                continue
            if start < column_index:
                blocks.append(aligned[:, start:column_index])
            duplicate_column = sp.csr_matrix(data.matrix[:, raw_indices].sum(axis=1))
            blocks.append(duplicate_column)
            start = column_index + 1
        if not blocks:
            return aligned
        if start < len(names):
            blocks.append(aligned[:, start:])
        return sp.hstack(blocks, format="csr")
    else:
        missing = [name for name in names if name not in id_index and name not in symbol_indices]
        ambiguous = [name for name in names if name in symbol_indices and len(symbol_indices[name]) != 1]
        detail = f"missing={missing[:5]}, ambiguous_symbols={ambiguous[:5]}"
        raise BuildDAVFLatentPairsError(f"raw Norman genes do not completely cover adapter.gene_names; {detail}")
    return data.matrix[:, raw_indices].tocsr()


def _target_genes(label: str, symbols: set[str]) -> tuple[str, ...]:
    if label in (CONTROL_LABEL,):
        raise BuildDAVFLatentPairsError("control is not a perturbation target")
    parts = tuple(label.split("_"))
    if len(parts) not in (1, 2) or any(not part or part not in symbols for part in parts):
        raise BuildDAVFLatentPairsError(
            f"cannot resolve perturbation label {label!r} to one or two Norman gene symbols"
        )
    return parts


def _resolve_target_tokens(
    data: Norman10xData,
    asset_gene_to_token: Mapping[str, int],
    asset_vocab_size: int,
    labels: Iterable[str] | None = None,
) -> dict[str, tuple[int, ...]]:
    symbol_to_ids: dict[str, list[str]] = {}
    for gene_id, symbol in zip(data.raw_gene_ids, data.raw_gene_symbols, strict=True):
        symbol_to_ids.setdefault(symbol, []).append(gene_id)
    symbols = set(symbol_to_ids)
    labels = sorted(set(labels) if labels is not None else {label for label in data.labels if label != CONTROL_LABEL})
    resolved: dict[str, tuple[int, ...]] = {}
    for label in labels:
        tokens: list[int] = []
        for symbol in _target_genes(label, symbols):
            ids = symbol_to_ids[symbol]
            if len(ids) != 1:
                raise BuildDAVFLatentPairsError(
                    f"perturbation symbol {symbol!r} maps to {len(ids)} raw gene IDs; mapping is ambiguous"
                )
            raw_id = ids[0]
            token = asset_gene_to_token.get(raw_id)
            if token is None:
                token = asset_gene_to_token.get(symbol)
            if token is None:
                raise BuildDAVFLatentPairsError(
                    f"perturbation gene {symbol!r} ({raw_id}) is absent from the PerturbGen asset"
                )
            if (
                isinstance(token, bool)
                or not isinstance(token, (int, np.integer))
                or int(token) < 0
                or int(token) >= asset_vocab_size
            ):
                raise BuildDAVFLatentPairsError(f"invalid PerturbGen token for {symbol!r}: {token!r}")
            tokens.append(int(token))
        resolved[label] = tuple(tokens)
    return resolved


def _split_resolvable_labels(
    data: Norman10xData,
) -> tuple[tuple[str, ...], Mapping[str, str]]:
    """Separate unobservable labels without inventing a gene feature.

    A guide label can be valid metadata while its target gene is absent from
    the filtered 10x feature table. Such a group cannot produce a supervised
    latent transition and is recorded as excluded instead of guessed.
    """

    symbols = set(data.raw_gene_symbols)
    valid: list[str] = []
    excluded: dict[str, str] = {}
    for label in sorted({label for label in data.labels if label != CONTROL_LABEL}):
        try:
            _target_genes(label, symbols)
        except BuildDAVFLatentPairsError:
            excluded[label] = "target_gene_absent_or_invalid_in_raw_gene_table"
        else:
            valid.append(label)
    if not valid:
        raise BuildDAVFLatentPairsError("Norman data contains no perturbation labels with observable raw target genes")
    return tuple(valid), excluded


def split_perturbation_labels(
    labels: Iterable[str],
    *,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, tuple[str, ...]]:
    """Split unique perturbation labels, never individual cells."""

    ratios = (float(train_ratio), float(val_ratio), float(test_ratio))
    if any(ratio <= 0 for ratio in ratios) or not np.isclose(sum(ratios), 1.0):
        raise BuildDAVFLatentPairsError("train/val/test ratios must be positive and sum to 1")
    unique = sorted(set(labels) - {CONTROL_LABEL})
    if len(unique) < 3:
        raise BuildDAVFLatentPairsError("at least three perturbation labels are required for train/val/test")
    shuffled = np.asarray(unique, dtype=object)[np.random.default_rng(seed).permutation(len(unique))]
    counts = np.floor(np.asarray(ratios) * len(unique)).astype(int)
    while counts.min() == 0:
        donor = int(np.argmax(counts))
        receiver = int(np.argmin(counts))
        if counts[donor] <= 1:
            raise BuildDAVFLatentPairsError("cannot create non-empty train/val/test perturbation splits")
        counts[donor] -= 1
        counts[receiver] += 1
    counts[2] += len(unique) - int(counts.sum())
    if counts[2] <= 0:
        raise BuildDAVFLatentPairsError("test perturbation split is empty")
    first = int(counts[0])
    second = first + int(counts[1])
    return {
        "train": tuple(str(value) for value in shuffled[:first]),
        "val": tuple(str(value) for value in shuffled[first:second]),
        "test": tuple(str(value) for value in shuffled[second:]),
    }


def _allocate_control_pools(
    data: Norman10xData,
    splits: Mapping[str, Sequence[str]],
    rng: np.random.Generator,
) -> dict[str, dict[str, np.ndarray]]:
    """Partition control cells across splits while allowing reuse within one split."""

    control_by_group: dict[str, np.ndarray] = {}
    for group in sorted(
        {data.pairing_groups[index] for index, label in enumerate(data.labels) if label == CONTROL_LABEL}
    ):
        control_by_group[group] = np.asarray(
            [
                index
                for index, label in enumerate(data.labels)
                if label == CONTROL_LABEL and data.pairing_groups[index] == group
            ],
            dtype=np.int64,
        )

    target_groups_by_split: dict[str, set[str]] = {}
    for split, labels in splits.items():
        label_set = set(labels)
        target_groups_by_split[split] = {
            data.pairing_groups[index] for index, label in enumerate(data.labels) if label in label_set
        }

    pools: dict[str, dict[str, np.ndarray]] = {split: {} for split in splits}
    for group, controls in control_by_group.items():
        active_splits = [split for split in splits if group in target_groups_by_split[split]]
        if not active_splits:
            continue
        if controls.size < len(active_splits):
            raise BuildDAVFLatentPairsError(
                f"Norman gemgroup {group!r} has {controls.size} control cells but needs "
                f"at least one disjoint control pool for each active split ({len(active_splits)})"
            )
        shuffled = controls[rng.permutation(controls.size)]
        for split, chunk in zip(active_splits, np.array_split(shuffled, len(active_splits)), strict=True):
            if chunk.size == 0:  # pragma: no cover - guarded by the size check above
                raise BuildDAVFLatentPairsError(f"empty control pool for split {split!r} and gemgroup {group!r}")
            pools[split][group] = np.asarray(chunk, dtype=np.int64)
    return pools


def _build_pair_arrays(
    *,
    latent: np.ndarray,
    data: Norman10xData,
    token_by_label: Mapping[str, tuple[int, ...]],
    split_labels: Sequence[str],
    rng: np.random.Generator,
    control_pools: Mapping[str, np.ndarray] | None = None,
) -> PairArrays:
    split_label_set = set(split_labels)
    target_indices = np.asarray(
        [index for index, label in enumerate(data.labels) if label in split_label_set], dtype=np.int64
    )
    control_indices = np.asarray(
        [index for index, label in enumerate(data.labels) if label == CONTROL_LABEL], dtype=np.int64
    )
    if target_indices.size == 0:
        raise BuildDAVFLatentPairsError(f"split has no target cells for labels {tuple(split_labels)}")
    if control_indices.size == 0:
        raise BuildDAVFLatentPairsError("Norman data contains no control cells")
    if control_pools is None:
        controls_by_group: dict[str, np.ndarray] = {}
        for group in sorted({data.pairing_groups[index] for index in control_indices}):
            controls_by_group[group] = control_indices[
                np.asarray([data.pairing_groups[index] == group for index in control_indices], dtype=bool)
            ]
    else:
        controls_by_group = {str(group): np.asarray(values, dtype=np.int64) for group, values in control_pools.items()}
    chosen_controls = np.empty(target_indices.size, dtype=np.int64)
    for row, target_index in enumerate(target_indices):
        group = data.pairing_groups[target_index]
        candidates = controls_by_group.get(group)
        if candidates is None or candidates.size == 0:
            raise BuildDAVFLatentPairsError(f"Norman gemgroup {group!r} has target cells but no control cells")
        chosen_controls[row] = rng.choice(candidates)
    labels = tuple(data.labels[index] for index in target_indices)
    z_0 = np.asarray(latent[chosen_controls], dtype=np.float32)
    z_1 = np.asarray(latent[target_indices], dtype=np.float32)
    if z_0.ndim != 2 or z_0.shape != z_1.shape or z_0.shape[1] != EXPECTED_LATENT_DIM:
        raise BuildDAVFLatentPairsError(
            f"ScVI encoder must return [N, {EXPECTED_LATENT_DIM}] latents, got {z_0.shape} and {z_1.shape}"
        )
    if not np.isfinite(z_0).all() or not np.isfinite(z_1).all():
        raise BuildDAVFLatentPairsError("ScVI encoder returned non-finite latent values")
    gene_ids = np.zeros((len(labels), MAX_PERTURBATION_GENES), dtype=np.int64)
    directions = np.zeros_like(gene_ids, dtype=np.int64)  # KO is direction 0.
    attention_mask = np.zeros_like(gene_ids, dtype=np.float32)
    for row, label in enumerate(labels):
        tokens = token_by_label[label]
        if not 1 <= len(tokens) <= MAX_PERTURBATION_GENES:
            raise BuildDAVFLatentPairsError(f"perturbation label {label!r} has an invalid target count")
        gene_ids[row, : len(tokens)] = tokens
        attention_mask[row, : len(tokens)] = 1.0
    return PairArrays(
        z_0=z_0,
        z_1=z_1,
        gene_ids=gene_ids,
        directions=directions,
        attention_mask=attention_mask,
        labels=labels,
        target_barcodes=tuple(data.barcodes[index] for index in target_indices),
        control_barcodes=tuple(data.barcodes[index] for index in chosen_controls),
    )


def _make_adata(
    matrix: sp.csr_matrix, gene_names: Sequence[str], adapter: Any, *, batch_value: str, dataset_value: str
):
    try:
        import anndata
    except ImportError as exc:
        raise ImportError("anndata is required to encode Norman cells with ScVIAdapter") from exc
    setup_args = getattr(getattr(adapter, "model", None), "registry_", {}).get("setup_args", {})
    obs: dict[str, list[str]] = {}
    batch_key = setup_args.get("batch_key")
    if batch_key:
        obs[str(batch_key)] = [batch_value] * matrix.shape[0]
    for key in setup_args.get("categorical_covariate_keys") or []:
        obs[str(key)] = [dataset_value] * matrix.shape[0]
    adata = anndata.AnnData(X=matrix, obs=obs)
    adata.var_names = list(gene_names)
    return adata


def _metadata(
    *,
    scvi_model: Path,
    gene_names: Sequence[str],
    embedding_asset: Path,
    asset: Any,
    seed: int,
    split: str,
    pair_arrays: PairArrays,
    excluded_labels: Mapping[str, str],
) -> str:
    value = {
        "schema_version": LATENT_DAVF_DATA_SCHEMA_VERSION,
        "scvi": {
            "model_path": str(scvi_model.resolve()),
            "latent_dim": EXPECTED_LATENT_DIM,
            "num_genes": EXPECTED_SCVI_GENES,
            "gene_names": list(gene_names),
        },
        "embedding_asset": {
            "path": str(embedding_asset.resolve()),
            "vocab_size": int(asset.vocab_size),
            "embedding_dim": int(asset.embedding_dim),
            "manifest": dict(asset.manifest),
        },
        "dataset": {
            "name": "Norman GSE133344",
            "intervention_type": "KO",
            "direction_code": 0,
            "split": split,
            "seed": int(seed),
            "pair_direction": "control_to_perturbation",
            "control_matching": "same_gemgroup",
            "control_pool_policy": "disjoint_across_splits_reuse_within_split",
            "raw_feature_alignment": "scVI symbols; duplicate raw symbol columns summed",
            "perturbation_labels": sorted(set(pair_arrays.labels)),
            "excluded_perturbation_labels": dict(sorted(excluded_labels.items())),
            "target_cells": len(pair_arrays.target_barcodes),
            "control_pairs": len(pair_arrays.control_barcodes),
            "directions": {"KO": 0},
        },
    }
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_split(path: Path, pair_arrays: PairArrays, metadata_json: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.asarray(metadata_json),
        z_0=pair_arrays.z_0,
        z_1=pair_arrays.z_1,
        gene_ids=pair_arrays.gene_ids,
        directions=pair_arrays.directions,
        attention_mask=pair_arrays.attention_mask,
    )


def build_latent_pairs(
    dataset_dir: str | Path,
    scvi_model: str | Path,
    embedding_asset: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    device: Optional[str] = "cpu",
    batch_value: str = "norman",
    dataset_value: str = "norman",
    adapter: Any = None,
    asset: Any = None,
) -> dict[str, dict[str, Any]]:
    """Build train/val/test NPZ files and return their summaries."""

    data = load_norman_10x(dataset_dir)
    scvi_path = Path(scvi_model).expanduser().resolve()
    asset_path = Path(embedding_asset).expanduser().resolve()
    if asset is None:
        asset = load_perturbgen_embedding_asset(asset_path)
    if adapter is None:
        adapter = ScVIAdapter.from_trained_model(
            scvi_path,
            config=ScVIAdapterConfig(
                model_path=str(scvi_path),
                n_latent=EXPECTED_LATENT_DIM,
                device=device,
            ),
        )
    if CONTROL_LABEL not in data.labels:
        raise BuildDAVFLatentPairsError("Norman data contains no control cells")
    if not any(label != CONTROL_LABEL for label in data.labels):
        raise BuildDAVFLatentPairsError("Norman data contains no perturbation cells")
    gene_names = tuple(str(name) for name in adapter.gene_names)
    if hasattr(adapter, "validate_compatibility"):
        adapter.validate_compatibility(
            expected_latent_dim=EXPECTED_LATENT_DIM,
            expected_num_genes=EXPECTED_SCVI_GENES,
            expected_gene_names=gene_names,
        )
    aligned = align_to_scvi_gene_order(data, gene_names)
    resolvable_labels, excluded_labels = _split_resolvable_labels(data)
    token_by_label = _resolve_target_tokens(
        data,
        asset.gene_to_token,
        int(asset.vocab_size),
        labels=resolvable_labels,
    )
    adata = _make_adata(
        aligned,
        gene_names,
        adapter,
        batch_value=batch_value,
        dataset_value=dataset_value,
    )
    latent = np.asarray(adapter.encode(adata))
    if latent.shape != (len(data.barcodes), EXPECTED_LATENT_DIM):
        raise BuildDAVFLatentPairsError(
            f"ScVI encoder returned {latent.shape}; expected ({len(data.barcodes)}, {EXPECTED_LATENT_DIM})"
        )
    splits = split_perturbation_labels(
        resolvable_labels,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    output_root = Path(output_dir).expanduser().resolve()
    pair_rng = np.random.default_rng(seed)
    report: dict[str, dict[str, Any]] = {}
    control_pools_by_split = _allocate_control_pools(data, splits, pair_rng)
    for split, labels in splits.items():
        arrays = _build_pair_arrays(
            latent=latent,
            data=data,
            token_by_label=token_by_label,
            split_labels=labels,
            rng=pair_rng,
            control_pools=control_pools_by_split[split],
        )
        output_path = output_root / f"{split}.npz"
        _write_split(
            output_path,
            arrays,
            _metadata(
                scvi_model=scvi_path,
                gene_names=gene_names,
                embedding_asset=asset_path,
                asset=asset,
                seed=seed,
                split=split,
                pair_arrays=arrays,
                excluded_labels=excluded_labels,
            ),
        )
        report[split] = {
            "path": str(output_path),
            "samples": len(arrays.labels),
            "perturbation_labels": list(labels),
        }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Norman GSE133344 latent DAVF pairs")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--norman-dir", help="directory containing the four GSE133344 10x files")
    input_group.add_argument("--geo-root", help="GEO root containing the GSE133344 directory")
    parser.add_argument("--scvi-model", required=True, help="trained scVI model directory")
    parser.add_argument("--embedding-asset", required=True, help="verified PerturbGen embedding asset directory")
    parser.add_argument("--output-dir", "--output", dest="output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="cpu", help="scVI device, default cpu")
    parser.add_argument("--batch-value", default="norman")
    parser.add_argument("--dataset-value", default="norman")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_dir = args.norman_dir or str(Path(args.geo_root) / "GSE133344")
    report = build_latent_pairs(
        dataset_dir,
        args.scvi_model,
        args.embedding_asset,
        args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        device=args.device,
        batch_value=args.batch_value,
        dataset_value=args.dataset_value,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BuildDAVFLatentPairsError",
    "EXPECTED_LATENT_DIM",
    "EXPECTED_SCVI_GENES",
    "Norman10xData",
    "align_to_scvi_gene_order",
    "build_latent_pairs",
    "build_parser",
    "load_norman_10x",
    "main",
    "parse_guide_identity",
    "split_perturbation_labels",
]
