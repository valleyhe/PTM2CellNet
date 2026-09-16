"""Strict loader and donor-level direction summary for GSE 10x cohorts.

This module is intentionally separate from the PerturbGen paired-donor
preparation path.  A public GSE normal/disease cohort supplies observational
state labels, not causal KO/KD labels.  The resulting AnnData is therefore a
state-evidence asset: it can train a dataset-specific scVI model and produce
independent disease-direction evidence, but it must not be converted into a
KO/KD latent-pair dataset by inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import gzip
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread
from scipy.stats import ttest_ind

from src.models.davf_checkpoint_contract import FORMAL_DAVF_NUM_GENES  # noqa: F401 - re-exported contract constant
from src.models.gene_vocabulary import normalize_ensembl_id


GSE_NORMAL_DISEASE_SCHEMA_VERSION = "ptm2cellnet.gse-normal-disease.v1"
GSE_COUNTS_LAYER = "counts"
DEFAULT_TARGET_COUNT_SCALE = 10_000.0


class GSENormalDiseaseError(ValueError):
    """Raised when a GSE normal/disease input violates the data contract."""


@dataclass(frozen=True)
class GSE10xSample:
    """One sample-level 10x matrix and its explicit state assignment."""

    sample: str
    donor: str
    state: str
    accession: str
    matrix_path: Path
    features_path: Path
    barcodes_path: Path
    tissue: str = ""

    def __post_init__(self) -> None:
        for field_name in ("sample", "donor", "state", "accession"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} must not be empty")
            object.__setattr__(self, field_name, value)
        if self.state not in {"normal", "disease"}:
            raise ValueError("state must be 'normal' or 'disease'")
        for field_name in ("matrix_path", "features_path", "barcodes_path"):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)).expanduser().resolve())
        object.__setattr__(self, "tissue", str(self.tissue).strip())


@dataclass(frozen=True)
class GSEPreparationResult:
    """Prepared AnnData and the provenance report written beside it."""

    adata: Any
    report: dict[str, Any]


@dataclass(frozen=True)
class _LoadedSample:
    spec: GSE10xSample
    matrix: sp.csr_matrix
    gene_ids: tuple[str, ...]
    gene_symbols: tuple[str, ...]
    barcodes: tuple[str, ...]
    raw_n_cells: int


_GSE10X_FILE_RE = re.compile(
    r"^(?P<accession>GSM\d+)_(?P<label>.+)_(?P<kind>matrix|features|barcodes)\."
    r"(?P<extension>mtx|tsv)(?P<compressed>\.gz)?$",
    re.IGNORECASE,
)


def _normalise_sample_label(value: Any) -> str:
    """Map labels such as ``HC-1`` and ``HC1`` to one comparison key."""

    text = str(value).strip()
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _require_file(path: Path, *, description: str) -> None:
    if not path.is_file():
        raise GSENormalDiseaseError(f"missing {description}: {path}")


def _read_barcodes(path: Path) -> tuple[str, ...]:
    _require_file(path, description="10x barcode file")
    try:
        with _open_text(path) as handle:
            values = tuple(row[0].strip() for row in csv.reader(handle, delimiter="\t") if row and row[0].strip())
    except (OSError, UnicodeError, csv.Error) as exc:
        raise GSENormalDiseaseError(f"cannot read barcode file {path}: {exc}") from exc
    if not values:
        raise GSENormalDiseaseError(f"10x barcode file is empty: {path}")
    if len(set(values)) != len(values):
        raise GSENormalDiseaseError(f"10x barcode file contains duplicate barcodes: {path}")
    return values


def _read_features(path: Path) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    """Read Gene Expression features and retain their original matrix rows."""

    _require_file(path, description="10x feature file")
    rows: list[list[str]] = []
    try:
        with _open_text(path) as handle:
            rows = [[cell.strip() for cell in row] for row in csv.reader(handle, delimiter="\t")]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise GSENormalDiseaseError(f"cannot read feature file {path}: {exc}") from exc
    rows = [row for row in rows if row and row[0]]
    if not rows:
        raise GSENormalDiseaseError(f"10x feature file is empty: {path}")

    if any(len(row) >= 3 for row in rows):
        selected = [(index, row) for index, row in enumerate(rows) if len(row) < 3 or row[2] == "Gene Expression"]
    else:
        selected = list(enumerate(rows))
    if not selected:
        raise GSENormalDiseaseError(f"feature file has no Gene Expression rows: {path}")

    gene_ids: list[str] = []
    gene_symbols: list[str] = []
    row_indices: list[int] = []
    for row_index, row in selected:
        if len(row) < 2 or not row[0] or not row[1]:
            raise GSENormalDiseaseError(f"feature row {row_index + 1} must contain gene id and symbol: {path}")
        try:
            gene_ids.append(normalize_ensembl_id(row[0]))
        except ValueError as exc:
            raise GSENormalDiseaseError(
                f"feature row {row_index + 1} has invalid Ensembl id {row[0]!r}: {path}"
            ) from exc
        gene_symbols.append(row[1])
        row_indices.append(row_index)
    if len(set(gene_ids)) != len(gene_ids):
        raise GSENormalDiseaseError(f"feature file contains duplicate Ensembl ids: {path}")
    return tuple(gene_ids), tuple(gene_symbols), tuple(row_indices)


def _validate_counts(matrix: sp.csr_matrix, *, source: str) -> None:
    values = np.asarray(matrix.data)
    if values.size == 0:
        raise GSENormalDiseaseError(f"{source} count matrix has no non-zero entries")
    if not np.isfinite(values).all() or (values < 0).any():
        raise GSENormalDiseaseError(f"{source} count matrix must contain finite non-negative values")
    if not np.allclose(values, np.round(values), atol=1e-6):
        raise GSENormalDiseaseError(f"{source} count matrix must contain integer-like raw counts")
    if np.any(np.asarray(matrix.sum(axis=1)).ravel() <= 0):
        raise GSENormalDiseaseError(f"{source} contains a cell with zero raw library size")


def _read_sample(
    spec: GSE10xSample,
    *,
    selected_barcodes: Sequence[str] | None = None,
) -> _LoadedSample:
    """Read one 10x sample and retain only explicitly annotated barcodes.

    GEO 10x supplementary matrices are often unfiltered: their barcode file
    can contain hundreds of thousands of barcodes while the accompanying
    annotation table covers only the cells that passed the study's annotation
    pipeline.  Those unannotated barcodes are not valid observations for a
    cell-type/state analysis and must be excluded before library-size checks.
    """

    for path, description in (
        (spec.matrix_path, "10x matrix file"),
        (spec.features_path, "10x feature file"),
        (spec.barcodes_path, "10x barcode file"),
    ):
        _require_file(path, description=description)
    gene_ids, gene_symbols, row_indices = _read_features(spec.features_path)
    barcodes = _read_barcodes(spec.barcodes_path)
    try:
        raw = mmread(str(spec.matrix_path))
    except (OSError, TypeError, ValueError) as exc:
        raise GSENormalDiseaseError(f"cannot parse 10x matrix {spec.matrix_path}: {exc}") from exc
    matrix = sp.csr_matrix(raw, dtype=np.float32)
    expected_shape = (len(row_indices), len(barcodes))
    if matrix.shape[0] != len(row_indices) or matrix.shape[1] != len(barcodes):
        raise GSENormalDiseaseError(
            f"10x matrix shape {matrix.shape} does not match feature/barcode rows "
            f"({len(row_indices)}, {len(barcodes)}): {spec.matrix_path}"
        )
    matrix = matrix[list(row_indices), :].T.tocsr()

    raw_n_cells = len(barcodes)
    if selected_barcodes is not None:
        requested = tuple(str(value).strip() for value in selected_barcodes)
        if not requested or any(not value for value in requested):
            raise GSENormalDiseaseError(f"annotated barcode selection is empty for sample {spec.sample}")
        if len(set(requested)) != len(requested):
            raise GSENormalDiseaseError(f"annotation contains duplicate barcodes for sample {spec.sample}")
        barcode_to_index = {barcode: index for index, barcode in enumerate(barcodes)}
        missing = [barcode for barcode in requested if barcode not in barcode_to_index]
        if missing:
            raise GSENormalDiseaseError(
                f"GSE annotation references {len(missing)} missing barcodes for sample {spec.sample}; "
                f"examples={missing[:3]}"
            )
        selected_set = set(requested)
        selected_indices = [index for index, barcode in enumerate(barcodes) if barcode in selected_set]
        matrix = matrix[selected_indices, :].tocsr()
        barcodes = tuple(barcodes[index] for index in selected_indices)

    if matrix.shape != (len(barcodes), len(gene_ids)):
        raise GSENormalDiseaseError(
            f"filtered cell-by-gene matrix has shape {matrix.shape}, expected ({len(barcodes)}, {len(gene_ids)})"
        )
    _validate_counts(matrix, source=spec.sample)
    return _LoadedSample(spec, matrix, gene_ids, gene_symbols, barcodes, raw_n_cells)


def discover_gse10x_files(raw_dir: str | Path) -> dict[str, dict[str, Path | str]]:
    """Discover per-sample GSE 10x files without guessing missing members."""

    root = Path(raw_dir).expanduser().resolve()
    if not root.is_dir():
        raise GSENormalDiseaseError(f"GSE raw directory not found: {root}")
    discovered: dict[str, dict[str, Path | str]] = {}
    for path in sorted(root.iterdir()):
        match = _GSE10X_FILE_RE.fullmatch(path.name)
        if match is None:
            continue
        key = _normalise_sample_label(match.group("label"))
        kind = match.group("kind").lower()
        if key in discovered and kind in discovered[key]:
            raise GSENormalDiseaseError(f"duplicate {kind} file for GSE sample {key}: {path}")
        discovered.setdefault(key, {})[kind] = path
        discovered[key]["sample_label"] = match.group("label")
        discovered[key]["accession"] = match.group("accession").upper()
    if not discovered:
        raise GSENormalDiseaseError(f"no per-sample GSE 10x files found in {root}")
    return discovered


def build_gse_sample_specs(
    raw_dir: str | Path,
    *,
    normal_samples: Sequence[str],
    disease_samples: Sequence[str],
) -> tuple[GSE10xSample, ...]:
    """Build explicit sample specs from discovered files and state lists."""

    discovered = discover_gse10x_files(raw_dir)
    normal_keys = tuple(_normalise_sample_label(value) for value in normal_samples if str(value).strip())
    disease_keys = tuple(_normalise_sample_label(value) for value in disease_samples if str(value).strip())
    if not normal_keys or not disease_keys:
        raise GSENormalDiseaseError("normal_samples and disease_samples must both be non-empty")
    overlap = set(normal_keys) & set(disease_keys)
    if overlap:
        raise GSENormalDiseaseError(f"a sample cannot be both normal and disease: {sorted(overlap)}")

    specs: list[GSE10xSample] = []
    for state, keys in (("normal", normal_keys), ("disease", disease_keys)):
        if len(set(keys)) != len(keys):
            raise GSENormalDiseaseError(f"duplicate {state} sample labels: {keys}")
        for key in keys:
            files = discovered.get(key)
            if files is None:
                raise GSENormalDiseaseError(f"requested {state} sample {key!r} was not found in {raw_dir}")
            missing = [kind for kind in ("matrix", "features", "barcodes") if kind not in files]
            if missing:
                raise GSENormalDiseaseError(f"sample {key!r} is missing 10x files: {missing}")
            label = str(files["sample_label"])
            accession = str(files["accession"])
            specs.append(
                GSE10xSample(
                    sample=key,
                    donor=key,
                    state=state,
                    accession=accession,
                    matrix_path=Path(str(files["matrix"])),
                    features_path=Path(str(files["features"])),
                    barcodes_path=Path(str(files["barcodes"])),
                    # The filename label is a sample identifier, not tissue
                    # metadata.  Do not silently infer biological attributes.
                    tissue="",
                )
            )
    return tuple(specs)


def _read_annotation(path: str | Path) -> pd.DataFrame:
    annotation_path = Path(path).expanduser().resolve()
    _require_file(annotation_path, description="GSE cell annotation")
    try:
        annotation = pd.read_csv(annotation_path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise GSENormalDiseaseError(f"cannot read GSE cell annotation {annotation_path}: {exc}") from exc
    required = {"sample", "cell_id", "annotation"}
    missing = sorted(required - set(annotation.columns))
    if missing:
        raise GSENormalDiseaseError(f"GSE annotation is missing columns: {missing}")
    annotation = annotation.copy()
    if annotation[["sample", "cell_id", "annotation"]].isna().any(axis=None):
        raise GSENormalDiseaseError("GSE annotation contains missing sample, cell_id or annotation values")
    annotation["sample"] = annotation["sample"].astype(str).map(_normalise_sample_label)
    annotation["cell_id"] = annotation["cell_id"].astype(str).str.strip()
    annotation["annotation"] = annotation["annotation"].astype(str).str.strip()
    if annotation[["sample", "cell_id", "annotation"]].eq("").any(axis=None):
        raise GSENormalDiseaseError("GSE annotation contains empty sample, cell_id or annotation values")
    duplicate_mask = annotation.duplicated(["sample", "cell_id"], keep=False)
    ambiguous_by_sample: dict[str, list[str]] = {}
    if duplicate_mask.any():
        duplicate_rows = annotation.loc[duplicate_mask]
        annotation_counts = duplicate_rows.groupby(["sample", "cell_id"], sort=False)["annotation"].nunique()
        ambiguous_keys = {
            (str(sample), str(cell_id)) for (sample, cell_id), count in annotation_counts.items() if int(count) > 1
        }
        if ambiguous_keys:
            for sample, cell_id in sorted(ambiguous_keys):
                ambiguous_by_sample.setdefault(sample, []).append(cell_id)
            annotation = annotation.loc[
                [key not in ambiguous_keys for key in zip(annotation["sample"], annotation["cell_id"], strict=True)]
            ]
        # Repeated identical labels carry no extra information and can be
        # reduced deterministically after conflicting cells are excluded.
        annotation = annotation.drop_duplicates(["sample", "cell_id"], keep="first").reset_index(drop=True)
    annotation.attrs["ambiguous_cells"] = {
        sample: list(sorted(cell_ids)) for sample, cell_ids in sorted(ambiguous_by_sample.items())
    }
    return cast(pd.DataFrame, annotation)


def _select_genes(
    loaded: Sequence[_LoadedSample],
    *,
    candidate_gene_ids: Iterable[str] | None,
    n_genes: int,
) -> tuple[str, ...]:
    if n_genes <= 0:
        raise GSENormalDiseaseError("n_genes must be positive")
    common = set(loaded[0].gene_ids)
    for sample in loaded[1:]:
        common.intersection_update(sample.gene_ids)
    if candidate_gene_ids is not None:
        try:
            candidates = {normalize_ensembl_id(value) for value in candidate_gene_ids}
        except ValueError as exc:
            raise GSENormalDiseaseError(f"candidate gene vocabulary contains an invalid Ensembl id: {exc}") from exc
        common.intersection_update(candidates)
    if len(common) < n_genes:
        raise GSENormalDiseaseError(f"only {len(common)} genes are eligible; {n_genes} are required")

    first_order = {gene_id: index for index, gene_id in enumerate(loaded[0].gene_ids)}
    total_detection: dict[str, float] = {gene_id: 0.0 for gene_id in common}
    for sample in loaded:
        index_by_gene = {gene_id: index for index, gene_id in enumerate(sample.gene_ids)}
        selected_indices = [index_by_gene[gene_id] for gene_id in common]
        detected = np.asarray((sample.matrix[:, selected_indices] > 0).sum(axis=0)).ravel()
        for gene_id, value in zip(common, detected, strict=True):
            total_detection[gene_id] += float(value)
    ordered = sorted(common, key=lambda gene_id: (-total_detection[gene_id], first_order[gene_id]))
    return tuple(ordered[:n_genes])


def _annotation_for_sample(annotation: pd.DataFrame, sample: GSE10xSample, barcodes: Sequence[str]) -> list[str]:
    subset = annotation[annotation["sample"] == _normalise_sample_label(sample.sample)]
    by_barcode = dict(zip(subset["cell_id"], subset["annotation"], strict=True))
    missing = [barcode for barcode in barcodes if barcode not in by_barcode]
    if missing:
        raise GSENormalDiseaseError(
            f"GSE annotation is missing {len(missing)} cells for sample {sample.sample}; examples={missing[:3]}"
        )
    return [by_barcode[barcode] for barcode in barcodes]


def prepare_gse_normal_disease(
    samples: Sequence[GSE10xSample],
    *,
    annotation_path: str | Path,
    candidate_gene_ids: Iterable[str] | None = None,
    n_genes: int = FORMAL_DAVF_NUM_GENES,
    dataset_accession: str = "GSE214695",
    embedding_asset_path: str | Path | None = None,
    embedding_manifest: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    min_donors: int = 3,
) -> GSEPreparationResult:
    """Prepare a strict observation-only normal/disease AnnData artifact."""

    if not samples:
        raise GSENormalDiseaseError("at least one normal and one disease sample are required")
    if min_donors < 1:
        raise GSENormalDiseaseError("min_donors must be positive")
    sample_names = [_normalise_sample_label(sample.sample) for sample in samples]
    if len(set(sample_names)) != len(sample_names):
        raise GSENormalDiseaseError("sample labels must be unique")
    donor_names = [sample.donor for sample in samples]
    if len(set(donor_names)) != len(donor_names):
        raise GSENormalDiseaseError("each GSE sample must map to one unique donor in this loader")
    normal_donors = {sample.donor for sample in samples if sample.state == "normal"}
    disease_donors = {sample.donor for sample in samples if sample.state == "disease"}
    if len(normal_donors) < min_donors or len(disease_donors) < min_donors:
        raise GSENormalDiseaseError(
            f"at least {min_donors} normal and disease donors are required; "
            f"got {len(normal_donors)} and {len(disease_donors)}"
        )

    annotation = _read_annotation(annotation_path)
    ambiguous_cells = {
        str(sample): [str(cell_id) for cell_id in cell_ids]
        for sample, cell_ids in annotation.attrs.get("ambiguous_cells", {}).items()
    }
    loaded_samples: list[_LoadedSample] = []
    for sample in samples:
        sample_annotation = annotation[annotation["sample"] == _normalise_sample_label(sample.sample)]
        if sample_annotation.empty:
            raise GSENormalDiseaseError(f"GSE annotation has no cells for sample {sample.sample}")
        loaded_samples.append(_read_sample(sample, selected_barcodes=tuple(sample_annotation["cell_id"])))
    loaded = tuple(loaded_samples)
    selected_gene_ids = _select_genes(loaded, candidate_gene_ids=candidate_gene_ids, n_genes=n_genes)
    selected_set = set(selected_gene_ids)
    first = loaded[0]
    common_gene_count = len(set.intersection(*(set(sample.gene_ids) for sample in loaded)))
    first_symbols = dict(zip(first.gene_ids, first.gene_symbols, strict=True))
    matrices: list[sp.csr_matrix] = []
    obs_rows: list[dict[str, str]] = []
    for loaded_sample in loaded:
        index_by_gene = {gene_id: index for index, gene_id in enumerate(loaded_sample.gene_ids)}
        indices = [index_by_gene[gene_id] for gene_id in selected_gene_ids]
        matrices.append(loaded_sample.matrix[:, indices].tocsr())
        cell_types = _annotation_for_sample(annotation, loaded_sample.spec, loaded_sample.barcodes)
        for _barcode, cell_type in zip(loaded_sample.barcodes, cell_types, strict=True):
            obs_rows.append(
                {
                    "sample": loaded_sample.spec.sample,
                    "sample_accession": loaded_sample.spec.accession,
                    "donor": loaded_sample.spec.donor,
                    "state": loaded_sample.spec.state,
                    "cell_type": cell_type,
                    "tissue": loaded_sample.spec.tissue,
                    "dataset": dataset_accession,
                    "davf_batch": loaded_sample.spec.sample,
                }
            )
    combined = sp.vstack(matrices, format="csr", dtype=np.float32)
    if combined.shape[1] != len(selected_gene_ids):
        raise GSENormalDiseaseError("combined GSE matrix does not match selected gene axis")
    if len(obs_rows) != combined.shape[0]:
        raise GSENormalDiseaseError("GSE observation metadata does not match combined matrix rows")

    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise ImportError("anndata is required to prepare a GSE normal/disease cohort") from exc

    obs_index = [f"{row['sample']}:{index}" for index, row in enumerate(obs_rows)]
    obs = pd.DataFrame(obs_rows, index=obs_index)
    var = pd.DataFrame(
        {
            "ensembl_id": list(selected_gene_ids),
            "gene_symbol": [first_symbols[gene_id] for gene_id in selected_gene_ids],
        },
        index=list(selected_gene_ids),
    )
    adata = ad.AnnData(X=combined, obs=obs, var=var)
    adata.layers[GSE_COUNTS_LAYER] = combined.copy()
    adata.obs["n_counts"] = np.asarray(combined.sum(axis=1)).ravel().astype(np.float64)
    adata.obs["davf_batch"] = adata.obs["davf_batch"].astype("category")
    shared_donors = sorted(normal_donors & disease_donors)
    sample_records = [
        {
            "sample": sample.spec.sample,
            "donor": sample.spec.donor,
            "state": sample.spec.state,
            "accession": sample.spec.accession,
            "n_cells": int(sample.matrix.shape[0]),
            "raw_n_cells": int(sample.raw_n_cells),
            "excluded_unannotated_cells": int(sample.raw_n_cells - sample.matrix.shape[0]),
            "tissue": sample.spec.tissue,
        }
        for sample in loaded
    ]
    preparation = {
        "schema_version": GSE_NORMAL_DISEASE_SCHEMA_VERSION,
        "dataset_accession": dataset_accession,
        "states": {"normal": "normal", "disease": "disease"},
        "counts_layer": GSE_COUNTS_LAYER,
        "gene_names": list(selected_gene_ids),
        "num_genes": len(selected_gene_ids),
        # A mapping is H5AD-serializable; a list of dictionaries is not.
        "samples": {record["sample"]: record for record in sample_records},
        "normal_donors": sorted(normal_donors),
        "disease_donors": sorted(disease_donors),
        "shared_donors": shared_donors,
        "ambiguous_annotation_cells": ambiguous_cells,
        "observational_only": True,
        "formal_perturbgen_ready": False,
        "formal_perturbgen_reason": "GSE normal/disease data has no causal perturbation labels",
        "embedding_asset": (
            {
                "path": str(Path(embedding_asset_path).expanduser().resolve()),
                "manifest": dict(embedding_manifest or {}),
            }
            if embedding_asset_path is not None
            else None
        ),
    }
    adata.uns["gse_normal_disease"] = preparation

    report = {
        "schema_version": GSE_NORMAL_DISEASE_SCHEMA_VERSION,
        "dataset_accession": dataset_accession,
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "gene_names": list(selected_gene_ids),
        "counts_layer": GSE_COUNTS_LAYER,
        "normal_donors": sorted(normal_donors),
        "disease_donors": sorted(disease_donors),
        "shared_donors": shared_donors,
        "ambiguous_annotation_cells": ambiguous_cells,
        "observational_only": True,
        "formal_perturbgen_ready": False,
        "samples": sample_records,
        "annotation": str(Path(annotation_path).expanduser().resolve()),
        "embedding_asset": preparation["embedding_asset"],
        "selected_from_common_genes": common_gene_count,
        "selected_asset_genes": len(selected_set),
    }
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(destination, compression=None)
        report["output"] = str(destination)
        report_path = destination.with_suffix(".manifest.json")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["manifest"] = str(report_path)
    return GSEPreparationResult(adata=adata, report=report)


def _validate_direction_input(adata: Any, *, counts_layer: str) -> tuple[Any, pd.DataFrame, tuple[str, ...]]:
    if counts_layer not in adata.layers:
        raise GSENormalDiseaseError(
            f"adata.layers[{counts_layer!r}] is required; GSE direction evidence must use raw counts"
        )
    counts = adata.layers[counts_layer]
    if getattr(counts, "shape", None) != adata.shape:
        raise GSENormalDiseaseError("GSE counts layer shape does not match AnnData")
    if hasattr(counts, "tocsr"):
        counts = counts.tocsr()
    values = np.asarray(counts.data if sp.issparse(counts) else counts)
    if values.size == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise GSENormalDiseaseError("GSE counts layer must contain finite non-negative values")
    if not np.allclose(values, np.round(values), atol=1e-6):
        raise GSENormalDiseaseError("GSE counts layer must contain integer-like raw counts")
    required_obs = {"cell_type", "state", "donor"}
    missing_obs = sorted(required_obs - set(adata.obs.columns))
    if missing_obs:
        raise GSENormalDiseaseError(f"GSE AnnData is missing required obs columns: {missing_obs}")
    if "ensembl_id" not in adata.var.columns:
        raise GSENormalDiseaseError("GSE AnnData is missing var['ensembl_id']")
    gene_names = tuple(normalize_ensembl_id(value) for value in adata.var["ensembl_id"])
    if len(set(gene_names)) != len(gene_names):
        raise GSENormalDiseaseError("GSE AnnData contains duplicate Ensembl ids")
    return counts, adata.obs.copy(), gene_names


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    clipped = np.clip(values, 0.0, 1.0)
    order = np.argsort(clipped, kind="mergesort")
    ranks = np.arange(1, len(clipped) + 1, dtype=np.float64)
    adjusted_sorted = clipped[order] * len(clipped) / ranks
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    result = np.empty_like(adjusted_sorted)
    result[order] = np.clip(adjusted_sorted, 0.0, 1.0)
    return result


def _donor_log2_means(counts: Any, indices: Iterable[int] | np.ndarray) -> np.ndarray:
    subset = counts[list(indices)]
    library = np.asarray(subset.sum(axis=1)).ravel().astype(np.float64)
    if np.any(library <= 0):
        raise GSENormalDiseaseError("direction evidence contains a zero-library cell")
    if sp.issparse(subset):
        normalized = subset.astype(np.float64).tocsr().multiply(DEFAULT_TARGET_COUNT_SCALE / library[:, None])
        normalized.data = np.log2(normalized.data + 1.0)
        return np.asarray(normalized.mean(axis=0)).ravel()
    normalized_dense = np.asarray(subset, dtype=np.float64) * (DEFAULT_TARGET_COUNT_SCALE / library[:, None])
    return np.asarray(np.log2(normalized_dense + 1.0).mean(axis=0))


def _donor_pseudobulk_log2(counts: Any, indices: Iterable[int] | np.ndarray) -> np.ndarray:
    """Donor pseudobulk profile: sum counts over cells, then normalize and log2.

    Unlike :func:`_donor_log2_means` (per-cell normalization averaged within a
    donor), this aggregation carries donor-level library composition into the
    test statistic; the estimand is chosen once per frozen research design and
    recorded in the DEG manifest, never switched silently.
    """

    subset = counts[list(indices)]
    bulk = np.asarray(subset.sum(axis=0)).ravel().astype(np.float64)
    library = float(bulk.sum())
    if library <= 0:
        raise GSENormalDiseaseError("direction evidence contains a zero-library donor pseudobulk")
    return np.log2(bulk * (DEFAULT_TARGET_COUNT_SCALE / library) + 1.0)


def summarize_normal_disease_directions(
    adata: Any,
    *,
    cell_types: Sequence[str] | None = None,
    counts_layer: str = GSE_COUNTS_LAYER,
    normal_state: str = "normal",
    disease_state: str = "disease",
    min_donors: int = 3,
    direction_epsilon: float = 1e-6,
) -> pd.DataFrame:
    """Calculate donor-level disease-vs-normal directions for each cell type."""

    if min_donors < 1:
        raise GSENormalDiseaseError("min_donors must be positive")
    if normal_state == disease_state:
        raise GSENormalDiseaseError("normal_state and disease_state must differ")
    counts, obs, gene_names = _validate_direction_input(adata, counts_layer=counts_layer)
    obs = obs.copy()
    for column in ("cell_type", "state", "donor"):
        obs[column] = obs[column].astype(str).str.strip()
        if obs[column].eq("").any():
            raise GSENormalDiseaseError(f"GSE AnnData obs[{column!r}] contains empty values")
    requested = tuple(str(value).strip() for value in cell_types or ())
    available = tuple(sorted(set(obs["cell_type"])))
    selected_types = requested or available
    unknown = sorted(set(selected_types) - set(available))
    if unknown:
        raise GSENormalDiseaseError(f"requested cell types are absent from GSE AnnData: {unknown}")

    var_symbols = (
        tuple(str(value) for value in adata.var["gene_symbol"]) if "gene_symbol" in adata.var.columns else gene_names
    )
    rows: list[pd.DataFrame] = []
    skipped: dict[str, str] = {}
    for cell_type in selected_types:
        cell_mask = obs["cell_type"].eq(cell_type).to_numpy()
        normal_donors = tuple(sorted(set(obs.loc[cell_mask & obs["state"].eq(normal_state), "donor"])))
        disease_donors = tuple(sorted(set(obs.loc[cell_mask & obs["state"].eq(disease_state), "donor"])))
        if len(normal_donors) < min_donors or len(disease_donors) < min_donors:
            skipped[cell_type] = f"normal_donors={len(normal_donors)}, disease_donors={len(disease_donors)}"
            continue
        donor_means: dict[tuple[str, str], np.ndarray] = {}
        for state, donors in ((normal_state, normal_donors), (disease_state, disease_donors)):
            for donor in donors:
                indices = np.flatnonzero(
                    cell_mask & obs["state"].eq(state).to_numpy() & obs["donor"].eq(donor).to_numpy()
                )
                if len(indices) == 0:
                    raise GSENormalDiseaseError(f"donor {donor!r} has no cells for cell type {cell_type!r}")
                donor_means[(state, donor)] = _donor_log2_means(counts, indices)
        normal_matrix = np.vstack([donor_means[(normal_state, donor)] for donor in normal_donors])
        disease_matrix = np.vstack([donor_means[(disease_state, donor)] for donor in disease_donors])
        delta = disease_matrix.mean(axis=0) - normal_matrix.mean(axis=0)
        test = ttest_ind(disease_matrix, normal_matrix, axis=0, equal_var=False, nan_policy="omit")
        p_values = np.asarray(test.pvalue, dtype=np.float64)
        invalid = ~np.isfinite(p_values)
        p_values[invalid] = np.where(np.isclose(delta[invalid], 0.0), 1.0, 0.0)
        fdr = _bh_adjust(p_values)
        pairwise_delta = disease_matrix[:, None, :] - normal_matrix[None, :, :]
        aggregate_sign = np.sign(delta)
        consistency = np.mean(np.sign(pairwise_delta) == aggregate_sign[None, None, :], axis=(0, 1))
        direction = np.where(delta > direction_epsilon, "up", np.where(delta < -direction_epsilon, "down", "neutral"))
        rows.append(
            pd.DataFrame(
                {
                    "cell_type": cell_type,
                    "ensembl_id": gene_names,
                    "gene_symbol": var_symbols,
                    "log2fc": delta.astype(np.float64),
                    "p_value": p_values,
                    "fdr": fdr,
                    "observed_direction": direction,
                    "donor_consistency": consistency.astype(np.float64),
                    "normal_donors": ",".join(normal_donors),
                    "disease_donors": ",".join(disease_donors),
                    "n_normal_donors": len(normal_donors),
                    "n_disease_donors": len(disease_donors),
                    "effect_scale": "donor_mean_log2_normalized_counts",
                }
            )
        )
    if not rows:
        detail = "; ".join(f"{key}: {value}" for key, value in sorted(skipped.items()))
        raise GSENormalDiseaseError(f"no cell type has at least {min_donors} donors in both states; {detail}")
    result = pd.concat(rows, ignore_index=True)
    result.attrs["skipped_cell_types"] = skipped
    result.attrs["counts_layer"] = counts_layer
    result.attrs["normal_state"] = normal_state
    result.attrs["disease_state"] = disease_state
    return result


__all__ = [
    "FORMAL_DAVF_NUM_GENES",
    "GSE10xSample",
    "GSE_COUNTS_LAYER",
    "GSE_NORMAL_DISEASE_SCHEMA_VERSION",
    "GSEPreparationResult",
    "GSENormalDiseaseError",
    "build_gse_sample_specs",
    "discover_gse10x_files",
    "prepare_gse_normal_disease",
    "summarize_normal_disease_directions",
]
