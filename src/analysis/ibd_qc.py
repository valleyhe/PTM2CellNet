"""Sample-wise loading, QC, and scVI integration for the IBD datasets.

This module deliberately processes one GEO library at a time.  The four
datasets use three different processed-count layouts (loose 10x triplets,
nested 10x tarballs, and 10x HDF5 files), so a single global ``read`` call
would make provenance and failure recovery difficult.
"""

from __future__ import annotations

import gzip
import io
import logging
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import IO, Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

from .ibd_dataset import load_metadata
from src.utils.dependency_check import require_extras


LOGGER = logging.getLogger(__name__)


def _require_anndata():
    require_extras(["anndata"], feature="IBD matrix processing")
    import anndata as ad

    return ad


def _require_scanpy():
    require_extras(["scanpy"], feature="GSE282122/scVI integration")
    import scanpy as sc

    return sc


def _strip_gzip_suffix(name: str) -> str:
    return name[:-3] if name.endswith(".gz") else name


def _read_gzip_text(stream: IO[bytes]) -> str:
    with gzip.GzipFile(fileobj=stream, mode="rb") as handle:
        return handle.read().decode("utf-8", errors="replace")


def _read_tsv_gzip(path_or_stream: Path | IO[bytes]) -> pd.DataFrame:
    if isinstance(path_or_stream, Path):
        return pd.read_csv(path_or_stream, sep="\t", header=None, dtype=str, compression="gzip")
    text = _read_gzip_text(path_or_stream)
    return pd.read_csv(io.StringIO(text), sep="\t", header=None, dtype=str)


def _read_matrix_gzip(path_or_stream: Path | IO[bytes]) -> sparse.csr_matrix:
    if isinstance(path_or_stream, Path):
        with gzip.open(path_or_stream, "rb") as handle:
            matrix = mmread(handle)
    else:
        with gzip.GzipFile(fileobj=path_or_stream, mode="rb") as handle:
            matrix = mmread(handle)
    return sparse.csr_matrix(matrix, dtype=np.float32)


def _feature_columns(features: pd.DataFrame) -> Tuple[pd.Index, pd.Index, pd.Series]:
    if features.shape[1] == 0:
        raise ValueError("10x features file is empty")
    gene_ids = features.iloc[:, 0].astype(str).str.strip()
    gene_symbols = features.iloc[:, 1].astype(str).str.strip() if features.shape[1] >= 2 else gene_ids.copy()
    feature_type = (
        features.iloc[:, 2].astype(str).str.strip()
        if features.shape[1] >= 3
        else pd.Series("Gene Expression", index=features.index)
    )
    return pd.Index(gene_ids), pd.Index(gene_symbols), feature_type


def _make_unique(values: Iterable[Any]) -> pd.Index:
    """Make feature names unique using the same suffix convention as Scanpy."""

    counts: Dict[str, int] = {}
    result: List[str] = []
    for raw in values:
        value = str(raw) or "unknown_feature"
        count = counts.get(value, 0)
        result.append(value if count == 0 else f"{value}-{count}")
        counts[value] = count + 1
    return pd.Index(result)


def _record_obs(metadata_row: Mapping[str, Any], barcodes: Iterable[Any]) -> pd.DataFrame:
    sample_id = str(metadata_row["sample_id"])
    obs = pd.DataFrame(index=pd.Index([f"{sample_id}:{barcode}" for barcode in barcodes]))
    obs["barcode"] = list(map(str, barcodes))
    for key, value in metadata_row.items():
        if key in {"sample_title", "archive_member", "characteristics_json"}:
            continue
        obs[key] = value
    return obs


def _build_adata_from_components(
    *,
    matrix: sparse.spmatrix,
    barcodes: Sequence[str],
    features: pd.DataFrame,
    metadata_row: Mapping[str, Any],
    selected_positions: Optional[Sequence[int]] = None,
    cell_annotations: Optional[pd.DataFrame] = None,
):
    ad = _require_anndata()
    gene_ids, gene_symbols, feature_type = _feature_columns(features)
    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    barcodes_array = np.asarray(list(map(str, barcodes)), dtype=object)
    if matrix.shape[1] == len(barcodes_array):
        # Matrix Market files from the public 10x exports are normally
        # features x barcodes; AnnData stores cells x features.
        matrix = matrix.T.tocsr()
    elif matrix.shape[0] != len(barcodes_array):
        raise ValueError(
            f"10x matrix/barcode dimensions disagree: matrix={matrix.shape}, barcodes={len(barcodes_array)}"
        )
    if selected_positions is not None:
        positions = np.asarray(selected_positions, dtype=np.int64)
        matrix = matrix[positions, :].tocsr()
        barcodes_array = barcodes_array[positions]
    if matrix.shape[0] != len(barcodes_array):
        raise ValueError(f"Selected matrix/barcode dimensions disagree: {matrix.shape}")
    if matrix.shape[1] != len(gene_ids):
        raise ValueError(f"10x matrix/feature dimensions disagree: matrix={matrix.shape}, features={len(gene_ids)}")

    var_names = _make_unique(gene_ids)
    var = pd.DataFrame(index=var_names)
    var["gene_id"] = gene_ids.to_numpy()
    var["gene_symbol"] = gene_symbols.to_numpy()
    var["feature_type"] = feature_type.to_numpy()
    obs = _record_obs(metadata_row, barcodes_array)
    if cell_annotations is not None and not cell_annotations.empty:
        annotation = cell_annotations.reindex(barcodes_array)
        for column in annotation.columns:
            obs[column] = annotation[column].to_numpy()
    result = ad.AnnData(X=matrix, obs=obs, var=var)
    result.obs_names_make_unique()
    result.var_names_make_unique()
    return result


def _read_disk_10x(
    *,
    matrix_path: Path,
    barcode_path: Path,
    feature_path: Path,
    metadata_row: Mapping[str, Any],
    selected_positions: Optional[Sequence[int]] = None,
    cell_annotations: Optional[pd.DataFrame] = None,
):
    barcodes = _read_tsv_gzip(barcode_path).iloc[:, 0].astype(str).tolist()
    features = _read_tsv_gzip(feature_path)
    matrix = _read_matrix_gzip(matrix_path)
    return _build_adata_from_components(
        matrix=matrix,
        barcodes=barcodes,
        features=features,
        metadata_row=metadata_row,
        selected_positions=selected_positions,
        cell_annotations=cell_annotations,
    )


def _member_by_suffix(tar: tarfile.TarFile, gsm: str, suffix: str) -> tarfile.TarInfo:
    candidates = [
        member for member in tar.getmembers() if Path(member.name).name.startswith(gsm) and member.name.endswith(suffix)
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one {suffix} member for {gsm}, found {[m.name for m in candidates]}")
    return candidates[0]


def _extract_stream(tar: tarfile.TarFile, member: tarfile.TarInfo) -> IO[bytes]:
    stream = tar.extractfile(member)
    if stream is None:
        raise ValueError(f"tar member {member.name} has no byte stream")
    return stream


def _read_gse231993_sample(tar: tarfile.TarFile, row: Mapping[Any, Any]):
    gsm = str(row["GSM"])
    barcode_member = _member_by_suffix(tar, gsm, "-barcodes.tsv.gz")
    feature_member = _member_by_suffix(tar, gsm, "-features.tsv.gz")
    matrix_member = _member_by_suffix(tar, gsm, "-matrix.mtx.gz")
    with _extract_stream(tar, barcode_member) as handle:
        barcodes = _read_tsv_gzip(handle).iloc[:, 0].astype(str).tolist()
    with _extract_stream(tar, feature_member) as handle:
        features = _read_tsv_gzip(handle)
    with _extract_stream(tar, matrix_member) as handle:
        matrix = _read_matrix_gzip(handle)
    return _build_adata_from_components(
        matrix=matrix,
        barcodes=barcodes,
        features=features,
        metadata_row=row,
    )


def _read_gse266616_sample(tar: tarfile.TarFile, row: Mapping[Any, Any]):
    gsm = str(row["GSM"])
    outer_candidates = [member for member in tar.getmembers() if Path(member.name).name.startswith(gsm)]
    if len(outer_candidates) != 1:
        raise ValueError(f"Expected one nested archive for {gsm}, found {outer_candidates}")

    matrix: Optional[sparse.csr_matrix] = None
    features: Optional[pd.DataFrame] = None
    barcodes: Optional[List[str]] = None
    outer_stream = tar.extractfile(outer_candidates[0])
    if outer_stream is None:
        raise ValueError(f"Could not extract nested archive for {gsm}")
    with outer_stream:
        with tarfile.open(fileobj=outer_stream, mode="r|gz") as inner:
            for member in inner:
                if not member.isfile():
                    continue
                base = Path(member.name).name
                handle = inner.extractfile(member)
                if handle is None:
                    continue
                with handle:
                    if base == "barcodes.tsv.gz":
                        barcodes = _read_tsv_gzip(cast(IO[bytes], handle)).iloc[:, 0].astype(str).tolist()
                    elif base == "features.tsv.gz":
                        features = _read_tsv_gzip(cast(IO[bytes], handle))
                    elif base == "matrix.mtx.gz":
                        matrix = _read_matrix_gzip(cast(IO[bytes], handle))
    if matrix is None or features is None or barcodes is None:
        raise ValueError(f"Incomplete nested 10x archive for {gsm}")
    return _build_adata_from_components(
        matrix=matrix,
        barcodes=barcodes,
        features=features,
        metadata_row=row,
    )


def _read_gse282122_sample(tar: tarfile.TarFile, row: Mapping[Any, Any]):
    sc = _require_scanpy()
    archive_member = str(row["archive_member"])
    expected = f"filtered_processed_data/{archive_member}/filtered_feature_bc_matrix.h5"
    matches = [member for member in tar.getmembers() if member.name == expected]
    if len(matches) != 1:
        raise ValueError(f"Could not find GSE282122 H5 member: {expected}")
    handle = tar.extractfile(matches[0])
    if handle is None:
        raise ValueError(f"Could not extract GSE282122 H5 member: {expected}")
    with handle, tempfile.NamedTemporaryFile(suffix=".h5") as temp:
        shutil.copyfileobj(handle, temp)
        temp.flush()
        result = sc.read_10x_h5(temp.name, gex_only=True)
    # read_10x_h5 normally exposes gene_ids; use them as the cross-dataset key.
    original_symbols = result.var_names.astype(str).to_numpy()
    if "gene_ids" in result.var:
        gene_ids = result.var["gene_ids"].astype(str).to_numpy()
    else:
        gene_ids = original_symbols.copy()
    result.var["gene_symbol"] = original_symbols
    unique_gene_ids = _make_unique(gene_ids)
    result.var_names = unique_gene_ids
    result.var["gene_id"] = gene_ids
    result.var_names_make_unique()
    obs = result.obs.copy()
    obs.index = [f"{row['sample_id']}:{barcode}" for barcode in obs.index.astype(str)]
    for key, value in row.items():
        if key not in {"sample_title", "archive_member", "characteristics_json"}:
            obs[key] = value
    result.obs = obs
    result.X = sparse.csr_matrix(result.X, dtype=np.float32)
    return result


def _gse214695_annotations(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "GSE214695_cell_annotation.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing GSE214695 cell annotation: {path}")
    annotation = pd.read_csv(path, dtype=str)
    required = {"sample", "cell_id", "annotation", "nanostring_reference"}
    missing = required.difference(annotation.columns)
    if missing:
        raise ValueError(f"GSE214695 annotation missing columns: {sorted(missing)}")
    return cast(pd.DataFrame, annotation.set_index("cell_id", drop=False))


def _read_gse214695_sample(
    raw_dir: Path,
    row: Mapping[Any, Any],
    annotations: pd.DataFrame,
):
    matrix_path = raw_dir / str(row["archive_member"])
    barcode_path = matrix_path.with_name(matrix_path.name.replace("_matrix.mtx.gz", "_barcodes.tsv.gz"))
    feature_path = matrix_path.with_name(matrix_path.name.replace("_matrix.mtx.gz", "_features.tsv.gz"))
    for path in (matrix_path, barcode_path, feature_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing GSE214695 10x member: {path}")
    barcodes = _read_tsv_gzip(barcode_path).iloc[:, 0].astype(str).tolist()
    sample_label = str(row["sample_title"])
    sample_annotations = annotations[annotations["sample"] == sample_label].copy()
    if sample_annotations.empty:
        raise ValueError(f"No cell annotations for GSE214695 sample {sample_label}")
    positions_by_barcode = {barcode: index for index, barcode in enumerate(barcodes)}
    selected_barcodes = [
        barcode for barcode in sample_annotations["cell_id"].astype(str) if barcode in positions_by_barcode
    ]
    if not selected_barcodes:
        raise ValueError(f"No annotated barcodes found in GSE214695 sample {sample_label}")
    selected_positions = [positions_by_barcode[barcode] for barcode in selected_barcodes]
    annotation_frame = sample_annotations.set_index("cell_id")[["annotation", "nanostring_reference"]]
    return _read_disk_10x(
        matrix_path=matrix_path,
        barcode_path=barcode_path,
        feature_path=feature_path,
        metadata_row=row,
        selected_positions=selected_positions,
        cell_annotations=annotation_frame,
    )


def _selected_rows(metadata: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "core":
        selected = metadata[metadata["include_atlas"]]
    elif scope == "core_plus_validation":
        selected = metadata[metadata["include_atlas"] | (metadata["analysis_role"] == "validation")]
    elif scope == "validation":
        selected = metadata[metadata["analysis_role"] == "validation"]
    elif scope == "all":
        selected = metadata
    else:
        raise ValueError(f"Unknown IBD scope {scope!r}; use core, validation, core_plus_validation, or all")
    return cast(pd.DataFrame, selected.sort_values(["dataset", "GSM"], kind="stable").reset_index(drop=True))


def iter_sample_adatas(
    metadata: pd.DataFrame,
    raw_root: Path | str = "data/raw",
    scope: str = "core",
) -> Iterator[Tuple[Mapping[Any, Any], Any]]:
    """Yield one ``(metadata_row, AnnData)`` pair at a time."""

    rows = _selected_rows(metadata, scope)
    root = Path(raw_root)
    annotations = _gse214695_annotations(root / "gse214695")
    for dataset, dataset_rows in rows.groupby("dataset", sort=False):
        records = dataset_rows.to_dict(orient="records")
        if dataset == "GSE214695":
            for row in records:
                yield row, _read_gse214695_sample(root / "gse214695", row, annotations)
        elif dataset == "GSE231993":
            archive = root / "gse231993" / "GSE231993_RAW.tar"
            with tarfile.open(archive, mode="r:") as tar:
                for row in records:
                    yield row, _read_gse231993_sample(tar, row)
        elif dataset == "GSE282122":
            archive = root / "gse282122" / "GSE282122_filtered_processed_data.tar.gz"
            with tarfile.open(archive, mode="r:gz") as tar:
                for row in records:
                    yield row, _read_gse282122_sample(tar, row)
        elif dataset == "GSE266616":
            archive = root / "gse266616" / "GSE266616_RAW.tar"
            with tarfile.open(archive, mode="r:") as tar:
                for row in records:
                    yield row, _read_gse266616_sample(tar, row)
        else:
            raise ValueError(f"Unsupported IBD dataset: {dataset}")


def _mad(values: np.ndarray) -> float:
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _robust_bounds(values: np.ndarray) -> Tuple[float, float, float, float]:
    median = float(np.median(values))
    mad = _mad(values)
    if mad == 0 or not np.isfinite(mad):
        return median, mad, -np.inf, np.inf
    return median, mad, max(0.0, median - 3.0 * mad), median + 3.0 * mad


def _group_positions(values: pd.Series) -> Iterator[Tuple[str, np.ndarray]]:
    """Yield integer row positions rather than AnnData index labels."""

    array = values.astype(str).to_numpy()
    for value in pd.unique(array):
        yield str(value), np.flatnonzero(array == value)


def add_sample_qc_metrics(adata: Any) -> Dict[str, Dict[str, float]]:
    """Calculate sample-specific robust QC metrics without dense conversion."""

    matrix = sparse.csr_matrix(adata.X)
    total_counts = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
    n_genes = matrix.getnnz(axis=1).astype(np.float64)
    symbols = adata.var.get("gene_symbol", pd.Series(adata.var_names, index=adata.var_names))
    symbols = symbols.astype(str).str.upper()
    mt_mask = symbols.str.startswith(("MT-", "MT."), na=False).to_numpy()
    ribo_mask = symbols.str.startswith(("RPL", "RPS"), na=False).to_numpy()
    mt_counts = (
        np.asarray(matrix[:, mt_mask].sum(axis=1)).ravel().astype(np.float64)
        if mt_mask.any()
        else np.zeros(adata.n_obs, dtype=np.float64)
    )
    ribo_counts = (
        np.asarray(matrix[:, ribo_mask].sum(axis=1)).ravel().astype(np.float64)
        if ribo_mask.any()
        else np.zeros(adata.n_obs, dtype=np.float64)
    )
    denominator = np.maximum(total_counts, 1.0)
    adata.obs["total_counts"] = total_counts
    adata.obs["n_genes_by_counts"] = n_genes
    adata.obs["pct_counts_mt"] = 100.0 * mt_counts / denominator
    adata.obs["pct_counts_ribo"] = 100.0 * ribo_counts / denominator

    summaries: Dict[str, Dict[str, float]] = {}
    qc_pass = np.ones(adata.n_obs, dtype=bool)
    reasons = np.full(adata.n_obs, "", dtype=object)
    sample_values = adata.obs["sample_id"].astype(str)
    for sample_id, indices in _group_positions(sample_values):
        metric_values = {
            "total_counts": total_counts[indices],
            "n_genes_by_counts": n_genes[indices],
            "pct_counts_mt": np.asarray(adata.obs.iloc[indices]["pct_counts_mt"], dtype=float),
        }
        bounds: Dict[str, Dict[str, float]] = {}
        failed_counts: Dict[str, int] = {}
        sample_pass = np.ones(len(indices), dtype=bool)
        local_reasons = np.full(len(indices), "", dtype=object)
        for metric, values in metric_values.items():
            median, mad, lower, upper = _robust_bounds(values)
            bounds[metric] = {
                "median": median,
                "mad": mad,
                "lower": float(lower),
                "upper": float(upper),
            }
            failed = (values < lower) | (values > upper)
            failed_counts[metric] = int(failed.sum())
            sample_pass &= ~failed
            for local_index in np.flatnonzero(failed):
                reason = metric
                local_reasons[local_index] = (
                    f"{local_reasons[local_index]};{reason}" if local_reasons[local_index] else reason
                )
        qc_pass[indices] = sample_pass
        reasons[indices] = local_reasons
        summaries[str(sample_id)] = {
            "raw_cells": float(len(indices)),
            "qc_cells": float(sample_pass.sum()),
            "qc_fraction": float(sample_pass.mean()) if len(indices) else 0.0,
            "median_total_counts": bounds["total_counts"]["median"],
            "median_n_genes": bounds["n_genes_by_counts"]["median"],
            "median_pct_mt": bounds["pct_counts_mt"]["median"],
            "total_counts_lower": bounds["total_counts"]["lower"],
            "total_counts_upper": bounds["total_counts"]["upper"],
            "n_genes_lower": bounds["n_genes_by_counts"]["lower"],
            "n_genes_upper": bounds["n_genes_by_counts"]["upper"],
            "pct_mt_upper": bounds["pct_counts_mt"]["upper"],
            "failed_total_counts": float(failed_counts["total_counts"]),
            "failed_n_genes": float(failed_counts["n_genes_by_counts"]),
            "failed_pct_mt": float(failed_counts["pct_counts_mt"]),
        }
        adata.uns.setdefault("qc_thresholds", {})[str(sample_id)] = bounds
    adata.obs["qc_pass"] = qc_pass
    adata.obs["qc_failure_reason"] = reasons
    adata.obs["doublet_pass"] = True
    adata.obs["analysis_pass"] = qc_pass
    return summaries


def _run_scrublet_by_sample(adata: Any) -> str:
    """Run Scrublet per sample."""

    require_extras(["scrublet"], feature="IBD Scrublet doublet detection")
    import scrublet as scr

    statuses: Dict[str, str] = {}
    for sample_id, indices in _group_positions(adata.obs["sample_id"]):
        try:
            detector = scr.Scrublet(sparse.csr_matrix(adata.X[indices]))
            scores, predicted = detector.scrub_doublets(verbose=False)
            adata.obs.iloc[indices, adata.obs.columns.get_loc("doublet_score")] = scores
            adata.obs.iloc[indices, adata.obs.columns.get_loc("predicted_doublet")] = predicted
            adata.obs.iloc[indices, adata.obs.columns.get_loc("doublet_pass")] = ~predicted
            statuses[str(sample_id)] = "ok"
        except Exception as exc:  # noqa: BLE001, pragma: no cover - algorithm/data dependent
            LOGGER.warning("Scrublet failed for %s: %s", sample_id, exc)
            statuses[str(sample_id)] = f"error:{type(exc).__name__}"
    adata.uns["doublet_detection"] = {"status": "completed", "method": "Scrublet", "samples": statuses}
    adata.obs["analysis_pass"] = adata.obs["qc_pass"].to_numpy() & adata.obs["doublet_pass"].to_numpy()
    return "completed"


def run_qc_on_adata(adata: Any, run_doublet: bool = False) -> Dict[str, Dict[str, float]]:
    """Add QC metrics and optionally run per-sample Scrublet."""

    summaries = add_sample_qc_metrics(adata)
    adata.obs["doublet_score"] = np.nan
    adata.obs["predicted_doublet"] = False
    if run_doublet:
        _run_scrublet_by_sample(adata)
    else:
        adata.uns["doublet_detection"] = {"status": "not_requested", "method": "Scrublet"}
    for sample_id, indices in _group_positions(adata.obs["sample_id"]):
        sample_obs = adata.obs.iloc[indices]
        sample_qc = sample_obs["qc_pass"].to_numpy(dtype=bool)
        sample_doublets = ~sample_obs["doublet_pass"].to_numpy(dtype=bool)
        sample_analysis = sample_qc & ~sample_doublets
        if sample_id in summaries:
            summaries[sample_id]["doublet_cells"] = float((sample_qc & sample_doublets).sum())
            summaries[sample_id]["analysis_cells"] = float(sample_analysis.sum())
            summaries[sample_id]["analysis_fraction"] = float(sample_analysis.mean()) if len(sample_analysis) else 0.0
    return summaries


def _concat_adatas(adatas: Sequence[Any]):
    if not adatas:
        raise ValueError("No AnnData objects were loaded")
    ad = _require_anndata()
    result = ad.concat(
        list(adatas),
        axis=0,
        join="inner",
        merge="same",
        index_unique=None,
        fill_value=0,
    )
    result.X = sparse.csr_matrix(result.X, dtype=np.float32)
    return result


def load_and_qc(
    metadata_path: Path | str,
    raw_root: Path | str = "data/raw",
    scope: str = "core",
    run_doublet: bool = False,
) -> Tuple[Any, pd.DataFrame]:
    """Load selected libraries, run sample-wise QC, and return filtered cells."""

    metadata = load_metadata(metadata_path)
    adatas: List[Any] = []
    summary_rows: List[Dict[str, Any]] = []
    total = len(_selected_rows(metadata, scope))
    for index, (row, adata) in enumerate(iter_sample_adatas(metadata, raw_root, scope), start=1):
        LOGGER.info(
            "Loading %s %s (%d/%d; %d cells)",
            row["dataset"],
            row["GSM"],
            index,
            total,
            adata.n_obs,
        )
        summary = run_qc_on_adata(adata, run_doublet=run_doublet)
        for sample_id, values in summary.items():
            summary_rows.append(
                {
                    "dataset": row["dataset"],
                    "GSM": row["GSM"],
                    "sample_id": sample_id,
                    **values,
                    "doublet_status": adata.uns.get("doublet_detection", {}).get("status", "unknown"),
                }
            )
        kept = adata.obs["analysis_pass"].to_numpy(dtype=bool)
        adatas.append(adata[kept].copy())
    result = _concat_adatas(adatas)
    result.layers["counts"] = result.X.copy()
    summary_frame = pd.DataFrame(summary_rows).sort_values(["dataset", "GSM"])
    result.uns["ibd_qc"] = {
        "scope": scope,
        "raw_counts_layer": "counts",
        "qc_rule": "sample-specific median +/- 3 MAD for total counts, detected genes, and mitochondrial fraction",
        "ambient_correction": "none",
        "doublet_requested": bool(run_doublet),
    }
    doublet_statuses = summary_frame["doublet_status"].astype(str).value_counts().to_dict()
    result.uns["doublet_detection"] = {
        "requested": bool(run_doublet),
        "status_counts": {str(key): int(value) for key, value in doublet_statuses.items()},
    }
    return result, summary_frame


def _subsample_by_sample(adata: Any, max_cells_per_sample: int, seed: int) -> Any:
    if max_cells_per_sample <= 0:
        return adata
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    sample_values = adata.obs["sample_id"].astype(str)
    for _, indices in _group_positions(sample_values):
        if len(indices) > max_cells_per_sample:
            indices = np.sort(rng.choice(indices, size=max_cells_per_sample, replace=False))
        selected.extend(indices.tolist())
    return adata[np.sort(np.asarray(selected, dtype=np.int64))].copy()


def integrate_scvi(
    adata: Any,
    *,
    max_cells_per_sample: int = 1000,
    max_epochs: int = 20,
    seed: int = 20260908,
) -> Tuple[Any, Any]:
    """Run a reproducible scVI integration on a per-sample capped dataset.

    The full QC object is not silently downsampled.  This function returns a
    separate integration object and records the cap in ``uns['ibd_integration']``.
    """

    sc = _require_scanpy()
    require_extras(["scvi"], feature="IBD scVI integration")
    import scvi

    integration = _subsample_by_sample(adata, max_cells_per_sample, seed)
    integration.layers["counts"] = integration.X.copy()
    scvi.settings.seed = seed
    scvi.model.SCVI.setup_anndata(
        integration,
        layer="counts",
        batch_key="dataset",
        categorical_covariate_keys=["sample_id"],
    )
    model = scvi.model.SCVI(integration, n_latent=30, gene_likelihood="nb")
    model.train(
        max_epochs=max_epochs,
        accelerator="gpu" if _cuda_available() else "cpu",
        devices=1,
        early_stopping=False,
    )
    integration.obsm["X_scVI"] = model.get_latent_representation()
    sc.pp.neighbors(integration, use_rep="X_scVI", random_state=seed)
    sc.tl.umap(integration, random_state=seed)
    integration.uns["ibd_integration"] = {
        "method": "scVI",
        "batch_key": "dataset",
        "covariate_keys": ["sample_id"],
        "max_cells_per_sample": int(max_cells_per_sample),
        "max_epochs": int(max_epochs),
        "seed": int(seed),
        "latent_key": "X_scVI",
    }
    return integration, model


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover - depends on environment
        return False


def write_qc_outputs(
    adata: Any,
    summary: pd.DataFrame,
    *,
    output_dir: Path | str = "data/processed/ibd",
    prefix: str = "ibd_core_qc",
) -> Tuple[Path, Path]:
    """Write the QC AnnData and sample audit table."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    h5ad_path = output / f"{prefix}.h5ad"
    summary_path = output / f"{prefix}_sample_summary.tsv"
    adata.write_h5ad(h5ad_path, compression="gzip")
    summary.to_csv(summary_path, sep="\t", index=False)
    return h5ad_path, summary_path


__all__ = [
    "add_sample_qc_metrics",
    "integrate_scvi",
    "iter_sample_adatas",
    "load_and_qc",
    "run_qc_on_adata",
    "write_qc_outputs",
]
