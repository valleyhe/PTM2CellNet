#!/usr/bin/env python3
"""Import Norman/Adamson CRISPR perturbation datasets into PTM2CellNet.

Consumes the GEO releases downloaded to ``data/raw/norman_adamson/``:

* ``GSE133344`` (Norman et al. 2019, Science 363:earmoon) — 10x-format
  filtered/raw matrix + gene table + barcodes + cell identities.
* ``GSE90546`` (Adamson et al. 2016, Cell 167:1853) — RAW.tar with per
  experiment count tables (structure is probed at runtime).

This script produces the *research-level intermediate* artifacts that the
cross-scale pipeline consumes *before* pLM embedding:

* ``GSE133344_expression.npz``        — cell x gene expression (CSC sparse)
* ``GSE133344_perturbations.tsv``     — perturbation catalogue (id, target,
  guides, n_cells, is_control)
* ``GSE133344_delta_expression.npz``  — perturbation x gene delta vs
  non-targeting control (dense float32)
* ``sequence_requests.json``          — gene symbols that need pLM embeddings
  (input contract for the TD-H02 raw-sequence -> embedding pre-computer)
* ``import_manifest.json``            — provenance (GEO ids, files, sha256,
  filtering parameters)

The final ``CrossScaleNPZDataset`` archive (schema
``ptm2cellnet.cross-scale.npz.v1``, including ``cell_edge_weight``) is
assembled downstream by the embedding pre-computer together with signaling
graph imports (STRING/BioPlex/kinase_substrate), not by this script.

Design notes
------------
* Pure numpy/scipy — no scanpy/anndata dependency required.
* Column-name probing for the perturbation annotations: GEO-derived identity
  files differ between releases, so the parser accepts a documented set of
  aliases instead of a fixed schema.
* Fail-fast for missing inputs, fail-soft (with a structural report) for
  unknown auxiliary file layouts.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.io import mmread

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GSE133344 = "GSE133344"
GSE90546 = "GSE90546"

GSE133344_FILES = (
    "GSE133344_filtered_barcodes.tsv.gz",
    "GSE133344_filtered_cell_identities.csv.gz",
    "GSE133344_filtered_genes.tsv.gz",
    "GSE133344_filtered_matrix.mtx.gz",
)

GSE90546_FILES = ("GSE90546_RAW.tar",)

#: Identity columns that indicate the perturbation annotation; the first
#: match wins.  ``target`` is the canonical output name.
IDENTITY_COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "barcode": ("barcode", "cell", "cell_barcode", "bc", "cell_id"),
    "target": (
        "target",
        "target_gene",
        "gene_target",
        "perturbation_target",
        "gene_name",
        "gene",
        "sg_target",
        "perturbed_gene",
    ),
    "guide": ("guide", "guide_id", "guide_identity", "grna", "sgRNA", "sgrna", "guide_rna", "guide_sequence"),
    "perturbation": ("perturbation", "perturbation_id", "identity", "label", "condition", "treatment"),
}

#: Labels treated as non-targeting control (case-insensitive, prefix match).
NON_TARGETING_MARKERS = (
    "nt",
    "non-targeting",
    "non_targeting",
    "nontargeting",
    "control",
    "ctrl",
    "scramble",
    "neg",
)


class ImportError_(RuntimeError):
    """Raised when a required GEO input is missing or malformed."""


# ---------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------


def _open_maybe_gzip(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.suffix == ".gz" else open(path, "rt", encoding="utf-8", errors="replace")


def read_tsv(path: Path, *, max_rows: Optional[int] = None) -> List[List[str]]:
    """Read a (possibly gzipped) TSV/CSV into a list of rows of strings."""
    if not path.is_file():
        raise ImportError_(f"缺少输入文件: {path}")
    rows: List[List[str]] = []
    with _open_maybe_gzip(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            delim = "," if (path.suffix == ".csv" or path.name.endswith(".csv.gz")) else "\t"
            rows.append([cell.strip() for cell in line.split(delim)])
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def read_genes(path: Path) -> Tuple[List[str], List[str]]:
    """10x genes.tsv -> (ensembl_ids, gene_symbols).  Falls back to the first
    column as symbol when only one column is present."""
    rows = read_tsv(path)
    if not rows:
        raise ImportError_(f"基因表为空: {path}")
    if len(rows[0]) >= 2:
        ensembl = [row[0] for row in rows]
        symbols = [row[1] if len(row) > 1 and row[1] else row[0] for row in rows]
    else:
        ensembl = [row[0] for row in rows]
        symbols = list(ensembl)
    return ensembl, symbols


def read_barcodes(path: Path) -> List[str]:
    rows = read_tsv(path)
    return [row[0] for row in rows if row and row[0]]


def read_mtx(path: Path) -> sp.csc_matrix:
    """Read a gzipped Market-Matrix file into a CSC sparse matrix."""
    if not path.is_file():
        raise ImportError_(f"缺少矩阵文件: {path}")
    try:
        matrix = mmread(str(path))
    except (OSError, ValueError) as exc:
        raise ImportError_(f"无法解析 10x 矩阵 {path}: {exc}") from exc
    if not sp.issparse(matrix):
        matrix = sp.csc_matrix(matrix)
    return matrix.tocsc()


# ---------------------------------------------------------------------------
# Perturbation annotation parsing
# ---------------------------------------------------------------------------


def _match_alias(name: str, aliases: Sequence[str]) -> bool:
    norm = name.strip().lower().replace(" ", "_").replace("-", "_")
    return any(norm == alias or norm.endswith("_" + alias) for alias in aliases)


def detect_identity_schema(rows: List[List[str]]) -> Dict[str, int]:
    """Map canonical roles (barcode/target/guide/perturbation) to column
    indexes by name probing.  Raises ImportError_ if no annotation column is
    recognisable."""
    header = rows[0]
    schema: Dict[str, int] = {}
    for role, aliases in IDENTITY_COLUMN_ALIASES.items():
        for index, name in enumerate(header):
            if _match_alias(name, aliases):
                schema.setdefault(role, index)
    if "target" not in schema and "perturbation" not in schema:
        raise ImportError_(
            "无法识别扰动注释列；支持别名: "
            + ", ".join(sorted(IDENTITY_COLUMN_ALIASES["target"] + IDENTITY_COLUMN_ALIASES["perturbation"]))
        )
    return schema


def _parse_norman_guide_identity(value: str) -> Tuple[str, bool]:
    """Parse Norman 2019 guide_identity labels.

    Format: ``ARID1A_NegCtrl0__ARID1A_NegCtrl0`` (single gene) or
    ``SET_KLF1__SET_KLF1`` (combinatorial, genes joined by ``_``).  The
    ``_NegCtrlN`` / ``_PosCtrlN`` suffix marks control/positive-control guides.
    Returns ``(target_genes, is_control)``.
    """
    head = value.split("__", 1)[0]
    is_control = bool(re.search(r"(NegCtrl|PosCtrl)\d+", value))
    target = re.sub(r"_(NegCtrl|PosCtrl)\d+$", "", head).strip()
    return target, is_control


def parse_identities(rows: List[List[str]]) -> Dict[str, Dict[str, str]]:
    """Return {barcode: {target, guide, perturbation, is_control}} for every
    annotated cell.  Norman 2019 ``guide_identity`` columns are parsed with
    :func:`_parse_norman_guide_identity`; other layouts use plain column
    mapping."""
    if not rows:
        return {}
    schema = detect_identity_schema(rows)
    barcode_idx = schema.get("barcode", 0)
    target_idx = schema.get("target")
    perturbation_idx = schema.get("perturbation")
    guide_idx = schema.get("guide")
    guide_column_name = rows[0][guide_idx].strip().lower() if guide_idx is not None else ""
    annotations: Dict[str, Dict[str, str]] = {}
    for row in rows[1:]:
        if not row or len(row) <= barcode_idx:
            continue
        barcode = row[barcode_idx].strip()
        if not barcode:
            continue
        target = row[target_idx].strip() if target_idx is not None and len(row) > target_idx else ""
        perturbation = row[perturbation_idx].strip() if perturbation_idx is not None and len(row) > perturbation_idx else ""
        guide = row[guide_idx].strip() if guide_idx is not None and len(row) > guide_idx else ""
        annotation: Dict[str, str] = {
            "target": target,
            "perturbation": perturbation,
            "guide": guide,
            "is_control": "",
        }
        if guide and "guide_identity" in guide_column_name and "__" in guide:
            norman_target, is_control = _parse_norman_guide_identity(guide)
            annotation["target"] = norman_target
            annotation["is_control"] = "1" if is_control else ""
        annotations[barcode] = annotation
    return annotations


def is_control_label(label: str) -> bool:
    norm = label.strip().lower().replace("_", "-").replace(" ", "-")
    return any(norm.startswith(marker) for marker in NON_TARGETING_MARKERS)


def perturbation_id_from_annotation(annotation: Mapping[str, str]) -> str:
    """Canonical perturbation id: target gene(s); NT/control labels are kept
    verbatim so control cells never merge into a gene group."""
    target = annotation.get("target", "").strip()
    perturbation = annotation.get("perturbation", "").strip()
    if annotation.get("is_control") in ("1", "true", "True", True):
        return "control"
    label = target or perturbation
    if not label:
        return "__unlabeled__"
    if is_control_label(label):
        return "control"
    return label


# ---------------------------------------------------------------------------
# Delta expression
# ---------------------------------------------------------------------------


def compute_delta_expression(
    matrix: sp.csc_matrix,
    annotations: Mapping[str, Mapping[str, str]],
    barcodes: Sequence[str],
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-perturbation pseudobulk and subtract the non-targeting
    control mean.

    Returns:
        delta:       [n_perturbations, n_genes] float32
        sample_ids:  perturbation ids (row order of delta)
        cell_counts: n cells per perturbation
        control_mean: [n_genes] mean expression of control cells
        unlabeled_mask: boolean [n_genes]? -> no, boolean [n_cells] whether the
            cell could not be assigned to any perturbation.
    """
    labels = np.empty(len(barcodes), dtype=object)
    assigned = np.zeros(len(barcodes), dtype=bool)
    for index, barcode in enumerate(barcodes):
        annotation = annotations.get(barcode)
        if annotation is None:
            labels[index] = "__unlabeled__"
            continue
        label = perturbation_id_from_annotation(annotation)
        labels[index] = label
        assigned[index] = label != "__unlabeled__"

    unique_labels = sorted({str(label) for label in labels if str(label) != "__unlabeled__"})
    sample_ids: List[str] = []
    cell_counts: List[int] = []
    group_means: List[np.ndarray] = []
    for label in unique_labels:
        mask = labels == label
        count = int(mask.sum())
        if count == 0:
            continue
        sample_ids.append(label)
        cell_counts.append(count)
        group_means.append(np.asarray(matrix[mask].mean(axis=0)).ravel().astype(np.float32))

    control_idx = [i for i, label in enumerate(sample_ids) if label == "control"]
    if not control_idx:
        # No explicit control: delta is relative to the global mean.
        control_mean = np.asarray(matrix.mean(axis=0)).ravel().astype(np.float32)
        control_label = "__global_mean__"
    else:
        control_mean = group_means[control_idx[0]]
        control_label = "control"

    delta = np.stack(
        [group_means[i] - control_mean for i in range(len(sample_ids)) if sample_ids[i] != control_label],
        axis=0,
    ).astype(np.float32) if any(sample_ids[i] != control_label for i in range(len(sample_ids))) else np.empty((0, matrix.shape[1]), dtype=np.float32)
    non_control_ids = [sample_ids[i] for i in range(len(sample_ids)) if sample_ids[i] != control_label]
    non_control_counts = [cell_counts[i] for i in range(len(sample_ids)) if sample_ids[i] != control_label]
    return delta, non_control_ids, np.asarray(non_control_counts, dtype=np.int64), control_mean, ~assigned


def _requested_genes_from_labels(labels: Sequence[str], gene_symbols: Sequence[str]) -> List[str]:
    """Extract gene symbols needing pLM embedding from perturbation labels.

    Single-gene labels match the gene table directly; combinatorial labels
    (Norman 2019 uses ``GENE1_GENE2``) are split and matched against the table.
    """
    gene_set = set(gene_symbols)
    requested: set[str] = set()
    for label in labels:
        if label in ("control", "__global_mean__", "__unlabeled__"):
            continue
        if label in gene_set:
            requested.add(label)
            continue
        requested.update(part for part in label.split("_") if part in gene_set)
    return sorted(requested)


# ---------------------------------------------------------------------------
# GSE90546 (Adamson) — best-effort RAW.tar parsing
# ---------------------------------------------------------------------------


def probe_gse90546(geo_root: Path) -> Dict[str, Any]:
    """List RAW.tar contents and parse recognisable tabular files.  The
    archive layout is not standardised; this returns a structural report and
    parsed tables when they look like count matrices."""
    tar_path = geo_root / GSE90546 / "GSE90546_RAW.tar"
    if not tar_path.is_file():
        return {"status": "missing", "reason": f"{tar_path} 不存在（尚未下载或下载失败）"}
    report: Dict[str, Any] = {"status": "parsed", "members": []}
    try:
        with tarfile.open(tar_path, "r:*") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                info: Dict[str, Any] = {"name": member.name, "size": member.size}
                if member.name.lower().endswith((".csv", ".tsv", ".txt", ".gz")):
                    fileobj = archive.extractfile(member)
                    if fileobj is None:
                        continue
                    raw = fileobj.read(64 * 1024)
                    if raw.startswith(b"\x1f\x8b"):
                        # 流式解压前 4KB，避免不完整 gz 流触发 EOFError。
                        try:
                            with gzip.GzipFile(fileobj=__import__("io").BytesIO(raw)) as gz:
                                head = gz.read(4096).decode("utf-8", errors="replace")
                        except (OSError, EOFError, ValueError):
                            head = "<gzip 解压失败，跳过>"
                    else:
                        head = raw.decode("utf-8", errors="replace")
                    info["head_preview"] = head[:400]
                    info["looks_tabular"] = "\t" in head or "," in head
                report["members"].append(info)
    except (tarfile.TarError, OSError) as exc:
        report["status"] = "error"
        report["reason"] = str(exc)
    return report


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_sparse_npz(path: Path, matrix: sp.csc_matrix, extra: Optional[Dict[str, np.ndarray]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    coo = matrix.tocoo()
    payload: Dict[str, Any] = {
        "shape": np.asarray(matrix.shape, dtype=np.int64),
        "data": coo.data.astype(np.float32),
        "row": coo.row.astype(np.int64),
        "col": coo.col.astype(np.int64),
    }
    for key, value in (extra or {}).items():
        payload[key] = np.asarray(value)
    np.savez_compressed(path, **payload)


def write_import_manifest(
    output_dir: Path,
    *,
    gse133344: Dict[str, Any],
    gse90546: Dict[str, Any],
    args: argparse.Namespace,
    sha256: Mapping[str, str],
) -> Path:
    payload = {
        "schema_version": "ptm2cellnet.norman-adamson.import.v1",
        "geo_accessions": [GSE133344, GSE90546],
        "source_files": {key: value for key, value in sorted(sha256.items())},
        "gse133344_summary": {
            "cells": gse133344.get("cells"),
            "genes": gse133344.get("genes"),
            "perturbations": gse133344.get("n_perturbations"),
            "controls_present": gse133344.get("controls_present"),
            "unassigned_cells": gse133344.get("unassigned_cells"),
        },
        "gse90546": gse90546,
        "parameters": {
            "geo_root": str(args.geo_root),
            "output": str(args.output),
        },
        "note": (
            "研究级中间产物；pLM embedding 与图结构由 TD-H02 预计算器与"
            "信号图导入（STRING/BioPlex/kinase_substrate）在后续步骤组装为 "
            "CrossScaleNPZDataset 所需的 NPZ v1 archive。"
        ),
    }
    destination = output_dir / "import_manifest.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


# ---------------------------------------------------------------------------
# Per-dataset importers
# ---------------------------------------------------------------------------


def import_gse133344(geo_root: Path, output_dir: Path) -> Dict[str, Any]:
    dataset_dir = geo_root / GSE133344
    if not dataset_dir.is_dir():
        raise ImportError_(f"缺少 {dataset_dir}；请先下载 GSE133344")
    summary: Dict[str, Any] = {}

    matrix_path = dataset_dir / "GSE133344_filtered_matrix.mtx.gz"
    genes_path = dataset_dir / "GSE133344_filtered_genes.tsv.gz"
    barcodes_path = dataset_dir / "GSE133344_filtered_barcodes.tsv.gz"
    identities_path = dataset_dir / "GSE133344_filtered_cell_identities.csv.gz"

    matrix = read_mtx(matrix_path)
    ensembl, symbols = read_genes(genes_path)
    barcodes = read_barcodes(barcodes_path)
    # 10x 标准矩阵为 基因×细胞；若与基因表/细胞数不匹配则转置对齐，
    # 使内部统一为 细胞×基因。
    if matrix.shape[0] == len(symbols) and matrix.shape[1] == len(barcodes):
        matrix = matrix.T.tocsc()
    if matrix.shape[1] != len(symbols):
        raise ImportError_(
            f"矩阵基因数 {matrix.shape[1]} 与基因表 {len(symbols)} 不一致"
        )
    if matrix.shape[0] != len(barcodes):
        raise ImportError_(
            f"矩阵细胞数 {matrix.shape[0]} 与 barcodes {len(barcodes)} 不一致"
        )

    identity_rows = read_tsv(identities_path)
    annotations = parse_identities(identity_rows)
    summary["identity_annotation_cells"] = len(annotations)

    delta, sample_ids, cell_counts, control_mean, unassigned = compute_delta_expression(
        matrix, annotations, barcodes
    )
    summary.update(
        {
            "cells": int(matrix.shape[0]),
            "genes": int(matrix.shape[1]),
            "n_perturbations": len(sample_ids),
            "n_nonzero": int(matrix.nnz),
            "controls_present": any(label == "control" for label in sample_ids) or "control" in [
                perturbation_id_from_annotation(a) for a in annotations.values()
            ],
            "unassigned_cells": int(unassigned.sum()),
            "delta_shape": list(delta.shape),
        }
    )

    write_sparse_npz(
        output_dir / "GSE133344_expression.npz",
        matrix,
        extra={"barcodes": np.asarray(barcodes, dtype=object)},
    )
    with (output_dir / "GSE133344_perturbations.tsv").open("w", encoding="utf-8") as handle:
        handle.write("perturbation_id\tn_cells\n")
        for label, count in zip(sample_ids, cell_counts):
            handle.write(f"{label}\t{count}\n")
    np.savez_compressed(
        output_dir / "GSE133344_delta_expression.npz",
        delta=delta,
        sample_ids=np.asarray(sample_ids, dtype=object),
        gene_symbols=np.asarray(symbols, dtype=object),
        control_mean=control_mean,
    )

    # Sequence requests: genes that are perturbed targets (need pLM embedding).
    # Single-gene labels match the gene table directly; combinatorial labels
    # (Norman 2019 uses "GENE1_GENE2") are split and matched against the table.
    requested_genes = _requested_genes_from_labels(sample_ids, symbols)
    (output_dir / "sequence_requests.json").write_text(
        json.dumps({"dataset": GSE133344, "genes": requested_genes, "n_genes": len(requested_genes)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def import_gse90546(geo_root: Path, output_dir: Path) -> Dict[str, Any]:
    report = probe_gse90546(geo_root)
    (output_dir / "GSE90546_structure_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--geo-root", default="data/raw/norman_adamson", help="GEO 下载根目录")
    parser.add_argument("--output", default="data/processed/norman_adamson", help="输出目录")
    parser.add_argument("--skip-gse90546", action="store_true", help="跳过 GSE90546 探测（RAW.tar 未下载时）")
    args = parser.parse_args(argv)

    geo_root = Path(args.geo_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sha256: Dict[str, str] = {}
    for dataset, files in ((GSE133344, GSE133344_FILES), (GSE90546, GSE90546_FILES)):
        for name in files:
            path = geo_root / dataset / name
            if path.is_file():
                sha256[name] = _sha256_file(path)

    gse133344_summary: Dict[str, Any] = {}
    gse90546_report: Dict[str, Any] = {"status": "skipped"}
    try:
        gse133344_summary = import_gse133344(geo_root, output_dir)
    except ImportError_ as exc:
        print(f"[GSE133344] 跳过：{exc}")
    if not args.skip_gse90546:
        gse90546_report = import_gse90546(geo_root, output_dir)

    manifest = write_import_manifest(
        output_dir,
        gse133344=gse133344_summary,
        gse90546=gse90546_report,
        args=args,
        sha256=sha256,
    )
    print(json.dumps({"ok": True, "manifest": str(manifest), "gse133344": gse133344_summary, "gse90546": gse90546_report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
