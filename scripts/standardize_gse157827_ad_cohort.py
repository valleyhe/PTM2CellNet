#!/usr/bin/env python3
"""Standardize the GSE157827 AD snRNA-seq cohort into the Gate-0 h5ad contract.

Inputs are the 21 per-sample 10x MTX triplets under ``data/AD/extracted/GSE157827``
and the GEO SOFT file. Two derivations are required by the 2026-09-14 audit
(``outputs/perturbgen/spike/20260914_ad_cohort_audit/evidence.json``) and are
recorded verbatim in the provenance JSON instead of being silently applied:

1. ``donor``: GEO carries no explicit donor id. The sample titles (AD1, AD2,
   AD4 ... NC3, NC7 ...) are non-consecutive subject-pool labels with exactly
   one library per label, which is the same structural evidence strength as
   the GSE174367 "unique subject covariate vector" derivation.
2. ``cell_type``: GEO publishes no per-nucleus annotation. Cell types are
   assigned data-driven by Leiden clustering plus fixed marker-module scores
   and mapped onto the GSE174367 seven-label space (ASC/EX/INH/MG/ODC/OPC/
   PER.END); nuclei whose best marker score is non-positive stay
   ``unassigned`` and are excluded from downstream per-cell-type DEG.

Counts stay raw integers; genes are canonical version-less ENSG with
duplicate rows summed; the output matches the GSE174367 standardized contract
(``X`` and ``layers['counts']`` both raw counts, ``obs[donor, state, cell_type,
sample, dataset]``, ``var[ensembl_id, gene_symbol]``).
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Sequence

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


GSE157827_DONOR_EVIDENCE = (
    "GEO sample titles (AD1, AD2, AD4, ..., NC3, NC7, ...) are non-consecutive "
    "subject-pool labels; exactly one library per label; audit policy forbids "
    "reinterpretation of consecutive library ordinals as donors, which this "
    "derivation does not rely on (see "
    "outputs/perturbgen/spike/20260914_ad_cohort_audit/evidence.json)"
)

#: Fixed marker modules; cluster-level mean scores map onto the GSE174367 labels.
MARKER_MODULES: dict[str, tuple[str, ...]] = {
    "EX": ("SLC17A7", "SATB2", "CAMK2A"),
    "INH": ("GAD1", "GAD2", "SLC32A1"),
    "ASC": ("AQP4", "GFAP", "SLC1A3"),
    "ODC": ("MBP", "PLP1", "MOBP"),
    "OPC": ("PDGFRA", "VCAN", "CSPG4"),
    "MG": ("CX3CR1", "P2RY12", "CSF1R"),
    "PER.END": ("PDGFRB", "RGS5", "CLDN5"),
}
STATE_MAP = {"AD": "disease", "healthy control": "normal"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted-dir", type=Path, required=True)
    parser.add_argument("--soft-gz", type=Path, required=True)
    parser.add_argument("--min-genes-per-nucleus", type=int, default=200)
    parser.add_argument("--leiden-resolution", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    return parser.parse_args(argv)


def _parse_soft_samples(soft_gz: Path) -> pd.DataFrame:
    sample_title = None
    rows: list[dict[str, str]] = []
    with gzip.open(soft_gz, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("^SAMPLE = "):
                sample_title = None
                accession = line.strip().split(" = ")[1]
            elif line.startswith("!Sample_title = "):
                sample_title = line.strip().split(" = ")[1]
            elif line.startswith("!Sample_characteristics_ch1 = ") and sample_title is not None:
                value = line.strip().split(" = ", 1)[1]
                if value.startswith("diagnosis:"):
                    diagnosis = value.split(":", 1)[1].strip()
                    state = STATE_MAP.get(diagnosis)
                    if state is None:
                        raise ValueError(f"unknown diagnosis {diagnosis!r} for {sample_title}")
                    rows.append(
                        {
                            "sample": sample_title,
                            "accession": accession,
                            "diagnosis": diagnosis,
                            "donor": sample_title,
                            "state": state,
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame.empty or len(frame) != 21:
        raise ValueError(f"expected 21 diagnosed samples in SOFT, got {len(frame)}")
    if frame["donor"].duplicated().any():
        raise ValueError("sample titles repeat across libraries; donor derivation would be invalid")
    return frame


def _read_sample(extracted_dir: Path, accession: str, sample: str) -> anndata.AnnData:
    barcode_path = extracted_dir / f"{accession}_{sample}_barcodes.tsv.gz"
    feature_path = extracted_dir / f"{accession}_{sample}_features.tsv.gz"
    matrix_path = extracted_dir / f"{accession}_{sample}_matrix.mtx.gz"
    for path in (barcode_path, feature_path, matrix_path):
        if not path.exists():
            raise FileNotFoundError(path)
    features = pd.read_csv(feature_path, sep="\t", header=None, names=["ensembl_id", "gene_symbol", "feature_type"])
    if not features["feature_type"].eq("Gene Expression").all():
        features = features[features["feature_type"] == "Gene Expression"].reset_index(drop=True)
    adata = sc.read_mtx(matrix_path).T
    adata.X = sp.csr_matrix(adata.X)
    barcodes = pd.read_csv(barcode_path, header=None)[0].astype(str)
    if adata.n_obs != len(barcodes):
        raise ValueError(f"{sample}: barcode/matrix row mismatch")
    adata.obs_names = pd.Index([f"{sample}_{bc}" for bc in barcodes], name=None)
    adata.var_names = pd.Index(features["ensembl_id"].astype(str), name=None)
    keep = adata.var_names.str.match(r"^ENSG\d+$")
    if not keep.all():
        adata = adata[:, keep.to_numpy()].copy()
        features = features.loc[keep.to_numpy()]
    adata.var = pd.DataFrame(
        {
            "ensembl_id": features["ensembl_id"].astype(str).to_numpy(),
            "gene_symbol": features["gene_symbol"].astype(str).to_numpy(),
        },
        index=pd.Index(features["ensembl_id"].astype(str), name=None),
    )
    return adata


def _sum_duplicate_genes(adata: anndata.AnnData) -> anndata.AnnData:
    """Collapse duplicate ENSG columns by summing (COO column remapping)."""

    ensembl = adata.var["ensembl_id"].astype(str).to_numpy()
    symbols = adata.var["gene_symbol"].astype(str).to_numpy()
    if len(set(ensembl)) == len(ensembl):
        adata.var.index = pd.Index(ensembl, name=None)
        return adata
    order = sorted(set(ensembl))
    new_index = {gene_id: position for position, gene_id in enumerate(order)}
    symbol_map: dict[str, str] = {}
    for gene_id, symbol in zip(ensembl, symbols, strict=True):
        symbol_map.setdefault(gene_id, symbol)
    coo = adata.X.tocoo()
    new_cols = np.asarray([new_index[gene_id] for gene_id in ensembl[coo.col]])
    merged = sp.coo_matrix(
        (coo.data, (coo.row, new_cols)),
        shape=(adata.n_obs, len(order)),
    ).tocsr()
    output = anndata.AnnData(
        X=merged,
        obs=adata.obs.copy(),
        var=pd.DataFrame(
            {
                "ensembl_id": order,
                "gene_symbol": [symbol_map[gene_id] for gene_id in order],
            },
            index=pd.Index(order, name=None),
        ),
    )
    return output


def _annotate_cell_types(adata: anndata.AnnData, *, resolution: float, seed: int) -> tuple[anndata.AnnData, dict]:
    work = adata.copy()
    sc.pp.highly_variable_genes(work, n_top_genes=2000, flavor="seurat_v3", layer="counts", subset=False)
    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)
    work = work[:, work.var["highly_variable"]].copy()
    sc.pp.scale(work, max_value=10)
    sc.tl.pca(work, n_comps=50, random_state=seed)
    sc.pp.neighbors(work, n_neighbors=15, random_state=seed)
    sc.tl.leiden(work, resolution=resolution, key_added="cluster", flavor="igraph", directed=False, random_state=seed)

    # score_genes matches on var_names; switch to gene-symbol names for scoring
    full = adata.copy()
    full.var_names = pd.Index(full.var["gene_symbol"].astype(str), name=None)
    full = full[:, ~full.var_names.duplicated()].copy()
    sc.pp.normalize_total(full, target_sum=1e4)
    sc.pp.log1p(full)
    for label, markers in MARKER_MODULES.items():
        present = [gene for gene in markers if gene in set(full.var_names)]
        if not present:
            raise ValueError(f"no marker of module {label} present: {markers}")
        sc.tl.score_genes(full, gene_list=present, score_name=f"score_{label}")
    score_columns = [f"score_{label}" for label in MARKER_MODULES]
    cluster_scores = full.obs.groupby(work.obs["cluster"]).mean(numeric_only=True)[score_columns]
    cluster_label = cluster_scores.idxmax(axis=1).str.removeprefix("score_")
    cluster_max = cluster_scores.max(axis=1)
    label_map = {
        cluster: (cluster_label[cluster] if cluster_max[cluster] > 0 else "unassigned")
        for cluster in cluster_scores.index
    }
    adata.obs["cell_type"] = work.obs["cluster"].astype(str).map(label_map).astype("category")

    provenance = {
        "method": "Leiden clustering on scanpy PCA neighbors + fixed marker-module scores (cluster-level argmax)",
        "markers": {label: list(markers) for label, markers in MARKER_MODULES.items()},
        "leiden_resolution": resolution,
        "seed": seed,
        "cluster_labels": {str(cluster): label_map[cluster] for cluster in sorted(label_map)},
        "n_unassigned_nuclei": int((adata.obs["cell_type"] == "unassigned").sum()),
    }
    return adata, provenance


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    samples = _parse_soft_samples(args.soft_gz)
    parts = []
    symbol_by_ensembl: dict[str, str] = {}
    for _, row in samples.iterrows():
        part = _read_sample(args.extracted_dir, row["accession"], row["sample"])
        part.obs["donor"] = row["donor"]
        part.obs["state"] = row["state"]
        part.obs["sample"] = row["sample"]
        for ensembl_id, symbol in zip(part.var["ensembl_id"], part.var["gene_symbol"], strict=True):
            symbol_by_ensembl.setdefault(str(ensembl_id), str(symbol))
        parts.append(part)
    adata = anndata.concat(parts, axis=0, join="outer", index_unique=None)
    adata.X = sp.csr_matrix(adata.X)
    adata.var = pd.DataFrame(
        {
            "ensembl_id": adata.var_names.astype(str),
            "gene_symbol": [symbol_by_ensembl.get(str(value), "") for value in adata.var_names],
        },
        index=pd.Index(adata.var_names.astype(str), name=None),
    )
    adata.X.data = np.round(adata.X.data).astype(np.int32)
    adata = _sum_duplicate_genes(adata)

    gene_counts = np.asarray((adata.X > 0).sum(axis=1)).ravel()
    n_before_qc = int(adata.n_obs)
    keep_nuclei = gene_counts >= args.min_genes_per_nucleus
    adata = adata[keep_nuclei].copy()
    adata.obs["dataset"] = "GSE157827"
    adata.layers["counts"] = adata.X.copy()

    adata, annotation_provenance = _annotate_cell_types(adata, resolution=args.leiden_resolution, seed=args.seed)

    n_after_qc = int(adata.n_obs)
    adata = adata[adata.obs["cell_type"] != "unassigned"].copy()
    ordered_columns = ["donor", "state", "cell_type", "sample", "dataset"]
    adata.obs = adata.obs[ordered_columns]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output, compression="gzip")

    donor_state = adata.obs.groupby("state")["donor"].nunique().to_dict()
    provenance = {
        "schema_version": "ptm2cellnet.ad-cohort-standardization/v1",
        "dataset": "GSE157827",
        "source": {
            "soft": str(args.soft_gz),
            "extracted_dir": str(args.extracted_dir),
            "supplementary_origin": "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE157nnn/GSE157827/suppl/",
        },
        "donor_derivation": GSE157827_DONOR_EVIDENCE,
        "state_mapping": STATE_MAP,
        "n_libraries": int(len(samples)),
        "n_nuclei_before_qc": n_before_qc,
        "n_nuclei_after_qc": n_after_qc,
        "n_nuclei_final": int(adata.n_obs),
        "min_genes_per_nucleus": args.min_genes_per_nucleus,
        "gene_axis": {
            "n_genes": int(adata.n_vars),
            "id_type": "canonical version-less ENSG",
            "duplicate_policy": "duplicate ENSG rows summed",
        },
        "cell_type_annotation": annotation_provenance,
        "donor_counts": donor_state,
        "cell_type_counts": adata.obs["cell_type"].value_counts().to_dict(),
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(args.output), "donors": donor_state}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
