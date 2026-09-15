#!/usr/bin/env python
"""Standardize the GSE174367 pooled snRNA-seq matrix into a Gate-0 cohort h5ad.

The AD cohort contract decision (lessons.md, Gate-0 ``between_donor``
pairing) allows case-control cohorts: one donor per state group, >= 3 donors
per group. GSE174367 provides a pooled filtered 10x H5 plus author cell
metadata (``SampleID``, ``Diagnosis``, ``Cell.Type`` and subject-level
covariates). This script converts those two files into one cohort h5ad with
canonical version-less Ensembl ids, ``layers["counts"]`` raw counts, and
``cell_type``/``state``/``donor`` obs columns, then runs the real Gate-0
preflight (``prepare_perturbgen_anndata``, ``pairing="between_donor"``) for
every cell type and records the evidence.

Donor derivation is explicit and evidence-checked, never silent: SampleID is
accepted as donor only because every sample carries one unique subject-level
covariate vector (Age/Sex/PMI/RIN/Tangle/Plaque; 18/18 distinct, constant
within sample). Any covariate collision aborts the run.

Usage:
    python scripts/standardize_gse174367_ad_cohort.py \
        [--matrix-h5 data/AD/raw_downloads/GSE174367/...h5] \
        [--cell-meta data/AD/raw_downloads/GSE174367/...csv.gz] \
        [--output-dir data/AD/standardized]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.integration.perturbgen.contracts import PerturbGenDataSpec
from src.integration.perturbgen.data_prep import prepare_perturbgen_anndata
from src.models.gene_vocabulary import normalize_ensembl_id

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPO_ROOT / "data/AD/raw_downloads/GSE174367/GSE174367_snRNA-seq_filtered_feature_bc_matrix.h5"
DEFAULT_CELL_META = REPO_ROOT / "data/AD/raw_downloads/GSE174367/GSE174367_snRNA-seq_cell_meta.csv.gz"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/AD/standardized"
EVIDENCE_DIR = REPO_ROOT / "outputs/perturbgen/spike" / (_dt.date.today().strftime("%Y%m%d") + "_gse174367_gate0")

DIAGNOSIS_TO_STATE = {"Control": "normal", "AD": "disease"}
SUBJECT_COVARIATES = ("Age", "Sex", "PMI", "RIN", "Tangle.Stage", "Plaque.Stage")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_donor_evidence(cell_meta: pd.DataFrame) -> dict[str, object]:
    """Validate SampleID-as-donor against author subject-level covariates."""

    unknown = sorted(set(cell_meta["Diagnosis"].astype(str)) - set(DIAGNOSIS_TO_STATE))
    if unknown:
        raise ValueError(f"unexpected Diagnosis values: {unknown}")
    # Diagnosis joins the per-sample cardinality check: a SampleID carrying two
    # diagnoses would silently take the first one into the state mapping and
    # corrupt the donor grouping (same contract as audit_ad_cohort_gate0.py).
    per_sample_cardinality = cell_meta.groupby("SampleID")[list(SUBJECT_COVARIATES) + ["Diagnosis"]].nunique()
    if int(per_sample_cardinality.to_numpy().max()) != 1:
        raise ValueError("subject covariates must be constant within each SampleID")
    subjects = cell_meta.drop_duplicates("SampleID")
    if subjects.duplicated(list(SUBJECT_COVARIATES)).any():
        raise ValueError("two samples share one subject covariate vector; SampleID is not a safe donor id")
    donor_diagnosis = subjects.groupby("Diagnosis")["SampleID"].nunique().to_dict()
    for state_donors in donor_diagnosis.values():
        if state_donors < 3:
            raise ValueError(f"each state group needs >= 3 donors: {donor_diagnosis}")
    return {
        "n_samples": int(len(subjects)),
        "unique_subject_covariate_vectors": int(len(subjects.drop_duplicates(list(SUBJECT_COVARIATES)))),
        "covariates": list(SUBJECT_COVARIATES),
        "donor_diagnosis": {str(k): int(v) for k, v in donor_diagnosis.items()},
        "derivation": "SampleID used as donor id; evidence = one unique subject covariate vector per sample",
    }


def _read_pooled_matrix(matrix_h5: Path) -> tuple[sp.csr_matrix, pd.DataFrame, np.ndarray, int, int]:
    """Read the pooled 10x H5 into cells x genes CSR with version-less ENSG."""

    with h5py.File(matrix_h5, "r") as handle:
        group = handle["matrix"]
        n_genes, n_cells = (int(value) for value in group["shape"][:])
        data = np.asarray(group["data"], dtype=np.int32)
        indices = np.asarray(group["indices"], dtype=np.int64)
        indptr = np.asarray(group["indptr"], dtype=np.int64)
        barcodes = np.asarray([value.decode() for value in group["barcodes"][:]])
        gene_ids = [value.decode() for value in group["features/id"][:]]
        gene_symbols = [value.decode() for value in group["features/name"][:]]
    if indptr.shape != (n_cells + 1,):
        raise ValueError(f"10x indptr length {indptr.shape} does not match {n_cells} cells")
    csc = sp.csc_matrix((data, indices, indptr), shape=(n_genes, n_cells))
    counts = csc.T.tocsr()

    # The GRCh38 10x reference annotates chrY pseudoautosomal copies of a
    # gene as ``ENSGxxx.N_PAR_Y``; every such entry shares the base ENSG of
    # its chrX copy (verified 45/45 in this file) and is merged by the
    # duplicate collapse below. Any other underscore suffix stays invalid.
    _PAR_Y_SUFFIX = "_PAR_Y"
    par_y_entries = sum(value.endswith(_PAR_Y_SUFFIX) for value in gene_ids)
    canonical = [
        normalize_ensembl_id(value[: -len(_PAR_Y_SUFFIX)] if value.endswith(_PAR_Y_SUFFIX) else value)
        for value in gene_ids
    ]
    unique_ids = list(dict.fromkeys(canonical))
    duplicates = len(canonical) - len(unique_ids)
    if duplicates:
        # Versioned duplicates of one gene are summed, not dropped.
        merged_index = {gene_id: position for position, gene_id in enumerate(unique_ids)}
        selector = sp.csr_matrix(
            (
                np.ones(len(canonical), dtype=np.int32),
                (np.arange(len(canonical)), [merged_index[value] for value in canonical]),
            ),
            shape=(len(canonical), len(unique_ids)),
        )
        counts = counts @ selector
    symbol_by_id: dict[str, str] = {}
    for gene_id, symbol in zip(canonical, gene_symbols, strict=True):
        symbol_by_id.setdefault(gene_id, symbol)
    var = pd.DataFrame(
        {
            "ensembl_id": unique_ids,
            "gene_symbol": [symbol_by_id[value] for value in unique_ids],
        },
        index=pd.Index(unique_ids, name="ensembl_id"),
    )
    return sp.csr_matrix(counts), var, barcodes, duplicates, int(par_y_entries)


def _build_cohort_anndata(
    counts: sp.csr_matrix,
    var: pd.DataFrame,
    barcodes: np.ndarray,
    cell_meta: pd.DataFrame,
) -> tuple[object, int]:
    import anndata as ad

    if not cell_meta["Barcode"].is_unique:
        raise ValueError("cell metadata Barcode must be unique")
    meta_by_barcode = cell_meta.set_index("Barcode")
    keep = np.isin(barcodes, meta_by_barcode.index.to_numpy())
    unannotated = int((~keep).sum())
    counts = counts[keep]
    kept_barcodes = barcodes[keep]
    selected = meta_by_barcode.loc[kept_barcodes]

    obs = pd.DataFrame(
        {
            "donor": selected["SampleID"].astype(str).to_numpy(),
            "state": selected["Diagnosis"].astype(str).map(DIAGNOSIS_TO_STATE).to_numpy(),
            "cell_type": selected["Cell.Type"].astype(str).to_numpy(),
            "sample": selected["SampleID"].astype(str).to_numpy(),
            "batch": selected["Batch"].astype(str).to_numpy(),
            "age": selected["Age"].astype(float).to_numpy(),
            "sex": selected["Sex"].astype(str).to_numpy(),
            "pmi": selected["PMI"].astype(float).to_numpy(),
            "rin": selected["RIN"].astype(float).to_numpy(),
            "tangle_stage": selected["Tangle.Stage"].astype(str).to_numpy(),
            "plaque_stage": selected["Plaque.Stage"].astype(str).to_numpy(),
        },
        index=pd.Index(
            [f"{sample}_{barcode}" for sample, barcode in zip(selected["SampleID"], kept_barcodes, strict=True)],
            name="obs_id",
        ),
    )
    obs["dataset"] = "GSE174367"
    if not obs.index.is_unique:
        raise ValueError("cohort obs_names must be unique")
    adata = ad.AnnData(X=sp.csr_matrix(counts, dtype=np.int32), obs=obs, var=var)
    adata.layers["counts"] = adata.X.copy()
    adata.uns["cohort_schema"] = "ptm2cellnet.ad-cohort.gse174367.v1"
    adata.uns["donor_derivation"] = "SampleID as donor; unique subject covariate vectors 18/18"
    return adata, unannotated


def main() -> int:
    """Run the standardization; exit codes are 0=PASS, 2=PARTIAL, 1=NO_CELL_TYPE_PASSED.

    PARTIAL means at least one cell type passed the Gate-0 preflight and at
    least one was rejected: the cohort artifact exists, but the cohort-level
    verdict is not PASS, so ``&&``-chained pipelines must not treat it as an
    unqualified success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-h5", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--cell-meta", type=Path, default=DEFAULT_CELL_META)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    cell_meta = pd.read_csv(args.cell_meta)
    donor_evidence = _load_donor_evidence(cell_meta)
    counts, var, barcodes, ensembl_duplicates, par_y_entries = _read_pooled_matrix(args.matrix_h5)
    adata, unannotated_cells = _build_cohort_anndata(counts, var, barcodes, cell_meta)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    h5ad_path = args.output_dir / "GSE174367_ad_cohort.h5ad"
    adata.write_h5ad(h5ad_path, compression="gzip")

    spec = PerturbGenDataSpec(pairing="between_donor")
    preflight: dict[str, object] = {}
    for cell_type in sorted(adata.obs["cell_type"].unique()):
        try:
            prepared = prepare_perturbgen_anndata(adata, cell_type=cell_type, spec=spec)
        except ValueError as exc:
            preflight[cell_type] = {"status": "REJECTED", "error": str(exc)[:300]}
        else:
            preflight[cell_type] = {
                "status": "PASS",
                "n_cells": prepared.report.n_cells,
                "n_genes": prepared.report.n_genes,
                "normal_donors": list(prepared.report.normal_donors),
                "disease_donors": list(prepared.report.disease_donors),
                "evaluable_donors": list(prepared.report.evaluable_donors),
            }
    passed = [name for name, item in preflight.items() if item["status"] == "PASS"]

    evidence = {
        "run_id": _dt.date.today().strftime("%Y%m%d") + "_gse174367_gate0",
        "collected_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "purpose": (
            "Standardize GSE174367 into a Gate-0 between_donor case-control cohort and run "
            "the real preflight for every cell type"
        ),
        "inputs": {
            "matrix_h5": {"path": str(args.matrix_h5), "sha256": _sha256(args.matrix_h5)},
            "cell_meta": {"path": str(args.cell_meta), "sha256": _sha256(args.cell_meta)},
        },
        "cohort_h5ad": str(h5ad_path),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "state_distribution": {str(k): int(v) for k, v in adata.obs["state"].value_counts().items()},
        "cells_without_metadata_dropped": unannotated_cells,
        "ensembl_par_y_entries_merged": par_y_entries,
        "ensembl_duplicates_summed": ensembl_duplicates,
        "donor_evidence": donor_evidence,
        "gate0_pairing": "between_donor",
        "preflight_by_cell_type": preflight,
        "verdict": "PASS"
        if len(passed) == len(preflight) and passed
        else ("PARTIAL" if passed else "NO_CELL_TYPE_PASSED"),
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    evidence_path = EVIDENCE_DIR / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False))

    provenance_path = args.output_dir / "GSE174367_ad_cohort_provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                key: evidence[key]
                for key in (
                    "inputs",
                    "cohort_h5ad",
                    "n_cells",
                    "n_genes",
                    "state_distribution",
                    "cells_without_metadata_dropped",
                    "ensembl_par_y_entries_merged",
                    "ensembl_duplicates_summed",
                    "donor_evidence",
                    "gate0_pairing",
                    "preflight_by_cell_type",
                    "verdict",
                )
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(
        json.dumps(
            {
                "h5ad": str(h5ad_path),
                "evidence": str(evidence_path),
                "verdict": evidence["verdict"],
                "cell_types_passed": f"{len(passed)}/{len(preflight)}",
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if evidence["verdict"] == "PASS":
        return 0
    if evidence["verdict"] == "PARTIAL":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
