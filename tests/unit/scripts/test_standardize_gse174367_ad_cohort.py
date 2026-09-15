"""Synthetic unit tests for the GSE174367 standardization contract (TD-14-06/F-13).

The script's pure functions carry the data-entry contract of the AD cohort:
SampleID-as-donor evidence, per-sample Diagnosis uniqueness (F-13), PAR_Y
collapse, and barcode alignment.  These tests never touch data/AD.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.standardize_gse174367_ad_cohort as standardize  # noqa: E402


def _cell_meta(rows: int = 6) -> pd.DataFrame:
    """Three Control + three AD samples, one unique covariate vector each."""

    records = []
    for index in range(rows):
        diagnosis = "Control" if index < 3 else "AD"
        records.append(
            {
                "SampleID": f"S{index}",
                "Diagnosis": diagnosis,
                "Cell.Type": "Neuron",
                "Barcode": f"BC{index:04d}",
                "Age": 60.0 + index,
                "Sex": "M" if index % 2 == 0 else "F",
                "PMI": 10.0 + index,
                "RIN": 5.0 + index * 0.1,
                "Tangle.Stage": f"{'I' * (index % 3 + 1)}",
                "Plaque.Stage": f"{'B' * (index % 2 + 1)}",
                "Batch": f"batch_{index % 2}",
            }
        )
    return pd.DataFrame(records)


class TestLoadDonorEvidence:
    def test_valid_meta_records_donor_diagnosis_counts(self):
        evidence = standardize._load_donor_evidence(_cell_meta())
        assert evidence["n_samples"] == 6
        assert evidence["unique_subject_covariate_vectors"] == 6
        assert evidence["donor_diagnosis"] == {"AD": 3, "Control": 3}

    def test_covariate_collision_between_samples_is_rejected(self):
        meta = _cell_meta()
        # S1 duplicate of S0's covariate vector → SampleID no longer a safe donor id.
        covariates = ["Age", "Sex", "PMI", "RIN", "Tangle.Stage", "Plaque.Stage"]
        for column in covariates:
            meta.loc[meta["SampleID"] == "S1", column] = meta.loc[meta["SampleID"] == "S0", column].iloc[0]
        with pytest.raises(ValueError, match="share one subject covariate vector"):
            standardize._load_donor_evidence(meta)

    def test_per_sample_covariate_drift_is_rejected(self):
        meta = _cell_meta()
        meta = pd.concat([meta, meta.iloc[[0]].assign(Age=99.0)], ignore_index=True)
        with pytest.raises(ValueError, match="constant within each SampleID"):
            standardize._load_donor_evidence(meta)

    def test_per_sample_diagnosis_ambiguity_is_rejected(self):
        """F-13: one SampleID with two Diagnosis labels must abort, not pass."""

        meta = _cell_meta()
        meta = pd.concat([meta, meta.iloc[[0]].assign(Diagnosis="AD")], ignore_index=True)
        with pytest.raises(ValueError, match="constant within each SampleID"):
            standardize._load_donor_evidence(meta)

    def test_small_state_group_is_rejected(self):
        meta = _cell_meta()
        meta.loc[meta["SampleID"] == "S3", "Diagnosis"] = "Control"
        with pytest.raises(ValueError, match="each state group needs >= 3 donors"):
            standardize._load_donor_evidence(meta)

    def test_unknown_diagnosis_is_rejected(self):
        meta = _cell_meta()
        meta.loc[meta["SampleID"] == "S0", "Diagnosis"] = "MCI"
        with pytest.raises(ValueError, match="unexpected Diagnosis values"):
            standardize._load_donor_evidence(meta)


class TestReadPooledMatrix:
    def _write_h5(self, path: Path, gene_ids: list[str]) -> None:
        import h5py

        n_genes, n_cells = len(gene_ids), 2
        csc = np.asarray([[3, 0], [0, 4], [5, 6]], dtype=np.int32)
        with h5py.File(path, "w") as handle:
            group = handle.create_group("matrix")
            group.create_dataset("shape", data=[n_genes, n_cells], dtype=np.int64)
            group.create_dataset("data", data=csc.ravel(), dtype=np.int32)
            group.create_dataset(
                "indices",
                data=np.tile(np.arange(n_genes, dtype=np.int64), n_cells),
            )
            group.create_dataset("indptr", data=[0, n_genes, 2 * n_genes], dtype=np.int64)
            group.create_dataset("barcodes", data=[f"BC{i}".encode() for i in range(n_cells)])
            group.create_dataset("features/id", data=[value.encode() for value in gene_ids])
            group.create_dataset("features/name", data=[f"SYM{i}".encode() for i in range(n_genes)])

    def test_par_y_copies_collapse_into_the_base_ensg_by_sum(self, tmp_path):
        pytest.importorskip("h5py")
        path = tmp_path / "matrix.h5"
        self._write_h5(path, ["ENSG00000223972.5", "ENSG00000223972.5_PAR_Y", "ENSG00000000003.1"])

        counts, var, _barcodes, duplicates, par_y_entries = standardize._read_pooled_matrix(path)

        assert var.index.tolist() == ["ENSG00000223972", "ENSG00000000003"]
        assert par_y_entries == 1
        assert duplicates == 1
        # Cell 0 carries 3 on the base copy only; cell 1 carries 4 on the base
        # copy and 5 on the PAR_Y copy, which must merge by sum into 9.
        assert counts[0, 0] == 3
        assert counts[0, 1] == 0
        assert counts[1, 0] == 9
        assert counts[1, 1] == 6


class TestBuildCohortAnnData:
    def test_unannotated_barcodes_are_dropped_and_counted(self, tmp_path):
        ad = pytest.importorskip("anndata")
        from scipy import sparse

        counts = sparse.csr_matrix(np.arange(8, dtype=np.int32).reshape(4, 2))
        var = pd.DataFrame(
            {"ensembl_id": ["ENSG00000000001", "ENSG00000000002"], "gene_symbol": ["A", "B"]},
            index=["ENSG00000000001", "ENSG00000000002"],
        )
        barcodes = np.asarray(["BC0000", "BC0001", "BC0002", "BC9999"])
        meta = _cell_meta(rows=3)
        meta["Barcode"] = ["BC0000", "BC0001", "BC0002"]

        adata, unannotated = standardize._build_cohort_anndata(counts, var, barcodes, meta)

        assert unannotated == 1
        assert adata.n_obs == 3
        assert set(adata.obs["donor"]) == {"S0", "S1", "S2"}
        assert set(adata.obs["state"]) == {"normal"}
        assert adata.obs.index.is_unique

    def test_duplicate_barcodes_in_metadata_are_rejected(self, tmp_path):
        from scipy import sparse

        counts = sparse.csr_matrix(np.arange(4, dtype=np.int32).reshape(2, 2))
        var = pd.DataFrame(
            {"ensembl_id": ["ENSG00000000001", "ENSG00000000002"], "gene_symbol": ["A", "B"]},
            index=["ENSG00000000001", "ENSG00000000002"],
        )
        barcodes = np.asarray(["BC0000", "BC0001"])
        meta = _cell_meta(rows=2)
        meta["Barcode"] = ["BC0000", "BC0000"]

        with pytest.raises(ValueError, match="Barcode must be unique"):
            standardize._build_cohort_anndata(counts, var, barcodes, meta)
