"""Unit tests for orientation-safe 10x loading and sample-wise QC."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

ad = pytest.importorskip("anndata")

from src.analysis.ibd_qc import _build_adata_from_components, run_qc_on_adata


def test_build_adata_transposes_feature_by_cell_matrix_and_selects_rows():
    features = pd.DataFrame(
        [["g1", "MT-ND1", "Gene Expression"], ["g2", "CD3D", "Gene Expression"]]
    )
    # Matrix Market convention: features x barcodes.
    matrix = sparse.csr_matrix(np.asarray([[1, 0, 2], [0, 3, 0]], dtype=np.float32))
    row = {
        "dataset": "GSE-test",
        "GSM": "GSM-test",
        "library_id": "library",
        "sample_id": "sample",
        "donor_id": "donor",
    }

    result = _build_adata_from_components(
        matrix=matrix,
        barcodes=["bc1", "bc2", "bc3"],
        features=features,
        metadata_row=row,
        selected_positions=[0, 2],
    )

    assert result.shape == (2, 2)
    assert result.obs["barcode"].tolist() == ["bc1", "bc3"]
    assert result.X.toarray().tolist() == [[1.0, 0.0], [2.0, 0.0]]


def test_qc_groups_by_sample_index_not_obs_labels():
    result = ad.AnnData(
        X=sparse.csr_matrix(np.asarray([[1, 0], [0, 2], [1, 1]], dtype=np.float32)),
        obs=pd.DataFrame(
            {"sample_id": ["s1", "s1", "s2"]},
            index=["s1:cell-a", "s1:cell-b", "s2:cell-a"],
        ),
        var=pd.DataFrame(
            {"gene_symbol": ["MT-ND1", "CD3D"]}, index=["g1", "g2"]
        ),
    )

    summary = run_qc_on_adata(result)

    assert set(summary) == {"s1", "s2"}
    assert all("analysis_cells" in values for values in summary.values())
    assert all("doublet_cells" in values for values in summary.values())
    assert result.obs["qc_pass"].dtype == bool
    assert result.obs["analysis_pass"].all()
