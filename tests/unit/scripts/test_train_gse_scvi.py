"""Contract tests for the GSE-specific scVI training entry point."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scripts.train_gse_scvi import validate_gse_scvi_input


def _valid_adata() -> ad.AnnData:
    n_obs = 12
    n_genes = 4018
    obs = pd.DataFrame(
        {
            "cell_type": ["T"] * n_obs,
            "state": ["normal"] * 6 + ["disease"] * 6,
            "donor": [f"N{index}" for index in range(3) for _ in range(2)]
            + [f"D{index}" for index in range(3) for _ in range(2)],
            "davf_batch": [f"sample_{index}" for index in range(n_obs)],
        },
        index=[f"cell_{index}" for index in range(n_obs)],
    )
    gene_names = [f"ENSG{index:011d}" for index in range(n_genes)]
    var = pd.DataFrame({"ensembl_id": gene_names}, index=gene_names)
    counts = sp.csr_matrix(np.ones((n_obs, n_genes), dtype=np.int32))
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.layers["counts"] = counts.copy()
    adata.uns["gse_normal_disease"] = {
        "schema_version": "ptm2cellnet.gse-normal-disease.v1",
    }
    return adata


def test_validate_gse_scvi_input_accepts_current_contract():
    report = validate_gse_scvi_input(_valid_adata())

    assert report["shape"] == [12, 4018]
    assert report["counts_layer"] == "counts"
    assert report["observational_only"] is True


def test_validate_gse_scvi_input_rejects_nonformal_gene_axis():
    adata = _valid_adata()[:, :4017].copy()

    with pytest.raises(ValueError, match="4018 genes"):
        validate_gse_scvi_input(adata)


def test_validate_gse_scvi_input_rejects_missing_raw_counts_layer():
    adata = _valid_adata()
    del adata.layers["counts"]

    with pytest.raises(ValueError, match="raw counts"):
        validate_gse_scvi_input(adata)
