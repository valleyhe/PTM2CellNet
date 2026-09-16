"""Unit tests for scVI-gene-axis context preparation."""

from types import SimpleNamespace

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import tempfile
from pathlib import Path

from src.data.scvi_context import (
    MISSING_GENE_POLICIES,
    ScviContextError,
    prepare_scvi_context,
    write_scvi_context,
)


AXIS = ("ENSG00000000001", "ENSG00000000002", "ENSG00000000003", "ENSG00000000004")


def _fake_adapter(mapping=("train_batch_0", "train_batch_1"), axis=AXIS):
    registry = {
        "setup_args": {"batch_key": "davf_batch"},
        "field_registries": {
            "batch": {"state_registry": {"categorical_mapping": np.asarray(mapping)}},
        },
    }
    config = SimpleNamespace(model_path="/checkpoints/scvi/davf_ko_test")
    return SimpleNamespace(gene_names=list(axis), model=SimpleNamespace(registry_=registry), config=config)


def _cohort():
    counts = sp.csr_matrix(
        np.array(
            [
                [10, 0, 5],
                [20, 1, 6],
                [30, 2, 7],
                [40, 3, 8],
            ],
            dtype=np.float32,
        )
    )
    obs = pd.DataFrame(
        {
            "cell_type": ["EX"] * 4,
            "state": ["normal", "normal", "disease", "disease"],
            "donor": ["N1", "N2", "D1", "D2"],
        },
        index=[f"cell-{i}" for i in range(4)],
    )
    var = pd.DataFrame(
        {
            # ENSG00000000002 is deliberately absent from the cohort.
            "ensembl_id": ["ENSG00000000001", "ENSG00000000003", "ENSG00000000004"],
            "gene_symbol": ["GENE1", "GENE3", "GENE4"],
        },
        index=["gene-1", "gene-3", "gene-4"],
    )
    adata = anndata.AnnData(X=counts, obs=obs, var=var)
    adata.layers["counts"] = counts.copy()
    return adata


def _aliases():
    path = Path(tempfile.mkdtemp()) / "aliases.tsv"
    rows = pd.DataFrame(
        {
            "ensembl_id": list(AXIS),
            "gene_symbol": [f"GENE{index}" for index in range(1, len(AXIS) + 1)],
        }
    )
    rows.to_csv(path, sep="\t", index=False)
    return path


def _prepare(**overrides):
    kwargs = dict(
        cohort=_cohort(),
        adapter=_fake_adapter(),
        davf_route="ko",
        davf_batch="train_batch_0",
        missing_gene_policy="zero_fill",
        batch_binding_rationale="cohort has no perturbation batches; bind to the reference control batch",
        gene_aliases_path=_aliases(),
    )
    kwargs.update(overrides)
    return prepare_scvi_context(**kwargs)


def test_zero_fill_aligns_axis_and_records_missing_gene():
    context, manifest = _prepare()

    assert tuple(context.var["ensembl_id"]) == AXIS
    assert context.shape == (4, len(AXIS))
    dense = np.asarray(context.layers["counts"].todense())
    # column order follows the axis; the missing gene becomes an all-zero column
    np.testing.assert_array_equal(dense[:, 0], [10, 20, 30, 40])
    np.testing.assert_array_equal(dense[:, 1], [0, 0, 0, 0])
    np.testing.assert_array_equal(dense[:, 2], [0, 1, 2, 3])
    np.testing.assert_array_equal(dense[:, 3], [5, 6, 7, 8])
    assert set(context.obs["davf_batch"]) == {"train_batch_0"}
    assert list(context.obs["donor"]) == ["N1", "N2", "D1", "D2"]
    assert manifest["missing_axis_genes"] == ["ENSG00000000002"]
    assert manifest["n_missing_axis_genes"] == 1
    assert manifest["n_dropped_cohort_genes"] == 0
    assert manifest["davf_route"] == "ko"
    assert manifest["davf_batch"] == "train_batch_0"
    assert manifest["gene_axis_size"] == len(AXIS)


def test_fail_policy_rejects_missing_axis_genes():
    with pytest.raises(ScviContextError, match="missing_gene_policy is 'fail'"):
        _prepare(missing_gene_policy="fail")


def test_rejects_batch_outside_registry():
    with pytest.raises(ScviContextError, match="not in the scVI batch registry"):
        _prepare(davf_batch="invented_batch")


def test_rejects_empty_binding_rationale():
    with pytest.raises(ScviContextError, match="batch_binding_rationale is required"):
        _prepare(batch_binding_rationale="   ")


def test_rejects_unknown_route_and_policy():
    with pytest.raises(ScviContextError, match="davf_route"):
        _prepare(davf_route="oe")
    with pytest.raises(ScviContextError, match="missing_gene_policy"):
        _prepare(missing_gene_policy="drop")


def test_policy_choices_are_frozen():
    assert MISSING_GENE_POLICIES == ("zero_fill", "fail")


def test_rejects_adapter_without_batch_registry():
    adapter = _fake_adapter()
    adapter.model = SimpleNamespace(registry_={"setup_args": {}, "field_registries": {}})
    with pytest.raises(ScviContextError, match="batch"):
        _prepare(adapter=adapter)


def test_rejects_alias_table_that_misses_axis_genes():
    path = Path(tempfile.mkdtemp()) / "partial.tsv"
    pd.DataFrame(
        {
            "ensembl_id": ["ENSG00000000001"],
            "gene_symbol": ["GENE1"],
        }
    ).to_csv(path, sep="\t", index=False)
    with pytest.raises(ScviContextError, match="does not cover"):
        _prepare(gene_aliases_path=path)


def test_write_scvi_context_roundtrip(tmp_path):
    context, manifest = _prepare()
    output = tmp_path / "context.h5ad"
    manifest_output = tmp_path / "manifest.json"
    write_scvi_context(context, manifest, output, manifest_output)

    loaded = anndata.read_h5ad(output)
    assert tuple(loaded.var["ensembl_id"]) == AXIS
    assert "counts" in loaded.layers
    assert set(loaded.obs["davf_batch"]) == {"train_batch_0"}
    payload = pd.read_json(manifest_output, typ="series")
    assert payload["schema_version"] == "ptm2cellnet.scvi-context/v1"
    assert payload["davf_batch"] == "train_batch_0"
