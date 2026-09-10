"""Tests for the real scPerturb DAVF preparation contract."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data.davf_scperturb import (
    DAVFScPerturbError,
    build_scperturb_latent_pairs,
    prepare_scperturb_anndata,
    split_target_cells,
    split_target_labels,
)


def _write_source(path: Path, *, modality: str = "KO") -> None:
    gene_ids = [f"ENSG{index:011d}" for index in range(4018)]
    gene_symbols = [f"G{index}" for index in range(4018)]
    gene_symbols[0] = "TARGET_A"
    gene_symbols[1] = "TARGET_B"
    matrix = sp.lil_matrix((8, 4018), dtype=np.float32)
    matrix[0, 0] = 4
    matrix[1, 0] = 3
    matrix[2, 1] = 2
    matrix[3, 1] = 1
    matrix[4, 4] = 9
    matrix[5, 4] = 8
    matrix[6, 4] = 7
    matrix[7, 4] = 6
    if modality == "KO":
        obs = pd.DataFrame(
            {
                "perturbation": ["control", "control", "TARGET_A", "TARGET_A", "TARGET_B", "TARGET_B", "unknown", "unknown"],
                "target": [np.nan, np.nan, "TARGET_A", "TARGET_A", "TARGET_B", "TARGET_B", "INTERGENIC", "INTERGENIC"],
                "batch": ["b0"] * 8,
            },
            index=[f"cell_{index}" for index in range(8)],
        )
    else:
        obs = pd.DataFrame(
            {
                "gene_id": ["non-targeting", "non-targeting", gene_ids[0], gene_ids[0], gene_ids[1], gene_ids[1], "bad", "bad"],
                "gene": ["non-targeting", "non-targeting", "TARGET_A", "TARGET_A", "TARGET_B", "TARGET_B", "bad", "bad"],
                "perturbation": ["control", "control", "TARGET_A", "TARGET_A", "TARGET_B", "TARGET_B", "bad", "bad"],
                "batch": ["b0"] * 8,
            },
            index=[f"cell_{index}" for index in range(8)],
        )
    var = pd.DataFrame({"gene_id": gene_ids, "ncells": np.arange(4018, 0, -1)}, index=gene_symbols)
    ad.AnnData(X=matrix.tocsr(), obs=obs, var=var).write_h5ad(path)


def test_split_target_labels_is_deterministic_and_label_disjoint():
    first = split_target_labels(["A", "A", "B", "C", "D"], seed=9)
    second = split_target_labels(["D", "C", "B", "A"], seed=9)
    assert first == second
    assert not set(first["train"]) & set(first["val"])
    assert not set(first["train"]) & set(first["test"])
    assert not set(first["val"]) & set(first["test"])


def test_split_target_cells_is_deterministic_and_keeps_observed_cells_disjoint():
    values = ["A"] * 10 + ["B"] * 4 + ["C"] * 2
    first = split_target_cells(values, seed=9)
    second = split_target_cells(values, seed=9)

    assert first == second
    assert not set(first["train"]) & set(first["val"])
    assert not set(first["train"]) & set(first["test"])
    assert not set(first["val"]) & set(first["test"])
    assert set(first["train"]) | set(first["val"]) | set(first["test"]) == set(range(len(values)))
    assert all(values[index] == "C" for index in first["train"] if index >= 14)
    assert not any(values[index] == "C" for split in ("val", "test") for index in first[split])


def test_split_target_cells_rejects_invalid_baseline_strategy():
    import pytest

    with pytest.raises(DAVFScPerturbError, match="control_baseline"):
        # The public pair builder validates this argument before reading data;
        # keep the helper-level regression close to the split contract.
        from src.data.davf_scperturb import build_scperturb_latent_pairs

        build_scperturb_latent_pairs(
            "missing.h5ad",
            scvi_model_path="missing-scvi",
            embedding_asset=object(),
            embedding_asset_path="missing-asset",
            output_dir="missing-output",
            modality="KO",
            control_baseline="invalid",
        )


def test_prepare_ko_retains_targets_and_writes_exact_gene_order(tmp_path):
    source = tmp_path / "ko.h5ad"
    _write_source(source, modality="KO")
    asset = {
        "ENSG00000000000": 17,
        "ENSG00000000001": 23,
    }
    report = prepare_scperturb_anndata(
        [source],
        modality="KO",
        asset_gene_to_token=asset,
        output_path=tmp_path / "prepared.h5ad",
        embedding_asset_path=tmp_path / "asset",
        embedding_manifest={"schema_version": 1},
    )

    prepared = ad.read_h5ad(tmp_path / "prepared.h5ad")
    assert prepared.shape == (6, 4018)
    assert tuple(prepared.var_names[:2]) == ("ENSG00000000000", "ENSG00000000001")
    assert set(prepared.obs["davf_target_ensembl"].astype(str)) == {"", "ENSG00000000000", "ENSG00000000001"}
    assert prepared.uns["davf_preparation"]["modality"] == "KO"
    assert json.loads((tmp_path / "prepared.manifest.json").read_text())["target_genes"] == [
        "ENSG00000000000",
        "ENSG00000000001",
    ]
    assert report["shape"] == [6, 4018]


def test_prepare_rejects_input_without_target_cells(tmp_path):
    source = tmp_path / "bad.h5ad"
    _write_source(source, modality="KO")
    source_data = ad.read_h5ad(source)
    source_data.obs["target"] = np.nan
    source_data.obs["perturbation"] = "control"
    source_data.write_h5ad(source)

    import pytest

    with pytest.raises(DAVFScPerturbError, match="no asset-resolvable target"):
        prepare_scperturb_anndata(
            [source],
            modality="KO",
            asset_gene_to_token={"ENSG00000000000": 17},
            output_path=tmp_path / "prepared.h5ad",
            embedding_asset_path=tmp_path / "asset",
            embedding_manifest={"schema_version": 1},
        )


def test_pair_builder_requires_matching_prepared_modality(tmp_path):
    source = tmp_path / "ko.h5ad"
    _write_source(source, modality="KO")
    prepare_scperturb_anndata(
        [source],
        modality="KO",
        asset_gene_to_token={"ENSG00000000000": 17, "ENSG00000000001": 23},
        output_path=tmp_path / "prepared.h5ad",
        embedding_asset_path=tmp_path / "asset",
        embedding_manifest={"schema_version": 1},
    )

    import pytest
    from types import SimpleNamespace

    with pytest.raises(DAVFScPerturbError, match="modality"):
        build_scperturb_latent_pairs(
            tmp_path / "prepared.h5ad",
            scvi_model_path=tmp_path / "scvi",
            embedding_asset=SimpleNamespace(),
            embedding_asset_path=tmp_path / "asset",
            output_dir=tmp_path / "pairs",
            modality="KD",
        )
