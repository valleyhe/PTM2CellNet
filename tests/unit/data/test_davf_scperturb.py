"""Tests for the real scPerturb DAVF preparation contract."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.data import davf_scperturb as davf
from src.data.davf_scperturb import (
    DAVFScPerturbError,
    build_scperturb_latent_pairs,
    prepare_scperturb_anndata,
    split_target_cells,
    split_target_labels,
)
from src.utils.dependency_check import DependencyStatus, MissingDependencyError


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
                "perturbation": [
                    "control",
                    "control",
                    "TARGET_A",
                    "TARGET_A",
                    "TARGET_B",
                    "TARGET_B",
                    "unknown",
                    "unknown",
                ],
                "target": [np.nan, np.nan, "TARGET_A", "TARGET_A", "TARGET_B", "TARGET_B", "INTERGENIC", "INTERGENIC"],
                "batch": ["b0"] * 8,
            },
            index=[f"cell_{index}" for index in range(8)],
        )
    else:
        obs = pd.DataFrame(
            {
                "gene_id": [
                    "non-targeting",
                    "non-targeting",
                    gene_ids[0],
                    gene_ids[0],
                    gene_ids[1],
                    gene_ids[1],
                    "bad",
                    "bad",
                ],
                "gene": [
                    "non-targeting",
                    "non-targeting",
                    "TARGET_A",
                    "TARGET_A",
                    "TARGET_B",
                    "TARGET_B",
                    "bad",
                    "bad",
                ],
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


def test_optional_preflight_accepts_available_anndata(monkeypatch):
    calls = []
    monkeypatch.setattr(davf, "require_extras", lambda names, **kwargs: calls.append((names, kwargs)))

    assert davf._require_anndata() is ad
    assert calls == [(["anndata"], {"feature": "DAVF scPerturb data preparation"})]


def test_optional_preflight_hard_fails_with_install_hint(monkeypatch):
    status = DependencyStatus(
        import_name="anndata",
        available=False,
        import_error=ModuleNotFoundError("No module named 'anndata'"),
        extra="analysis",
        install_name="anndata",
    )

    def raise_missing(*args, **kwargs):
        raise MissingDependencyError('pip install -e ".[analysis]"', [status])

    monkeypatch.setattr(davf, "require_extras", raise_missing)
    with pytest.raises(MissingDependencyError, match=r"\.\[analysis\]"):
        davf._require_anndata()


class _FakeScVIAdapterForDonors:
    """Deterministic stand-in for the scVI encoder used by donor-bound pairs."""

    def __init__(self, gene_names: tuple[str, ...]) -> None:
        self.gene_names = gene_names

    @classmethod
    def from_trained_model(cls, *_args, **_kwargs):
        return cls(_kwargs["adata"].var_names.tolist())

    def validate_compatibility(self, **_kwargs) -> None:
        return None

    def encode(self, adata, batch_size=512):
        rng = np.random.default_rng(1234)
        return rng.normal(size=(adata.n_obs, 64)).astype(np.float32)


def _prepared_donor_anndata(path: Path) -> None:
    gene_ids = [f"ENSG{index:011d}" for index in range(4018)]
    matrix = sp.csr_matrix((10, 4018), dtype=np.float32)
    obs = pd.DataFrame(
        {
            "davf_target_ensembl": [
                "ENSG00000000000",
                "ENSG00000000000",
                "ENSG00000000001",
                "ENSG00000000001",
                "",
                "",
                "ENSG00000000000",
                "ENSG00000000001",
                "",
                "",
            ],
            "davf_batch": ["b0"] * 10,
            "donor": ["d1", "d1", "d2", "d2", "d1", "d2", "d3", "d4", "d3", "d4"],
        },
        index=[f"cell_{index}" for index in range(10)],
    )
    prepared = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=gene_ids))
    prepared.uns["davf_preparation"] = {"modality": "KO"}
    prepared.write_h5ad(path)


def _donor_asset():
    from types import SimpleNamespace

    return SimpleNamespace(
        gene_to_token={"ENSG00000000000": 17, "ENSG00000000001": 23},
        vocab_size=50,
        embedding_dim=64,
        manifest={"schema_version": 1},
    )


def _build_donor_bound_pairs(tmp_path, monkeypatch, donor_split, **overrides):
    prepared_path = tmp_path / "prepared.h5ad"
    _prepared_donor_anndata(prepared_path)
    monkeypatch.setattr(
        "src.models.scvi_adapter.ScVIAdapter",
        _FakeScVIAdapterForDonors,
    )
    kwargs = dict(
        scvi_model_path=tmp_path / "scvi",
        embedding_asset=_donor_asset(),
        embedding_asset_path=tmp_path / "asset",
        output_dir=tmp_path / "pairs",
        modality="KO",
        donor_obs_column="donor",
        donor_split=donor_split,
    )
    kwargs.update(overrides)
    return build_scperturb_latent_pairs(prepared_path, **kwargs)


_DONOR_SPLIT_PAYLOAD = {
    "schema_version": "ptm2cellnet.donor_split/v1",
    "train_donors": ["d1", "d2"],
    "held_out_donors": ["d3", "d4"],
    "sha256": "0" * 64,
}


def test_donor_bound_pairs_keep_train_and_held_out_donors_in_separate_npz(tmp_path, monkeypatch):
    report = _build_donor_bound_pairs(tmp_path, monkeypatch, _DONOR_SPLIT_PAYLOAD)

    assert set(report) == {"train", "val", "test"}
    for split, allowed, pool in (
        ("train", {"d1", "d2"}, "train_donors"),
        ("val", {"d1", "d2"}, "train_donors"),
        ("test", {"d3", "d4"}, "held_out_donors"),
    ):
        with np.load(report[split]["path"], allow_pickle=False) as data:
            donors = set(data["target_donors"].tolist())
            assert donors
            assert donors <= allowed
            metadata = json.loads(data["metadata_json"].item())
            assert metadata["donor_split"] == _DONOR_SPLIT_PAYLOAD
            assert metadata["dataset"]["donor_rows"] == data["target_donors"].tolist()
            assert metadata["dataset"]["donor_pool"] == pool
            assert metadata["dataset"]["donor_obs_column"] == "donor"

    manifest = json.loads((tmp_path / "pairs" / "pair_manifest.json").read_text(encoding="utf-8"))
    assert manifest["donor_split"] == _DONOR_SPLIT_PAYLOAD
    assert manifest["donor_obs_column"] == "donor"


def test_donor_bound_pairs_reject_donor_outside_both_pools(tmp_path, monkeypatch):
    payload = dict(_DONOR_SPLIT_PAYLOAD)
    payload["held_out_donors"] = ["d9"]
    with pytest.raises(DAVFScPerturbError, match="unassigned/missing donors"):
        _build_donor_bound_pairs(tmp_path, monkeypatch, payload)


def test_donor_bound_pairs_require_explicit_donor_column(tmp_path, monkeypatch):
    with pytest.raises(DAVFScPerturbError, match="no donor column"):
        _build_donor_bound_pairs(
            tmp_path,
            monkeypatch,
            _DONOR_SPLIT_PAYLOAD,
            donor_obs_column="missing_donor",
        )


def test_donor_bound_pairs_reject_leaking_donor_lists(tmp_path, monkeypatch):
    payload = dict(_DONOR_SPLIT_PAYLOAD)
    payload["held_out_donors"] = ["d1", "d3", "d4"]
    with pytest.raises(DAVFScPerturbError, match="leaks donors into both pools"):
        _build_donor_bound_pairs(tmp_path, monkeypatch, payload)


def _prepared_state_anndata(path: Path, held_out_states: list[str]) -> None:
    """Donor-bound prepared AnnData with a two-value ``state`` obs column.

    Train donors d1/d2 contribute both states; held-out donors d3/d4 carry
    ``held_out_states`` (one cell of each target/none combination per donor
    keeps the control pools satisfiable).
    """
    gene_ids = [f"ENSG{index:011d}" for index in range(4018)]
    matrix = sp.csr_matrix((10, 4018), dtype=np.float32)
    obs = pd.DataFrame(
        {
            "davf_target_ensembl": [
                "ENSG00000000000",
                "ENSG00000000000",
                "ENSG00000000001",
                "ENSG00000000001",
                "",
                "",
                "ENSG00000000000",
                "ENSG00000000001",
                "",
                "",
            ],
            "davf_batch": ["b0"] * 10,
            "donor": ["d1", "d1", "d2", "d2", "d1", "d2", "d3", "d4", "d3", "d4"],
            "state": ["normal", "disease", "normal", "disease", "normal", "disease"] + held_out_states * 2,
        },
        index=[f"cell_{index}" for index in range(10)],
    )
    prepared = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=gene_ids))
    prepared.uns["davf_preparation"] = {"modality": "KO"}
    prepared.write_h5ad(path)


def _build_state_bound_pairs(tmp_path, monkeypatch, held_out_states: list[str], **overrides):
    prepared_path = tmp_path / "prepared.h5ad"
    _prepared_state_anndata(prepared_path, held_out_states)
    monkeypatch.setattr(
        "src.models.scvi_adapter.ScVIAdapter",
        _FakeScVIAdapterForDonors,
    )
    kwargs = dict(
        scvi_model_path=tmp_path / "scvi",
        embedding_asset=_donor_asset(),
        embedding_asset_path=tmp_path / "asset",
        output_dir=tmp_path / "pairs",
        modality="KO",
        donor_obs_column="donor",
        donor_split=_DONOR_SPLIT_PAYLOAD,
        state_obs_column="state",
        require_state_coverage=True,
    )
    kwargs.update(overrides)
    return build_scperturb_latent_pairs(prepared_path, **kwargs)


def test_require_state_coverage_rejects_single_state_held_out_pool(tmp_path, monkeypatch):
    with pytest.raises(DAVFScPerturbError, match="fewer than two states"):
        _build_state_bound_pairs(tmp_path, monkeypatch, ["normal", "normal"])


def test_require_state_coverage_passes_and_records_distribution(tmp_path, monkeypatch):
    report = _build_state_bound_pairs(tmp_path, monkeypatch, ["normal", "disease"])
    assert set(report) == {"train", "val", "test"}
    manifest = json.loads((tmp_path / "pairs" / "pair_manifest.json").read_text(encoding="utf-8"))
    assert manifest["state_obs_column"] == "state"
    assert manifest["held_out_state_coverage"] == {"disease": 2, "normal": 2}


def test_require_state_coverage_needs_explicit_state_column(tmp_path, monkeypatch):
    with pytest.raises(DAVFScPerturbError, match="needs an explicit state_obs_column"):
        _build_state_bound_pairs(
            tmp_path,
            monkeypatch,
            ["normal", "disease"],
            state_obs_column=None,
        )
    with pytest.raises(DAVFScPerturbError, match="no state column"):
        _build_state_bound_pairs(
            tmp_path,
            monkeypatch,
            ["normal", "disease"],
            state_obs_column="missing_state",
        )


def test_state_coverage_check_is_opt_in(tmp_path, monkeypatch):
    report = _build_state_bound_pairs(
        tmp_path,
        monkeypatch,
        ["normal", "normal"],
        require_state_coverage=False,
    )
    assert set(report) == {"train", "val", "test"}
    manifest = json.loads((tmp_path / "pairs" / "pair_manifest.json").read_text(encoding="utf-8"))
    assert "held_out_state_coverage" not in manifest


def test_require_state_coverage_without_donor_split_is_rejected(tmp_path, monkeypatch):
    prepared_path = tmp_path / "prepared.h5ad"
    _prepared_state_anndata(prepared_path, ["normal", "disease"])
    monkeypatch.setattr(
        "src.models.scvi_adapter.ScVIAdapter",
        _FakeScVIAdapterForDonors,
    )
    with pytest.raises(DAVFScPerturbError, match="only applies to donor-bound splits"):
        build_scperturb_latent_pairs(
            prepared_path,
            scvi_model_path=tmp_path / "scvi",
            embedding_asset=_donor_asset(),
            embedding_asset_path=tmp_path / "asset",
            output_dir=tmp_path / "pairs",
            modality="KO",
            state_obs_column="state",
            require_state_coverage=True,
        )
