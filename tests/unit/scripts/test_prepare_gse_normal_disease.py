"""Tests for the GSE preparation CLI contract."""

from __future__ import annotations

from types import SimpleNamespace

from scripts.prepare_gse_normal_disease import _asset_gene_ids, parse_args


def test_asset_gene_ids_excludes_perturbgen_special_tokens():
    asset = SimpleNamespace(
        gene_to_token={
            "<cls>": 0,
            "<eos>": 1,
            "ENSG00000000001": 2,
            "ENSG00000000002.7": 3,
        }
    )

    assert _asset_gene_ids(asset) == ("ENSG00000000001", "ENSG00000000002")


def test_parse_args_keeps_explicit_sample_lists():
    args = parse_args(
        [
            "--raw-dir",
            "raw",
            "--annotation",
            "annotation.csv.gz",
            "--normal-samples",
            "HC1,HC2",
            "--disease-samples",
            "CD1,CD2",
            "--embedding-asset",
            "asset",
            "--output",
            "prepared.h5ad",
        ]
    )

    assert args.normal_samples == "HC1,HC2"
    assert args.disease_samples == "CD1,CD2"
    assert args.dataset_accession == "GSE214695"

