"""CLI tests for building aggregate and donor-level AD DEG tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

anndata = pytest.importorskip("anndata")

from scripts.build_ad_deg_table import main  # noqa: E402


def _write_config(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "ptm2cellnet.ptm-research-config/v1",
                "research_objective": "association",
                "reference_axis": "disease_minus_normal",
                "contrast": "disease-minus-normal",
                "primary_activity_method": "KSTAR",
                "network_release": "2026-08",
                "cell_types": ["neuron"],
                "cohort_h5ad": "cohort.h5ad",
                "cohort_pairing": "between_donor",
                "species": "9606",
                "ptm_cohort": "CPTAC_TEST",
                "deg_max_fdr": 0.05,
                "min_donors_per_state": 2,
                "replicate_policy": "mean",
                "propagation": {"max_depth": 1, "decay": 0.5, "gene_edge_types": ["tf_regulation"]},
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_adata(path: Path, *, missing_obs: str | None = None) -> Path:
    counts = np.array([[2000, 8000], [3000, 7000], [6000, 4000], [7000, 3000]], dtype=np.int64)
    obs = pd.DataFrame(
        {
            "cell_type": ["neuron"] * 4,
            "state": ["normal", "normal", "disease", "disease"],
            "donor": ["N1", "N2", "D1", "D2"],
        },
        index=["cell-1", "cell-2", "cell-3", "cell-4"],
    )
    if missing_obs is not None:
        obs = obs.drop(columns=missing_obs)
    anndata.AnnData(
        X=counts,
        layers={"counts": counts.copy()},
        obs=obs,
        var=pd.DataFrame(
            {
                "ensembl_id": ["ENSG00000000001", "ENSG00000000002"],
                "gene_symbol": ["GENE1", "GENE2"],
            },
            index=["gene-1", "gene-2"],
        ),
    ).write_h5ad(path)
    return path


def _cli_args(config: Path, output_dir: Path) -> list[str]:
    return [
        "--config",
        str(config),
        "--output-tsv",
        str(output_dir / "aggregate.tsv"),
        "--donor-level-output",
        str(output_dir / "donor.csv"),
        "--manifest-output",
        str(output_dir / "manifest.json"),
    ]


def test_main_writes_ad_deg_outputs_from_frozen_config(tmp_path: Path):
    config = _write_config(tmp_path / "ptm_research_config.yaml")
    cohort = _write_adata(tmp_path / "cohort.h5ad")
    output_dir = tmp_path / "outputs"

    assert main(_cli_args(config, output_dir)) == 0

    aggregate_output = output_dir / "aggregate.tsv"
    donor_output = output_dir / "donor.csv"
    manifest_output = output_dir / "manifest.json"
    assert aggregate_output.is_file()
    assert donor_output.is_file()
    assert manifest_output.is_file()

    manifest = json.loads(manifest_output.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "ptm2cellnet.ad-deg-manifest/v1"
    assert manifest["input_h5ad"] == str(cohort.resolve())
    assert manifest["columns"] == {"cell_type": "cell_type", "state": "state", "donor": "donor"}
    assert manifest["cell_types"] == ["neuron"]
    assert manifest["cohort_pairing"] == "between_donor"
    assert manifest["audit"]["normal_state"] == "normal"
    assert manifest["audit"]["disease_state"] == "disease"
    assert manifest["audit"]["normal_reference"] == "per cell type normal donor-level log2(normalized counts) mean"
    assert manifest["audit"]["effect_scale"] == "disease donor mean minus normal donor mean"
    assert manifest["output_contracts"] == {
        "aggregate": {
            "path": str(aggregate_output.resolve()),
            "format": "tsv",
            "columns": [
                "cell_type",
                "ensembl_id",
                "gene_symbol",
                "log2fc",
                "fdr",
                "observed_direction",
                "n_normal_donors",
                "n_disease_donors",
            ],
        },
        "donor_level": {
            "path": str(donor_output.resolve()),
            "format": "csv",
            "columns": ["cell_type", "donor", "ensembl_id", "gene_symbol", "log2fc", "fdr"],
        },
    }

    aggregate = pd.read_csv(aggregate_output, sep="\t")
    assert list(aggregate.columns) == [
        "cell_type",
        "ensembl_id",
        "gene_symbol",
        "log2fc",
        "fdr",
        "observed_direction",
        "n_normal_donors",
        "n_disease_donors",
    ]
    assert len(aggregate.columns) == 8


def test_main_returns_one_for_missing_config(tmp_path: Path, capsys):
    assert main(_cli_args(tmp_path / "missing.yaml", tmp_path / "outputs")) == 1
    assert capsys.readouterr().err


def test_main_returns_one_for_missing_cohort(tmp_path: Path, capsys):
    config = _write_config(tmp_path / "ptm_research_config.yaml")

    assert main(_cli_args(config, tmp_path / "outputs")) == 1
    assert capsys.readouterr().err


def test_main_returns_one_for_missing_obs(tmp_path: Path, capsys):
    config = _write_config(tmp_path / "ptm_research_config.yaml")
    _write_adata(tmp_path / "cohort.h5ad", missing_obs="donor")

    assert main(_cli_args(config, tmp_path / "outputs")) == 1
    assert capsys.readouterr().err
