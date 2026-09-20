"""CLI tests for PTM smoke generation and the 1/3/4/5 contract chain."""

from __future__ import annotations

import json

import pandas as pd
import pytest

anndata = pytest.importorskip("anndata")

from scripts.generate_ptm_smoke import main
from src.analysis.ptm_smoke import SMOKE_METHOD_VERSION, SMOKE_NETWORK_RELEASE


def test_cli_writes_bundle_without_pipeline(tmp_path, capsys):
    output_dir = tmp_path / "bundle"
    assert main(["--output-dir", str(output_dir)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["biology_pass"] is False
    assert payload["kstar_ran"] is False
    assert "pipeline" not in payload
    assert (output_dir / "inputs" / "ptm_site_quantification.tsv").is_file()


def test_cli_run_pipeline_is_contract_pass_not_biology(tmp_path, capsys):
    output_dir = tmp_path / "run"
    assert main(["--output-dir", str(output_dir), "--run-pipeline"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["biology_pass"] is False
    assert payload["kstar_ran"] is False
    report = json.loads((output_dir / "pipeline" / "smoke_pipeline_report.json").read_text(encoding="utf-8"))
    assert report["stage_2_skipped"]
    scores = pd.read_csv(output_dir / "pipeline" / "ptm_global_gene_scores.tsv", sep="\t")
    assert (scores["prediction_status"] == "direction_only").all()
    assert set(scores["ptm_cohort"]) == {"PTM_SMOKE_20260918"}
    activity = pd.read_csv(output_dir / "inputs" / "ptm_activity.smoke_stub.tsv", sep="\t")
    assert set(activity["method_version"]) == {SMOKE_METHOD_VERSION}
    network = pd.read_csv(output_dir / "inputs" / "signed_network.smoke.tsv", sep="\t")
    assert set(network["release"]) == {SMOKE_NETWORK_RELEASE}
    spec = json.loads((output_dir / "pipeline" / "specs" / "candidate_spec_EX.json").read_text(encoding="utf-8"))
    assert spec["observed_admission_rule"] == "signed_direction_without_fdr_cutoff"
    assert spec["candidates"]
    assert all(row["observed_significant"] is False for row in spec["candidates"])
    summary = json.loads((output_dir / "pipeline" / "specs" / "build_summary.json").read_text(encoding="utf-8"))
    inh = summary["cell_types"]["INH"]
    assert inh["n_candidates"] == 0
    assert inh["exploratory_sources"]
