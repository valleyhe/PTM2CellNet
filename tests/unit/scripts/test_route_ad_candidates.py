"""CLI tests for AD candidate routing."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analysis.ad_candidate_routing import FROZEN_AD_CANDIDATES
from scripts.route_ad_candidates import main


def _axis_tsv(path: Path) -> Path:
    pd.DataFrame(
        [
            {
                "gene_symbol": symbol,
                "ensembl_id": ensembl_id,
                "embedding_vocab": True,
                "ko_in_axis": symbol == "APOE",
                "kd_in_axis": False,
                "scperturb_perturbed_in": "FrangiehIzar2021_RNA" if symbol == "APOE" else "",
            }
            for symbol, ensembl_id in FROZEN_AD_CANDIDATES
        ]
    ).to_csv(path, sep="\t", index=False)
    return path


def _deg_tsv(path: Path) -> Path:
    rows = []
    for symbol, ensembl_id in FROZEN_AD_CANDIDATES:
        rows.append(
            {
                "cell_type": "EX",
                "ensembl_id": ensembl_id,
                "gene_symbol": symbol,
                "log2fc": 0.2,
                "fdr": 0.52,
                "observed_direction": "up",
                "n_normal_donors": 7,
                "n_disease_donors": 11,
            }
        )
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def test_cli_writes_exploratory_split_without_formal_invocations(tmp_path, capsys):
    axis = _axis_tsv(tmp_path / "axis.tsv")
    deg = _deg_tsv(tmp_path / "deg.tsv")
    anchor = tmp_path / "anchor.json"
    anchor.write_text(json.dumps({"verdict": "fail"}), encoding="utf-8")
    geneformer = tmp_path / "geneformer.json"
    geneformer.write_text(
        json.dumps(
            {
                "evidence_kind": "network_counterfactual",
                "lineage_boundary": "supplementary_only",
                "candidates": {
                    "ENSG00000130203": {"predicted_direction": None},
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "route"
    exit_code = main(
        [
            "--axis-audit-tsv",
            str(axis),
            "--deg-table",
            str(deg),
            "--anchor-backtest",
            str(anchor),
            "--external-evidence-json",
            str(geneformer),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["n_formal_invocations"] == 0
    assert printed["biology_pass"] is False
    payload = json.loads((output_dir / "candidate_routing_report.json").read_text(encoding="utf-8"))
    assert payload["coverage_claim"] == "five_candidates_ko_only"
    by_key = {(row["gene_symbol"], row["intervention"]): row for row in payload["rows"]}
    assert by_key[("APOE", "KO")]["selected_route"] == "APOE_KO_ENGINEERING"
    assert ("MAPT", "KD") not in by_key
    assert all(row["intervention"] == "KO" for row in payload["rows"])
    assert payload["research_decision"]["public_perturbation_policy"] == "out_of_scope"
    assert payload["research_decision"]["biology_pass"] is False
    assert (output_dir / "observed_gate_decision.json").exists()
    assert "foundation_ranking_only_no_expression_direction" in by_key[("APP", "KO")]["reasons"]


def test_cli_missing_audit_exits_nonzero(tmp_path):
    exit_code = main(
        [
            "--axis-audit-tsv",
            str(tmp_path / "missing.tsv"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 1
