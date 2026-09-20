"""Regression tests for the source-gene DEG gate in candidate-spec assembly."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts import build_celltype_candidate_specs as script


def _config(path, extra_lines=()):
    path.write_text(
        "\n".join(
            [
                "schema_version: ptm2cellnet.ptm-research-config/v1",
                "research_objective: association",
                "reference_axis: disease_minus_normal",
                'contrast: "disease-minus-normal"',
                "primary_activity_method: KSTAR",
                'network_release: "2026-08"',
                "cell_types: [EX]",
                "cohort_h5ad: cohort.h5ad",
                "cohort_pairing: between_donor",
                'species: "9606"',
                "ptm_cohort: TEST",
                "deg_max_fdr: 0.05",
                "min_donors_per_state: 3",
                "replicate_policy: mean",
                "propagation:",
                "  max_depth: 2",
                "  decay: 0.5",
                "  gene_edge_types: [tf_regulation]",
                "semantic_context:",
                '  context: "TEST {cell_type}"',
                '  intervention: "KO"',
                '  comparison_baseline: "disease vs normal"',
                '  reference_axis: "disease_minus_normal"',
                '  research_objective: "association"',
                '  evidence_source: "test"',
                '  cohort: "TEST"',
                *extra_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )


def _inputs(
    tmp_path,
    *,
    observed_direction="neutral",
    fdr=0.01,
    normal_donors=3,
    disease_donors=3,
    extra_config=(),
):
    ensembl_id = "ENSG00000000001"
    config = tmp_path / "config.yaml"
    _config(config, extra_lines=extra_config)
    proposals = tmp_path / "proposals.tsv"
    pd.DataFrame(
        [
            {
                "protein_id": "P12345",
                "position": 9,
                "ptm_type": "phosphorylation",
                "gene_symbol": "KIN",
                "ensembl_id": ensembl_id,
                "source_activity_id": "KIN",
                "proposed_direction": "up",
                "site_probability": 0.9,
                "provenance": "test",
            }
        ]
    ).to_csv(proposals, sep="\t", index=False)
    deg = tmp_path / "deg.tsv"
    pd.DataFrame(
        [
            {
                "cell_type": "EX",
                "ensembl_id": ensembl_id,
                "gene_symbol": "KIN",
                "log2fc": 0.5,
                "fdr": fdr,
                "observed_direction": observed_direction,
                "n_normal_donors": normal_donors,
                "n_disease_donors": disease_donors,
            }
        ]
    ).to_csv(deg, sep="\t", index=False)
    target_manifest = tmp_path / "target_set_manifest.json"
    target_manifest.write_text(
        json.dumps(
            {"cell_types": {"EX": {"sources": {"KIN": {"targets": [{"target_ensembl_id": "ENSG00000000002"}]}}}}}
        ),
        encoding="utf-8",
    )
    context = tmp_path / "cohort.h5ad"
    context.write_text("placeholder", encoding="utf-8")
    return config, proposals, deg, target_manifest, context


def test_source_deg_gate_keeps_failed_source_exploratory_even_with_target_set(tmp_path, monkeypatch):
    config, proposals, deg, target_manifest, context = _inputs(
        tmp_path,
        observed_direction="neutral",
        fdr=0.2,
        normal_donors=2,
        disease_donors=2,
    )
    monkeypatch.setattr(script, "_first_cell_index_per_type", lambda *_args: {"EX": 0})
    output_dir = tmp_path / "specs"

    assert (
        script.main(
            [
                "--config",
                str(config),
                "--source-proposals-tsv",
                str(proposals),
                "--deg-table",
                str(deg),
                "--target-set-manifest",
                str(target_manifest),
                "--context-h5ad",
                str(context),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    summary = json.loads((output_dir / "build_summary.json").read_text(encoding="utf-8"))
    cell_summary = summary["cell_types"]["EX"]
    assert cell_summary["n_candidates"] == 0
    assert len(cell_summary["exploratory_sources"]) == 1
    assert "formal gate" in cell_summary["exploratory_sources"][0]["reason"]
    assert not (output_dir / "candidate_spec_EX.json").exists()


def test_signed_admission_emits_candidate_without_relabeling_fdr(tmp_path, monkeypatch):
    config, proposals, deg, target_manifest, context = _inputs(
        tmp_path,
        observed_direction="up",
        fdr=0.52,
        extra_config=(
            "observed_admission_rule: signed_direction_without_fdr_cutoff",
            "kd_policy: merged_into_ko_out_of_scope",
            "public_perturbation_policy: out_of_scope",
        ),
    )
    monkeypatch.setattr(script, "_first_cell_index_per_type", lambda *_args: {"EX": 0})
    output_dir = tmp_path / "specs"

    assert (
        script.main(
            [
                "--config",
                str(config),
                "--source-proposals-tsv",
                str(proposals),
                "--deg-table",
                str(deg),
                "--target-set-manifest",
                str(target_manifest),
                "--context-h5ad",
                str(context),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    spec = json.loads((output_dir / "candidate_spec_EX.json").read_text(encoding="utf-8"))
    assert spec["observed_admission_rule"] == "signed_direction_without_fdr_cutoff"
    assert spec["kd_policy"] == "merged_into_ko_out_of_scope"
    candidate = spec["candidates"][0]
    assert candidate["observed_direction"] == "up"
    assert candidate["observed_fdr"] == pytest.approx(0.52)
    assert candidate["observed_significant"] is False
