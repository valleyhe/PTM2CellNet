"""Regression tests for the frozen contrast binding in the global-score CLI."""

from __future__ import annotations

import json

import pandas as pd

from scripts.build_ptm_global_gene_scores import main


def _config(path):
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
                "cohort_h5ad: unused.h5ad",
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
                "",
            ]
        ),
        encoding="utf-8",
    )


def _activity_table(path):
    pd.DataFrame(
        [
            {
                "activity_unit": "zscore",
                "regulator_id": "KIN",
                "regulator_type": "kinase",
                "condition_or_contrast": "disease-minus-normal",
                "activity_score": 2.0,
                "activity_direction": "up",
                "activity_pvalue": 0.01,
                "activity_qvalue": 0.02,
                "n_substrates": 3,
                "network_coverage": 1.0,
                "method": "KSTAR",
                "method_version": "1.0",
                "input_manifest": "input.json",
            },
            {
                "activity_unit": "zscore",
                "regulator_id": "KIN",
                "regulator_type": "kinase",
                "condition_or_contrast": "normal-minus-disease",
                "activity_score": 3.0,
                "activity_direction": "up",
                "activity_pvalue": 0.01,
                "activity_qvalue": 0.02,
                "n_substrates": 3,
                "network_coverage": 1.0,
                "method": "KSTAR",
                "method_version": "1.0",
                "input_manifest": "input.json",
            },
        ]
    ).to_csv(path, sep="\t", index=False)


def _network_inputs(tmp_path):
    network = tmp_path / "network.tsv"
    pd.DataFrame(
        [
            {
                "source_id": "KIN",
                "target_id": "TF",
                "edge_type": "protein_protein",
                "effect_sign": "+1",
                "site": "",
                "species": "9606",
                "evidence": "test",
                "confidence": 1.0,
                "release": "2026-08",
            },
            {
                "source_id": "TF",
                "target_id": "GENE_A",
                "edge_type": "tf_regulation",
                "effect_sign": "+1",
                "site": "s1",
                "species": "9606",
                "evidence": "test",
                "confidence": 1.0,
                "release": "2026-08",
            },
        ]
    ).to_csv(network, sep="\t", index=False)
    id_map = tmp_path / "network_id_map.tsv"
    id_map.write_text(
        "network_id\tgene_symbol\tensembl_id\nGENE_A\tGENEA\tENSG00000000001\n",
        encoding="utf-8",
    )
    return network, id_map


def _run_args(config, activity, network, id_map, output, manifest, *extra):
    return [
        "--config",
        str(config),
        "--activity-tsv",
        str(activity),
        "--network-tsv",
        str(network),
        "--network-id-map",
        str(id_map),
        *extra,
        "--output-tsv",
        str(output),
        "--manifest-output",
        str(manifest),
    ]


def test_missing_condition_uses_frozen_config_contrast_and_manifest(tmp_path):
    config = tmp_path / "config.yaml"
    activity = tmp_path / "activity.tsv"
    _config(config)
    _activity_table(activity)
    network, id_map = _network_inputs(tmp_path)
    output = tmp_path / "scores.tsv"
    manifest = tmp_path / "manifest.json"

    assert main(_run_args(config, activity, network, id_map, output, manifest)) == 0

    assert len(pd.read_csv(output, sep="\t")) == 1
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["sources"]["activity_table"]["condition_or_contrast"] == "disease-minus-normal"


def test_explicit_non_frozen_contrast_fails(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    activity = tmp_path / "activity.tsv"
    _config(config)
    _activity_table(activity)
    network, id_map = _network_inputs(tmp_path)

    assert (
        main(
            _run_args(
                config,
                activity,
                network,
                id_map,
                tmp_path / "scores.tsv",
                tmp_path / "manifest.json",
                "--condition-or-contrast",
                "normal-minus-disease",
            )
        )
        == 1
    )
    assert "输入契约失败" in capsys.readouterr().err


def _benchmark_table(path, effects):
    pd.DataFrame({"regulator_id": list(effects), "perturbation_effect": [effects[key] for key in effects]}).to_csv(
        path, sep="\t", index=False
    )


def test_activity_benchmark_without_preregistered_criteria_fails(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    activity = tmp_path / "activity.tsv"
    _config(config)
    _activity_table(activity)
    network, id_map = _network_inputs(tmp_path)
    benchmark = tmp_path / "benchmark.tsv"
    _benchmark_table(benchmark, {"KIN": -2.0})

    assert (
        main(
            _run_args(
                config,
                activity,
                network,
                id_map,
                tmp_path / "scores.tsv",
                tmp_path / "manifest.json",
                "--activity-benchmark",
                str(benchmark),
            )
        )
        == 1
    )
    assert "pre-registered" in capsys.readouterr().err


def _config_with_benchmark(path, criteria_lines):
    _config(path)
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "propagation:",
        "\n".join(["activity_benchmark:", *criteria_lines, "propagation:"]),
    )
    path.write_text(text, encoding="utf-8")


def _activity_table_multi(path):
    rows = []
    for regulator, score in (("KIN", 2.0), ("KIN2", 1.0)):
        rows.append(
            {
                "activity_unit": "zscore",
                "regulator_id": regulator,
                "regulator_type": "kinase",
                "condition_or_contrast": "disease-minus-normal",
                "activity_score": score,
                "activity_direction": "up",
                "activity_pvalue": 0.01,
                "activity_qvalue": 0.02,
                "n_substrates": 3,
                "network_coverage": 1.0,
                "method": "KSTAR",
                "method_version": "1.0",
                "input_manifest": "input.json",
            }
        )
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def test_activity_benchmark_pass_records_gate(tmp_path):
    config = tmp_path / "config.yaml"
    activity = tmp_path / "activity.tsv"
    _config_with_benchmark(
        config,
        ["  min_paired_regulators: 1", "  min_direction_concordance: 0.9", "  min_abs_spearman: 0.5", "  seed: 3"],
    )
    _activity_table_multi(activity)
    network, id_map = _network_inputs(tmp_path)
    benchmark = tmp_path / "benchmark.tsv"
    _benchmark_table(benchmark, {"KIN": 2.0, "KIN2": 1.0})
    output = tmp_path / "scores.tsv"
    manifest = tmp_path / "manifest.json"

    assert (
        main(
            _run_args(
                config,
                activity,
                network,
                id_map,
                output,
                manifest,
                "--activity-benchmark",
                str(benchmark),
            )
        )
        == 0
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    gate = payload["sources"]["activity_table"]["admission"]["benchmark_gate"]
    assert gate["available"] is True
    assert gate["passed"] is True
    assert payload["sources"]["activity_benchmark"]["sha256"]


def test_activity_benchmark_fail_keeps_exploratory_output(tmp_path):
    config = tmp_path / "config.yaml"
    activity = tmp_path / "activity.tsv"
    _config_with_benchmark(
        config,
        ["  min_paired_regulators: 1", "  min_direction_concordance: 0.9", "  min_abs_spearman: 0.5", "  seed: 3"],
    )
    _activity_table_multi(activity)
    network, id_map = _network_inputs(tmp_path)
    benchmark = tmp_path / "benchmark.tsv"
    _benchmark_table(benchmark, {"KIN": -2.0, "KIN2": -1.0})
    output = tmp_path / "scores.tsv"
    manifest = tmp_path / "manifest.json"

    assert (
        main(
            _run_args(
                config,
                activity,
                network,
                id_map,
                output,
                manifest,
                "--activity-benchmark",
                str(benchmark),
            )
        )
        == 0
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    gate = payload["sources"]["activity_table"]["admission"]["benchmark_gate"]
    assert gate["available"] is True
    assert gate["passed"] is False
    assert "exploratory" in gate["effect"]


def test_formal_mode_rejects_benchmark_fail(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    activity = tmp_path / "activity.tsv"
    _config_with_benchmark(
        config,
        ["  min_paired_regulators: 1", "  min_direction_concordance: 0.9", "  min_abs_spearman: 0.5", "  seed: 3"],
    )
    text = config.read_text(encoding="utf-8").replace("ptm_cohort: TEST", "ptm_cohort: REAL\nmode: formal")
    config.write_text(text, encoding="utf-8")
    _activity_table_multi(activity)
    network, id_map = _network_inputs(tmp_path)
    benchmark = tmp_path / "benchmark.tsv"
    _benchmark_table(benchmark, {"KIN": -2.0, "KIN2": -1.0})

    assert (
        main(
            _run_args(
                config,
                activity,
                network,
                id_map,
                tmp_path / "scores.tsv",
                tmp_path / "manifest.json",
                "--activity-benchmark",
                str(benchmark),
            )
        )
        == 1
    )
    assert "formal mode rejects" in capsys.readouterr().err
