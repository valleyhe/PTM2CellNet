"""Tests for the independent kinase-perturbation activity benchmark (方案 §5.2, U-01)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.ptm_activity import PTMActivityContractError
from src.analysis.ptm_activity_admission import select_activities_for_propagation
from src.analysis.ptm_activity_benchmark import (
    evaluate_activity_benchmark,
    load_activity_benchmark_table,
)
from src.analysis.ptm_research_config import (
    ActivityBenchmarkCriteria,
    PTMResearchConfigError,
    parse_ptm_research_config,
)


def _criteria(**overrides) -> ActivityBenchmarkCriteria:
    kwargs = {
        "min_paired_regulators": 3,
        "min_direction_concordance": 0.7,
        "min_abs_spearman": 0.3,
        "bootstrap_iterations": 50,
        "seed": 7,
    }
    kwargs.update(overrides)
    return ActivityBenchmarkCriteria(**kwargs)


def _activity_row(**overrides) -> dict:
    row = {
        "regulator_id": "GSK3B",
        "condition_or_contrast": "disease-minus-normal",
        "activity_score": 1.5,
        "activity_qvalue": 0.05,
        "n_substrates": 12,
        "network_coverage": 0.8,
        "method": "KSTAR",
    }
    row.update(overrides)
    return row


def _activity_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _benchmark_frame(effects: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"regulator_id": list(effects), "perturbation_effect": [effects[key] for key in effects]})


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"min_paired_regulators": 0}, "min_paired_regulators"),
        ({"min_direction_concordance": 0.0}, "min_direction_concordance"),
        ({"min_direction_concordance": 1.5}, "min_direction_concordance"),
        ({"min_abs_spearman": -0.1}, "min_abs_spearman"),
        ({"min_abs_spearman": 1.5}, "min_abs_spearman"),
        ({"bootstrap_iterations": -1}, "bootstrap_iterations"),
        ({"ci_level": 1.0}, "ci_level"),
        ({"seed": -1}, "seed"),
    ],
)
def test_criteria_rejects_out_of_range_values(kwargs, match):
    with pytest.raises(PTMResearchConfigError, match=match):
        _criteria(**kwargs)


def test_criteria_requires_explicit_thresholds():
    with pytest.raises(TypeError):
        ActivityBenchmarkCriteria()


def test_load_benchmark_table_validates_contract(tmp_path):
    path = tmp_path / "benchmark.tsv"
    _benchmark_frame({"GSK3B": -0.8, "CDK5": 1.2}).to_csv(path, sep="\t", index=False)
    frame = load_activity_benchmark_table(path)
    assert set(frame["regulator_id"]) == {"GSK3B", "CDK5"}

    missing = tmp_path / "missing.tsv"
    _benchmark_frame({"GSK3B": -0.8}).drop(columns=["perturbation_effect"]).to_csv(missing, sep="\t", index=False)
    with pytest.raises(PTMActivityContractError, match="requires columns"):
        load_activity_benchmark_table(missing)

    empty = tmp_path / "empty.tsv"
    _benchmark_frame({}).to_csv(empty, sep="\t", index=False)
    with pytest.raises(PTMActivityContractError, match="must not be empty"):
        load_activity_benchmark_table(empty)

    dup = tmp_path / "dup.tsv"
    pd.DataFrame({"regulator_id": ["GSK3B", "GSK3B"], "perturbation_effect": [1.0, 2.0]}).to_csv(
        dup, sep="\t", index=False
    )
    with pytest.raises(PTMActivityContractError, match="duplicate rows"):
        load_activity_benchmark_table(dup)

    nonfinite = tmp_path / "nonfinite.tsv"
    pd.DataFrame({"regulator_id": ["GSK3B"], "perturbation_effect": [float("nan")]}).to_csv(
        nonfinite, sep="\t", index=False
    )
    with pytest.raises(PTMActivityContractError, match="must be finite"):
        load_activity_benchmark_table(nonfinite)


def _concordant_tables():
    activity = _activity_frame(
        [
            _activity_row(regulator_id="GSK3B", activity_score=-1.5),
            _activity_row(regulator_id="CDK5", activity_score=2.0),
            _activity_row(regulator_id="MAPK1", activity_score=0.5),
        ]
    )
    benchmark = _benchmark_frame({"GSK3B": -0.8, "CDK5": 1.2, "MAPK1": 0.4, "UNPAIRED": 3.0})
    return activity, benchmark


def test_perfect_agreement_passes():
    activity, benchmark = _concordant_tables()
    report = evaluate_activity_benchmark(
        activity,
        benchmark,
        criteria=_criteria(),
        method="KSTAR",
        condition_or_contrast="disease-minus-normal",
    )
    assert report.passed
    assert report.verdict == "PASS"
    assert report.n_paired == 3
    assert report.direction_concordance == pytest.approx(1.0)
    assert report.spearman_rho == pytest.approx(1.0, abs=1e-12)
    assert report.paired_regulators == ("CDK5", "GSK3B", "MAPK1")
    low, high = report.bootstrap["direction_concordance_ci"]
    assert 0.0 <= low <= high <= 1.0


def test_opposite_directions_fail():
    activity, benchmark = _concordant_tables()
    flipped = _benchmark_frame({"GSK3B": 0.8, "CDK5": -1.2, "MAPK1": -0.4})
    report = evaluate_activity_benchmark(activity, flipped, criteria=_criteria(), method="KSTAR")
    assert not report.passed
    assert report.direction_concordance == pytest.approx(0.0)
    assert report.spearman_rho < 0.0


def test_too_few_paired_regulators_fails():
    activity, benchmark = _concordant_tables()
    single = _benchmark_frame({"GSK3B": -0.8})
    report = evaluate_activity_benchmark(activity, single, criteria=_criteria(min_paired_regulators=2))
    assert not report.passed
    assert report.n_paired == 1


def test_no_overlap_hard_fails():
    activity, benchmark = _concordant_tables()
    disjoint = _benchmark_frame({"OTHER1": 1.0, "OTHER2": -1.0})
    with pytest.raises(PTMActivityContractError, match="share no regulator_id"):
        evaluate_activity_benchmark(activity, disjoint, criteria=_criteria(), method="KSTAR")


def test_unknown_method_scope_hard_fails():
    activity, benchmark = _concordant_tables()
    with pytest.raises(PTMActivityContractError, match="no rows for benchmark method"):
        evaluate_activity_benchmark(activity, benchmark, criteria=_criteria(), method="PhosR")


def test_evaluation_is_deterministic_under_fixed_seed():
    activity, benchmark = _concordant_tables()
    first = evaluate_activity_benchmark(activity, benchmark, criteria=_criteria(), method="KSTAR")
    second = evaluate_activity_benchmark(activity, benchmark, criteria=_criteria(), method="KSTAR")
    assert first == second


def test_bootstrap_disabled_records_note():
    activity, benchmark = _concordant_tables()
    report = evaluate_activity_benchmark(
        activity, benchmark, criteria=_criteria(bootstrap_iterations=0), method="KSTAR"
    )
    assert report.bootstrap["iterations"] == 0
    assert "bootstrap disabled" in report.bootstrap["note"]


def test_input_hash_distinguishes_content_and_is_stable():
    activity, benchmark = _concordant_tables()
    first = evaluate_activity_benchmark(activity, benchmark, criteria=_criteria(), method="KSTAR")
    second = evaluate_activity_benchmark(activity, benchmark, criteria=_criteria(), method="KSTAR")
    assert first.activity_input_hash == second.activity_input_hash
    assert first.benchmark_input_hash == second.benchmark_input_hash
    perturbed = benchmark.copy()
    perturbed.loc[perturbed.index[0], "perturbation_effect"] = 99.0
    third = evaluate_activity_benchmark(activity, perturbed, criteria=_criteria(), method="KSTAR")
    assert third.benchmark_input_hash != first.benchmark_input_hash
    payload = first.as_dict()
    assert payload["schema_version"] == "ptm2cellnet.activity-benchmark/v1"
    assert payload["criteria"]["seed"] == 7


def test_conflicting_activity_scores_hard_fail():
    activity, benchmark = _concordant_tables()
    conflicting = pd.concat(
        [
            activity,
            _activity_frame([_activity_row(regulator_id="GSK3B", activity_score=-2.0)]),
        ],
        ignore_index=True,
    )
    with pytest.raises(PTMActivityContractError, match="conflicting activity scores"):
        evaluate_activity_benchmark(conflicting, benchmark, criteria=_criteria(), method="KSTAR")


def test_admission_gate_reflects_benchmark_report():
    activity, benchmark = _concordant_tables()
    report = evaluate_activity_benchmark(activity, benchmark, criteria=_criteria(), method="KSTAR")
    selection = select_activities_for_propagation(
        _activity_frame(
            [
                _activity_row(regulator_id="GSK3B", activity_score=-1.5),
                _activity_row(regulator_id="CDK5", activity_score=2.0),
            ]
        ),
        method="KSTAR",
        condition_or_contrast="disease-minus-normal",
        benchmark_report=report,
    )
    payload = selection.as_dict()
    assert payload["benchmark_gate"]["available"] is True
    assert payload["benchmark_gate"]["passed"] is True
    assert "report" in payload["benchmark_gate"]

    without = select_activities_for_propagation(
        _activity_frame([_activity_row(regulator_id="GSK3B", activity_score=-1.5)]),
        method="KSTAR",
        condition_or_contrast="disease-minus-normal",
    )
    assert without.as_dict()["benchmark_gate"]["available"] is False


def test_config_parses_optional_benchmark_block():
    payload = {
        "schema_version": "ptm2cellnet.ptm-research-config/v1",
        "research_objective": "association",
        "reference_axis": "disease_minus_normal",
        "contrast": "disease-minus-normal",
        "primary_activity_method": "KSTAR",
        "network_release": "2026-08",
        "cell_types": ["EX"],
        "cohort_h5ad": "unused.h5ad",
        "cohort_pairing": "between_donor",
        "species": "9606",
        "ptm_cohort": "TEST",
        "deg_max_fdr": 0.05,
        "min_donors_per_state": 3,
        "replicate_policy": "mean",
        "propagation": {"max_depth": 2, "decay": 0.5, "gene_edge_types": ["tf_regulation"]},
    }
    assert parse_ptm_research_config(payload).activity_benchmark is None

    with_block = dict(
        payload,
        activity_benchmark={
            "min_paired_regulators": 5,
            "min_direction_concordance": 0.75,
            "min_abs_spearman": 0.2,
        },
    )
    criteria = parse_ptm_research_config(with_block).activity_benchmark
    assert criteria is not None
    assert criteria.min_paired_regulators == 5
    assert criteria.bootstrap_iterations == 2000

    with pytest.raises(PTMResearchConfigError, match="pre-registered values"):
        parse_ptm_research_config(dict(payload, activity_benchmark={"min_abs_spearman": 0.2}))

    with pytest.raises(PTMResearchConfigError, match="unknown keys"):
        parse_ptm_research_config(
            dict(
                payload,
                activity_benchmark={
                    "min_paired_regulators": 5,
                    "min_direction_concordance": 0.75,
                    "min_abs_spearman": 0.2,
                    "unexpected": 1,
                },
            )
        )


def test_bootstrap_ci_brackets_point_estimate_for_large_samples():
    rng = np.random.default_rng(11)
    scores = rng.normal(size=200)
    effects = scores + rng.normal(scale=0.5, size=200)
    activity = _activity_frame(
        [_activity_row(regulator_id=f"KIN{i}", activity_score=float(scores[i])) for i in range(200)]
    )
    benchmark = _benchmark_frame({f"KIN{i}": float(effects[i]) for i in range(200)})
    report = evaluate_activity_benchmark(
        activity, benchmark, criteria=_criteria(min_paired_regulators=100), method="KSTAR"
    )
    assert report.direction_concordance == pytest.approx(float(np.mean(np.sign(scores) == np.sign(effects))))
    low, high = report.bootstrap["spearman_rho_ci"]
    assert low <= report.spearman_rho <= high
