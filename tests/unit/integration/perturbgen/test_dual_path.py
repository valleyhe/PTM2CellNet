from types import SimpleNamespace

import pytest

from src.integration.perturbgen.dual_path import (
    evaluate_dual_path_candidate,
    evaluate_path_results,
)


def _result(
    *,
    path: str,
    mode: str,
    seed: int,
    rescue: float,
    donor_consistency: float = 2 / 3,
    evaluable_donors: int = 3,
    matched_null_count: int = 99,
    status: str = "evaluable",
) -> dict:
    return {
        "status": status,
        "path": path,
        "mode": mode,
        "rescue_excl_target": rescue,
        "evaluable_donors": evaluable_donors,
        "donor_consistency": donor_consistency,
        "seed": seed,
        "reason_code": None,
        "metadata": {
            "matched_null_count": matched_null_count,
            "empirical_pvalue": 0.01,
        },
    }


def _ko_path(path_name: str, rescue_by_mode: dict[str, tuple[float, float, float]]) -> list[dict]:
    rows: list[dict] = []
    for mode_name, rescues in rescue_by_mode.items():
        for seed, rescue in enumerate(rescues, start=1):
            rows.append(_result(path=path_name, mode=mode_name, seed=seed, rescue=rescue))
    return rows


def test_evaluate_path_results_passes_with_strict_majority_and_two_of_three_modes() -> None:
    rows = _ko_path(
        "source_intervention",
        {
            "mask": (0.5, 0.4, 0.2),
            "pad": (0.4, 0.3, 0.2),
            "delete": (0.3, 0.2, 0.1),
        },
    )

    decision = evaluate_path_results(rows, observed_direction="up", intervention_type="KO")

    assert decision.verdict == "pass"
    assert decision.primary_mode == "mask"
    assert decision.seed_count == 3
    assert decision.median_rescue == 0.4


def test_evaluate_path_results_fails_when_donor_consistency_is_not_strict_majority() -> None:
    rows = [
        _result(path="source_intervention", mode="mask", seed=1, rescue=0.5, donor_consistency=0.5),
        _result(path="source_intervention", mode="mask", seed=2, rescue=0.4, donor_consistency=0.5),
        _result(path="source_intervention", mode="mask", seed=3, rescue=0.3, donor_consistency=0.5),
        _result(path="source_intervention", mode="pad", seed=1, rescue=0.5, donor_consistency=0.5),
        _result(path="source_intervention", mode="pad", seed=2, rescue=0.4, donor_consistency=0.5),
        _result(path="source_intervention", mode="pad", seed=3, rescue=0.3, donor_consistency=0.5),
        _result(path="source_intervention", mode="delete", seed=1, rescue=0.5, donor_consistency=0.5),
        _result(path="source_intervention", mode="delete", seed=2, rescue=0.4, donor_consistency=0.5),
        _result(path="source_intervention", mode="delete", seed=3, rescue=0.3, donor_consistency=0.5),
    ]

    decision = evaluate_path_results(rows, observed_direction="up", intervention_type="KO")

    assert decision.verdict == "fail"
    assert "seed_1_donor_consistency_not_strict_majority" in decision.reasons


def test_evaluate_path_results_fails_if_only_delete_mode_is_positive() -> None:
    rows = _ko_path(
        "source_intervention",
        {
            "mask": (-0.2, -0.1, -0.1),
            "pad": (-0.1, -0.2, -0.3),
            "delete": (0.5, 0.4, 0.3),
        },
    )

    decision = evaluate_path_results(rows, observed_direction="up", intervention_type="KO")

    assert decision.verdict == "fail"
    assert "ko_modes_not_two_of_three_positive" in decision.reasons


def test_evaluate_path_results_marks_20_null_as_inconclusive_smoke() -> None:
    rows = _ko_path(
        "source_intervention",
        {
            "mask": (0.5, 0.4, 0.2),
            "pad": (0.4, 0.3, 0.2),
            "delete": (0.3, 0.2, 0.1),
        },
    )
    for row in rows:
        row["metadata"]["matched_null_count"] = 20

    decision = evaluate_path_results(rows, observed_direction="up", intervention_type="KO")

    assert decision.verdict == "inconclusive"
    assert "seed_1_smoke_only_null" in decision.reasons


def test_evaluate_dual_path_candidate_requires_both_paths_to_pass() -> None:
    source_rows = _ko_path(
        "source_intervention",
        {
            "mask": (0.5, 0.4, 0.2),
            "pad": (0.4, 0.3, 0.2),
            "delete": (0.3, 0.2, 0.1),
        },
    )
    within_rows = _ko_path(
        "within_state",
        {
            "mask": (0.6, 0.5, 0.3),
            "pad": (0.4, 0.4, 0.2),
            "delete": (0.2, 0.2, 0.1),
        },
    )

    decision = evaluate_dual_path_candidate(
        source_rows + within_rows,
        observed_direction="up",
        intervention_type="KO",
        q_value=0.01,
        candidate_gene="STAT3",
        unperturbed_quality_status="pass",
    )

    assert decision.verdict == "pass"
    assert decision.candidate_gene == "STAT3"
    assert decision.scientific_acceptance is False
    assert decision.evaluation_mode == "engineering"
    assert len(decision.path_decisions) == 2


def test_evaluate_dual_path_candidate_fails_when_one_path_is_negative() -> None:
    source_rows = _ko_path(
        "source_intervention",
        {
            "mask": (0.5, 0.4, 0.2),
            "pad": (0.4, 0.3, 0.2),
            "delete": (0.3, 0.2, 0.1),
        },
    )
    within_rows = _ko_path(
        "within_state",
        {
            "mask": (-0.5, -0.4, -0.3),
            "pad": (-0.4, -0.3, -0.2),
            "delete": (-0.3, -0.2, -0.1),
        },
    )

    decision = evaluate_dual_path_candidate(
        source_rows + within_rows,
        observed_direction="up",
        intervention_type="KO",
        q_value=0.01,
        unperturbed_quality_status="pass",
    )

    assert decision.verdict == "fail"
    assert "within_state_fail" in decision.reasons


def test_evaluate_dual_path_candidate_is_inconclusive_when_a_path_is_not_evaluable() -> None:
    source_rows = _ko_path(
        "source_intervention",
        {
            "mask": (0.5, 0.4, 0.2),
            "pad": (0.4, 0.3, 0.2),
            "delete": (0.3, 0.2, 0.1),
        },
    )
    within_rows = _ko_path(
        "within_state",
        {
            "mask": (0.6, 0.5, 0.3),
            "pad": (0.4, 0.4, 0.2),
            "delete": (0.2, 0.2, 0.1),
        },
    )
    within_rows[0]["evaluable_donors"] = 2

    decision = evaluate_dual_path_candidate(
        source_rows + within_rows,
        observed_direction="up",
        intervention_type="KO",
        q_value=0.01,
        unperturbed_quality_status="pass",
    )

    assert decision.verdict == "inconclusive"
    assert "within_state_inconclusive" in decision.reasons


def test_dual_path_requires_unperturbed_quality_evidence() -> None:
    rows = _ko_path(
        "source_intervention",
        {"mask": (0.5, 0.4, 0.2), "pad": (0.4, 0.3, 0.2), "delete": (0.3, 0.2, 0.1)},
    ) + _ko_path(
        "within_state",
        {"mask": (0.5, 0.4, 0.2), "pad": (0.4, 0.3, 0.2), "delete": (0.3, 0.2, 0.1)},
    )
    decision = evaluate_dual_path_candidate(
        rows,
        observed_direction="up",
        intervention_type="KO",
        q_value=0.01,
    )
    assert decision.verdict == "inconclusive"
    assert "unperturbed_quality_inconclusive" in decision.reasons


def test_path_rejects_duplicate_mode_seed_rows() -> None:
    rows = _ko_path(
        "source_intervention",
        {"mask": (0.5, 0.4, 0.2), "pad": (0.4, 0.3, 0.2), "delete": (0.3, 0.2, 0.1)},
    )
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_path_results(rows + [dict(rows[0])], observed_direction="up", intervention_type="KO")


def test_evaluate_path_results_accepts_equivalent_object_with_top_level_null_fields() -> None:
    rows = [
        SimpleNamespace(
            status="evaluable",
            path="source_intervention",
            mode=mode_name,
            rescue_excl_target=rescue,
            evaluable_donors=3,
            donor_consistency=2 / 3,
            seed=seed,
            reason_code=None,
            matched_null_count=99,
            empirical_pvalue=0.01,
        )
        for mode_name, rescues in {
            "mask": (0.5, 0.4, 0.2),
            "pad": (0.4, 0.3, 0.2),
            "delete": (0.3, 0.2, 0.1),
        }.items()
        for seed, rescue in enumerate(rescues, start=1)
    ]

    decision = evaluate_path_results(rows, observed_direction="up", intervention_type="KO")

    assert decision.verdict == "pass"


def test_kd_up_mask_only_passes_without_sensitivity_modes() -> None:
    rows = []
    for path_name, rescues in (
        ("source_intervention", (0.5, 0.4, 0.2)),
        ("within_state", (0.6, 0.5, 0.3)),
    ):
        for seed, rescue in enumerate(rescues, start=1):
            rows.append(_result(path=path_name, mode="mask", seed=seed, rescue=rescue))
    decision = evaluate_dual_path_candidate(
        rows,
        observed_direction="up",
        intervention_type="KD",
        q_value=0.01,
        unperturbed_quality_status="pass",
    )
    assert decision.verdict == "pass"
    assert all(item.primary_mode == "mask" for item in decision.path_decisions)


def test_ko_up_mask_only_is_inconclusive() -> None:
    rows = [
        _result(path="source_intervention", mode="mask", seed=seed, rescue=rescue)
        for seed, rescue in enumerate((0.5, 0.4, 0.2), start=1)
    ]
    path_decision = evaluate_path_results(rows, observed_direction="up", intervention_type="KO")
    assert path_decision.verdict == "inconclusive"
    assert "missing_mode_pad" in path_decision.reasons


def test_formal_mode_rejects_uniform_pvalue_even_when_paths_pass() -> None:
    rows = _ko_path(
        "source_intervention",
        {"mask": (0.5, 0.4, 0.2), "pad": (0.4, 0.3, 0.2), "delete": (0.3, 0.2, 0.1)},
    ) + _ko_path(
        "within_state",
        {"mask": (0.6, 0.5, 0.3), "pad": (0.4, 0.4, 0.2), "delete": (0.2, 0.2, 0.1)},
    )
    decision = evaluate_dual_path_candidate(
        rows,
        observed_direction="up",
        intervention_type="KO",
        q_value=0.01,
        unperturbed_quality_status="pass",
        evaluation_mode="formal",
        evidence_class="synthetic",
        pvalue_source="uniform",
    )
    assert decision.verdict == "inconclusive"
    assert decision.scientific_acceptance is False
    assert "formal_rejects_synthetic_pvalue" in decision.reasons
    assert "formal_requires_empirical_null" in decision.reasons
