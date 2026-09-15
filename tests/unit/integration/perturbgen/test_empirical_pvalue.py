"""U-02: candidate-level empirical p aggregation."""

from __future__ import annotations

import pytest

from src.integration.perturbgen.empirical_pvalue import (
    CONSERVATIVE_MAX_ESTIMAND,
    EmpiricalPvalueError,
    aggregate_candidate_empirical_pvalues,
    primary_mode_for_direction,
)


def _runs(*, extra: list[dict] | None = None) -> list[dict]:
    rows = []
    for path in ("source_intervention", "within_state"):
        for seed, pvalue in ((1, 0.02), (2, 0.04), (3, 0.01)):
            rows.append(
                {
                    "path": path,
                    "mode": "mask",
                    "seed": seed,
                    "empirical_pvalue": pvalue,
                }
            )
        for seed in (1, 2, 3):
            rows.append(
                {
                    "path": path,
                    "mode": "pad",
                    "seed": seed,
                    "empirical_pvalue": 0.9,
                }
            )
    if extra:
        rows.extend(extra)
    return rows


def test_primary_mode_follows_observed_direction() -> None:
    assert primary_mode_for_direction("up") == "mask"
    assert primary_mode_for_direction("down") == "overexpress"


def test_aggregation_takes_max_of_required_primary_runs_and_ignores_pad() -> None:
    payload = aggregate_candidate_empirical_pvalues(
        _runs(),
        "KO",
        "up",
        ensembl_id="ENSG00000168610.7",
    )
    assert payload["pvalue"] == 0.04
    assert payload["estimand"] == CONSERVATIVE_MAX_ESTIMAND
    assert payload["required_run_count"] == 6
    assert payload["ensembl_id"] == "ENSG00000168610"
    assert payload["provenance"]["aggregation"] == "max"
    assert all(item["mode"] == "mask" for item in payload["provenance"]["runs"])


def test_missing_required_run_is_an_error_not_a_fabricated_p() -> None:
    rows = [
        row for row in _runs() if not (row["path"] == "within_state" and row["seed"] == 3 and row["mode"] == "mask")
    ]
    with pytest.raises(EmpiricalPvalueError, match="missing finite empirical_pvalue"):
        aggregate_candidate_empirical_pvalues(rows, "KD", "up", ensembl_id="ENSG00000168610")


def test_nested_path_result_pvalue_is_accepted() -> None:
    rows = []
    for path in ("source_intervention", "within_state"):
        for seed in (0, 1, 2):
            rows.append(
                {
                    "path": path,
                    "mode": "overexpress",
                    "seed": seed,
                    "path_result": {"empirical_pvalue": 0.03},
                }
            )
    payload = aggregate_candidate_empirical_pvalues(
        rows,
        "KO",
        "down",
        ensembl_id="ENSG00000168610",
    )
    assert payload["pvalue"] == 0.03
    assert payload["primary_mode"] == "overexpress"
