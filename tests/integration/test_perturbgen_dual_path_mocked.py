import pytest

from src.integration.perturbgen.dual_path import evaluate_dual_path_candidate
from src.integration.perturbgen.reports import (
    build_candidate_report_payload,
    build_candidate_summary_dataframe,
    save_report_artifacts,
)
from src.integration.perturbgen.results import benjamini_hochberg, evaluate_null_calibration

pytestmark = pytest.mark.integration


def _path_rows(path_name: str, matched_null_count: int, base: float) -> list[dict]:
    rows: list[dict] = []
    for mode_name, offset in (("mask", 0.0), ("pad", -0.1), ("delete", -0.2)):
        for seed, rescue in enumerate((base + offset, base + offset - 0.1, base + offset - 0.2), start=1):
            calibration = evaluate_null_calibration(rescue, [0.05] * matched_null_count)
            rows.append(
                {
                    "status": "evaluable",
                    "path": path_name,
                    "mode": mode_name,
                    "rescue_excl_target": rescue,
                    "evaluable_donors": 3,
                    "donor_consistency": 2 / 3,
                    "seed": seed,
                    "reason_code": None,
                    "metadata": {
                        "matched_null_count": calibration.matched_null_count,
                        "empirical_pvalue": calibration.empirical_pvalue,
                    },
                }
            )
    return rows


def test_mocked_dual_path_pipeline_passes_and_writes_replayable_report(tmp_path) -> None:
    q_value = benjamini_hochberg([0.01])[0]
    decision = evaluate_dual_path_candidate(
        _path_rows("source_intervention", 99, 0.6) + _path_rows("within_state", 99, 0.5),
        observed_direction="up",
        intervention_type="KO",
        q_value=q_value,
        candidate_gene="STAT3",
        unperturbed_quality_status="pass",
    )

    payload = build_candidate_report_payload(
        decision,
        manifest={"run_id": "mocked-001", "replay": {"config": "cfg.yaml", "stage_manifest": "manifest.json"}},
        candidate={"gene_symbol": "STAT3"},
    )
    artifacts = save_report_artifacts(payload, tmp_path)
    summary = build_candidate_summary_dataframe([payload])

    assert decision.verdict == "pass"
    assert payload["manifest"]["replay"]["stage_manifest"] == "manifest.json"
    assert summary.iloc[0]["run_id"] == "mocked-001"
    assert tmp_path.joinpath("candidate_report.md").exists()
    assert artifacts["json"].endswith("candidate_report.json")


def test_mocked_dual_path_pipeline_marks_smoke_only_null_as_inconclusive() -> None:
    q_value = benjamini_hochberg([0.01])[0]
    decision = evaluate_dual_path_candidate(
        _path_rows("source_intervention", 20, 0.6) + _path_rows("within_state", 20, 0.5),
        observed_direction="up",
        intervention_type="KO",
        q_value=q_value,
        candidate_gene="STAT3",
        unperturbed_quality_status="pass",
    )

    assert decision.verdict == "inconclusive"
    assert "source_intervention_inconclusive" in decision.reasons
