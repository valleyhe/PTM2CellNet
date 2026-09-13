from src.integration.perturbgen.dual_path import evaluate_dual_path_candidate
from src.integration.perturbgen.reports import (
    build_candidate_report_payload,
    build_candidate_summary_dataframe,
    build_path_detail_dataframe,
    render_candidate_markdown,
    save_report_artifacts,
)


def _result(path: str, mode: str, seed: int, rescue: float) -> dict:
    return {
        "status": "evaluable",
        "path": path,
        "mode": mode,
        "rescue_excl_target": rescue,
        "evaluable_donors": 3,
        "donor_consistency": 2 / 3,
        "seed": seed,
        "reason_code": None,
        "metadata": {"matched_null_count": 99, "empirical_pvalue": 0.01},
    }


def _decision():
    rows = []
    for path_name, base in (("source_intervention", 0.5), ("within_state", 0.6)):
        for mode_name, offset in (("mask", 0.0), ("pad", -0.1), ("delete", -0.2)):
            rows.extend(
                [
                    _result(path_name, mode_name, 1, base + offset),
                    _result(path_name, mode_name, 2, base + offset - 0.1),
                    _result(path_name, mode_name, 3, base + offset - 0.2),
                ]
            )
    return evaluate_dual_path_candidate(
        rows,
        observed_direction="up",
        intervention_type="KO",
        q_value=0.01,
        candidate_gene="STAT3",
        unperturbed_quality_status="pass",
    )


def test_build_candidate_report_payload_keeps_manifest_for_replay() -> None:
    payload = build_candidate_report_payload(
        _decision(),
        manifest={"run_id": "run-001", "stage_manifest": "abc.json"},
        candidate={"gene_symbol": "STAT3", "cell_type": "T cell"},
    )

    assert payload["candidate_gene"] == "STAT3"
    assert payload["manifest"]["run_id"] == "run-001"
    assert payload["verdict"] == "pass"


def test_build_candidate_summary_dataframe_extracts_both_paths() -> None:
    payload = build_candidate_report_payload(
        _decision(),
        manifest={"run_id": "run-001"},
        candidate={"gene_symbol": "STAT3"},
    )

    frame = build_candidate_summary_dataframe([payload])

    assert frame.iloc[0]["candidate_gene"] == "STAT3"
    assert frame.iloc[0]["source_verdict"] == "pass"
    assert frame.iloc[0]["within_state_verdict"] == "pass"


def test_build_path_detail_dataframe_expands_seed_rows() -> None:
    payload = build_candidate_report_payload(
        _decision(),
        manifest={"run_id": "run-001"},
        candidate={"gene_symbol": "STAT3"},
    )

    frame = build_path_detail_dataframe(payload)

    assert set(frame["path"]) == {"source_intervention", "within_state"}
    assert frame["seed"].nunique() == 3


def test_render_candidate_markdown_contains_verdict_and_paths() -> None:
    payload = build_candidate_report_payload(
        _decision(),
        manifest={"run_id": "run-001"},
        candidate={"gene_symbol": "STAT3"},
    )

    markdown = render_candidate_markdown(payload)

    assert "PerturbGen Dual-Path Report" in markdown
    assert "Candidate: STAT3" in markdown
    assert "source_intervention: pass" in markdown


def test_save_report_artifacts_writes_all_files(tmp_path) -> None:
    payload = build_candidate_report_payload(
        _decision(),
        manifest={"run_id": "run-001"},
        candidate={"gene_symbol": "STAT3"},
    )

    artifacts = save_report_artifacts(payload, tmp_path)

    assert set(artifacts) == {"json", "summary_csv", "detail_csv", "markdown"}
    for path in artifacts.values():
        assert tmp_path.joinpath(path.split(str(tmp_path) + "/")[-1]).exists()
