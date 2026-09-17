from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.integration.perturbgen.formal_verification import verify_formal_workflow_a


def _passing_e2e_report() -> dict:
    return {
        "schema_version": "davf_perturbgen_e2e/v1",
        "candidates": [
            {
                "status": "pass",
                "invocation": {"gene_symbol": "APOE", "ensembl_id": "ENSG00000130203"},
            }
        ],
        "perturbgen_runs": [{"ensembl_id": "ENSG00000130203", "gene_symbol": "APOE"}],
        "statistical_evidence": {"status": "assembled", "scientific_acceptance": True},
    }


def _write_json(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_dry_run_report_fails_e2e_gate(tmp_path: Path) -> None:
    report = _passing_e2e_report()
    report["perturbgen_runs"] = []
    report["statistical_evidence"] = {"status": "inconclusive", "reason": "statistical_evidence_not_assembled"}

    result = verify_formal_workflow_a(e2e_report=_write_json(tmp_path, "e2e.json", report))

    assert result["verdict"] == "fail"
    reasons = result["checks"]["e2e_gate"]["reasons"]
    assert "no_perturbgen_runs" in reasons
    assert "statistical_evidence_not_assembled" in reasons


def test_report_without_passing_invocation_fails_e2e_gate(tmp_path: Path) -> None:
    report = _passing_e2e_report()
    report["candidates"] = [{"status": "inconclusive", "invocation": None}]

    result = verify_formal_workflow_a(e2e_report=_write_json(tmp_path, "e2e.json", report))

    assert result["checks"]["e2e_gate"]["status"] == "fail"
    assert any(reason.startswith("no_passing_invocation") for reason in result["checks"]["e2e_gate"]["reasons"])


def test_missing_optional_inputs_render_blocked_verdict(tmp_path: Path) -> None:
    result = verify_formal_workflow_a(e2e_report=_write_json(tmp_path, "e2e.json", _passing_e2e_report()))

    assert result["checks"]["e2e_gate"]["status"] == "pass"
    assert result["verdict"] == "blocked"
    assert set(result["missing_inputs"]) == {
        "frozen_cohort",
        "matched_null",
        "unperturbed_quality",
        "acceptance_replay",
    }


def test_matched_null_manifest_pass_and_fail(tmp_path: Path) -> None:
    values = [round(0.001 * index, 6) for index in range(99)]
    null_ids = [f"ENSG00000{index:06d}" for index in range(99)]
    good = {
        "schema_version": "perturbgen_null_distribution/v1",
        "candidate_ensembl_id": "ENSG00000130203",
        "path": "source_intervention",
        "mode": "mask",
        "seed": 0,
        "required_count": 99,
        "values": values,
        "null_ensembl_ids": null_ids,
    }

    result = verify_formal_workflow_a(
        e2e_report=_write_json(tmp_path, "e2e.json", _passing_e2e_report()),
        null_distribution_manifests=[_write_json(tmp_path, "null.json", good)],
    )

    assert result["checks"]["matched_null"]["status"] == "pass"
    assert result["checks"]["matched_null"]["manifests"][0]["value_count"] == 99

    short = dict(good, values=values[:50], null_ensembl_ids=null_ids[:50])
    result = verify_formal_workflow_a(
        e2e_report=_write_json(tmp_path, "e2e2.json", _passing_e2e_report()),
        null_distribution_manifests=[_write_json(tmp_path, "null_short.json", short)],
    )

    assert result["checks"]["matched_null"]["status"] == "fail"
    assert any("null_distribution_manifests" in reason for reason in result["checks"]["matched_null"]["reasons"])


def test_unperturbed_quality_from_explicit_payload(tmp_path: Path) -> None:
    good = {"source": "extract_unperturbed_quality_from_h5ad", "status": "pass"}
    result = verify_formal_workflow_a(
        e2e_report=_write_json(tmp_path, "e2e.json", _passing_e2e_report()),
        quality_payload=_write_json(tmp_path, "quality.json", good),
    )
    assert result["checks"]["unperturbed_quality"]["status"] == "pass"

    hand_filled = {"source": "hand_filled", "status": "pass"}
    result = verify_formal_workflow_a(
        e2e_report=_write_json(tmp_path, "e2e2.json", _passing_e2e_report()),
        quality_payload=_write_json(tmp_path, "quality_bad.json", hand_filled),
    )
    assert result["checks"]["unperturbed_quality"]["status"] == "fail"


def test_unperturbed_quality_from_eval_input_candidates(tmp_path: Path) -> None:
    eval_input = {
        "schema_version": "perturbgen_dual_path_eval/v1",
        "candidates": [
            {
                "gene_symbol": "APOE",
                "unperturbed_quality_status": "pass",
                "unperturbed_quality_source": "extract_unperturbed_quality_from_h5ad",
            }
        ],
    }

    result = verify_formal_workflow_a(
        e2e_report=_write_json(tmp_path, "e2e.json", _passing_e2e_report()),
        eval_input=_write_json(tmp_path, "eval_input.json", eval_input),
    )

    assert result["checks"]["unperturbed_quality"]["status"] == "pass"

    hand_filled = dict(eval_input)
    hand_filled["candidates"] = [dict(eval_input["candidates"][0], unperturbed_quality_source="hand_filled")]
    result = verify_formal_workflow_a(
        e2e_report=_write_json(tmp_path, "e2e2.json", _passing_e2e_report()),
        eval_input=_write_json(tmp_path, "eval_input_bad.json", hand_filled),
    )
    assert result["checks"]["unperturbed_quality"]["status"] == "fail"
    assert any("hand-filled" in reason for reason in result["checks"]["unperturbed_quality"]["reasons"])


def test_frozen_cohort_check_with_real_manifest(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.h5ad"
    cohort.write_bytes(b"cohort-bytes")
    manifest = {
        "schema_version": "ptm2cellnet.frozen-cohort/v1",
        "cohort_h5ad": str(cohort),
        "cohort_sha256": hashlib.sha256(cohort.read_bytes()).hexdigest(),
        "cell_type": "EX",
        "donor_obs_column": "donor",
        "train_donors": ["D1", "D2", "D3"],
        "held_out_donors": ["D4", "D5", "D6"],
        "candidates": [
            {
                "gene_symbol": "APOE",
                "ensembl_id": "ENSG00000130203",
                "intervention_type": "KO",
                "modes": ["mask", "pad", "delete"],
                "seeds": [0, 1, 2],
                "matched_nulls": 99,
            }
        ],
        "created_at": "2026-09-16T00:00:00+00:00",
        "pairing": "between_donor",
    }

    result = verify_formal_workflow_a(
        e2e_report=_write_json(tmp_path, "e2e.json", _passing_e2e_report()),
        frozen_manifest=_write_json(tmp_path, "frozen.json", manifest),
    )

    assert result["checks"]["frozen_cohort"]["status"] == "pass"
    assert result["checks"]["frozen_cohort"]["n_train_donors"] == 3
    assert result["checks"]["acceptance_replay"]["status"] == "blocked"

    leaked = dict(manifest, held_out_donors=["D1", "D5", "D6"])
    result = verify_formal_workflow_a(
        e2e_report=_write_json(tmp_path, "e2e2.json", _passing_e2e_report()),
        frozen_manifest=_write_json(tmp_path, "frozen_leak.json", leaked),
    )
    assert result["checks"]["frozen_cohort"]["status"] == "fail"
    assert any("donor leakage" in reason for reason in result["checks"]["frozen_cohort"]["reasons"])


def test_acceptance_replay_blocked_until_all_three_inputs_exist(tmp_path: Path) -> None:
    result = verify_formal_workflow_a(
        e2e_report=_write_json(tmp_path, "e2e.json", _passing_e2e_report()),
        eval_input=_write_json(tmp_path, "eval_input.json", {"schema_version": "perturbgen_dual_path_eval/v1"}),
    )

    assert result["checks"]["acceptance_replay"]["status"] == "blocked"
    assert "frozen_manifest_not_supplied" in result["checks"]["acceptance_replay"]["reasons"]
    assert "report_manifest_not_supplied" in result["checks"]["acceptance_replay"]["reasons"]
