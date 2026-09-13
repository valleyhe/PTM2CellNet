"""U-01: matched-null batch stage generation without fake invocations."""

from __future__ import annotations

import json

import pytest

from src.integration.perturbgen.null_generation import (
    NULL_STAGE_PLAN_SCHEMA_VERSION,
    NullGenerationError,
    NullStageRecord,
    candidate_identity_from_e2e_report,
    collect_null_stage_records,
    run_matched_null_stages,
)
from src.integration.perturbgen.null_selection import FEATURE_ORDER

CANDIDATE = "ENSG00000168610"
PATH = "source_intervention"


def _features(values=(1.0, 0.5, 0.1, 2.0)) -> dict:
    return {
        "feature_order": list(FEATURE_ORDER),
        "mean_expression": values[0],
        "detection_rate": values[1],
        "fc": values[2],
        "token_rank": values[3],
        "matching_vector": list(values),
    }


def _selection(count: int = 99) -> dict:
    selected = []
    for index in range(count):
        ensembl_id = f"ENSG{index + 4:011d}"
        selected.append(
            {
                "ensembl_id": ensembl_id,
                "gene_symbol": f"NULL{index}",
                "match_features": _features(),
                "distance": float(index),
            }
        )
    return {
        "schema_version": "perturbgen_null_selection/v1",
        "source": {"cohort": "synthetic"},
        "target": {"ensembl_id": CANDIDATE, "features": _features((2.0, 0.8, 1.5, 1.0))},
        "excluded_ids": [CANDIDATE, "ENSG00000000002"],
        "excluded_candidate_ids": ["ENSG00000000002"],
        "required_count": 99,
        "selected_nulls": selected,
    }


def _e2e_report() -> dict:
    return {
        "schema_version": "davf_perturbgen_e2e/v1",
        "candidates": [
            {
                "status": "pass",
                "invocation": {
                    "gene_symbol": "STAT3",
                    "ensembl_id": CANDIDATE,
                    "intervention_type": "KO",
                },
            }
        ],
    }


def _executor(request) -> NullStageRecord:
    null_id = request["null_ensembl_id"]
    return NullStageRecord(
        null_ensembl_id=null_id,
        null_gene_symbol=request["null_gene_symbol"],
        rescue_excl_target=0.01,
        stage_manifest=f"/tmp/{null_id}/stage_manifest.json",
        result_h5ad=f"/tmp/{null_id}/result.h5ad",
        result_h5ad_sha256="a" * 64,
    )


def test_candidate_identity_requires_passing_invocation() -> None:
    identity = candidate_identity_from_e2e_report(_e2e_report())
    assert identity == {"gene_symbol": "STAT3", "ensembl_id": CANDIDATE}
    failed = _e2e_report()
    failed["candidates"][0]["status"] = "fail"
    with pytest.raises(NullGenerationError, match="no passing invocation"):
        candidate_identity_from_e2e_report(failed)


def test_dry_run_plans_99_null_identities_without_invocation(tmp_path) -> None:
    payload = run_matched_null_stages(
        _selection(),
        None,
        _e2e_report(),
        PATH,
        "mask",
        7,
        tmp_path / "nulls",
        dry_run=True,
        execute=False,
    )
    assert payload["schema_version"] == NULL_STAGE_PLAN_SCHEMA_VERSION
    assert payload["required_count"] == 99
    assert len(payload["planned_stages"]) == 99
    assert payload["execute"] is False
    assert "invocation" not in payload
    assert all(item["null_ensembl_id"] != CANDIDATE for item in payload["planned_stages"])


def test_stage_executor_collects_formal_null_distribution(tmp_path) -> None:
    output = tmp_path / "distribution.json"
    payload = run_matched_null_stages(
        _selection(),
        None,
        _e2e_report(),
        PATH,
        "mask",
        7,
        tmp_path / "nulls",
        stage_executor=_executor,
        output_path=output,
    )
    assert payload["schema_version"] == "perturbgen_null_distribution/v1"
    assert payload["required_count"] == 99
    assert len(payload["values"]) == 99
    assert len(payload["result_h5ad_sha256"]) == 99
    assert all(digest == "a" * 64 for digest in payload["result_h5ad_sha256"])
    assert json.loads(output.read_text(encoding="utf-8"))["candidate_ensembl_id"] == CANDIDATE


def test_gpu_path_requires_rescue_extractor_and_runner(tmp_path) -> None:
    with pytest.raises(NullGenerationError, match="base_config is required"):
        run_matched_null_stages(
            _selection(),
            None,
            _e2e_report(),
            PATH,
            "mask",
            7,
            tmp_path / "nulls",
            execute=True,
        )


def test_collect_rejects_empty_sha(tmp_path) -> None:
    records = [
        NullStageRecord(
            null_ensembl_id=f"ENSG{index + 4:011d}",
            null_gene_symbol=f"NULL{index}",
            rescue_excl_target=0.1,
            stage_manifest="m.json",
            result_h5ad="r.h5ad",
            result_h5ad_sha256="" if index == 0 else "b" * 64,
        )
        for index in range(99)
    ]
    with pytest.raises(NullGenerationError, match="sha256"):
        collect_null_stage_records(
            records,
            candidate_ensembl_id=CANDIDATE,
            path=PATH,
            mode="mask",
            seed=7,
        )
