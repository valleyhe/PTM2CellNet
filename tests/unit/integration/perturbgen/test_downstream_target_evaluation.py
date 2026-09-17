"""Tests for downstream target-set sidecar loading and delta evaluation (方案 §5.5)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.integration.perturbgen.downstream_target_evaluation import (
    DownstreamTargetEvaluationError,
    driver_target_gate_to_payload,
    evaluate_driver_target_gate,
    evaluate_target_set_deltas,
    evaluation_to_payload,
    load_downstream_target_sidecar,
)


def _sidecar_payload() -> dict:
    return {
        "schema_version": "ptm2cellnet.downstream-target-sidecar/v1",
        "cell_type": "EX",
        "sources": {
            "ENSG00000082701": {
                "source_activity_id": "GSK3B",
                "gene_symbol": "GSK3B",
                "position": 9,
                "ptm_type": "phosphorylation",
                "proposed_direction": "down",
                "site_probability": 0.9,
                "provenance": "literature:test",
                "targets": [
                    {
                        "target_ensembl_id": "ENSG00000186868",
                        "target_gene_symbol": "MAPT",
                        "gene_score": -0.51,
                        "predicted_gene_direction": "down",
                        "observed_log2fc": -0.7,
                        "observed_fdr": 0.01,
                        "observed_direction": "down",
                    },
                    {
                        "target_ensembl_id": "ENSG00000000009",
                        "target_gene_symbol": "GENE9",
                        "gene_score": 0.2,
                        "predicted_gene_direction": "up",
                        "observed_log2fc": -0.1,
                        "observed_fdr": 0.01,
                        "observed_direction": "down",
                    },
                ],
            }
        },
        "lineage": {"target_set_manifest": "target_set_manifest.json"},
    }


def _write_sidecar(tmp_path, payload=None) -> object:
    path = tmp_path / "downstream_targets_EX.json"
    path.write_text(json.dumps(payload or _sidecar_payload()), encoding="utf-8")
    return path


def _delta_frame() -> pd.DataFrame:
    return pd.DataFrame(
        np.array(
            [
                [-0.8, -0.6],  # ENSG00000186868 mean < 0 -> down
                [0.3, 0.5],  # ENSG00000000009 mean > 0 -> up
            ]
        ),
        index=["ENSG00000186868", "ENSG00000000009"],
        columns=["run1", "run2"],
    )


class TestLoadSidecar:
    def test_valid_sidecar_loads(self, tmp_path):
        sidecar = load_downstream_target_sidecar(_write_sidecar(tmp_path))
        assert sidecar.cell_type == "EX"
        source = sidecar.source_by_ensembl("ENSG00000082701")
        assert len(source.targets) == 2

    def test_wrong_schema_fails(self, tmp_path):
        payload = _sidecar_payload()
        payload["schema_version"] = "other/v9"
        with pytest.raises(DownstreamTargetEvaluationError, match="schema_version"):
            load_downstream_target_sidecar(_write_sidecar(tmp_path, payload))

    def test_empty_targets_fail(self, tmp_path):
        payload = _sidecar_payload()
        payload["sources"]["ENSG00000082701"]["targets"] = []
        with pytest.raises(DownstreamTargetEvaluationError, match="non-empty frozen target list"):
            load_downstream_target_sidecar(_write_sidecar(tmp_path, payload))

    def test_duplicate_targets_fail(self, tmp_path):
        payload = _sidecar_payload()
        payload["sources"]["ENSG00000082701"]["targets"].append(payload["sources"]["ENSG00000082701"]["targets"][0])
        with pytest.raises(DownstreamTargetEvaluationError, match="duplicate target"):
            load_downstream_target_sidecar(_write_sidecar(tmp_path, payload))

    def test_invalid_direction_fails(self, tmp_path):
        payload = _sidecar_payload()
        payload["sources"]["ENSG00000082701"]["proposed_direction"] = "flat"
        with pytest.raises(DownstreamTargetEvaluationError, match="proposed_direction"):
            load_downstream_target_sidecar(_write_sidecar(tmp_path, payload))

    def test_source_lookup_miss_fails(self, tmp_path):
        sidecar = load_downstream_target_sidecar(_write_sidecar(tmp_path))
        with pytest.raises(DownstreamTargetEvaluationError, match="no source"):
            sidecar.source_by_ensembl("ENSG00000000001")


class TestEvaluateTargetSetDeltas:
    def test_directions_and_concordance_are_field_separate(self, tmp_path):
        sidecar = load_downstream_target_sidecar(_write_sidecar(tmp_path))
        evaluations = evaluate_target_set_deltas(_delta_frame(), sidecar)
        evaluation = evaluations["ENSG00000082701"]
        mapt = next(d for d in evaluation.deltas if d.target_ensembl_id == "ENSG00000186868")
        assert mapt.delta_direction == "down"
        assert mapt.matches_predicted is True
        assert mapt.matches_observed is True
        gene9 = next(d for d in evaluation.deltas if d.target_ensembl_id == "ENSG00000000009")
        assert gene9.delta_direction == "up"
        assert gene9.matches_predicted is True  # predicted up
        assert gene9.matches_observed is False  # observed down — kept separate (方案 §3.1)
        assert evaluation.n_matching_predicted == 2
        assert evaluation.n_matching_observed == 1

    def test_missing_targets_are_explicit(self, tmp_path):
        sidecar = load_downstream_target_sidecar(_write_sidecar(tmp_path))
        frame = _delta_frame().drop(index="ENSG00000000009")
        evaluations = evaluate_target_set_deltas(frame, sidecar)
        evaluation = evaluations["ENSG00000082701"]
        assert evaluation.missing_targets == ("ENSG00000000009",)
        assert evaluation.n_missing_targets == 1
        assert len(evaluation.deltas) == 1

    def test_zero_mean_delta_is_indeterminate(self, tmp_path):
        sidecar = load_downstream_target_sidecar(_write_sidecar(tmp_path))
        frame = _delta_frame()
        frame.loc["ENSG00000186868"] = [0.5, -0.5]
        evaluations = evaluate_target_set_deltas(frame, sidecar)
        evaluation = evaluations["ENSG00000082701"]
        mapt = next(d for d in evaluation.deltas if d.target_ensembl_id == "ENSG00000186868")
        assert mapt.delta_direction == "indeterminate"
        assert mapt.matches_predicted is False
        assert evaluation.n_indeterminate == 1

    def test_duplicate_gene_ids_fail(self, tmp_path):
        sidecar = load_downstream_target_sidecar(_write_sidecar(tmp_path))
        frame = pd.concat([_delta_frame(), _delta_frame()])
        with pytest.raises(DownstreamTargetEvaluationError, match="duplicated"):
            evaluate_target_set_deltas(frame, sidecar)

    def test_non_finite_delta_fails(self, tmp_path):
        sidecar = load_downstream_target_sidecar(_write_sidecar(tmp_path))
        frame = _delta_frame()
        frame.iloc[0, 0] = np.nan
        with pytest.raises(DownstreamTargetEvaluationError, match="non-finite"):
            evaluate_target_set_deltas(frame, sidecar)

    def test_versioned_ensembl_index_is_canonicalized(self, tmp_path):
        sidecar = load_downstream_target_sidecar(_write_sidecar(tmp_path))
        frame = _delta_frame().rename(index={"ENSG00000186868": "ENSG00000186868.7"})
        evaluations = evaluate_target_set_deltas(frame, sidecar)
        # The versioned id canonicalizes to the sidecar id and both targets match
        evaluation = evaluations["ENSG00000082701"]
        assert evaluation.n_missing_targets == 0
        assert {d.target_ensembl_id for d in evaluation.deltas} == {"ENSG00000186868", "ENSG00000000009"}

    def test_gene_id_column_is_excluded_from_numeric_delta_values(self, tmp_path):
        sidecar = load_downstream_target_sidecar(_write_sidecar(tmp_path))
        frame = _delta_frame().rename_axis("gene_id").reset_index()
        evaluations = evaluate_target_set_deltas(frame, sidecar, gene_id_column="gene_id")
        assert evaluations["ENSG00000082701"].n_missing_targets == 0

    def test_payload_serializes_without_verdicts(self, tmp_path):
        sidecar = load_downstream_target_sidecar(_write_sidecar(tmp_path))
        evaluations = evaluate_target_set_deltas(_delta_frame(), sidecar)
        payload = evaluation_to_payload(evaluations, cell_type="EX", delta_matrix_source="stub.h5ad")
        assert payload["schema_version"] == "ptm2cellnet.target-set-evaluation/v1"
        assert "not causal validation" in payload["note"]
        assert payload["sources"]["ENSG00000082701"]["n_matching_predicted"] == 2


def _source_gate(**overrides) -> dict:
    payload = {
        "status": "pass",
        "gene_symbol": "GSK3B",
        "ensembl_id": "ENSG00000082701",
        "proposed_direction": "down",
        "davf_direction": "down",
        "observed_direction": "down",
        "corrective_action": "oe",
        "reasons": (),
    }
    payload.update(overrides)
    return payload


class TestDriverTargetGate:
    def test_classifies_mixed_concordance_with_field_separate_fractions(self, tmp_path):
        sidecar = load_downstream_target_sidecar(_write_sidecar(tmp_path))
        evaluation = evaluate_target_set_deltas(_delta_frame(), sidecar)["ENSG00000082701"]
        evidence = evaluate_driver_target_gate(
            _source_gate(),
            evaluation,
            semantic_context={"context": "GSE174367 EX cells", "intervention": "KO"},
        )
        # MAPT down matches observed down, GENE9 up disagrees with observed down.
        assert evidence.driver_target_status == "mixed"
        assert evidence.n_with_observed == 2
        assert evidence.n_matching_observed == 1
        assert evidence.concordance_fraction_observed == pytest.approx(0.5)
        assert evidence.n_with_predicted == 2
        assert evidence.n_matching_predicted == 2
        assert evidence.concordance_fraction_predicted == pytest.approx(1.0)
        # Source side is re-recorded untouched, never overwritten by target counts.
        assert evidence.source_gate_status == "pass"
        assert evidence.source_has_own_deg is True
        assert evidence.source_observed_direction == "down"
        assert evidence.semantic_context == {"context": "GSE174367 EX cells", "intervention": "KO"}

    def test_all_concordant_and_none_concordant_classification(self, tmp_path):
        sidecar = load_downstream_target_sidecar(_write_sidecar(tmp_path))
        all_down = _delta_frame().copy()
        all_down.iloc[1] = [-0.4, -0.2]  # GENE9 now down as well
        evaluation = evaluate_target_set_deltas(all_down, sidecar)["ENSG00000082701"]
        evidence = evaluate_driver_target_gate(_source_gate(), evaluation)
        assert evidence.driver_target_status == "all_concordant_observed"
        assert evidence.concordance_fraction_observed == pytest.approx(1.0)

        all_up = _delta_frame().copy()
        all_up.iloc[0] = [0.7, 0.9]  # both targets up while observed says down
        evaluation = evaluate_target_set_deltas(all_up, sidecar)["ENSG00000082701"]
        evidence = evaluate_driver_target_gate(_source_gate(), evaluation)
        assert evidence.driver_target_status == "none_concordant_observed"
        assert evidence.concordance_fraction_observed == pytest.approx(0.0)

    def test_missing_source_observation_does_not_block_target_evaluation(self, tmp_path):
        sidecar = load_downstream_target_sidecar(_write_sidecar(tmp_path))
        evaluation = evaluate_target_set_deltas(_delta_frame(), sidecar)["ENSG00000082701"]
        evidence = evaluate_driver_target_gate(_source_gate(observed_direction=None), evaluation)
        # The source side records its own missing observation; the target side
        # still evaluates against the targets' observed directions (方案 §3.1).
        assert evidence.source_has_own_deg is False
        assert evidence.driver_target_status == "mixed"
        assert evidence.concordance_fraction_observed == pytest.approx(0.5)

    def test_not_evaluable_without_any_evaluable_targets(self, tmp_path):
        sidecar = load_downstream_target_sidecar(_write_sidecar(tmp_path))
        missing_frame = _delta_frame().rename(
            index={"ENSG00000186868": "ENSG00000000005", "ENSG00000000009": "ENSG00000000006"}
        )
        evaluation = evaluate_target_set_deltas(missing_frame, sidecar)["ENSG00000082701"]
        assert evaluation.n_missing_targets == 2
        evidence = evaluate_driver_target_gate(_source_gate(), evaluation)
        assert evidence.driver_target_status == "not_evaluable"
        assert evidence.n_evaluable_targets == 0
        assert evidence.concordance_fraction_observed is None

    def test_payload_separates_source_and_target_and_states_boundary(self, tmp_path):
        sidecar = load_downstream_target_sidecar(_write_sidecar(tmp_path))
        evaluation = evaluate_target_set_deltas(_delta_frame(), sidecar)["ENSG00000082701"]
        payload = driver_target_gate_to_payload(
            evaluate_driver_target_gate(
                _source_gate(reasons=("observed_expression_not_significant",)),
                evaluation,
                semantic_context={"cohort": "GSE174367"},
            )
        )
        assert payload["schema_version"] == "ptm2cellnet.driver-target-gate/v1"
        assert payload["source"]["gate_kind"] == "source_three_way"
        assert payload["source"]["gate_status"] == "pass"
        assert payload["source"]["gate_reasons"] == ["observed_expression_not_significant"]
        assert payload["source"]["source_has_own_deg"] is True
        assert payload["target_set"]["n_matching_observed"] == 1
        assert payload["target_set"]["concordance_fraction_observed"] == pytest.approx(0.5)
        assert payload["driver_target_status"] == "mixed"
        assert payload["semantic_context"] == {"cohort": "GSE174367"}
        assert "never replaces" in payload["note"]

    def test_source_identity_mismatch_fails(self, tmp_path):
        sidecar = load_downstream_target_sidecar(_write_sidecar(tmp_path))
        evaluation = evaluate_target_set_deltas(_delta_frame(), sidecar)["ENSG00000082701"]
        with pytest.raises(DownstreamTargetEvaluationError, match="source mismatch"):
            evaluate_driver_target_gate(_source_gate(ensembl_id="ENSG00000130203"), evaluation)
        with pytest.raises(DownstreamTargetEvaluationError, match="source gene mismatch"):
            evaluate_driver_target_gate(_source_gate(gene_symbol="OTHER"), evaluation)

    def test_invalid_source_gate_payload_fails(self, tmp_path):
        sidecar = load_downstream_target_sidecar(_write_sidecar(tmp_path))
        evaluation = evaluate_target_set_deltas(_delta_frame(), sidecar)["ENSG00000082701"]
        with pytest.raises(DownstreamTargetEvaluationError, match="status must be one of"):
            evaluate_driver_target_gate(_source_gate(status="maybe"), evaluation)
        with pytest.raises(DownstreamTargetEvaluationError, match="requires a source ensembl_id"):
            evaluate_driver_target_gate(_source_gate(ensembl_id=""), evaluation)
        with pytest.raises(DownstreamTargetEvaluationError, match="must be 'up', 'down' or null"):
            evaluate_driver_target_gate(_source_gate(davf_direction="flat"), evaluation)
