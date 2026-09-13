from dataclasses import asdict
import json

import pytest

import scripts.run_perturbgen_pipeline as pipeline
from scripts.run_perturbgen_pipeline import _apply_path
from src.integration.perturbgen.contracts import CandidateEvidence, DAVFDirectionEvidence
from src.integration.perturbgen.orchestrator import PerturbGenInvocation


def _config():
    return {
        "stages": {
            "perturb": {
                "perturb_config": {
                    "trainer": {
                        "genes_to_perturb": ["STAT3"],
                        "perturbation_mode": "overexpress",
                        "perturbation_sequence": ["src"],
                        "pert_tps": [1],
                    },
                    "datamodule": {"pert_tps": [1]},
                }
            }
        },
        "pipeline": {"random_seed": 42},
    }


def test_apply_source_path_removes_target_timepoints():
    config = _config()
    _apply_path(config, "source_intervention")
    perturb = config["stages"]["perturb"]["perturb_config"]
    assert perturb["trainer"]["perturbation_sequence"] == ["src"]
    assert "pert_tps" not in perturb["trainer"]
    assert "pert_tps" not in perturb["datamodule"]


def test_apply_within_state_uses_target_and_requires_pert_tps():
    config = _config()
    _apply_path(config, "within_state")
    assert config["stages"]["perturb"]["perturb_config"]["trainer"]["perturbation_sequence"] == ["tgt"]

    bad = _config()
    bad["stages"]["perturb"]["perturb_config"]["trainer"].pop("pert_tps")
    with pytest.raises(ValueError, match="requires.*pert_tps"):
        _apply_path(bad, "within_state")


def _stub_pipeline_main(monkeypatch):
    monkeypatch.setattr(pipeline, "load_pipeline_config", lambda _path: _config())
    monkeypatch.setattr(pipeline, "_build_selected_plans", lambda *_args: ())

    class _Runner:
        def __init__(self, **_kwargs):
            pass

        def run_pipeline(self, *_args, **_kwargs):
            return []

    monkeypatch.setattr(pipeline, "PerturbGenRunner", _Runner)


def _valid_gate_record(config_path):
    evidence = DAVFDirectionEvidence(
        gene_symbol="STAT3",
        ensembl_id="ENSG00000168610",
        predicted_direction="down",
        predicted_delta=-1.0,
        model_source="davf-test",
        checkpoint_provenance="checkpoint-test",
        embedding_provenance="embedding-test",
        confidence=0.5,
    )
    candidate = CandidateEvidence(
        gene_symbol="STAT3",
        ensembl_id="ENSG00000168610",
        cell_type="T cell",
        ptm_context="STAT3:S12",
        observed_log2fc=-1.0,
        observed_fdr=0.01,
        observed_direction="down",
        davf_action="oe",
        davf_score=0.5,
        davf_provenance="candidate-test",
        proposed_direction="down",
        davf_predicted_direction="down",
        davf_predicted_delta=-1.0,
        direction_gate_status="pass",
    )
    invocation = PerturbGenInvocation(
        intervention_type="KO",
        gene_symbol="STAT3",
        ensembl_id="ENSG00000168610",
        target_token_id=7,
        perturbation_mode="overexpress",
        paths=("source_intervention", "within_state"),
        candidate=candidate,
        davf_evidence=evidence,
        perturbgen_config_path=config_path,
        seed=42,
    )
    return {
        "status": "pass",
        "candidate": asdict(candidate),
        "davf_evidence": asdict(evidence),
        "invocation": invocation.to_dict(),
    }


def test_perturb_stage_requires_explicit_e2e_gate_report(tmp_path):
    with pytest.raises(SystemExit):
        pipeline.main(
            [
                "--config",
                str(tmp_path / "config.yaml"),
                "--stages",
                "perturb",
            ]
        )


def test_training_only_does_not_require_e2e_gate(tmp_path, monkeypatch):
    _stub_pipeline_main(monkeypatch)
    assert pipeline.main(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--stages",
            "tokenise",
        ]
    ) == 0


def test_valid_explicit_e2e_gate_report_allows_perturb_stage(tmp_path, monkeypatch):
    _stub_pipeline_main(monkeypatch)
    report = tmp_path / "e2e-gate.json"
    report.write_text(json.dumps({"candidates": [_valid_gate_record(tmp_path / "config.yaml")]}), encoding="utf-8")

    assert pipeline.main(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--stages",
            "perturb",
            "--e2e-gate-report",
            str(report),
        ]
    ) == 0


def test_e2e_gate_report_rejects_unbound_davf_score(tmp_path):
    report = tmp_path / "mismatched-e2e-gate.json"
    record = _valid_gate_record(tmp_path / "config.yaml")
    record["invocation"]["davf_evidence"]["confidence"] = 0.4
    report.write_text(json.dumps({"candidates": [record]}), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly match"):
        pipeline._validate_e2e_gate_report(
            report,
            expected_binding=pipeline._config_gate_binding(
                _config(), path=None, config_path=tmp_path / "config.yaml"
            ),
        )


def test_failed_e2e_gate_report_is_rejected(tmp_path, monkeypatch):
    _stub_pipeline_main(monkeypatch)
    report = tmp_path / "failed-e2e-gate.json"
    report.write_text(
        json.dumps(
            {
                "preparation": {
                    "status": "fail",
                    "candidate": {"direction_gate_status": "pass"},
                    "invocation": {"gene_symbol": "STAT3"},
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        pipeline.main(
            [
                "--config",
                str(tmp_path / "config.yaml"),
                "--stages",
                "perturb",
                "--e2e-gate-report",
                str(report),
            ]
        )
