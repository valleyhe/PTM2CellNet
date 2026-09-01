from __future__ import annotations

import pytest

from src.integration.perturbgen.contracts import (
    CandidateEvidence,
    DAVFDirectionEvidence,
    Direction,
    PTMSiteDirectionProposal,
)
from src.integration.perturbgen.direction_gate import (
    build_direction_gated_candidate,
    evaluate_direction_gate,
)
from src.integration.perturbgen.mainline import evaluate_davf_perturbgen_candidate


def _proposal(direction: Direction = "up") -> PTMSiteDirectionProposal:
    return PTMSiteDirectionProposal(
        gene_symbol="TP53",
        ensembl_id="ENSG00000141510.18",
        position=473,
        ptm_type="phosphorylation",
        proposed_direction=direction,
        site_probability=0.95,
        provenance="ptm-site-test",
    )


def _davf(
    direction: Direction = "up",
    *,
    model_source: str = "davf-checkpoint",
) -> DAVFDirectionEvidence:
    return DAVFDirectionEvidence(
        gene_symbol="TP53",
        ensembl_id="ENSG00000141510.18",
        predicted_direction=direction,
        predicted_delta=0.8 if direction == "up" else -0.8,
        model_source=model_source,
        checkpoint_provenance="checkpoint-v1",
        embedding_provenance="embedding-v1",
        confidence=0.9,
    )


def test_direction_gate_passes_for_three_agreeing_directions() -> None:
    result = evaluate_direction_gate(
        _proposal("up"),
        _davf("up"),
        observed_direction="up",
        observed_fdr=0.01,
    )

    assert result.status == "pass"
    assert result.proposed_direction == "up"
    assert result.davf_direction == "up"
    assert result.observed_direction == "up"
    assert result.corrective_action == "ko"
    assert result.reasons == ()


@pytest.mark.parametrize(
    ("proposal_direction", "davf_direction", "observed_direction"),
    [
        ("up", "down", "up"),
        ("down", "up", "up"),
        ("up", "up", "down"),
    ],
)
def test_direction_gate_fails_when_complete_directions_disagree(
    proposal_direction: Direction,
    davf_direction: Direction,
    observed_direction: Direction,
) -> None:
    result = evaluate_direction_gate(
        _proposal(proposal_direction),
        _davf(davf_direction),
        observed_direction=observed_direction,
        observed_fdr=0.01,
    )

    assert result.status == "fail"
    assert "direction_evidence_disagreement" in result.reasons
    assert result.corrective_action is None


@pytest.mark.parametrize(
    ("missing_field", "expected_reason"),
    [
        ("proposal", "missing_ptm_direction_proposal"),
        ("davf", "missing_davf_direction_evidence"),
        ("observed_direction", "missing_observed_direction"),
        ("observed_fdr", "missing_observed_fdr"),
    ],
)
def test_direction_gate_is_inconclusive_when_evidence_is_missing(
    missing_field: str,
    expected_reason: str,
) -> None:
    proposal = None if missing_field == "proposal" else _proposal("up")
    davf_evidence = None if missing_field == "davf" else _davf("up")
    observed_direction = None if missing_field == "observed_direction" else "up"
    observed_fdr = None if missing_field == "observed_fdr" else 0.01

    result = evaluate_direction_gate(
        proposal,
        davf_evidence,
        observed_direction=observed_direction,
        observed_fdr=observed_fdr,
    )

    assert result.status == "inconclusive"
    assert expected_reason in result.reasons
    assert result.corrective_action is None


def test_zero_fallback_davf_evidence_cannot_pass() -> None:
    result = evaluate_direction_gate(
        _proposal("up"),
        _davf("up", model_source="zero_fallback"),
        observed_direction="up",
        observed_fdr=0.01,
    )

    assert result.status == "inconclusive"
    assert "davf_model_asset_unavailable" in result.reasons
    assert result.corrective_action is None


def test_near_zero_davf_delta_is_inconclusive() -> None:
    evidence = DAVFDirectionEvidence(
        gene_symbol="TP53",
        ensembl_id="ENSG00000141510",
        predicted_direction=None,
        predicted_delta=0.0,
        model_source="davf-checkpoint",
        checkpoint_provenance="checkpoint-v1",
        embedding_provenance="embedding-v1",
    )

    result = evaluate_direction_gate(
        _proposal("up"),
        evidence,
        observed_direction="up",
        observed_fdr=0.01,
    )

    assert result.status == "inconclusive"
    assert result.reasons == ("missing_davf_predicted_direction",)


def test_build_direction_gated_candidate_constructs_evidence_after_pass() -> None:
    gate, candidate = build_direction_gated_candidate(
        _proposal("up"),
        _davf("up"),
        cell_type="CD14 Monocyte",
        ptm_context="AKT1:S473",
        observed_log2fc=1.2,
        observed_fdr=0.01,
        observed_direction="up",
    )

    assert gate.status == "pass"
    assert isinstance(candidate, CandidateEvidence)
    assert candidate is not None
    assert candidate.gene_symbol == "TP53"
    assert candidate.ensembl_id == "ENSG00000141510"
    assert candidate.davf_action == "ko"
    assert candidate.davf_score is None
    assert candidate.davf_predicted_delta == pytest.approx(0.8)
    assert candidate.davf_provenance == "checkpoint-v1"
    assert candidate.direction_gate_status == "pass"
    assert candidate.proposed_direction == "up"
    assert candidate.davf_predicted_direction == "up"


@pytest.mark.parametrize("failure", ["mismatch", "missing_davf", "zero_fallback"])
def test_build_direction_gated_candidate_returns_no_candidate_without_pass(
    failure: str,
) -> None:
    proposal = _proposal("up")
    davf_evidence = _davf("up")
    if failure == "mismatch":
        davf_evidence = _davf("down")
    elif failure == "missing_davf":
        davf_evidence = None
    else:
        davf_evidence = _davf("up", model_source="zero_fallback")

    gate, candidate = build_direction_gated_candidate(
        proposal,
        davf_evidence,
        cell_type="CD14 Monocyte",
        ptm_context="AKT1:S473",
        observed_log2fc=1.2,
        observed_fdr=0.01,
        observed_direction="up",
    )

    assert gate.status != "pass"
    assert candidate is None


@pytest.mark.parametrize("failure", ["mismatch", "missing_davf", "zero_fallback"])
def test_mainline_does_not_produce_dual_path_when_direction_gate_does_not_pass(
    failure: str,
) -> None:
    proposal = _proposal("up")
    davf_evidence = _davf("up")
    expected_verdict = "inconclusive"
    if failure == "mismatch":
        davf_evidence = _davf("down")
        expected_verdict = "fail"
    elif failure == "missing_davf":
        davf_evidence = None
    else:
        davf_evidence = _davf("up", model_source="zero_fallback")

    decision = evaluate_davf_perturbgen_candidate(
        proposal,
        davf_evidence,
        cell_type="CD14 Monocyte",
        ptm_context="AKT1:S473",
        observed_log2fc=1.2,
        observed_fdr=0.01,
        observed_direction="up",
        path_results=(object(),),
        q_value=0.01,
    )

    assert decision.verdict == expected_verdict
    assert decision.direction_gate.status == expected_verdict
    assert decision.candidate is None
    assert decision.dual_path is None
