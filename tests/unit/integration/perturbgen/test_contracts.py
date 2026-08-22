from dataclasses import FrozenInstanceError
import math

import pytest

from src.integration.perturbgen.contracts import (
    CandidateEvidence,
    CandidateScreeningResult,
    DualPathVerdict,
    PathResult,
    PerturbGenDataSpec,
    PreparedPerturbationReport,
)


def _candidate(**overrides):
    payload = {
        "gene_symbol": "tp53",
        "ensembl_id": "ENSG00000141510.18",
        "cell_type": "CD14 Monocyte",
        "ptm_context": "AKT1:S473",
        "observed_log2fc": 1.2,
        "observed_fdr": 0.01,
        "observed_direction": "up",
        "davf_action": "oe",
        "davf_score": 0.8,
        "davf_provenance": "davf-v1",
    }
    payload.update(overrides)
    return CandidateEvidence(**payload)


def test_candidate_evidence_is_frozen_and_separates_direction_from_davf_action():
    candidate = _candidate()
    assert candidate.gene_symbol == "TP53"
    assert candidate.ensembl_id == "ENSG00000141510"
    assert candidate.observed_direction == "up"
    assert candidate.davf_action == "oe"

    with pytest.raises(FrozenInstanceError):
        candidate.observed_direction = "down"


def test_candidate_evidence_rejects_invalid_direction_and_fdr():
    with pytest.raises(ValueError, match="observed_direction"):
        _candidate(observed_direction="ko")
    with pytest.raises(ValueError, match="observed_fdr"):
        _candidate(observed_fdr=1.5)
    with pytest.raises(ValueError, match="observed_log2fc must be finite"):
        _candidate(observed_log2fc=math.nan)
    with pytest.raises(ValueError, match="observed_fdr"):
        _candidate(observed_fdr=math.nan)
    with pytest.raises(ValueError, match="requires observed_log2fc"):
        _candidate(observed_direction="down", observed_log2fc=1.0)


def test_data_spec_requires_at_least_three_donors():
    with pytest.raises(ValueError, match="min_donors"):
        PerturbGenDataSpec(min_donors=2)


def test_prepared_report_requires_three_evaluable_donors():
    with pytest.raises(ValueError, match="at least 3 evaluable donors"):
        PreparedPerturbationReport(
            cell_type="Mono",
            normal_state="normal",
            disease_state="disease",
            evaluable_donors=("d1", "d2"),
            n_cells=20,
            n_genes=10,
        )


def test_candidate_screening_result_requires_reason_codes_for_non_evaluable():
    candidate = _candidate()
    with pytest.raises(ValueError, match="reason_codes"):
        CandidateScreeningResult(candidate=candidate, status="failed")

    result = CandidateScreeningResult(
        candidate=candidate,
        status="inconclusive",
        reason_codes=("gene_not_detected_in_disease_state",),
    )
    assert result.status == "inconclusive"


def test_path_result_and_dual_path_verdict_enforce_tri_state_contract():
    result = PathResult(
        status="evaluable",
        path="source_intervention",
        mode="mask",
        rescue_excl_target=0.3,
        evaluable_donors=3,
        donor_consistency=2 / 3,
        seed=0,
        output_h5ad="outputs/run1.h5ad",
        matched_null_count=99,
        empirical_pvalue=0.02,
    )
    assert result.status == "evaluable"

    with pytest.raises(ValueError, match="reason_code"):
        PathResult(
            status="failed",
            path="within_state",
            mode="delete",
            rescue_excl_target=None,
            evaluable_donors=0,
            donor_consistency=None,
            seed=0,
        )

    assert DualPathVerdict(verdict="pass", q_value=0.03).verdict == "pass"
    with pytest.raises(ValueError, match="provide reasons"):
        DualPathVerdict(verdict="inconclusive")
