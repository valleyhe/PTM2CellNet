"""Tests for the frozen 2026-09-18 AD research-decision schema."""

from __future__ import annotations

import pytest

from src.analysis.ad_research_decision import (
    ADResearchDecisionError,
    FROZEN_AD_RESEARCH_DECISION,
    OBSERVED_ADMISSION_FDR_CUTOFF,
    OBSERVED_ADMISSION_SIGNED_DIRECTION,
    load_ad_research_decision,
    observed_significance_required,
    parse_ad_research_decision,
    write_ad_research_decision,
)


def test_frozen_decision_is_option_three_ko_only_no_public_inventory():
    decision = FROZEN_AD_RESEARCH_DECISION
    assert decision.observed_admission_rule == OBSERVED_ADMISSION_SIGNED_DIRECTION
    assert decision.deg_reporting_max_fdr == 0.05
    assert decision.kd_policy == "merged_into_ko_out_of_scope"
    assert decision.public_perturbation_policy == "out_of_scope"
    assert decision.biology_pass is False
    assert observed_significance_required(decision.observed_admission_rule) is False
    payload = decision.to_payload()
    assert payload["observed_significance_required"] is False
    assert payload["biology_pass"] is False


def test_unmarked_rule_keeps_legacy_fdr_cutoff():
    assert observed_significance_required(None) is True
    assert observed_significance_required("") is True
    assert observed_significance_required(OBSERVED_ADMISSION_FDR_CUTOFF) is True


def test_cannot_claim_biology_pass_or_relax_reporting_fdr():
    payload = FROZEN_AD_RESEARCH_DECISION.to_payload()
    payload["biology_pass"] = True
    with pytest.raises(ADResearchDecisionError, match="biology_pass"):
        parse_ad_research_decision(payload)
    payload = FROZEN_AD_RESEARCH_DECISION.to_payload()
    payload["deg_reporting_max_fdr"] = 0.2
    with pytest.raises(ADResearchDecisionError, match="0.05"):
        parse_ad_research_decision(payload)


def test_round_trip_json(tmp_path):
    path = write_ad_research_decision(FROZEN_AD_RESEARCH_DECISION, tmp_path / "observed_gate_decision.json")
    loaded = load_ad_research_decision(path)
    assert loaded == FROZEN_AD_RESEARCH_DECISION
