"""Tests for the frozen activity-admission gate (方案 §4.4/§5.2, TD-02)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.ptm_activity import PTMActivityContractError
from src.analysis.ptm_activity_admission import select_activities_for_propagation
from src.analysis.ptm_research_config import ActivityAdmissionPolicy, PTMResearchConfigError


def _row(**overrides) -> dict:
    row = {
        "regulator_id": "GSK3B",
        "condition_or_contrast": "disease-minus-normal",
        "activity_score": 1.5,
        "activity_qvalue": 0.05,
        "n_substrates": 12,
        "network_coverage": 0.8,
        "method": "KSTAR",
    }
    row.update(overrides)
    return row


def test_policy_hash_is_stable_and_content_addressed():
    base = ActivityAdmissionPolicy()
    strict = ActivityAdmissionPolicy(max_activity_qvalue=0.05, min_substrates=1, min_network_coverage=0.1)
    assert base.policy_hash != strict.policy_hash
    assert ActivityAdmissionPolicy().policy_hash == base.policy_hash
    assert strict.as_dict() == {"max_activity_qvalue": 0.05, "min_substrates": 1, "min_network_coverage": 0.1}


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"max_activity_qvalue": 0.0}, "max_activity_qvalue"),
        ({"max_activity_qvalue": 1.5}, "max_activity_qvalue"),
        ({"min_substrates": -1}, "min_substrates"),
        ({"min_network_coverage": -0.1}, "min_network_coverage"),
    ],
)
def test_policy_rejects_out_of_range_thresholds(kwargs, match):
    with pytest.raises(PTMResearchConfigError, match=match):
        ActivityAdmissionPolicy(**kwargs)


def test_default_policy_admits_every_finite_row():
    frame = pd.DataFrame(
        [
            _row(),
            _row(regulator_id="CDK5", activity_qvalue=1.0, n_substrates=0, network_coverage=0.0),
        ]
    )
    selection = select_activities_for_propagation(frame, method="KSTAR", condition_or_contrast="disease-minus-normal")
    assert set(selection.admitted) == {"GSK3B", "CDK5"}
    assert selection.rejected.empty
    payload = selection.as_dict()
    assert payload["policy_hash"] == ActivityAdmissionPolicy().policy_hash
    assert payload["benchmark_gate"]["available"] is False


def test_strict_policy_rejects_rows_with_reasons():
    frame = pd.DataFrame(
        [
            _row(),  # admitted
            _row(regulator_id="CDK5", activity_qvalue=0.9),
            _row(regulator_id="MAPK", n_substrates=0),
            _row(regulator_id="AKT1", network_coverage=0.05),
        ]
    )
    policy = ActivityAdmissionPolicy(max_activity_qvalue=0.05, min_substrates=1, min_network_coverage=0.1)
    selection = select_activities_for_propagation(
        frame, method="KSTAR", condition_or_contrast="disease-minus-normal", policy=policy
    )
    assert set(selection.admitted) == {"GSK3B"}
    reasons = dict(zip(selection.rejected["regulator_id"], selection.rejected["reason"], strict=True))
    assert reasons["CDK5"] == "activity_qvalue_above_max"
    assert reasons["MAPK"] == "n_substrates_below_min"
    assert reasons["AKT1"] == "network_coverage_below_min"
    assert selection.as_dict()["n_rejected"] == 3


def test_all_rows_rejected_hard_fails():
    frame = pd.DataFrame([_row(regulator_id="CDK5", activity_qvalue=0.9)])
    policy = ActivityAdmissionPolicy(max_activity_qvalue=0.05)
    with pytest.raises(PTMActivityContractError, match="rejected every regulator"):
        select_activities_for_propagation(
            frame, method="KSTAR", condition_or_contrast="disease-minus-normal", policy=policy
        )


def test_missing_admission_columns_hard_fail():
    frame = pd.DataFrame([_row()]).drop(columns=["activity_qvalue"])
    with pytest.raises(PTMActivityContractError, match="activity_qvalue"):
        select_activities_for_propagation(frame, method="KSTAR", condition_or_contrast="disease-minus-normal")
