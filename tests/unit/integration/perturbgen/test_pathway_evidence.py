"""U-06: pathway evidence is secondary and cannot write a dual-path verdict."""

from __future__ import annotations

import pytest

from src.integration.perturbgen.pathway_evidence import (
    PathwayEvidenceError,
    assert_not_used_for_dual_path,
    evaluate_pathway_evidence,
)


def test_default_backend_is_inconclusive_secondary_evidence() -> None:
    payload = evaluate_pathway_evidence(["ENSG00000168610", "ENSG00000177606"])
    assert payload["affects_dual_path_verdict"] is False
    assert payload["evidence_class"] == "secondary"
    assert payload["status"] == "inconclusive"
    assert "pathway_backend_not_configured" in payload["reasons"]
    assert_not_used_for_dual_path(payload)


def test_gseapy_absence_or_missing_gmt_stays_inconclusive() -> None:
    payload = evaluate_pathway_evidence(["G1"], backend="gseapy")
    assert payload["affects_dual_path_verdict"] is False
    assert payload["status"] == "inconclusive"
    assert payload["reasons"]


def test_cannot_claim_pathway_as_hard_gate() -> None:
    with pytest.raises(PathwayEvidenceError, match="must not affect"):
        assert_not_used_for_dual_path(
            {"affects_dual_path_verdict": True, "evidence_class": "secondary"}
        )
