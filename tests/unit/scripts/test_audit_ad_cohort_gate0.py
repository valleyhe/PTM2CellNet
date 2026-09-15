"""Synthetic unit tests for the AD Gate-0 audit verdict logic (TD-14-06/F-14).

The audit functions that read data/AD stay untested here; these tests pin the
contract-facing pure parts: the dual-pairing requirement wording, the
structural analysis shape, and the derived verdict (F-14 regression net).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.audit_ad_cohort_gate0 as audit  # noqa: E402


def test_gate0_requirements_cover_both_pairing_semantics():
    joined = " ".join(audit.GATE0_REQUIREMENTS)
    assert "within_donor pairing" in joined
    assert "between_donor pairing" in joined
    assert ">= 3 donors shared" in joined
    assert "donor-disjoint state groups" in joined


def test_structural_analysis_reports_both_pairing_tracks():
    analysis = audit.structural_shared_donor_analysis()
    assert analysis["max_achievable_shared_donors_within_donor_pairing"] == 0
    assert set(analysis["between_donor_status"]) == {
        "GSE147528",
        "GSE157827",
        "GSE174367",
        "GSE188545",
    }
    assert analysis["between_donor_status"]["GSE174367"].startswith("standardized")
    assert analysis["decision_record"].startswith("lessons.md L-2026-0914-01")


def test_verdict_unlocks_only_when_a_cohort_is_standardized():
    blocked_audit = {"gate0_status": audit.GATE0_STATUS_BLOCKED}
    assert audit.compute_verdict({"GSE157827": blocked_audit}) == "GATE0_BLOCKED_SEMANTICS_AND_LABELS"

    standardized_audit = {"gate0_status": audit.GATE0_STATUS_STANDARDIZED}
    verdict = audit.compute_verdict({"GSE174367": standardized_audit, "GSE157827": blocked_audit})
    assert verdict == audit.GATE0_VERDICT
    assert "GSE174367" in verdict
