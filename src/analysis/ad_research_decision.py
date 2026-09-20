"""Frozen 2026-09-18 AD research decisions (observed gate / KD / public Perturb-seq).

Rivera's three choices:

1. Observed gate branch 3 — revise formal *admission* semantics to signed
   disease−normal direction concordance. Donor-level Welch+BH and
   ``deg_max_fdr=0.05`` remain the reporting/significance label; FDR is not
   an admission/three-way-gate pass-fail cutoff, and FDR>0.05 is never
   relabelled significant.
2. KD is merged into KO for these AD candidates (KO-only loss-of-function).
   DAVF labels 0/1 stay distinct; KO is not inverted into KD.
3. Public Perturb-seq inventory (B1/B3 transfer) is out of scope.

This module is the auditable schema. It does not emit PerturbGen invocations
and does not claim formal biology PASS.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

AD_RESEARCH_DECISION_SCHEMA_VERSION = "ptm2cellnet.ad-research-decision/v1"

OBSERVED_GATE_BRANCH_REVISE_SEMANTICS = "revise_semantics"
OBSERVED_ADMISSION_FDR_CUTOFF = "fdr_cutoff"
OBSERVED_ADMISSION_SIGNED_DIRECTION = "signed_direction_without_fdr_cutoff"
KD_POLICY_SEPARATE_ROUTES = "separate_routes"
KD_POLICY_MERGED_INTO_KO = "merged_into_ko_out_of_scope"
PUBLIC_PERTURBATION_INVENTORY_OPTIONAL = "inventory_optional"
PUBLIC_PERTURBATION_OUT_OF_SCOPE = "out_of_scope"

VALID_OBSERVED_GATE_BRANCHES = (OBSERVED_GATE_BRANCH_REVISE_SEMANTICS,)
VALID_OBSERVED_ADMISSION_RULES = (
    OBSERVED_ADMISSION_FDR_CUTOFF,
    OBSERVED_ADMISSION_SIGNED_DIRECTION,
)
VALID_KD_POLICIES = (KD_POLICY_SEPARATE_ROUTES, KD_POLICY_MERGED_INTO_KO)
VALID_PUBLIC_PERTURBATION_POLICIES = (
    PUBLIC_PERTURBATION_INVENTORY_OPTIONAL,
    PUBLIC_PERTURBATION_OUT_OF_SCOPE,
)
DEG_REPORTING_MAX_FDR = 0.05


class ADResearchDecisionError(ValueError):
    """Raised when an AD research-decision payload violates the freeze."""


@dataclass(frozen=True)
class ADResearchDecision:
    """Auditable freeze of the 2026-09-18 AD mainline choices."""

    schema_version: str
    observed_gate_branch: str
    observed_admission_rule: str
    deg_reporting_max_fdr: float
    kd_policy: str
    public_perturbation_policy: str
    biology_pass: bool
    decided_at: str
    notes: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != AD_RESEARCH_DECISION_SCHEMA_VERSION:
            raise ADResearchDecisionError(
                f"schema_version must be {AD_RESEARCH_DECISION_SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )
        if self.observed_gate_branch not in VALID_OBSERVED_GATE_BRANCHES:
            raise ADResearchDecisionError(f"observed_gate_branch must be one of {VALID_OBSERVED_GATE_BRANCHES}")
        if self.observed_admission_rule not in VALID_OBSERVED_ADMISSION_RULES:
            raise ADResearchDecisionError(f"observed_admission_rule must be one of {VALID_OBSERVED_ADMISSION_RULES}")
        if self.observed_gate_branch == OBSERVED_GATE_BRANCH_REVISE_SEMANTICS:
            if self.observed_admission_rule != OBSERVED_ADMISSION_SIGNED_DIRECTION:
                raise ADResearchDecisionError(
                    f"revise_semantics requires observed_admission_rule={OBSERVED_ADMISSION_SIGNED_DIRECTION!r}"
                )
        if not isinstance(self.deg_reporting_max_fdr, float) or self.deg_reporting_max_fdr != DEG_REPORTING_MAX_FDR:
            raise ADResearchDecisionError(
                f"deg_reporting_max_fdr is frozen at {DEG_REPORTING_MAX_FDR}; reporting FDR is not relaxed"
            )
        if self.kd_policy not in VALID_KD_POLICIES:
            raise ADResearchDecisionError(f"kd_policy must be one of {VALID_KD_POLICIES}")
        if self.public_perturbation_policy not in VALID_PUBLIC_PERTURBATION_POLICIES:
            raise ADResearchDecisionError(
                f"public_perturbation_policy must be one of {VALID_PUBLIC_PERTURBATION_POLICIES}"
            )
        if self.biology_pass:
            raise ADResearchDecisionError("this freeze does not authorize biology_pass=true")
        if not str(self.decided_at).strip():
            raise ADResearchDecisionError("decided_at must not be empty")

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["observed_significance_required"] = observed_significance_required(self.observed_admission_rule)
        payload["boundary"] = (
            "FDR 0.05 remains the BH significance label; signed-direction admission "
            "does not relabel FDR>0.05 as significant; KD is not inferred from KO; "
            "public Perturb-seq is out of scope; formal biology PASS remains false"
        )
        return payload


FROZEN_AD_RESEARCH_DECISION = ADResearchDecision(
    schema_version=AD_RESEARCH_DECISION_SCHEMA_VERSION,
    observed_gate_branch=OBSERVED_GATE_BRANCH_REVISE_SEMANTICS,
    observed_admission_rule=OBSERVED_ADMISSION_SIGNED_DIRECTION,
    deg_reporting_max_fdr=DEG_REPORTING_MAX_FDR,
    kd_policy=KD_POLICY_MERGED_INTO_KO,
    public_perturbation_policy=PUBLIC_PERTURBATION_OUT_OF_SCOPE,
    biology_pass=False,
    decided_at="2026-09-18",
    notes=(
        "Rivera 2026-09-18: observed gate option 3; KO/KD merged to KO-only; "
        "do not search public Perturb-seq. Reporting FDR stays 0.05."
    ),
)


def legacy_ad_research_decision() -> ADResearchDecision:
    """Pre-freeze behaviour: FDR admission, separate KD rows, optional inventory.

    This is not the AD mainline freeze. Tests that document historical B1/KD
    routing may pass it explicitly; production AD CLIs must not.
    """

    return ADResearchDecision(
        schema_version=AD_RESEARCH_DECISION_SCHEMA_VERSION,
        observed_gate_branch=OBSERVED_GATE_BRANCH_REVISE_SEMANTICS,
        observed_admission_rule=OBSERVED_ADMISSION_SIGNED_DIRECTION,
        deg_reporting_max_fdr=DEG_REPORTING_MAX_FDR,
        kd_policy=KD_POLICY_SEPARATE_ROUTES,
        public_perturbation_policy=PUBLIC_PERTURBATION_INVENTORY_OPTIONAL,
        biology_pass=False,
        decided_at="2026-09-18",
        notes="legacy routing compatibility: KD rows and optional inventory remain selectable",
    )


def parse_ad_research_decision(payload: Mapping[str, Any]) -> ADResearchDecision:
    """Validate an already-parsed decision mapping."""

    required = (
        "schema_version",
        "observed_gate_branch",
        "observed_admission_rule",
        "deg_reporting_max_fdr",
        "kd_policy",
        "public_perturbation_policy",
        "biology_pass",
        "decided_at",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ADResearchDecisionError("research decision is missing keys: " + ", ".join(missing))
    fdr = payload["deg_reporting_max_fdr"]
    if isinstance(fdr, bool) or not isinstance(fdr, (int, float)):
        raise ADResearchDecisionError("deg_reporting_max_fdr must be numeric")
    return ADResearchDecision(
        schema_version=str(payload["schema_version"]).strip(),
        observed_gate_branch=str(payload["observed_gate_branch"]).strip(),
        observed_admission_rule=str(payload["observed_admission_rule"]).strip(),
        deg_reporting_max_fdr=float(fdr),
        kd_policy=str(payload["kd_policy"]).strip(),
        public_perturbation_policy=str(payload["public_perturbation_policy"]).strip(),
        biology_pass=bool(payload["biology_pass"]),
        decided_at=str(payload["decided_at"]).strip(),
        notes=str(payload.get("notes") or ""),
    )


def load_ad_research_decision(path: str | Path) -> ADResearchDecision:
    """Load and validate ``observed_gate_decision.json``."""

    import json

    resolved = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ADResearchDecisionError(f"research decision root must be a JSON object: {resolved}")
    return parse_ad_research_decision(payload)


def write_ad_research_decision(decision: ADResearchDecision, path: str | Path) -> Path:
    """Write the auditable decision JSON (never sets biology_pass)."""

    import json

    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(decision.to_payload(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return resolved


def observed_significance_required(rule: str | None) -> bool:
    """Return whether FDR≤0.05 is an admission blocker.

    ``None`` / empty keep the historical FDR-cutoff so unmarked candidate specs
    cannot silently switch semantics.
    """

    if rule is None or str(rule).strip() == "":
        return True
    normalized = str(rule).strip()
    if normalized == OBSERVED_ADMISSION_FDR_CUTOFF:
        return True
    if normalized == OBSERVED_ADMISSION_SIGNED_DIRECTION:
        return False
    raise ADResearchDecisionError(f"unknown observed_admission_rule: {normalized!r}")


__all__ = [
    "AD_RESEARCH_DECISION_SCHEMA_VERSION",
    "ADResearchDecision",
    "ADResearchDecisionError",
    "DEG_REPORTING_MAX_FDR",
    "FROZEN_AD_RESEARCH_DECISION",
    "KD_POLICY_MERGED_INTO_KO",
    "KD_POLICY_SEPARATE_ROUTES",
    "OBSERVED_ADMISSION_FDR_CUTOFF",
    "OBSERVED_ADMISSION_SIGNED_DIRECTION",
    "OBSERVED_GATE_BRANCH_REVISE_SEMANTICS",
    "PUBLIC_PERTURBATION_INVENTORY_OPTIONAL",
    "PUBLIC_PERTURBATION_OUT_OF_SCOPE",
    "legacy_ad_research_decision",
    "load_ad_research_decision",
    "observed_significance_required",
    "parse_ad_research_decision",
    "write_ad_research_decision",
]
