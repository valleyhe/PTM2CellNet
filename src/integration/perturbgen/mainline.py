"""Executable decision boundary for the DAVF × PerturbGen mainline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .contracts import (
    CandidateEvidence,
    DAVFDirectionEvidence,
    DirectionGateResult,
    DualPathVerdictValue,
    ObservedDirection,
    PTMSiteDirectionProposal,
)
from .direction_gate import build_direction_gated_candidate
from .dual_path import DualPathDecision, evaluate_dual_path_candidate


@dataclass(frozen=True)
class MainlineDecision:
    """Combined direction-gate and dual-path result."""

    verdict: DualPathVerdictValue
    direction_gate: DirectionGateResult
    candidate: CandidateEvidence | None
    dual_path: DualPathDecision | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_davf_perturbgen_candidate(
    proposal: PTMSiteDirectionProposal | None,
    davf_evidence: DAVFDirectionEvidence | None,
    *,
    cell_type: str,
    ptm_context: str,
    observed_log2fc: float,
    observed_fdr: float,
    observed_direction: ObservedDirection | None,
    path_results: Sequence[Mapping[str, Any] | object] = (),
    q_value: float | None = None,
    unperturbed_quality_status: str = "inconclusive",
    max_observed_fdr: float = 0.05,
    formal_null_min: int = 99,
    smoke_null_count: int = 20,
    expected_seed_count: int = 3,
) -> MainlineDecision:
    """Run the agreed mainline decision sequence.

    PerturbGen evaluation is never called for a direction-gate failure or an
    inconclusive DAVF/normal-disease evidence set.  This is the central
    semantic boundary that keeps downstream utility from rescuing an invalid
    direction claim.
    """

    gate, candidate = build_direction_gated_candidate(
        proposal,
        davf_evidence,
        cell_type=cell_type,
        ptm_context=ptm_context,
        observed_log2fc=observed_log2fc,
        observed_fdr=observed_fdr,
        observed_direction=observed_direction,
        max_observed_fdr=max_observed_fdr,
    )
    if gate.status != "pass":
        return MainlineDecision(
            verdict=gate.status,
            direction_gate=gate,
            candidate=None,
            dual_path=None,
            reasons=gate.reasons,
        )

    if candidate is None or observed_direction is None:
        raise RuntimeError("passing direction gate did not produce a candidate")
    if not path_results:
        reasons = ("missing_perturbgen_path_results",)
        return MainlineDecision(
            verdict="inconclusive",
            direction_gate=gate,
            candidate=candidate,
            dual_path=None,
            reasons=reasons,
        )

    dual_path = evaluate_dual_path_candidate(
        path_results,
        observed_direction=observed_direction,
        q_value=q_value,
        candidate_gene=candidate.gene_symbol,
        formal_null_min=formal_null_min,
        smoke_null_count=smoke_null_count,
        expected_seed_count=expected_seed_count,
        unperturbed_quality_status=unperturbed_quality_status,
    )
    return MainlineDecision(
        verdict=dual_path.verdict,
        direction_gate=gate,
        candidate=candidate,
        dual_path=dual_path,
        reasons=dual_path.reasons,
    )
