"""Direction gate for the DAVF → PerturbGen mainline."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import (
    CandidateEvidence,
    DAVFDirectionEvidence,
    Direction,
    DirectionGateResult,
    DirectionGateStatus,
    DavfAction,
    ObservedDirection,
    PTMSiteDirectionProposal,
    SemanticContext,
    normalize_semantic_context,
)


def direction_from_delta(delta: float, *, epsilon: float = 0.0) -> Direction | None:
    """Convert a signed gene-expression delta to a direction.

    A delta inside ``[-epsilon, epsilon]`` is not assigned a direction.  This
    keeps a near-zero DAVF result from being promoted to a biological claim.
    """

    if not math.isfinite(delta):
        raise ValueError("delta must be finite")
    if epsilon < 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and >= 0")
    if abs(delta) <= epsilon:
        return None
    return "up" if delta > 0 else "down"


def evaluate_direction_gate(
    proposal: PTMSiteDirectionProposal | None,
    davf_evidence: DAVFDirectionEvidence | None,
    *,
    observed_direction: ObservedDirection | None,
    observed_fdr: float | None,
    max_observed_fdr: float = 0.05,
) -> DirectionGateResult:
    """Require agreement between PTM proposal, DAVF, and independent data.

    The gate is intentionally strict:

    - missing evidence is ``inconclusive``;
    - a configured fallback/random DAVF result is not scientific evidence;
    - disagreement among complete directions is ``fail``;
    - only complete three-way agreement returns ``pass``.
    """

    if not 0.0 <= max_observed_fdr <= 1.0:
        raise ValueError("max_observed_fdr must be within [0, 1]")
    if observed_fdr is not None and (not math.isfinite(observed_fdr) or not 0.0 <= observed_fdr <= 1.0):
        raise ValueError("observed_fdr must be within [0, 1]")

    gene_symbol = proposal.gene_symbol if proposal is not None else None
    ensembl_id = proposal.ensembl_id if proposal is not None else None
    proposal_direction = proposal.proposed_direction if proposal is not None else None
    davf_direction = davf_evidence.predicted_direction if davf_evidence is not None else None
    reasons: list[str] = []

    if proposal is None:
        reasons.append("missing_ptm_direction_proposal")
    if davf_evidence is None:
        reasons.append("missing_davf_direction_evidence")
    elif davf_evidence.predicted_direction is None:
        reasons.append("missing_davf_predicted_direction")
    if observed_direction is None:
        reasons.append("missing_observed_direction")
    if observed_fdr is None:
        reasons.append("missing_observed_fdr")
    elif observed_fdr > max_observed_fdr:
        reasons.append("observed_expression_not_significant")

    if proposal is not None and davf_evidence is not None:
        if proposal.gene_symbol != davf_evidence.gene_symbol:
            reasons.append("direction_evidence_gene_symbol_mismatch")
        if proposal.ensembl_id != davf_evidence.ensembl_id:
            reasons.append("direction_evidence_ensembl_mismatch")
        if davf_evidence.model_source in {
            "zero_fallback",
            "synthetic_fallback",
            "random",
        }:
            reasons.append("davf_model_asset_unavailable")
        if not davf_evidence.checkpoint_provenance:
            reasons.append("missing_davf_checkpoint_provenance")
        if not davf_evidence.embedding_provenance:
            reasons.append("missing_davf_embedding_provenance")

    directions: Sequence[Direction | None] = (
        proposal_direction,
        davf_direction,
        observed_direction,
    )
    if any(direction is None for direction in directions):
        status: DirectionGateStatus = "inconclusive"
    elif any(direction != directions[0] for direction in directions[1:]):
        reasons.append("direction_evidence_disagreement")
        status = "fail"
    elif reasons:
        status = "inconclusive"
    else:
        status = "pass"

    corrective_action: DavfAction | None = None
    if status == "pass":
        # The downstream PerturbGen action follows the observed disease
        # direction, not the mapper's PTM action code:
        # disease-up → source removal (KO/mask), disease-down → OE.
        corrective_action = "ko" if observed_direction == "up" else "oe"

    return DirectionGateResult(
        status=status,
        gene_symbol=gene_symbol,
        ensembl_id=ensembl_id,
        proposed_direction=proposal_direction,
        davf_direction=davf_direction,
        observed_direction=observed_direction,
        corrective_action=corrective_action,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def build_direction_gated_candidate(
    proposal: PTMSiteDirectionProposal | None,
    davf_evidence: DAVFDirectionEvidence | None,
    *,
    cell_type: str,
    ptm_context: str,
    observed_log2fc: float,
    observed_fdr: float,
    observed_direction: ObservedDirection | None,
    max_observed_fdr: float = 0.05,
    semantic_context: SemanticContext | Mapping[str, Any] | None = None,
) -> tuple[DirectionGateResult, CandidateEvidence | None]:
    """Build ``CandidateEvidence`` only after the direction gate passes."""

    gate = evaluate_direction_gate(
        proposal,
        davf_evidence,
        observed_direction=observed_direction,
        observed_fdr=observed_fdr,
        max_observed_fdr=max_observed_fdr,
    )
    if gate.status != "pass":
        return gate, None
    if proposal is None or davf_evidence is None or gate.corrective_action is None:
        raise RuntimeError("passing direction gate has incomplete evidence")
    if observed_direction is None:
        raise RuntimeError("passing direction gate requires an observed direction")

    candidate = CandidateEvidence(
        gene_symbol=proposal.gene_symbol,
        ensembl_id=proposal.ensembl_id,
        cell_type=cell_type,
        ptm_context=ptm_context,
        observed_log2fc=observed_log2fc,
        observed_fdr=observed_fdr,
        observed_direction=observed_direction,
        davf_action=gate.corrective_action,
        davf_score=davf_evidence.confidence,
        davf_provenance=davf_evidence.checkpoint_provenance,
        proposed_direction=gate.proposed_direction,
        davf_predicted_direction=gate.davf_direction,
        davf_predicted_delta=davf_evidence.predicted_delta,
        direction_gate_status=gate.status,
        semantic_context=normalize_semantic_context(semantic_context),
    )
    return gate, candidate
