"""PerturbGen integration contracts and data-prep helpers."""

from .contracts import (
    CandidateEvidence,
    CandidateScreeningResult,
    DAVFDirectionEvidence,
    DirectionGateResult,
    DualPathVerdict,
    PathResult,
    PTMSiteDirectionProposal,
    PerturbGenDataSpec,
    PreparedPerturbationData,
    PreparedPerturbationReport,
)
from .data_prep import prepare_perturbgen_anndata, screen_candidate_for_perturbation
from .direction_gate import (
    build_direction_gated_candidate,
    direction_from_delta,
    evaluate_direction_gate,
)
from .mainline import MainlineDecision, evaluate_davf_perturbgen_candidate
from .orchestrator import (
    DAVFPerturbGenE2EError,
    DAVFPerturbGenOrchestrator,
    DAVFPerturbGenPreparation,
    PerturbGenInvocation,
    build_candidate_stage_plans,
    materialize_candidate_config,
    merge_route_preparations,
    merge_route_reports,
)

__all__ = [
    "CandidateEvidence",
    "CandidateScreeningResult",
    "DAVFDirectionEvidence",
    "DirectionGateResult",
    "DualPathVerdict",
    "PathResult",
    "PTMSiteDirectionProposal",
    "PerturbGenDataSpec",
    "PreparedPerturbationData",
    "PreparedPerturbationReport",
    "prepare_perturbgen_anndata",
    "screen_candidate_for_perturbation",
    "build_direction_gated_candidate",
    "direction_from_delta",
    "evaluate_direction_gate",
    "MainlineDecision",
    "evaluate_davf_perturbgen_candidate",
    "DAVFPerturbGenE2EError",
    "DAVFPerturbGenOrchestrator",
    "DAVFPerturbGenPreparation",
    "PerturbGenInvocation",
    "build_candidate_stage_plans",
    "materialize_candidate_config",
    "merge_route_preparations",
    "merge_route_reports",
]
