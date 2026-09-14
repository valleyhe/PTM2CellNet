"""Strict contracts for DAVF × PerturbGen integration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import Any, Literal

from src.models.gene_vocabulary import normalize_ensembl_id, normalize_gene_symbol

Direction = Literal["up", "down"]
ObservedDirection = Direction
DirectionGateStatus = Literal["pass", "fail", "inconclusive"]
DavfAction = Literal["ko", "kd", "oe"]
PathStatus = Literal["evaluable", "failed", "inconclusive"]
DualPathVerdictValue = Literal["pass", "fail", "inconclusive"]
PerturbationMode = Literal["mask", "pad", "delete", "overexpress"]
PathKind = Literal["source_intervention", "within_state"]

_VALID_DIRECTIONS = {"up", "down"}
_VALID_OBSERVED_DIRECTIONS = _VALID_DIRECTIONS
_VALID_DIRECTION_GATE_STATUSES = {"pass", "fail", "inconclusive"}
_VALID_DAVF_ACTIONS = {"ko", "kd", "oe"}
_VALID_PATH_STATUSES = {"evaluable", "failed", "inconclusive"}
_VALID_VERDICTS = {"pass", "fail", "inconclusive"}
_VALID_MODES = {"mask", "pad", "delete", "overexpress"}
_VALID_PATHS = {"source_intervention", "within_state"}

_SEMANTIC_CONTEXT_FIELDS = (
    "context",
    "intervention",
    "comparison_baseline",
    "reference_axis",
    "research_objective",
    "evidence_source",
    "cohort",
)
_VALID_RESEARCH_OBJECTIVES = {"association", "replication", "reversal"}


@dataclass(frozen=True)
class SemanticContext:
    """Shared semantic provenance required by formal PerturbGen invocations."""

    context: str
    intervention: str
    comparison_baseline: str
    reference_axis: str
    research_objective: Literal["association", "replication", "reversal"]
    evidence_source: str
    cohort: str

    def __post_init__(self) -> None:
        for field_name in _SEMANTIC_CONTEXT_FIELDS:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"semantic_context.{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())
        object.__setattr__(self, "intervention", self.intervention.upper())
        objective = self.research_objective.lower()
        if objective not in _VALID_RESEARCH_OBJECTIVES:
            raise ValueError("semantic_context.research_objective must be one of association, replication, reversal")
        object.__setattr__(self, "research_objective", objective)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SemanticContext":
        missing = [field_name for field_name in _SEMANTIC_CONTEXT_FIELDS if field_name not in value]
        if missing:
            raise ValueError("semantic_context is missing required fields: " + ", ".join(missing))
        return cls(**{field_name: value[field_name] for field_name in _SEMANTIC_CONTEXT_FIELDS})


def normalize_semantic_context(
    value: SemanticContext | Mapping[str, Any] | None,
) -> SemanticContext | None:
    """Normalize a context mapping without supplying missing semantic values."""

    if value is None or isinstance(value, SemanticContext):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("semantic_context must be a mapping")
    return SemanticContext.from_mapping(value)


@dataclass(frozen=True)
class PTMSiteDirectionProposal:
    """Direction proposed by the upstream PTM-site analysis.

    The existing PTM-site classifier predicts site presence.  It does not
    produce an expression-direction label, so the mainline accepts that
    direction as an explicit, provenance-bearing artifact instead of
    silently deriving it from a KO/KD/OE action code.
    """

    gene_symbol: str
    ensembl_id: str
    position: int
    ptm_type: str
    proposed_direction: Direction
    site_probability: float
    provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "gene_symbol", normalize_gene_symbol(self.gene_symbol))
        object.__setattr__(self, "ensembl_id", normalize_ensembl_id(self.ensembl_id))
        object.__setattr__(self, "ptm_type", str(self.ptm_type).strip())
        object.__setattr__(self, "provenance", str(self.provenance).strip())
        if self.position < 1:
            raise ValueError("position must be >= 1")
        if not self.ptm_type:
            raise ValueError("ptm_type must not be empty")
        if self.proposed_direction not in _VALID_DIRECTIONS:
            raise ValueError("proposed_direction must be 'up' or 'down'")
        if not math.isfinite(self.site_probability) or not 0.0 <= self.site_probability <= 1.0:
            raise ValueError("site_probability must be within [0, 1]")
        if not self.provenance:
            raise ValueError("provenance must not be empty")


@dataclass(frozen=True)
class DAVFDirectionEvidence:
    """Gene-level expression-direction evidence emitted by DAVF.

    ``davf_action`` is deliberately absent.  Action encoding remains the
    responsibility of :class:`PTMDirectionMapper`; this object records only
    the decoded gene-expression delta used by the direction gate.

    ``confidence`` is a bounded relative effect-size proxy, not a calibrated
    probability.
    """

    gene_symbol: str
    ensembl_id: str
    predicted_direction: Direction | None
    predicted_delta: float
    model_source: str
    checkpoint_provenance: str
    embedding_provenance: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "gene_symbol", normalize_gene_symbol(self.gene_symbol))
        object.__setattr__(self, "ensembl_id", normalize_ensembl_id(self.ensembl_id))
        object.__setattr__(self, "model_source", "" if self.model_source is None else str(self.model_source).strip())
        object.__setattr__(
            self,
            "checkpoint_provenance",
            "" if self.checkpoint_provenance is None else str(self.checkpoint_provenance).strip(),
        )
        object.__setattr__(
            self,
            "embedding_provenance",
            "" if self.embedding_provenance is None else str(self.embedding_provenance).strip(),
        )
        if not math.isfinite(self.predicted_delta):
            raise ValueError("predicted_delta must be finite")
        if self.predicted_direction not in _VALID_DIRECTIONS and self.predicted_direction is not None:
            raise ValueError("predicted_direction must be 'up', 'down' or None")
        if self.predicted_direction == "up" and self.predicted_delta <= 0:
            raise ValueError("predicted_direction='up' requires predicted_delta > 0")
        if self.predicted_direction == "down" and self.predicted_delta >= 0:
            raise ValueError("predicted_direction='down' requires predicted_delta < 0")
        if not self.model_source:
            raise ValueError("model_source must not be empty")
        if not self.checkpoint_provenance:
            raise ValueError("checkpoint_provenance must not be empty")
        if not self.embedding_provenance:
            raise ValueError("embedding_provenance must not be empty")
        if self.confidence is not None:
            try:
                confidence = float(self.confidence)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("confidence must be within [0, 1]") from exc
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be within [0, 1]")
            object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class DirectionGateResult:
    """Decision that controls whether a candidate may enter PerturbGen."""

    status: DirectionGateStatus
    gene_symbol: str | None
    ensembl_id: str | None
    proposed_direction: Direction | None
    davf_direction: Direction | None
    observed_direction: Direction | None
    corrective_action: DavfAction | None
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.status not in _VALID_DIRECTION_GATE_STATUSES:
            raise ValueError(f"invalid direction gate status: {self.status!r}")
        if self.gene_symbol is not None:
            object.__setattr__(self, "gene_symbol", normalize_gene_symbol(self.gene_symbol))
        if self.ensembl_id is not None:
            object.__setattr__(self, "ensembl_id", normalize_ensembl_id(self.ensembl_id))
        for field_name in ("proposed_direction", "davf_direction", "observed_direction"):
            value = getattr(self, field_name)
            if value is not None and value not in _VALID_DIRECTIONS:
                raise ValueError(f"{field_name} must be 'up', 'down' or None")
        if self.corrective_action is not None and self.corrective_action not in _VALID_DAVF_ACTIONS:
            raise ValueError("corrective_action must be 'ko', 'kd', 'oe' or None")
        object.__setattr__(self, "reasons", tuple(str(reason) for reason in self.reasons))
        if self.status == "pass":
            if self.reasons:
                raise ValueError("passing direction gate must not carry reasons")
            if None in (
                self.gene_symbol,
                self.ensembl_id,
                self.proposed_direction,
                self.davf_direction,
                self.observed_direction,
            ):
                raise ValueError("passing direction gate requires complete evidence")
            if self.corrective_action is None:
                raise ValueError("passing direction gate requires corrective_action")
        elif not self.reasons:
            raise ValueError("failed/inconclusive direction gate must provide reasons")


@dataclass(frozen=True)
class CandidateEvidence:
    """Observed expression evidence plus DAVF-suggested action.

    ``observed_direction`` is the measured disease-vs-normal expression change.
    ``davf_action`` is a separate model-side perturbation suggestion and must
    not be used as a surrogate for observed expression direction.
    """

    gene_symbol: str
    ensembl_id: str
    cell_type: str
    ptm_context: str
    observed_log2fc: float
    observed_fdr: float
    observed_direction: ObservedDirection
    davf_action: DavfAction | None
    davf_score: float | None
    davf_provenance: str
    proposed_direction: Direction | None = None
    davf_predicted_direction: Direction | None = None
    davf_predicted_delta: float | None = None
    direction_gate_status: DirectionGateStatus | None = None
    direction_gate_reasons: tuple[str, ...] = field(default_factory=tuple)
    semantic_context: SemanticContext | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "gene_symbol", normalize_gene_symbol(self.gene_symbol))
        object.__setattr__(self, "ensembl_id", normalize_ensembl_id(self.ensembl_id))
        object.__setattr__(self, "cell_type", str(self.cell_type).strip())
        object.__setattr__(self, "ptm_context", str(self.ptm_context).strip())
        object.__setattr__(
            self,
            "davf_provenance",
            "" if self.davf_provenance is None else str(self.davf_provenance).strip(),
        )
        object.__setattr__(self, "direction_gate_reasons", tuple(self.direction_gate_reasons))
        object.__setattr__(self, "semantic_context", normalize_semantic_context(self.semantic_context))

        if not self.cell_type:
            raise ValueError("cell_type must not be empty")
        if not self.ptm_context:
            raise ValueError("ptm_context must not be empty")
        if self.observed_direction not in _VALID_OBSERVED_DIRECTIONS:
            raise ValueError(f"observed_direction must be 'up' or 'down', got {self.observed_direction!r}")
        if not math.isfinite(self.observed_log2fc):
            raise ValueError("observed_log2fc must be finite")
        if self.observed_direction == "up" and self.observed_log2fc <= 0:
            raise ValueError("observed_direction='up' requires observed_log2fc > 0")
        if self.observed_direction == "down" and self.observed_log2fc >= 0:
            raise ValueError("observed_direction='down' requires observed_log2fc < 0")
        if not math.isfinite(self.observed_fdr) or not 0.0 <= self.observed_fdr <= 1.0:
            raise ValueError("observed_fdr must be within [0, 1]")
        if self.davf_action is not None and self.davf_action not in _VALID_DAVF_ACTIONS:
            raise ValueError("davf_action must be one of 'ko', 'kd', 'oe' or None")
        if self.davf_score is not None and (
            not isinstance(self.davf_score, (int, float)) or not math.isfinite(float(self.davf_score))
        ):
            raise ValueError("davf_score must be finite numeric or None")
        if self.davf_score is not None and not self.davf_provenance:
            raise ValueError("davf_provenance must not be empty when davf_score is provided")
        if self.davf_predicted_delta is not None and not math.isfinite(self.davf_predicted_delta):
            raise ValueError("davf_predicted_delta must be finite or None")
        for field_name in ("proposed_direction", "davf_predicted_direction"):
            value = getattr(self, field_name)
            if value is not None and value not in _VALID_DIRECTIONS:
                raise ValueError(f"{field_name} must be 'up', 'down' or None")
        if self.direction_gate_status is not None:
            if self.direction_gate_status not in _VALID_DIRECTION_GATE_STATUSES:
                raise ValueError("direction_gate_status must be 'pass', 'fail', 'inconclusive' or None")
            if self.direction_gate_status != "pass":
                raise ValueError("CandidateEvidence may only be created after a passing direction gate")
            if self.proposed_direction != self.observed_direction:
                raise ValueError("passing candidate proposal must match observed_direction")
            if self.davf_predicted_direction != self.observed_direction:
                raise ValueError("passing DAVF direction must match observed_direction")
            if self.davf_action is None:
                raise ValueError("passing candidate must define davf_action")
            if self.direction_gate_reasons:
                raise ValueError("passing candidate must not carry direction_gate_reasons")


VALID_COHORT_PAIRINGS = ("within_donor", "between_donor")


@dataclass(frozen=True)
class PerturbGenDataSpec:
    """Dataset contract for PerturbGen-ready AnnData.

    ``pairing`` freezes how donors relate to the two states:

    - ``within_donor`` (default, perturbation cohorts): the same donor
      contributes cells to both states; Gate-0 requires >= ``min_donors``
      donors shared across the two states.
    - ``between_donor`` (case-control cohorts): each donor belongs to
      exactly one state; Gate-0 requires >= ``min_donors`` donors in each
      state and rejects any donor observed in both states as a labeling
      error.
    """

    counts_layer: str = "counts"
    ensembl_id_col: str = "ensembl_id"
    cell_type_col: str = "cell_type"
    state_col: str = "state"
    donor_col: str = "donor"
    n_counts_col: str = "n_counts"
    normal_state: str = "normal"
    disease_state: str = "disease"
    min_donors: int = 3
    pairing: str = "within_donor"

    def __post_init__(self) -> None:
        object.__setattr__(self, "pairing", str(self.pairing).strip())
        if self.pairing not in VALID_COHORT_PAIRINGS:
            raise ValueError(f"pairing must be one of {VALID_COHORT_PAIRINGS}")
        text_fields = (
            "counts_layer",
            "ensembl_id_col",
            "cell_type_col",
            "state_col",
            "donor_col",
            "n_counts_col",
            "normal_state",
            "disease_state",
        )
        for field_name in text_fields:
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} must not be empty")
            object.__setattr__(self, field_name, value)
        if self.min_donors < 3:
            raise ValueError("min_donors must be >= 3")
        if self.normal_state == self.disease_state:
            raise ValueError("normal_state and disease_state must differ")


@dataclass(frozen=True)
class PreparedPerturbationReport:
    """Validated donor/state summary for one target cell type.

    ``normal_donors``/``disease_donors`` carry the two disjoint donor groups
    in ``between_donor`` mode and stay empty in ``within_donor`` mode, where
    ``evaluable_donors`` holds the shared donors and ``normal_only``/
    ``disease_only`` record the single-state exclusions.
    """

    cell_type: str
    normal_state: str
    disease_state: str
    evaluable_donors: tuple[str, ...]
    normal_only_donors: tuple[str, ...] = field(default_factory=tuple)
    disease_only_donors: tuple[str, ...] = field(default_factory=tuple)
    n_cells: int = 0
    n_genes: int = 0
    pairing: str = "within_donor"
    normal_donors: tuple[str, ...] = field(default_factory=tuple)
    disease_donors: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_type", str(self.cell_type).strip())
        object.__setattr__(self, "normal_state", str(self.normal_state).strip())
        object.__setattr__(self, "disease_state", str(self.disease_state).strip())
        object.__setattr__(
            self, "evaluable_donors", tuple(sorted({str(x).strip() for x in self.evaluable_donors if str(x).strip()}))
        )
        object.__setattr__(
            self,
            "normal_only_donors",
            tuple(sorted({str(x).strip() for x in self.normal_only_donors if str(x).strip()})),
        )
        object.__setattr__(
            self,
            "disease_only_donors",
            tuple(sorted({str(x).strip() for x in self.disease_only_donors if str(x).strip()})),
        )
        object.__setattr__(self, "pairing", str(self.pairing).strip())
        object.__setattr__(
            self, "normal_donors", tuple(sorted({str(x).strip() for x in self.normal_donors if str(x).strip()}))
        )
        object.__setattr__(
            self, "disease_donors", tuple(sorted({str(x).strip() for x in self.disease_donors if str(x).strip()}))
        )

        if not self.cell_type:
            raise ValueError("cell_type must not be empty")
        if not self.normal_state:
            raise ValueError("normal_state must not be empty")
        if not self.disease_state:
            raise ValueError("disease_state must not be empty")
        if self.normal_state == self.disease_state:
            raise ValueError("normal_state and disease_state must differ")
        if len(self.evaluable_donors) < 3:
            raise ValueError("prepared report requires at least 3 evaluable donors")
        overlap = set(self.normal_only_donors) & set(self.disease_only_donors)
        if overlap:
            raise ValueError(f"donors cannot be both normal_only and disease_only: {sorted(overlap)}")
        if self.pairing not in VALID_COHORT_PAIRINGS:
            raise ValueError(f"pairing must be one of {VALID_COHORT_PAIRINGS}")
        group_overlap = set(self.normal_donors) & set(self.disease_donors)
        if group_overlap:
            raise ValueError(f"between_donor groups must be disjoint: {sorted(group_overlap)}")
        if self.pairing == "between_donor":
            if not self.normal_donors or not self.disease_donors:
                raise ValueError("between_donor report requires non-empty normal_donors and disease_donors")
            if self.normal_only_donors or self.disease_only_donors:
                raise ValueError("between_donor report must not carry normal_only/disease_only donors")
        elif self.normal_donors or self.disease_donors:
            raise ValueError("within_donor report must not carry normal_donors/disease_donors")
        if self.n_cells <= 0:
            raise ValueError("n_cells must be > 0")
        if self.n_genes <= 0:
            raise ValueError("n_genes must be > 0")

    @property
    def evaluable_donor_count(self) -> int:
        return len(self.evaluable_donors)


@dataclass(frozen=True)
class PreparedPerturbationData:
    """Validated AnnData bundle for later PerturbGen stages."""

    adata: Any
    spec: PerturbGenDataSpec
    report: PreparedPerturbationReport

    def __post_init__(self) -> None:
        if self.adata is None:
            raise ValueError("adata must not be None")


@dataclass(frozen=True)
class CandidateScreeningResult:
    """Candidate-specific preflight result with explicit tri-state status."""

    candidate: CandidateEvidence
    status: PathStatus
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    recommended_mode: PerturbationMode | None = None
    evaluable_donors: tuple[str, ...] = field(default_factory=tuple)
    matched_gene_symbol: str | None = None
    matched_ensembl_id: str | None = None
    normal_detected_cells: int = 0
    disease_detected_cells: int = 0
    normal_mean_counts: float | None = None
    disease_mean_counts: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        object.__setattr__(
            self, "evaluable_donors", tuple(sorted({str(x).strip() for x in self.evaluable_donors if str(x).strip()}))
        )
        if self.status not in _VALID_PATH_STATUSES:
            raise ValueError(f"invalid status: {self.status!r}")
        if self.recommended_mode is not None and self.recommended_mode not in _VALID_MODES:
            raise ValueError(f"invalid recommended_mode: {self.recommended_mode!r}")
        if self.status == "evaluable":
            if self.reason_codes:
                raise ValueError("evaluable candidate must not carry reason_codes")
            if self.recommended_mode is None:
                raise ValueError("evaluable candidate must define recommended_mode")
            if not self.matched_gene_symbol or not self.matched_ensembl_id:
                raise ValueError("evaluable candidate must define matched gene identifiers")
            if len(self.evaluable_donors) < 3:
                raise ValueError("evaluable candidate requires at least 3 evaluable donors")
        else:
            if not self.reason_codes:
                raise ValueError("non-evaluable candidate must carry reason_codes")


@dataclass(frozen=True)
class PathResult:
    """Formal result contract for one PerturbGen path."""

    status: PathStatus
    path: PathKind
    mode: PerturbationMode
    rescue_excl_target: float | None
    evaluable_donors: int
    donor_consistency: float | None
    seed: int
    output_h5ad: str | None = None
    reason_code: str | None = None
    matched_null_count: int | None = None
    empirical_pvalue: float | None = None

    def __post_init__(self) -> None:
        if self.status not in _VALID_PATH_STATUSES:
            raise ValueError(f"invalid status: {self.status!r}")
        if self.path not in _VALID_PATHS:
            raise ValueError(f"invalid path: {self.path!r}")
        if self.mode not in _VALID_MODES:
            raise ValueError(f"invalid mode: {self.mode!r}")
        if self.evaluable_donors < 0:
            raise ValueError("evaluable_donors must be >= 0")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.matched_null_count is not None and self.matched_null_count < 0:
            raise ValueError("matched_null_count must be >= 0")
        if self.empirical_pvalue is not None and (
            not math.isfinite(self.empirical_pvalue) or not 0.0 <= self.empirical_pvalue <= 1.0
        ):
            raise ValueError("empirical_pvalue must be within [0, 1]")
        if self.donor_consistency is not None and not 0.0 <= self.donor_consistency <= 1.0:
            raise ValueError("donor_consistency must be within [0, 1]")
        if self.rescue_excl_target is not None and not math.isfinite(self.rescue_excl_target):
            raise ValueError("rescue_excl_target must be finite")
        if self.donor_consistency is not None and not math.isfinite(self.donor_consistency):
            raise ValueError("donor_consistency must be finite")
        if self.status == "evaluable":
            if self.rescue_excl_target is None:
                raise ValueError("evaluable path result requires rescue_excl_target")
            if self.donor_consistency is None:
                raise ValueError("evaluable path result requires donor_consistency")
            if self.evaluable_donors < 3:
                raise ValueError("evaluable path result requires at least 3 donors")
            if self.reason_code is not None:
                raise ValueError("evaluable path result must not carry reason_code")
        elif not self.reason_code:
            raise ValueError("non-evaluable path result must carry reason_code")


@dataclass(frozen=True)
class DualPathVerdict:
    """Top-level PASS/FAIL/INCONCLUSIVE verdict contract."""

    verdict: DualPathVerdictValue
    reasons: tuple[str, ...] = field(default_factory=tuple)
    q_value: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if self.verdict not in _VALID_VERDICTS:
            raise ValueError(f"invalid verdict: {self.verdict!r}")
        if self.q_value is not None and (not math.isfinite(self.q_value) or not 0.0 <= self.q_value <= 1.0):
            raise ValueError("q_value must be within [0, 1]")
        if self.verdict != "pass" and not self.reasons:
            raise ValueError("fail/inconclusive verdict must provide reasons")
