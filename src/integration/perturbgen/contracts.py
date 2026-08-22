"""Strict contracts for DAVF × PerturbGen integration."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal

from src.models.gene_vocabulary import normalize_ensembl_id, normalize_gene_symbol

ObservedDirection = Literal["up", "down"]
DavfAction = Literal["ko", "kd", "oe"]
PathStatus = Literal["evaluable", "failed", "inconclusive"]
DualPathVerdictValue = Literal["pass", "fail", "inconclusive"]
PerturbationMode = Literal["mask", "pad", "delete", "overexpress"]
PathKind = Literal["source_intervention", "within_state"]

_VALID_OBSERVED_DIRECTIONS = {"up", "down"}
_VALID_DAVF_ACTIONS = {"ko", "kd", "oe"}
_VALID_PATH_STATUSES = {"evaluable", "failed", "inconclusive"}
_VALID_VERDICTS = {"pass", "fail", "inconclusive"}
_VALID_MODES = {"mask", "pad", "delete", "overexpress"}
_VALID_PATHS = {"source_intervention", "within_state"}


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "gene_symbol", normalize_gene_symbol(self.gene_symbol))
        object.__setattr__(self, "ensembl_id", normalize_ensembl_id(self.ensembl_id))
        object.__setattr__(self, "cell_type", str(self.cell_type).strip())
        object.__setattr__(self, "ptm_context", str(self.ptm_context).strip())
        object.__setattr__(self, "davf_provenance", str(self.davf_provenance).strip())

        if not self.cell_type:
            raise ValueError("cell_type must not be empty")
        if not self.ptm_context:
            raise ValueError("ptm_context must not be empty")
        if self.observed_direction not in _VALID_OBSERVED_DIRECTIONS:
            raise ValueError(
                "observed_direction must be 'up' or 'down', "
                f"got {self.observed_direction!r}"
            )
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
            not isinstance(self.davf_score, (int, float))
            or not math.isfinite(float(self.davf_score))
        ):
            raise ValueError("davf_score must be finite numeric or None")
        if self.davf_score is not None and not self.davf_provenance:
            raise ValueError("davf_provenance must not be empty when davf_score is provided")


@dataclass(frozen=True)
class PerturbGenDataSpec:
    """Dataset contract for PerturbGen-ready AnnData."""

    counts_layer: str = "counts"
    ensembl_id_col: str = "ensembl_id"
    cell_type_col: str = "cell_type"
    state_col: str = "state"
    donor_col: str = "donor"
    n_counts_col: str = "n_counts"
    normal_state: str = "normal"
    disease_state: str = "disease"
    min_donors: int = 3

    def __post_init__(self) -> None:
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
    """Validated donor/state summary for one target cell type."""

    cell_type: str
    normal_state: str
    disease_state: str
    evaluable_donors: tuple[str, ...]
    normal_only_donors: tuple[str, ...] = field(default_factory=tuple)
    disease_only_donors: tuple[str, ...] = field(default_factory=tuple)
    n_cells: int = 0
    n_genes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_type", str(self.cell_type).strip())
        object.__setattr__(self, "normal_state", str(self.normal_state).strip())
        object.__setattr__(self, "disease_state", str(self.disease_state).strip())
        object.__setattr__(self, "evaluable_donors", tuple(sorted({str(x).strip() for x in self.evaluable_donors if str(x).strip()})))
        object.__setattr__(self, "normal_only_donors", tuple(sorted({str(x).strip() for x in self.normal_only_donors if str(x).strip()})))
        object.__setattr__(self, "disease_only_donors", tuple(sorted({str(x).strip() for x in self.disease_only_donors if str(x).strip()})))

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
        object.__setattr__(self, "evaluable_donors", tuple(sorted({str(x).strip() for x in self.evaluable_donors if str(x).strip()})))
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
            not math.isfinite(self.empirical_pvalue)
            or not 0.0 <= self.empirical_pvalue <= 1.0
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
        if self.q_value is not None and (
            not math.isfinite(self.q_value) or not 0.0 <= self.q_value <= 1.0
        ):
            raise ValueError("q_value must be within [0, 1]")
        if self.verdict != "pass" and not self.reasons:
            raise ValueError("fail/inconclusive verdict must provide reasons")
