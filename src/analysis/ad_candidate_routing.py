"""AD candidate routing for the observed-gate / no-local-KO-KD plan (方案 §6).

This module freezes the five AD candidates, assigns each gene × KO (KD is
merged out of scope under the 2026-09-18 decision) to a documented route, and
writes exploratory sidecars. It does not build a PerturbGen invocation, does
not relax the reporting FDR of 0.05, does not relabel FDR>0.05 as significant,
does not infer KD from KO, does not search public Perturb-seq, and does not
treat embedding-vocabulary hits as DAVF training-axis coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.analysis.ad_research_decision import (
    ADResearchDecision,
    FROZEN_AD_RESEARCH_DECISION,
    KD_POLICY_MERGED_INTO_KO,
    OBSERVED_ADMISSION_FDR_CUTOFF,
    OBSERVED_ADMISSION_SIGNED_DIRECTION,
    PUBLIC_PERTURBATION_OUT_OF_SCOPE,
)
from src.models.gene_vocabulary import normalize_ensembl_id, normalize_gene_symbol

AD_CANDIDATE_ROUTING_SCHEMA_VERSION = "ptm2cellnet.ad-candidate-routing/v1"
LINEAGE_BOUNDARY_SUPPLEMENTARY_ONLY = "supplementary_only"

FROZEN_AD_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("APOE", "ENSG00000130203"),
    ("APP", "ENSG00000142192"),
    ("PSEN1", "ENSG00000080815"),
    ("BACE1", "ENSG00000186318"),
    ("MAPT", "ENSG00000186868"),
)
FROZEN_AD_ENSEMBL: dict[str, str] = {symbol: ensembl_id for symbol, ensembl_id in FROZEN_AD_CANDIDATES}
FROZEN_AD_SYMBOLS: dict[str, str] = {ensembl_id: symbol for symbol, ensembl_id in FROZEN_AD_CANDIDATES}

VALID_INTERVENTIONS = ("KO", "KD")
VALID_STATUSES = (
    "blocked",
    "failed",
    "inconclusive",
    "unavailable",
    "direction_only",
    "exploratory_unvalidated",
    "exploratory_supported",
    "engineering_verification",
)
VALID_ROUTES = ("B1", "B2", "B3", "B4", "B5", "B6", "APOE_KO_ENGINEERING")
VALID_RESEARCH_OBJECTIVES = ("association", "replication", "reversal")
VALID_KD_MECHANISMS = frozenset({"crispri", "partial_knockdown", "shrna", "dose", "kd"})
VALID_KO_MECHANISMS = frozenset({"crispr_ko", "crispr_cas9", "ko", "knockout"})
HUMAN_SPECIES = frozenset({"human", "homo sapiens", "9606", "hsapiens"})

AXIS_REQUIRED_COLUMNS = (
    "gene_symbol",
    "ensembl_id",
    "embedding_vocab",
    "ko_in_axis",
    "kd_in_axis",
)
INVENTORY_REQUIRED_COLUMNS = (
    "dataset_id",
    "species",
    "cell_type",
    "state",
    "intervention",
    "mechanism",
    "target_ensembl_id",
    "target_gene_symbol",
    "has_control",
    "has_perturbed",
)
GRN_REQUIRED_COLUMNS = ("target_ensembl_id", "has_tf_support", "has_direct_path")
DEG_MIN_COLUMNS = ("cell_type", "ensembl_id", "fdr", "observed_direction")

_PROTECTED_EXTERNAL_BASENAMES = frozenset(
    {
        "gears_predictions.h5ad",
        "gears_predictions.manifest.json",
        "gears_evidence.json",
        "geneformer_isp_EX.h5ad",
        "geneformer_isp_EX.manifest.json",
        "geneformer_evidence.json",
        "apoe_anchor_backtest.json",
    }
)


class ADCandidateRoutingError(ValueError):
    """Raised when candidate routing inputs violate the §6 contract."""


@dataclass(frozen=True)
class AxisCoverage:
    gene_symbol: str
    ensembl_id: str
    embedding_vocab: bool
    ko_in_axis: bool
    kd_in_axis: bool
    scperturb_perturbed_in: tuple[str, ...]


@dataclass(frozen=True)
class ObservedCandidateStatus:
    ensembl_id: str
    min_fdr: float | None
    observed_significant: bool
    observed_gate_pass: bool
    cell_types_passing: tuple[str, ...]


@dataclass(frozen=True)
class ObservedGateSummary:
    max_fdr: float
    n_rows: int
    n_significant_rows: int
    min_fdr: float | None
    per_candidate: dict[str, ObservedCandidateStatus]
    admission_rule: str = OBSERVED_ADMISSION_SIGNED_DIRECTION


@dataclass(frozen=True)
class InventoryMatch:
    dataset_id: str
    species: str
    cell_type: str
    state: str
    intervention: str
    mechanism: str
    target_ensembl_id: str
    has_dose_or_time: bool


@dataclass(frozen=True)
class CandidateRouteRow:
    gene_symbol: str
    ensembl_id: str
    intervention: str
    selected_route: str | None
    status: str
    lineage_boundary: str
    formal_invocation_allowed: bool
    predicted_direction_allowed: bool
    evidence_kind: str | None
    validation_status: str
    embedding_vocab: bool
    in_route_axis: bool
    has_local_target_pair: bool
    observed_fdr: float | None
    observed_significant: bool
    observed_gate_pass: bool
    public_target_specific: bool
    grn_status: str
    exploratory_score: float | None
    reasons: tuple[str, ...]
    semantic_context: dict[str, str]
    observed_semantic_context: dict[str, str]
    formal_preconditions: dict[str, bool]


@dataclass(frozen=True)
class CandidateRoutingReport:
    schema_version: str
    research_objective: str
    coverage_claim: str
    n_formal_invocations: int
    n_observed_significant_rows: int
    observed_min_fdr: float | None
    anchor_verdict: str
    may_enter_lineage: bool
    rows: tuple[CandidateRouteRow, ...]
    blocked: tuple[CandidateRouteRow, ...]
    sources: dict[str, str] = field(default_factory=dict)
    research_decision: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "research_objective": self.research_objective,
            "coverage_claim": self.coverage_claim,
            "research_decision": dict(self.research_decision),
            "n_formal_invocations": self.n_formal_invocations,
            "n_observed_significant_rows": self.n_observed_significant_rows,
            "observed_min_fdr": self.observed_min_fdr,
            "anchor_verdict": self.anchor_verdict,
            "may_enter_lineage": self.may_enter_lineage,
            "lineage_boundary": LINEAGE_BOUNDARY_SUPPLEMENTARY_ONLY,
            "biology_pass": False,
            "sources": dict(self.sources),
            "rows": [_row_to_payload(row) for row in self.rows],
            "blocked": [_row_to_payload(row) for row in self.blocked],
            "boundary": (
                "AD candidate routing encodes 方案 §6 route selection only; "
                "it never emits a PerturbGen invocation, never merges observed "
                "disease-minus-normal with intervention direction, never infers KD "
                "from KO, never admits failed or unapproved external assets "
                "into formal lineage, and never relabels FDR>0.05 as significant"
            ),
        }


def _row_to_payload(row: CandidateRouteRow) -> dict[str, Any]:
    return {
        "gene_symbol": row.gene_symbol,
        "ensembl_id": row.ensembl_id,
        "intervention": row.intervention,
        "selected_route": row.selected_route,
        "status": row.status,
        "lineage_boundary": row.lineage_boundary,
        "formal_invocation_allowed": row.formal_invocation_allowed,
        "predicted_direction_allowed": row.predicted_direction_allowed,
        "evidence_kind": row.evidence_kind,
        "validation_status": row.validation_status,
        "embedding_vocab": row.embedding_vocab,
        "in_route_axis": row.in_route_axis,
        "has_local_target_pair": row.has_local_target_pair,
        "observed_fdr": row.observed_fdr,
        "observed_significant": row.observed_significant,
        "observed_gate_pass": row.observed_gate_pass,
        "public_target_specific": row.public_target_specific,
        "grn_status": row.grn_status,
        "exploratory_score": row.exploratory_score,
        "reasons": list(row.reasons),
        "semantic_context": dict(row.semantic_context),
        "observed_semantic_context": dict(row.observed_semantic_context),
        "formal_preconditions": dict(row.formal_preconditions),
    }


def _as_bool(value: object, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ADCandidateRoutingError(f"{field} must be boolean; got empty")
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ADCandidateRoutingError(f"{field} must be boolean; got {value!r}")


def _nonempty(value: object) -> str:
    return str(value).strip() if value is not None and not (isinstance(value, float) and math.isnan(value)) else ""


def frozen_ad_candidates() -> tuple[tuple[str, str], ...]:
    """Return the frozen five-candidate catalog."""

    return FROZEN_AD_CANDIDATES


def lineage_admission(*, anchor_verdict: str | None, evidence_kind: str | None = None) -> tuple[str, bool]:
    """External evidence cannot enter formal lineage until a dedicated consumer exists.

    A passing anchor is still not admission: 方案 §6.4 forbids adding a formal
    consumer for the current GEARS/Geneformer assets. Missing or failed anchors
    are also refused.
    """

    del evidence_kind
    if anchor_verdict not in {None, "", "not_provided", "pass", "fail"}:
        raise ADCandidateRoutingError("anchor_verdict must be pass, fail, or not_provided")
    return LINEAGE_BOUNDARY_SUPPLEMENTARY_ONLY, False


def exploratory_score(
    *,
    validation_calibration: float | None,
    context_similarity: float | None,
    model_agreement: float | None,
    network_support: float | None,
) -> float | None:
    """Informal ranking score; any unavailable factor returns None instead of a default."""

    factors = (validation_calibration, context_similarity, model_agreement, network_support)
    if any(factor is None for factor in factors):
        return None
    product = 1.0
    for factor in factors:
        assert factor is not None
        if not math.isfinite(factor) or factor < 0.0:
            raise ADCandidateRoutingError("exploratory score factors must be finite and >= 0")
        product *= float(factor)
    return product


def load_axis_audit(path: str | Path) -> dict[str, AxisCoverage]:
    """Load the frozen DAVF axis-coverage audit; vocabulary is never treated as the axis."""

    resolved = Path(path).expanduser().resolve(strict=True)
    frame = pd.read_csv(resolved, sep="\t")
    missing = [column for column in AXIS_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ADCandidateRoutingError(f"axis audit is missing columns: {missing}")
    coverage: dict[str, AxisCoverage] = {}
    for record in frame.to_dict("records"):
        symbol = normalize_gene_symbol(str(record["gene_symbol"]))
        ensembl_id = normalize_ensembl_id(str(record["ensembl_id"]))
        expected = FROZEN_AD_ENSEMBL.get(symbol)
        if expected is None:
            raise ADCandidateRoutingError(f"axis audit contains a gene outside the frozen AD catalog: {symbol}")
        if expected != ensembl_id:
            raise ADCandidateRoutingError(
                f"axis audit Ensembl mismatch for {symbol}: expected {expected}, got {ensembl_id}"
            )
        if ensembl_id in coverage:
            raise ADCandidateRoutingError(f"axis audit contains duplicate candidate {ensembl_id}")
        perturbed_raw = _nonempty(record.get("scperturb_perturbed_in"))
        perturbed = tuple(part for part in perturbed_raw.split(";") if part) if perturbed_raw else ()
        coverage[ensembl_id] = AxisCoverage(
            gene_symbol=symbol,
            ensembl_id=ensembl_id,
            embedding_vocab=_as_bool(record["embedding_vocab"], field="embedding_vocab"),
            ko_in_axis=_as_bool(record["ko_in_axis"], field="ko_in_axis"),
            kd_in_axis=_as_bool(record["kd_in_axis"], field="kd_in_axis"),
            scperturb_perturbed_in=perturbed,
        )
    missing_candidates = [symbol for symbol, ensembl_id in FROZEN_AD_CANDIDATES if ensembl_id not in coverage]
    if missing_candidates:
        raise ADCandidateRoutingError(f"axis audit is missing frozen candidates: {missing_candidates}")
    return coverage


def load_public_perturbation_inventory(path: str | Path | None) -> tuple[InventoryMatch, ...]:
    """Load public perturbation inventory rows; missing path means no public coverage."""

    if path is None:
        return ()
    resolved = Path(path).expanduser().resolve(strict=True)
    frame = pd.read_csv(resolved, sep="\t")
    missing = [column for column in INVENTORY_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ADCandidateRoutingError(f"public perturbation inventory is missing columns: {missing}")
    matches: list[InventoryMatch] = []
    seen_keys: set[tuple[str, str, str]] = set()
    symbol_to_ensembl: dict[str, str] = {}
    for record in frame.to_dict("records"):
        dataset_id = _nonempty(record["dataset_id"])
        species = _nonempty(record["species"])
        intervention = _nonempty(record["intervention"]).upper()
        mechanism = _nonempty(record["mechanism"]).lower()
        target = normalize_ensembl_id(str(record["target_ensembl_id"]))
        symbol = normalize_gene_symbol(str(record["target_gene_symbol"]))
        if not dataset_id or not species or not mechanism:
            raise ADCandidateRoutingError("inventory rows require dataset_id, species, and mechanism")
        if intervention not in {"KO", "KD", "OE"}:
            raise ADCandidateRoutingError(f"inventory intervention must be KO, KD, or OE; got {intervention!r}")
        if intervention == "KO" and mechanism in VALID_KD_MECHANISMS:
            raise ADCandidateRoutingError(
                f"inventory {dataset_id} mixes KO intervention with KD mechanism {mechanism!r}"
            )
        if intervention == "KD" and mechanism in VALID_KO_MECHANISMS:
            raise ADCandidateRoutingError(
                f"inventory {dataset_id} mixes KD intervention with KO mechanism {mechanism!r}"
            )
        if intervention == "KD" and mechanism not in VALID_KD_MECHANISMS:
            raise ADCandidateRoutingError(
                "KD inventory mechanism must be CRISPRi/partial_knockdown/shRNA/dose; "
                f"got {mechanism!r} in {dataset_id}"
            )
        if intervention == "KO" and mechanism not in VALID_KO_MECHANISMS:
            raise ADCandidateRoutingError(
                f"KO inventory mechanism must be an explicit knockout mechanism; got {mechanism!r} in {dataset_id}"
            )
        expected_ensembl = FROZEN_AD_ENSEMBL.get(symbol)
        if expected_ensembl is not None and expected_ensembl != target:
            raise ADCandidateRoutingError(
                f"inventory mapping conflict for {symbol}: frozen Ensembl is {expected_ensembl}, got {target}"
            )
        expected_symbol = FROZEN_AD_SYMBOLS.get(target)
        if expected_symbol is not None and expected_symbol != symbol:
            raise ADCandidateRoutingError(
                f"inventory mapping conflict for {target}: frozen symbol is {expected_symbol}, got {symbol}"
            )
        previous = symbol_to_ensembl.get(symbol)
        if previous is not None and previous != target:
            raise ADCandidateRoutingError(f"inventory maps {symbol} to both {previous} and {target}")
        symbol_to_ensembl[symbol] = target
        has_control = _as_bool(record["has_control"], field="has_control")
        has_perturbed = _as_bool(record["has_perturbed"], field="has_perturbed")
        if not has_control or not has_perturbed:
            if target in FROZEN_AD_SYMBOLS:
                raise ADCandidateRoutingError(
                    f"inventory {dataset_id} for {target} is missing control/perturbed pairing"
                )
            continue
        key = (dataset_id, target, intervention)
        if key in seen_keys:
            raise ADCandidateRoutingError(f"inventory contains duplicate {dataset_id}/{target}/{intervention}")
        seen_keys.add(key)
        dose = _nonempty(record.get("dose"))
        time_point = _nonempty(record.get("time"))
        matches.append(
            InventoryMatch(
                dataset_id=dataset_id,
                species=species,
                cell_type=_nonempty(record["cell_type"]),
                state=_nonempty(record["state"]),
                intervention=intervention,
                mechanism=mechanism,
                target_ensembl_id=target,
                has_dose_or_time=bool(dose or time_point),
            )
        )
    return tuple(matches)


def load_grn_support(path: str | Path | None) -> dict[str, str]:
    """Return per-target GRN status: supported, insufficient, or empty if not provided."""

    if path is None:
        return {}
    resolved = Path(path).expanduser().resolve(strict=True)
    frame = pd.read_csv(resolved, sep="\t")
    missing = [column for column in GRN_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ADCandidateRoutingError(f"GRN support table is missing columns: {missing}")
    status: dict[str, str] = {}
    for record in frame.to_dict("records"):
        ensembl_id = normalize_ensembl_id(str(record["target_ensembl_id"]))
        has_tf = _as_bool(record["has_tf_support"], field="has_tf_support")
        has_path = _as_bool(record["has_direct_path"], field="has_direct_path")
        if has_tf and has_path:
            status[ensembl_id] = "supported"
        else:
            status[ensembl_id] = "insufficient"
    return status


def summarize_observed_gate(
    deg_frame: pd.DataFrame | None,
    *,
    max_fdr: float = 0.05,
    candidates: Sequence[tuple[str, str]] = FROZEN_AD_CANDIDATES,
    admission_rule: str = OBSERVED_ADMISSION_SIGNED_DIRECTION,
) -> ObservedGateSummary:
    """Summarize donor-level observed status without relaxing the frozen FDR label.

    ``admission_rule`` controls whether FDR≤0.05 is required to *admit* a
    signed disease−normal direction. Significance counts always use ``max_fdr``.
    """

    if not 0.0 < max_fdr <= 1.0:
        raise ADCandidateRoutingError("observed max_fdr must be within (0, 1]")
    if admission_rule not in {OBSERVED_ADMISSION_FDR_CUTOFF, OBSERVED_ADMISSION_SIGNED_DIRECTION}:
        raise ADCandidateRoutingError(f"unknown observed admission_rule: {admission_rule!r}")
    per_candidate: dict[str, ObservedCandidateStatus]
    if deg_frame is None:
        per_candidate = {
            ensembl_id: ObservedCandidateStatus(
                ensembl_id=ensembl_id,
                min_fdr=None,
                observed_significant=False,
                observed_gate_pass=False,
                cell_types_passing=(),
            )
            for _, ensembl_id in candidates
        }
        return ObservedGateSummary(
            max_fdr=max_fdr,
            n_rows=0,
            n_significant_rows=0,
            min_fdr=None,
            per_candidate=per_candidate,
            admission_rule=admission_rule,
        )

    missing = [column for column in DEG_MIN_COLUMNS if column not in deg_frame.columns]
    if missing:
        raise ADCandidateRoutingError(f"observed DEG table is missing columns: {missing}")
    frame = deg_frame.copy()
    frame["ensembl_id"] = frame["ensembl_id"].map(lambda value: normalize_ensembl_id(str(value)))
    frame["fdr"] = pd.to_numeric(frame["fdr"], errors="raise")
    if not frame["fdr"].apply(lambda value: math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0).all():
        raise ADCandidateRoutingError("observed DEG fdr must be finite and within [0, 1]")
    directions = frame["observed_direction"].astype(str).str.strip().str.lower()
    n_significant = int(((frame["fdr"] <= max_fdr) & directions.isin(("up", "down"))).sum())
    min_fdr = float(frame["fdr"].min()) if len(frame) else None
    per_candidate = {}
    for _, ensembl_id in candidates:
        subset = frame[frame["ensembl_id"] == ensembl_id]
        if subset.empty:
            per_candidate[ensembl_id] = ObservedCandidateStatus(
                ensembl_id=ensembl_id,
                min_fdr=None,
                observed_significant=False,
                observed_gate_pass=False,
                cell_types_passing=(),
            )
            continue
        candidate_min = float(subset["fdr"].min())
        subset_directions = subset["observed_direction"].astype(str).str.strip().str.lower()
        signed = subset[subset_directions.isin(("up", "down"))]
        significant = subset[(subset["fdr"] <= max_fdr) & subset_directions.isin(("up", "down"))]
        observed_significant = not significant.empty
        if admission_rule == OBSERVED_ADMISSION_FDR_CUTOFF:
            admitted = significant
            observed_gate_pass = observed_significant
        else:
            admitted = signed
            observed_gate_pass = not signed.empty
        per_candidate[ensembl_id] = ObservedCandidateStatus(
            ensembl_id=ensembl_id,
            min_fdr=candidate_min,
            observed_significant=observed_significant,
            observed_gate_pass=observed_gate_pass,
            cell_types_passing=tuple(sorted(str(value) for value in admitted["cell_type"].astype(str).unique())),
        )
    return ObservedGateSummary(
        max_fdr=max_fdr,
        n_rows=int(len(frame)),
        n_significant_rows=n_significant,
        min_fdr=min_fdr,
        per_candidate=per_candidate,
        admission_rule=admission_rule,
    )


def parse_anchor_verdict(payload: Mapping[str, Any] | None) -> str:
    if payload is None:
        return "not_provided"
    verdict = str(payload.get("verdict") or "").strip().lower()
    if verdict not in {"pass", "fail"}:
        raise ADCandidateRoutingError("anchor backtest verdict must be pass or fail")
    return verdict


def _inventory_for(
    inventory: Sequence[InventoryMatch],
    *,
    intervention: str,
    target_ensembl_id: str,
    human_only: bool = True,
) -> tuple[InventoryMatch, ...]:
    selected = []
    for row in inventory:
        if row.intervention != intervention or row.target_ensembl_id != target_ensembl_id:
            continue
        species = row.species.strip().lower()
        if human_only and species not in HUMAN_SPECIES:
            continue
        selected.append(row)
    return tuple(selected)


def _semantic_context(
    *,
    intervention: str,
    research_objective: str,
    evidence_source: str,
    context: str,
    cohort: str,
) -> dict[str, str]:
    return {
        "context": context,
        "intervention": intervention,
        "comparison_baseline": "matched_control",
        "reference_axis": "intervention_minus_control",
        "research_objective": research_objective,
        "evidence_source": evidence_source,
        "cohort": cohort,
    }


def _observed_semantic_context(*, research_objective: str, cohort: str) -> dict[str, str]:
    return {
        "context": f"{cohort} donor-level AD case-control",
        "intervention": "none",
        "comparison_baseline": "donor-level disease vs normal",
        "reference_axis": "disease_minus_normal",
        "research_objective": research_objective,
        "evidence_source": "observed donor-level DEG",
        "cohort": cohort,
    }


def _build_row(
    *,
    coverage: AxisCoverage,
    intervention: str,
    selected_route: str | None,
    status: str,
    evidence_kind: str | None,
    validation_status: str,
    reasons: Sequence[str],
    observed: ObservedCandidateStatus,
    has_local_target_pair: bool,
    public_target_specific: bool,
    grn_status: str,
    research_objective: str,
    observed_cohort: str,
    evidence_source: str,
    predicted_direction_allowed: bool,
) -> CandidateRouteRow:
    if status not in VALID_STATUSES:
        raise ADCandidateRoutingError(f"invalid routing status {status!r}")
    if selected_route is not None and selected_route not in VALID_ROUTES:
        raise ADCandidateRoutingError(f"invalid route {selected_route!r}")
    in_route_axis = coverage.ko_in_axis if intervention == "KO" else coverage.kd_in_axis
    if coverage.embedding_vocab and not in_route_axis:
        reasons = ("vocabulary_is_not_training_axis", *tuple(reasons))
    reasons = tuple(dict.fromkeys(reasons))
    if not observed.observed_significant:
        reasons = (*reasons, "observed_not_bh_significant")
    formal_preconditions = {
        "observed_gate": observed.observed_gate_pass,
        "route_axis": in_route_axis,
        "local_target_pair": has_local_target_pair,
        "three_way_gate": False,
        "held_out_davf": False,
    }
    return CandidateRouteRow(
        gene_symbol=coverage.gene_symbol,
        ensembl_id=coverage.ensembl_id,
        intervention=intervention,
        selected_route=selected_route,
        status=status,
        lineage_boundary=LINEAGE_BOUNDARY_SUPPLEMENTARY_ONLY,
        formal_invocation_allowed=False,
        predicted_direction_allowed=predicted_direction_allowed,
        evidence_kind=evidence_kind,
        validation_status=validation_status,
        embedding_vocab=coverage.embedding_vocab,
        in_route_axis=in_route_axis,
        has_local_target_pair=has_local_target_pair,
        observed_fdr=observed.min_fdr,
        observed_significant=observed.observed_significant,
        observed_gate_pass=observed.observed_gate_pass,
        public_target_specific=public_target_specific,
        grn_status=grn_status,
        exploratory_score=None,
        reasons=reasons,
        semantic_context=_semantic_context(
            intervention=intervention,
            research_objective=research_objective,
            evidence_source=evidence_source,
            context=f"AD candidate {coverage.gene_symbol} {intervention}",
            cohort="route-specific; see evidence_source",
        ),
        observed_semantic_context=_observed_semantic_context(
            research_objective=research_objective, cohort=observed_cohort
        ),
        formal_preconditions=formal_preconditions,
    )


def _route_kd(
    coverage: AxisCoverage,
    *,
    observed: ObservedCandidateStatus,
    inventory: Sequence[InventoryMatch],
    grn_status: str,
    contract_to_apoe_ko: bool,
    research_objective: str,
    observed_cohort: str,
    anchor_fail: bool,
) -> CandidateRouteRow:
    # Local scPerturb "perturbed_in" lists are not KD-mechanism-typed. A Frangieh
    # KO hit must not become a KD pair. KD coverage comes only from CRISPRi /
    # partial-knockdown / dose inventory rows.
    if contract_to_apoe_ko:
        return _build_row(
            coverage=coverage,
            intervention="KD",
            selected_route="B6",
            status="blocked",
            evidence_kind=None,
            validation_status="not_evaluated",
            reasons=("candidate_contraction_apoe_ko_only", "kd_unavailable_under_contraction"),
            observed=observed,
            has_local_target_pair=False,
            public_target_specific=False,
            grn_status=grn_status,
            research_objective=research_objective,
            observed_cohort=observed_cohort,
            evidence_source="candidate contraction B6",
            predicted_direction_allowed=False,
        )
    kd_inventory = _inventory_for(inventory, intervention="KD", target_ensembl_id=coverage.ensembl_id)
    if kd_inventory:
        route = "B3" if any(row.has_dose_or_time for row in kd_inventory) else "B1"
        reasons = ["public_kd_transfer_unvalidated"]
        if anchor_fail:
            reasons.append("current_k562_anchor_fail_not_reused")
        return _build_row(
            coverage=coverage,
            intervention="KD",
            selected_route=route,
            status="exploratory_unvalidated",
            evidence_kind="trained_response",
            validation_status="not_evaluated",
            reasons=reasons,
            observed=observed,
            has_local_target_pair=False,
            public_target_specific=True,
            grn_status=grn_status,
            research_objective=research_objective,
            observed_cohort=observed_cohort,
            evidence_source="public CRISPRi/partial knockdown/dose inventory",
            predicted_direction_allowed=False,
        )
    reasons = ["kd_requires_real_crispri_or_partial_knockdown"]
    if coverage.ko_in_axis or bool(coverage.scperturb_perturbed_in):
        reasons.append("cannot_infer_kd_from_ko")
    return _build_row(
        coverage=coverage,
        intervention="KD",
        selected_route=None,
        status="unavailable",
        evidence_kind=None,
        validation_status="not_evaluated",
        reasons=reasons,
        observed=observed,
        has_local_target_pair=False,
        public_target_specific=False,
        grn_status=grn_status,
        research_objective=research_objective,
        observed_cohort=observed_cohort,
        evidence_source="KD unavailable",
        predicted_direction_allowed=False,
    )


def _route_ko(
    coverage: AxisCoverage,
    *,
    observed: ObservedCandidateStatus,
    inventory: Sequence[InventoryMatch],
    grn_status: str,
    contract_to_apoe_ko: bool,
    research_objective: str,
    observed_cohort: str,
    anchor_fail: bool,
    has_foundation_asset: bool,
) -> CandidateRouteRow:
    has_local_ko_pair = coverage.ko_in_axis and bool(coverage.scperturb_perturbed_in)
    if contract_to_apoe_ko and coverage.gene_symbol != "APOE":
        return _build_row(
            coverage=coverage,
            intervention="KO",
            selected_route="B6",
            status="blocked",
            evidence_kind=None,
            validation_status="not_evaluated",
            reasons=("candidate_contraction_apoe_ko_only",),
            observed=observed,
            has_local_target_pair=has_local_ko_pair,
            public_target_specific=False,
            grn_status=grn_status,
            research_objective=research_objective,
            observed_cohort=observed_cohort,
            evidence_source="candidate contraction B6",
            predicted_direction_allowed=False,
        )
    if has_local_ko_pair:
        reasons = ["local_ko_axis_engineering_only", "not_biology_pass"]
        if not observed.observed_gate_pass:
            reasons.append("observed_gate_inconclusive")
        if anchor_fail:
            reasons.append("external_assets_anchor_fail_supplementary_only")
        return _build_row(
            coverage=coverage,
            intervention="KO",
            selected_route="APOE_KO_ENGINEERING",
            status="engineering_verification",
            evidence_kind="trained_response",
            validation_status="not_evaluated",
            reasons=reasons,
            observed=observed,
            has_local_target_pair=True,
            public_target_specific=False,
            grn_status=grn_status,
            research_objective=research_objective,
            observed_cohort=observed_cohort,
            evidence_source="local route-specific KO asset",
            predicted_direction_allowed=True,
        )
    ko_inventory = _inventory_for(inventory, intervention="KO", target_ensembl_id=coverage.ensembl_id)
    if ko_inventory:
        route = "B3" if any(row.has_dose_or_time for row in ko_inventory) else "B1"
        reasons = ["public_ko_transfer_unvalidated", "no_local_target_specific_pair"]
        if anchor_fail:
            reasons.append("current_k562_anchor_fail_not_reused")
        return _build_row(
            coverage=coverage,
            intervention="KO",
            selected_route=route,
            status="exploratory_unvalidated",
            evidence_kind="trained_response",
            validation_status="not_evaluated",
            reasons=reasons,
            observed=observed,
            has_local_target_pair=False,
            public_target_specific=True,
            grn_status=grn_status,
            research_objective=research_objective,
            observed_cohort=observed_cohort,
            evidence_source="public target-specific KO inventory",
            predicted_direction_allowed=False,
        )
    reasons = ["no_public_target_specific_data", "no_local_target_specific_pair"]
    if grn_status == "insufficient":
        reasons.append("grn_insufficient_no_zero_fill")
    if anchor_fail:
        reasons.append("current_gears_anchor_fail_not_lineage")
    if has_foundation_asset:
        reasons.append("foundation_ranking_only_no_expression_direction")
    evidence_kind = "go_extrapolation"
    selected_route = "B2"
    if grn_status == "supported":
        selected_route = "B4"
        evidence_kind = "network_counterfactual"
        reasons.append("ad_grn_network_counterfactual_only")
    return _build_row(
        coverage=coverage,
        intervention="KO",
        selected_route=selected_route,
        status="direction_only",
        evidence_kind=evidence_kind,
        validation_status="heldout_fail" if anchor_fail else "not_evaluated",
        reasons=reasons,
        observed=observed,
        has_local_target_pair=False,
        public_target_specific=False,
        grn_status=grn_status,
        research_objective=research_objective,
        observed_cohort=observed_cohort,
        evidence_source="unseen GEARS/GRN/foundation exploratory only",
        predicted_direction_allowed=False,
    )


def inspect_external_assets(assets: Sequence[Mapping[str, Any]]) -> tuple[bool, bool]:
    """Return (has_foundation_asset, refuses_predicted_direction_for_counterfactual)."""

    has_foundation = False
    for asset in assets:
        kind = str(asset.get("evidence_kind") or asset.get("design", {}).get("evidence_kind") or "").strip()
        if kind == "network_counterfactual":
            has_foundation = True
            candidates = asset.get("candidates")
            if isinstance(candidates, Mapping):
                for payload in candidates.values():
                    if isinstance(payload, Mapping) and payload.get("predicted_direction") not in {None, ""}:
                        raise ADCandidateRoutingError(
                            "network_counterfactual assets must not carry predicted_direction; "
                            "influence scores are not expression up/down"
                        )
        boundary = str(asset.get("lineage_boundary") or "").strip()
        if boundary and boundary != LINEAGE_BOUNDARY_SUPPLEMENTARY_ONLY:
            raise ADCandidateRoutingError(
                f"external evidence lineage_boundary must remain supplementary_only; got {boundary!r}"
            )
    return has_foundation, True


def route_frozen_ad_candidates(
    *,
    axis_coverage: Mapping[str, AxisCoverage],
    observed: ObservedGateSummary,
    inventory: Sequence[InventoryMatch] = (),
    grn_support: Mapping[str, str] | None = None,
    external_assets: Sequence[Mapping[str, Any]] = (),
    anchor_verdict: str = "not_provided",
    contract_to_apoe_ko: bool = False,
    research_objective: str = "association",
    observed_cohort: str = "GSE174367",
    sources: Mapping[str, str] | None = None,
    decision: ADResearchDecision | None = None,
) -> CandidateRoutingReport:
    """Assign every frozen candidate × in-scope intervention to a §6 route without emitting invocations."""

    frozen = decision if decision is not None else FROZEN_AD_RESEARCH_DECISION
    if research_objective not in VALID_RESEARCH_OBJECTIVES:
        raise ADCandidateRoutingError(f"research_objective must be one of {VALID_RESEARCH_OBJECTIVES}")
    if observed.max_fdr != 0.05 or frozen.deg_reporting_max_fdr != 0.05:
        raise ADCandidateRoutingError("routing refuses to change the frozen observed FDR threshold of 0.05")
    if observed.admission_rule != frozen.observed_admission_rule:
        raise ADCandidateRoutingError(
            "observed admission_rule must match the frozen research decision: "
            f"summary={observed.admission_rule!r} decision={frozen.observed_admission_rule!r}"
        )
    if frozen.public_perturbation_policy == PUBLIC_PERTURBATION_OUT_OF_SCOPE and inventory:
        raise ADCandidateRoutingError(
            "public Perturb-seq inventory is out of scope under the frozen 2026-09-18 decision"
        )
    lineage_boundary, may_enter = lineage_admission(anchor_verdict=anchor_verdict)
    if lineage_boundary != LINEAGE_BOUNDARY_SUPPLEMENTARY_ONLY or may_enter:
        raise ADCandidateRoutingError("routing cannot admit external evidence into formal lineage")
    has_foundation, _ = inspect_external_assets(external_assets)
    grn = dict(grn_support or {})
    rows: list[CandidateRouteRow] = []
    for symbol, ensembl_id in FROZEN_AD_CANDIDATES:
        coverage = axis_coverage.get(ensembl_id)
        if coverage is None:
            raise ADCandidateRoutingError(f"axis coverage missing frozen candidate {symbol}")
        if coverage.gene_symbol != symbol:
            raise ADCandidateRoutingError(f"axis coverage symbol mismatch for {ensembl_id}")
        observed_status = observed.per_candidate[ensembl_id]
        grn_status = grn.get(ensembl_id, "not_provided")
        rows.append(
            _route_ko(
                coverage,
                observed=observed_status,
                inventory=inventory,
                grn_status=grn_status,
                contract_to_apoe_ko=contract_to_apoe_ko,
                research_objective=research_objective,
                observed_cohort=observed_cohort,
                anchor_fail=anchor_verdict == "fail",
                has_foundation_asset=has_foundation,
            )
        )
        if frozen.kd_policy != KD_POLICY_MERGED_INTO_KO:
            rows.append(
                _route_kd(
                    coverage,
                    observed=observed_status,
                    inventory=inventory,
                    grn_status=grn_status,
                    contract_to_apoe_ko=contract_to_apoe_ko,
                    research_objective=research_objective,
                    observed_cohort=observed_cohort,
                    anchor_fail=anchor_verdict == "fail",
                )
            )
    if frozen.public_perturbation_policy == PUBLIC_PERTURBATION_OUT_OF_SCOPE and any(
        row.selected_route in {"B1", "B3"} for row in rows
    ):
        raise ADCandidateRoutingError("B1/B3 public transfer is out of scope under the frozen decision")
    if any(row.formal_invocation_allowed for row in rows):
        raise ADCandidateRoutingError("routing must never set formal_invocation_allowed")
    if any(row.semantic_context["reference_axis"] == row.observed_semantic_context["reference_axis"] for row in rows):
        raise ADCandidateRoutingError("intervention and observed reference axes must stay field-separated")
    blocked = tuple(row for row in rows if row.status in {"blocked", "unavailable", "failed"})
    if contract_to_apoe_ko:
        coverage_claim = "apoe_ko_only"
    elif frozen.kd_policy == KD_POLICY_MERGED_INTO_KO:
        coverage_claim = "five_candidates_ko_only"
    else:
        coverage_claim = "five_candidates_exploratory_split"
    return CandidateRoutingReport(
        schema_version=AD_CANDIDATE_ROUTING_SCHEMA_VERSION,
        research_objective=research_objective,
        coverage_claim=coverage_claim,
        n_formal_invocations=0,
        n_observed_significant_rows=observed.n_significant_rows,
        observed_min_fdr=observed.min_fdr,
        anchor_verdict=anchor_verdict,
        may_enter_lineage=False,
        rows=tuple(rows),
        blocked=blocked,
        sources=dict(sources or {}),
        research_decision=frozen.to_payload(),
    )


def write_routing_outputs(report: CandidateRoutingReport, output_dir: str | Path) -> dict[str, Path]:
    """Write candidates, semantic contexts, ranking, blocked items, and the routing report."""

    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    collisions = sorted(name for name in _PROTECTED_EXTERNAL_BASENAMES if (directory / name).exists())
    # Existing fail assets in the same folder are left untouched; refuse to replace them.
    for name in collisions:
        if name in {
            "candidates.tsv",
            "semantic_context.json",
            "candidate_routing_report.json",
            "exploratory_candidate_ranking.tsv",
            "blocked_items.tsv",
            "manifest.json",
        }:
            raise ADCandidateRoutingError(f"refusing to overwrite protected external asset {name}")

    candidates_path = directory / "candidates.tsv"
    pd.DataFrame(
        [
            {"gene_symbol": symbol, "ensembl_id": ensembl_id, "frozen": True}
            for symbol, ensembl_id in FROZEN_AD_CANDIDATES
        ]
    ).to_csv(candidates_path, sep="\t", index=False)

    semantic_path = directory / "semantic_context.json"
    semantic_path.write_text(
        json.dumps(
            {
                "schema_version": AD_CANDIDATE_ROUTING_SCHEMA_VERSION,
                "per_row": [
                    {
                        "ensembl_id": row.ensembl_id,
                        "intervention": row.intervention,
                        "intervention_semantic_context": row.semantic_context,
                        "observed_semantic_context": row.observed_semantic_context,
                    }
                    for row in report.rows
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    ranking_path = directory / "exploratory_candidate_ranking.tsv"
    pd.DataFrame([_row_to_payload(row) for row in report.rows]).to_csv(ranking_path, sep="\t", index=False)

    blocked_path = directory / "blocked_items.tsv"
    pd.DataFrame([_row_to_payload(row) for row in report.blocked]).to_csv(blocked_path, sep="\t", index=False)

    report_path = directory / "candidate_routing_report.json"
    report_path.write_text(json.dumps(report.to_payload(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    decision_path = directory / "observed_gate_decision.json"
    decision_path.write_text(
        json.dumps(dict(report.research_decision), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": AD_CANDIDATE_ROUTING_SCHEMA_VERSION,
                "coverage_claim": report.coverage_claim,
                "n_formal_invocations": 0,
                "may_enter_lineage": False,
                "biology_pass": False,
                "research_decision": dict(report.research_decision),
                "outputs": {
                    "candidates": str(candidates_path),
                    "semantic_context": str(semantic_path),
                    "ranking": str(ranking_path),
                    "blocked": str(blocked_path),
                    "report": str(report_path),
                    "observed_gate_decision": str(decision_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "candidates": candidates_path,
        "semantic_context": semantic_path,
        "ranking": ranking_path,
        "blocked": blocked_path,
        "report": report_path,
        "observed_gate_decision": decision_path,
        "manifest": manifest_path,
    }


__all__ = [
    "ADCandidateRoutingError",
    "AD_CANDIDATE_ROUTING_SCHEMA_VERSION",
    "AxisCoverage",
    "CandidateRouteRow",
    "CandidateRoutingReport",
    "FROZEN_AD_CANDIDATES",
    "LINEAGE_BOUNDARY_SUPPLEMENTARY_ONLY",
    "ObservedGateSummary",
    "exploratory_score",
    "frozen_ad_candidates",
    "lineage_admission",
    "load_axis_audit",
    "load_grn_support",
    "load_public_perturbation_inventory",
    "parse_anchor_verdict",
    "route_frozen_ad_candidates",
    "summarize_observed_gate",
    "write_routing_outputs",
]
