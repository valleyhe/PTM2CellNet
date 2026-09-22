"""Frozen research-design contract for the PTM-activity → AD intersection mainline.

Implements stage 0 of ``docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md``
(§7 阶段 0): before any real PTM asset is processed, the reference axis,
contrast, research objective, activity method, network release, AD DEG
thresholds, propagation parameters and the frozen cell-type list must be
recorded in a ``ptm_research_config.yaml`` and validated fail-fast here.

Every downstream CLI (``run_ptm_activity.py``, ``build_ptm_global_gene_scores.py``,
``build_ptm_ad_intersections.py``, ``build_celltype_candidate_specs.py``)
consumes this same frozen config so the thresholds cannot drift between
stages (方案 §8.7: PTM-side thresholds must not be tuned on AD DEG results).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

PTM_RESEARCH_CONFIG_SCHEMA_VERSION = "ptm2cellnet.ptm-research-config/v1"

VALID_RESEARCH_OBJECTIVES = ("association", "replication", "reversal")
VALID_REPLICATE_POLICIES = ("fail", "mean")
#: Frozen donor-level DEG estimands (方案 §7.4-U4): ``per_cell_log2_mean``
#: normalizes each cell before the donor average; ``pseudobulk_counts`` sums
#: raw counts within a donor before normalization;
#: ``pseudobulk_counts_centered`` is the multi-cohort variant that subtracts
#: each (cell type, cohort) normal-donor baseline before the test. Switching
#: estimands is a research-design decision that must be recorded in the DEG
#: manifest.
VALID_DONOR_AGGREGATIONS = ("per_cell_log2_mean", "pseudobulk_counts", "pseudobulk_counts_centered")
#: Direction fields that must stay separate per 方案 §3.1; the config freezes
#: the *reference axis* used to interpret them, it never merges them.
VALID_REFERENCE_AXES = (
    "disease_minus_normal",
    "contrast_specific",
)
#: Observed-arm *admission* (2026-09-18 option 3) is independent of the frozen
#: BH reporting cutoff ``deg_max_fdr``. Legacy configs omit the field and keep
#: FDR as both label and admission rule.
VALID_OBSERVED_ADMISSION_RULES = ("fdr_cutoff", "signed_direction_without_fdr_cutoff")
VALID_KD_POLICIES = ("separate_routes", "merged_into_ko_out_of_scope")
VALID_PUBLIC_PERTURBATION_POLICIES = ("inventory_optional", "out_of_scope")
#: Formal/exploratory intake boundary (TD-04): ``formal`` rejects every
#: ``PENDING*`` placeholder asset; ``exploratory`` keeps the sentinel and the
#: downstream lineage boundaries stay ``may_enter_lineage=false``.
VALID_MODES = ("exploratory", "formal")


class PTMResearchConfigError(ValueError):
    """Raised when the frozen research design violates the plan contract."""


@dataclass(frozen=True)
class PropagationConfig:
    """Signed-network propagation parameters (方案 §5.3).

    ``max_depth`` bounds the simple-path search; ``decay`` is the per-hop
    weight factor applied to every edge regardless of confidence;
    ``gene_edge_types`` freezes which ``edge_type`` values terminate a path
    at a gene (e.g. ``tf_regulation``). Genes reached only through other
    edge types are intermediate network nodes, not scored genes.
    """

    max_depth: int
    decay: float
    gene_edge_types: tuple[str, ...]
    max_paths_per_seed: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_depth, bool) or not isinstance(self.max_depth, int) or self.max_depth < 1:
            raise PTMResearchConfigError("propagation.max_depth must be an integer >= 1")
        if not isinstance(self.decay, float) or not 0.0 < self.decay <= 1.0:
            raise PTMResearchConfigError("propagation.decay must be within (0, 1]")
        types = tuple(str(value).strip() for value in self.gene_edge_types if str(value).strip())
        if not types:
            raise PTMResearchConfigError("propagation.gene_edge_types must contain at least one edge type")
        object.__setattr__(self, "gene_edge_types", types)
        if self.max_paths_per_seed is not None and (
            isinstance(self.max_paths_per_seed, bool)
            or not isinstance(self.max_paths_per_seed, int)
            or self.max_paths_per_seed < 1
        ):
            raise PTMResearchConfigError("propagation.max_paths_per_seed must be a positive integer or None")


@dataclass(frozen=True)
class ActivityAdmissionPolicy:
    """Frozen activity-admission thresholds executed before signed propagation.

    方案 §4.4/§5.2 要求 PTM score 侧使用*预先冻结*的 activity q-value /
    substrate / coverage 阈值，且阈值一经登记不得在结果上事后调整
    (方案 §8.7)。默认值是显式登记的宽松阈值：在独立 kinase-perturbation
    benchmark 校准前 PTM 侧保持 direction_only，不按 q/coverage 隐式收紧；
    研究负责人在 frozen config 中显式收紧后此 gate 立即生效。
    """

    max_activity_qvalue: float = 1.0
    min_substrates: int = 0
    min_network_coverage: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.max_activity_qvalue, bool) or not isinstance(self.max_activity_qvalue, (int, float)):
            raise PTMResearchConfigError("activity_admission.max_activity_qvalue must be numeric")
        if not 0.0 < float(self.max_activity_qvalue) <= 1.0:
            raise PTMResearchConfigError("activity_admission.max_activity_qvalue must be within (0, 1]")
        if isinstance(self.min_substrates, bool) or not isinstance(self.min_substrates, int) or self.min_substrates < 0:
            raise PTMResearchConfigError("activity_admission.min_substrates must be an integer >= 0")
        if isinstance(self.min_network_coverage, bool) or not isinstance(self.min_network_coverage, (int, float)):
            raise PTMResearchConfigError("activity_admission.min_network_coverage must be numeric")
        if not 0.0 <= float(self.min_network_coverage) <= 1.0:
            raise PTMResearchConfigError("activity_admission.min_network_coverage must be within [0, 1]")
        object.__setattr__(self, "max_activity_qvalue", float(self.max_activity_qvalue))
        object.__setattr__(self, "min_network_coverage", float(self.min_network_coverage))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def policy_hash(self) -> str:
        """Stable content hash so a score manifest can pin the exact policy."""

        payload = json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ActivityBenchmarkCriteria:
    """Pre-registered thresholds for the independent activity benchmark (方案 §5.2 第 4–5 条).

    与 ``ActivityAdmissionPolicy`` 同级冻结在 config 中，但三个判据阈值
    *没有宽松默认*：必须由研究负责人显式预注册 (方案 §8.7)，未注册时
    ``activity_benchmark`` 段为空，传播 CLI 拒绝 benchmark 评估而不是替
    用户挑阈值。``perturbation_effect`` 语义：kinase 扰动后活性的真实
    变化，正=升高、负=降低。
    """

    min_paired_regulators: int
    min_direction_concordance: float
    min_abs_spearman: float
    bootstrap_iterations: int = 2000
    ci_level: float = 0.95
    seed: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_paired_regulators, bool)
            or not isinstance(self.min_paired_regulators, int)
            or self.min_paired_regulators < 1
        ):
            raise PTMResearchConfigError("activity_benchmark.min_paired_regulators must be an integer >= 1")
        if isinstance(self.min_direction_concordance, bool) or not isinstance(
            self.min_direction_concordance, (int, float)
        ):
            raise PTMResearchConfigError("activity_benchmark.min_direction_concordance must be numeric")
        if not 0.0 < float(self.min_direction_concordance) <= 1.0:
            raise PTMResearchConfigError("activity_benchmark.min_direction_concordance must be within (0, 1]")
        if isinstance(self.min_abs_spearman, bool) or not isinstance(self.min_abs_spearman, (int, float)):
            raise PTMResearchConfigError("activity_benchmark.min_abs_spearman must be numeric")
        if not 0.0 <= float(self.min_abs_spearman) <= 1.0:
            raise PTMResearchConfigError("activity_benchmark.min_abs_spearman must be within [0, 1]")
        if (
            isinstance(self.bootstrap_iterations, bool)
            or not isinstance(self.bootstrap_iterations, int)
            or self.bootstrap_iterations < 0
        ):
            raise PTMResearchConfigError("activity_benchmark.bootstrap_iterations must be an integer >= 0")
        if isinstance(self.ci_level, bool) or not isinstance(self.ci_level, (int, float)):
            raise PTMResearchConfigError("activity_benchmark.ci_level must be numeric")
        if not 0.0 < float(self.ci_level) < 1.0:
            raise PTMResearchConfigError("activity_benchmark.ci_level must be within (0, 1)")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise PTMResearchConfigError("activity_benchmark.seed must be an integer >= 0")
        object.__setattr__(self, "min_direction_concordance", float(self.min_direction_concordance))
        object.__setattr__(self, "min_abs_spearman", float(self.min_abs_spearman))
        object.__setattr__(self, "ci_level", float(self.ci_level))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PTMResearchConfig:
    """Frozen stage-0 research design (方案 §7 阶段 0)."""

    schema_version: str
    research_objective: str
    reference_axis: str
    contrast: str
    primary_activity_method: str
    network_release: str
    cell_types: tuple[str, ...]
    cohort_h5ad: str
    cohort_pairing: str
    species: str
    ptm_cohort: str
    deg_max_fdr: float
    min_donors_per_state: int
    replicate_policy: str
    propagation: PropagationConfig
    sensitivity_activity_method: str | None = None
    deg_donor_aggregation: str = "per_cell_log2_mean"
    observed_admission_rule: str = "fdr_cutoff"
    kd_policy: str = "separate_routes"
    public_perturbation_policy: str = "inventory_optional"
    semantic_context: Mapping[str, str] = field(default_factory=dict)
    activity_admission: ActivityAdmissionPolicy = field(default_factory=ActivityAdmissionPolicy)
    #: Optional pre-registered benchmark thresholds; ``None`` keeps
    #: ``benchmark_gate.available=false`` (exploratory-only, 方案 §5.2 第 5 条).
    activity_benchmark: ActivityBenchmarkCriteria | None = None
    mode: str = "exploratory"
    #: Optional path to the signed-network release manifest that pins the exact
    #: asset (sha256/rows) behind ``network_release`` (TD-07); when set, the
    #: propagation CLI must pass ``verify_network_release_binding`` before use.
    network_release_manifest: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != PTM_RESEARCH_CONFIG_SCHEMA_VERSION:
            raise PTMResearchConfigError(
                f"schema_version must be {PTM_RESEARCH_CONFIG_SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )
        if self.research_objective not in VALID_RESEARCH_OBJECTIVES:
            raise PTMResearchConfigError(f"research_objective must be one of {', '.join(VALID_RESEARCH_OBJECTIVES)}")
        if self.reference_axis not in VALID_REFERENCE_AXES:
            raise PTMResearchConfigError(
                f"reference_axis must be one of {', '.join(VALID_REFERENCE_AXES)} (方案 §3.1 方向字段不得合并)"
            )
        for name in ("contrast", "primary_activity_method", "network_release", "cohort_h5ad", "species", "ptm_cohort"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise PTMResearchConfigError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value)
        if self.cohort_pairing not in ("within_donor", "between_donor"):
            raise PTMResearchConfigError("cohort_pairing must be 'within_donor' or 'between_donor'")
        cell_types = tuple(str(value).strip() for value in self.cell_types if str(value).strip())
        if not cell_types:
            raise PTMResearchConfigError("cell_types must contain at least one frozen cell type")
        if len(set(cell_types)) != len(cell_types):
            raise PTMResearchConfigError("cell_types must not contain duplicates")
        object.__setattr__(self, "cell_types", cell_types)
        if not isinstance(self.deg_max_fdr, float) or not 0.0 < self.deg_max_fdr <= 1.0:
            raise PTMResearchConfigError("deg_max_fdr must be within (0, 1]")
        if isinstance(self.min_donors_per_state, bool) or not isinstance(self.min_donors_per_state, int):
            raise PTMResearchConfigError("min_donors_per_state must be an integer")
        if self.min_donors_per_state < 1:
            raise PTMResearchConfigError("min_donors_per_state must be >= 1")
        if self.replicate_policy not in VALID_REPLICATE_POLICIES:
            raise PTMResearchConfigError(
                f"replicate_policy must be one of {', '.join(VALID_REPLICATE_POLICIES)}; "
                "duplicate (sample, protein, residue, ptm_type) rows cannot be resolved silently (方案 §5.1)"
            )
        if self.deg_donor_aggregation not in VALID_DONOR_AGGREGATIONS:
            raise PTMResearchConfigError(
                f"deg_donor_aggregation must be one of {', '.join(VALID_DONOR_AGGREGATIONS)}; "
                f"got {self.deg_donor_aggregation!r}"
            )
        if self.observed_admission_rule not in VALID_OBSERVED_ADMISSION_RULES:
            raise PTMResearchConfigError(
                f"observed_admission_rule must be one of {', '.join(VALID_OBSERVED_ADMISSION_RULES)}"
            )
        if self.kd_policy not in VALID_KD_POLICIES:
            raise PTMResearchConfigError(f"kd_policy must be one of {', '.join(VALID_KD_POLICIES)}")
        if self.public_perturbation_policy not in VALID_PUBLIC_PERTURBATION_POLICIES:
            raise PTMResearchConfigError(
                "public_perturbation_policy must be one of " + ", ".join(VALID_PUBLIC_PERTURBATION_POLICIES)
            )
        if not isinstance(self.propagation, PropagationConfig):
            raise PTMResearchConfigError("propagation must be a PropagationConfig")
        if not isinstance(self.activity_admission, ActivityAdmissionPolicy):
            raise PTMResearchConfigError("activity_admission must be an ActivityAdmissionPolicy")
        if self.activity_benchmark is not None and not isinstance(self.activity_benchmark, ActivityBenchmarkCriteria):
            raise PTMResearchConfigError("activity_benchmark must be an ActivityBenchmarkCriteria or omitted")
        if self.mode not in VALID_MODES:
            raise PTMResearchConfigError(f"mode must be one of {', '.join(VALID_MODES)}")
        if self.mode == "formal":
            for name in ("ptm_cohort", "cohort_h5ad"):
                value = str(getattr(self, name)).strip()
                if value.upper().startswith("PENDING"):
                    raise PTMResearchConfigError(
                        f"formal mode rejects placeholder {name}={value!r}; a real registered asset is required "
                        "(exploratory runs keep the sentinel)"
                    )
        if self.network_release_manifest is not None:
            manifest_value = str(self.network_release_manifest).strip()
            if not manifest_value:
                raise PTMResearchConfigError("network_release_manifest must be a non-empty path or omitted")
            object.__setattr__(self, "network_release_manifest", manifest_value)
        if self.semantic_context:
            required = (
                "context",
                "intervention",
                "comparison_baseline",
                "reference_axis",
                "research_objective",
                "evidence_source",
                "cohort",
            )
            missing = [key for key in required if key not in self.semantic_context]
            if missing:
                raise PTMResearchConfigError("semantic_context template is missing fields: " + ", ".join(missing))
            frozen = {**self.semantic_context}
            if frozen["research_objective"] != self.research_objective:
                raise PTMResearchConfigError(
                    "semantic_context.research_objective must match the frozen research_objective"
                )
            object.__setattr__(self, "semantic_context", frozen)

    def semantic_context_for_cell_type(self, cell_type: str) -> dict[str, str]:
        """Render the seven-field semantic context for one cell type.

        ``{cell_type}`` placeholders are substituted; a template without
        placeholders applies verbatim to every cell type. This mapping is
        what the generated candidate specs carry into the existing E2E
        semantic-context gate (方案 §3.1).
        """

        if not self.semantic_context:
            raise PTMResearchConfigError(
                "semantic_context template is required to emit candidate specs (方案 §3.1 七字段)"
            )
        rendered = {key: str(value).replace("{cell_type}", cell_type) for key, value in self.semantic_context.items()}
        return rendered


def load_ptm_research_config(path: str | Path) -> PTMResearchConfig:
    """Load and validate the frozen research-design YAML (fail-fast)."""

    resolved = Path(path).expanduser().resolve(strict=True)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PTMResearchConfigError(f"research config root must be a YAML mapping: {resolved}")
    return parse_ptm_research_config(payload)


def parse_ptm_research_config(payload: Mapping[str, Any]) -> PTMResearchConfig:
    """Build a config from an already-parsed mapping."""

    missing: list[str] = []
    for key in (
        "schema_version",
        "research_objective",
        "reference_axis",
        "contrast",
        "primary_activity_method",
        "network_release",
        "cell_types",
        "cohort_h5ad",
        "cohort_pairing",
        "species",
        "ptm_cohort",
        "deg_max_fdr",
        "min_donors_per_state",
        "replicate_policy",
        "propagation",
    ):
        if key not in payload:
            missing.append(key)
    if missing:
        raise PTMResearchConfigError("research config is missing keys: " + ", ".join(missing))
    propagation_payload = payload["propagation"]
    if not isinstance(propagation_payload, Mapping):
        raise PTMResearchConfigError("propagation must be a mapping")
    for key in ("max_depth", "decay", "gene_edge_types"):
        if key not in propagation_payload:
            raise PTMResearchConfigError(f"propagation is missing key: {key}")
    max_depth = propagation_payload["max_depth"]
    decay = propagation_payload["decay"]
    if isinstance(max_depth, bool) or not isinstance(max_depth, int):
        raise PTMResearchConfigError("propagation.max_depth must be an integer")
    if isinstance(decay, bool) or not isinstance(decay, (int, float)):
        raise PTMResearchConfigError("propagation.decay must be numeric")
    propagation = PropagationConfig(
        max_depth=max_depth,
        decay=float(decay),
        gene_edge_types=tuple(propagation_payload["gene_edge_types"]),
        max_paths_per_seed=propagation_payload.get("max_paths_per_seed"),
    )
    semantic_context = payload.get("semantic_context") or {}
    if not isinstance(semantic_context, Mapping):
        raise PTMResearchConfigError("semantic_context must be a mapping")
    admission_payload = payload.get("activity_admission")
    if admission_payload is None:
        admission = ActivityAdmissionPolicy()
    elif isinstance(admission_payload, Mapping):
        unknown = sorted(set(admission_payload) - {"max_activity_qvalue", "min_substrates", "min_network_coverage"})
        if unknown:
            raise PTMResearchConfigError(f"activity_admission has unknown keys: {', '.join(unknown)}")
        admission = ActivityAdmissionPolicy(
            **{key: value for key, value in admission_payload.items() if value is not None},
        )
    else:
        raise PTMResearchConfigError("activity_admission must be a mapping")
    benchmark_payload = payload.get("activity_benchmark")
    if benchmark_payload is None:
        benchmark = None
    elif isinstance(benchmark_payload, Mapping):
        required = ("min_paired_regulators", "min_direction_concordance", "min_abs_spearman")
        missing_benchmark = [key for key in required if benchmark_payload.get(key) is None]
        if missing_benchmark:
            raise PTMResearchConfigError(
                "activity_benchmark requires pre-registered values for " + ", ".join(missing_benchmark) + " (方案 §8.7)"
            )
        unknown_benchmark = sorted(
            set(benchmark_payload)
            - {
                "min_paired_regulators",
                "min_direction_concordance",
                "min_abs_spearman",
                "bootstrap_iterations",
                "ci_level",
                "seed",
            }
        )
        if unknown_benchmark:
            raise PTMResearchConfigError(f"activity_benchmark has unknown keys: {', '.join(unknown_benchmark)}")
        benchmark = ActivityBenchmarkCriteria(
            **{key: value for key, value in benchmark_payload.items() if value is not None}
        )
    else:
        raise PTMResearchConfigError("activity_benchmark must be a mapping")
    return PTMResearchConfig(
        schema_version=payload["schema_version"],
        research_objective=str(payload["research_objective"]).strip(),
        reference_axis=str(payload["reference_axis"]).strip(),
        contrast=str(payload["contrast"]).strip(),
        primary_activity_method=str(payload["primary_activity_method"]).strip(),
        network_release=str(payload["network_release"]).strip(),
        cell_types=tuple(payload["cell_types"]),
        cohort_h5ad=str(payload["cohort_h5ad"]),
        cohort_pairing=str(payload["cohort_pairing"]),
        species=str(payload["species"]),
        ptm_cohort=str(payload["ptm_cohort"]),
        deg_max_fdr=payload["deg_max_fdr"],
        min_donors_per_state=payload["min_donors_per_state"],
        replicate_policy=str(payload["replicate_policy"]).strip(),
        propagation=propagation,
        sensitivity_activity_method=payload.get("sensitivity_activity_method"),
        deg_donor_aggregation=str(payload.get("deg_donor_aggregation", "per_cell_log2_mean")).strip(),
        observed_admission_rule=str(payload.get("observed_admission_rule", "fdr_cutoff")).strip(),
        kd_policy=str(payload.get("kd_policy", "separate_routes")).strip(),
        public_perturbation_policy=str(payload.get("public_perturbation_policy", "inventory_optional")).strip(),
        semantic_context=dict(semantic_context),
        activity_admission=admission,
        activity_benchmark=benchmark,
        mode=str(payload.get("mode", "exploratory")).strip(),
        network_release_manifest=(
            str(payload["network_release_manifest"]).strip()
            if payload.get("network_release_manifest") is not None
            else None
        ),
    )
