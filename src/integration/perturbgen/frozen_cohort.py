"""M6 frozen-cohort scientific acceptance contracts (proposal §7.2 M6).

The statistical kernel (seeds, matched nulls, BH-FDR, bootstrap CI) already
lives in ``results.py``/``dual_path.py``; what was missing is the manifest
contract and the independent-replay entry point (analysis
``project_analysis_20260910.md`` §4.1 N-4):

* a frozen cohort manifest binding the cohort h5ad (hash), the explicit
  train/held-out donor split, and the candidate list with its seed/mode/null
  execution plan;
* donor-leakage auditing (a donor may not build the disease signature and
  judge the rescue at the same time);
* an acceptance plan matrix (candidate × path × seed × mode) derived from
  the manifest so executions and replays can be checked against it;
* an independent replay that re-runs the dual-path verdicts from the
  evaluation input and compares them with a produced report.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from numbers import Real
import re
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

import pandas as pd

from .contracts import VALID_COHORT_PAIRINGS
from .dual_path import evaluate_dual_path_candidate
from .empirical_pvalue import EmpiricalPvalueError, aggregate_candidate_empirical_pvalues

FROZEN_COHORT_SCHEMA_VERSION = "ptm2cellnet.frozen-cohort/v1"

_VALID_PATHS = ("source_intervention", "within_state")
_VALID_MODES = ("mask", "pad", "delete", "overexpress")
_VALID_INTERVENTIONS = ("KO", "KD")
_KO_MODES = frozenset(("mask", "pad", "delete"))
_FORMAL_MIN_NULLS = 99
_FORMAL_MIN_SEEDS = 3
_EVAL_INPUT_SCHEMA_VERSION = "perturbgen_dual_path_eval/v1"
_VALID_EVALUATION_MODES = {"engineering", "formal"}
_SYNTHETIC_PVALUE_SOURCES = {"uniform", "hand_filled", "external_table"}
_PATH_TO_SEQUENCE = {"source_intervention": "src", "within_state": "tgt"}
_H5AD_NAME_PATTERN = re.compile(r"_g(?P<gene>[^_]+)_s(?P<sequence>src|tgt)_t(?P<mode>[A-Za-z]+)\.h5ad$")
_REPLAY_PARAMETER_FIELDS = (
    "target_gene",
    "donor_obs_column",
    "var_gene_column",
    "deg_donor_column",
    "deg_gene_column",
    "deg_effect_column",
    "deg_fdr_column",
    "fdr_threshold",
    "min_training_donors",
    "min_evaluable_donors",
    "top_k",
    "bootstrap_iterations",
    "bootstrap_seed",
)


class FrozenCohortError(ValueError):
    """Raised when a frozen-cohort contract is violated."""


@dataclass(frozen=True)
class FrozenCandidate:
    """One candidate of the frozen acceptance cohort."""

    gene_symbol: str
    ensembl_id: str
    intervention_type: str
    modes: tuple[str, ...] = ("mask",)
    seeds: tuple[int, ...] = (0, 1, 2)
    matched_nulls: int = _FORMAL_MIN_NULLS

    def __post_init__(self) -> None:
        if not self.gene_symbol.strip() or not self.ensembl_id.strip():
            raise FrozenCohortError("candidate gene_symbol and ensembl_id must not be empty")
        route = str(self.intervention_type).strip().upper()
        if route not in _VALID_INTERVENTIONS:
            raise FrozenCohortError(f"candidate intervention_type must be one of {_VALID_INTERVENTIONS}")
        object.__setattr__(self, "intervention_type", route)
        modes = tuple(dict.fromkeys(str(mode).strip() for mode in self.modes))
        if not modes or any(mode not in _VALID_MODES for mode in modes):
            raise FrozenCohortError(f"candidate modes must be a subset of {_VALID_MODES}")
        if "overexpress" in modes and len(modes) != 1:
            raise FrozenCohortError(
                f"candidate {self.gene_symbol} cannot mix the OE overexpress mode with KO sensitivity modes"
            )
        if "overexpress" not in modes and "mask" not in modes:
            raise FrozenCohortError(f"candidate {self.gene_symbol} must include the 'mask' primary mode")
        if route == "KD" and any(mode in _KO_MODES - {"mask"} for mode in modes):
            raise FrozenCohortError("KD candidates cannot declare KO pad/delete sensitivity modes")
        try:
            seeds = tuple(sorted({int(seed) for seed in self.seeds}))
        except (TypeError, ValueError) as exc:
            raise FrozenCohortError(f"candidate {self.gene_symbol} seeds must be integers") from exc
        if any(seed < 0 for seed in seeds):
            raise FrozenCohortError(f"candidate {self.gene_symbol} seeds must be non-negative")
        if len(seeds) < _FORMAL_MIN_SEEDS:
            raise FrozenCohortError(f"candidate {self.gene_symbol} requires at least {_FORMAL_MIN_SEEDS} seeds")
        if isinstance(self.matched_nulls, bool) or int(self.matched_nulls) < _FORMAL_MIN_NULLS:
            raise FrozenCohortError(
                f"candidate {self.gene_symbol} requires at least {_FORMAL_MIN_NULLS} matched nulls for a formal verdict"
            )
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "matched_nulls", int(self.matched_nulls))

    @property
    def formal_plan_complete(self) -> bool:
        """Whether this candidate carries the full mode plan for its route."""

        if self.modes == ("overexpress",):
            return True
        if self.intervention_type == "KD":
            return self.modes == ("mask",)
        return set(self.modes) == _KO_MODES


@dataclass(frozen=True)
class FrozenCohortManifest:
    """Frozen cohort + candidate plan with verifiable provenance."""

    cohort_h5ad: str
    cohort_sha256: str
    cell_type: str
    donor_obs_column: str
    train_donors: tuple[str, ...]
    held_out_donors: tuple[str, ...]
    candidates: tuple[FrozenCandidate, ...]
    created_at: str
    source_config: str | None = None
    pairing: str = "within_donor"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.cohort_h5ad or not self.cohort_sha256:
            raise FrozenCohortError("cohort_h5ad and cohort_sha256 are required")
        cell_type = str(self.cell_type).strip()
        donor_obs_column = str(self.donor_obs_column).strip()
        if not cell_type:
            raise FrozenCohortError("cell_type must not be empty")
        if not donor_obs_column:
            raise FrozenCohortError("donor_obs_column must not be empty")
        object.__setattr__(self, "cell_type", cell_type)
        object.__setattr__(self, "donor_obs_column", donor_obs_column)
        pairing = str(self.pairing).strip()
        if pairing not in VALID_COHORT_PAIRINGS:
            raise FrozenCohortError(f"pairing must be one of {VALID_COHORT_PAIRINGS}, got {pairing!r}")
        object.__setattr__(self, "pairing", pairing)
        if not self.train_donors or not self.held_out_donors:
            raise FrozenCohortError("frozen cohort requires explicit train and held-out donor lists")
        train = _normalise_donors(self.train_donors, "training")
        held_out = _normalise_donors(self.held_out_donors, "held-out")
        object.__setattr__(self, "train_donors", train)
        object.__setattr__(self, "held_out_donors", held_out)
        if len(train) < 2:
            raise FrozenCohortError("at least 2 training donors are required")
        if len(held_out) < 3:
            raise FrozenCohortError("at least 3 held-out donors are required")
        self.audit_donor_leakage()
        if not self.candidates:
            raise FrozenCohortError("frozen cohort requires at least one candidate")
        candidate_ids = [candidate.ensembl_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise FrozenCohortError("frozen cohort candidates must have unique ensembl_id values")

    def audit_donor_leakage(self) -> None:
        """A donor must not build the signature and judge the rescue."""

        overlap = sorted(set(self.train_donors) & set(self.held_out_donors))
        if overlap:
            raise FrozenCohortError(f"donor leakage: donors appear in both train and held-out splits: {overlap}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": FROZEN_COHORT_SCHEMA_VERSION,
            "cohort_h5ad": self.cohort_h5ad,
            "cohort_sha256": self.cohort_sha256,
            "cell_type": self.cell_type,
            "donor_obs_column": self.donor_obs_column,
            "train_donors": list(self.train_donors),
            "held_out_donors": list(self.held_out_donors),
            "candidates": [
                {
                    "gene_symbol": candidate.gene_symbol,
                    "ensembl_id": candidate.ensembl_id,
                    "intervention_type": candidate.intervention_type,
                    "modes": list(candidate.modes),
                    "seeds": list(candidate.seeds),
                    "matched_nulls": candidate.matched_nulls,
                }
                for candidate in self.candidates
            ],
            "created_at": self.created_at,
            "source_config": self.source_config,
            "pairing": self.pairing,
            "notes": list(self.notes),
        }


def _normalise_donors(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise FrozenCohortError(f"{label} donors must be a sequence")
    labels = tuple(str(value).strip() for value in values)
    if not labels or any(not value for value in labels):
        raise FrozenCohortError(f"{label} donors must not contain empty labels")
    if len(labels) != len(set(labels)):
        raise FrozenCohortError(f"{label} donors must be unique")
    return labels


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_frozen_manifest(
    *,
    cohort_h5ad: str | Path,
    cell_type: str,
    donor_obs_column: str,
    train_donors: Sequence[str],
    held_out_donors: Sequence[str],
    candidates_csv: str | Path,
    created_at: str,
    modes: Sequence[str] = ("mask", "pad", "delete"),
    seeds: Sequence[int] = (0, 1, 2),
    matched_nulls: int = _FORMAL_MIN_NULLS,
    source_config: str | None = None,
    pairing: str = "within_donor",
    notes: Sequence[str] = (),
) -> FrozenCohortManifest:
    """Build a frozen manifest from explicit splits and a candidate CSV.

    The candidate CSV needs ``gene_symbol``, ``ensembl_id`` and
    ``intervention_type`` columns; the mode/seed/null plan is uniform and
    recorded per candidate so the acceptance matrix is reproducible.
    ``pairing`` freezes the donor/state design the cohort was accepted under
    (``within_donor`` shared donors vs ``between_donor`` case-control) and is
    propagated into every downstream Gate-0 validation of this manifest.
    """

    resolved = Path(cohort_h5ad).expanduser().resolve(strict=True)
    candidate_rows = _load_candidates_csv(candidates_csv)
    default_modes = tuple(modes)
    candidates = tuple(
        FrozenCandidate(
            gene_symbol=row["gene_symbol"],
            ensembl_id=row["ensembl_id"],
            intervention_type=row["intervention_type"],
            modes=(tuple(mode.strip() for mode in row["modes"].split(",")) if row.get("modes", "") else default_modes),
            seeds=tuple(seeds),
            matched_nulls=matched_nulls,
        )
        for row in candidate_rows
    )
    return FrozenCohortManifest(
        cohort_h5ad=str(resolved),
        cohort_sha256=sha256_file(resolved),
        cell_type=cell_type,
        donor_obs_column=donor_obs_column,
        train_donors=tuple(train_donors),
        held_out_donors=tuple(held_out_donors),
        candidates=candidates,
        created_at=created_at,
        source_config=source_config,
        pairing=pairing,
        notes=tuple(notes),
    )


def load_frozen_manifest(path: str | Path) -> FrozenCohortManifest:
    """Load and validate a frozen manifest JSON."""

    resolved = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FrozenCohortError(f"frozen manifest root must be an object: {resolved}")
    if payload.get("schema_version") != FROZEN_COHORT_SCHEMA_VERSION:
        raise FrozenCohortError(f"unsupported frozen manifest schema_version {payload.get('schema_version')!r}")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise FrozenCohortError("frozen manifest requires a non-empty candidates list")
    candidates = tuple(
        FrozenCandidate(
            gene_symbol=str(row.get("gene_symbol", "")),
            ensembl_id=str(row.get("ensembl_id", "")),
            intervention_type=str(row.get("intervention_type", "")),
            modes=tuple(row.get("modes", ("mask",))),
            seeds=tuple(row.get("seeds", ())),
            matched_nulls=int(row.get("matched_nulls", 0)),
        )
        for row in raw_candidates
    )
    manifest = FrozenCohortManifest(
        cohort_h5ad=str(payload.get("cohort_h5ad", "")),
        cohort_sha256=str(payload.get("cohort_sha256", "")),
        cell_type=str(payload.get("cell_type", "")),
        donor_obs_column=str(payload.get("donor_obs_column", "donor")),
        train_donors=tuple(payload.get("train_donors", ())),
        held_out_donors=tuple(payload.get("held_out_donors", ())),
        candidates=candidates,
        created_at=str(payload.get("created_at", "")),
        source_config=payload.get("source_config"),
        pairing=str(payload.get("pairing", "within_donor")),
        notes=tuple(payload.get("notes", ())),
    )
    current_hash = sha256_file(manifest.cohort_h5ad)
    if current_hash != manifest.cohort_sha256:
        raise FrozenCohortError(
            "cohort h5ad hash drifted from the frozen manifest: "
            f"{manifest.cohort_h5ad} is {current_hash}, manifest records {manifest.cohort_sha256}"
        )
    return manifest


def build_acceptance_plan(manifest: FrozenCohortManifest) -> dict[str, Any]:
    """Derive the candidate × path × seed × mode execution matrix."""

    entries: list[dict[str, Any]] = []
    for candidate in manifest.candidates:
        for path_kind in _VALID_PATHS:
            for seed in candidate.seeds:
                for mode in candidate.modes:
                    entries.append(
                        {
                            "gene_symbol": candidate.gene_symbol,
                            "ensembl_id": candidate.ensembl_id,
                            "intervention_type": candidate.intervention_type,
                            "path": path_kind,
                            "seed": seed,
                            "mode": mode,
                            "matched_nulls": candidate.matched_nulls,
                        }
                    )
    return {
        "schema_version": "ptm2cellnet.frozen-acceptance-plan/v1",
        "cohort_h5ad": manifest.cohort_h5ad,
        "cohort_sha256": manifest.cohort_sha256,
        "cell_type": manifest.cell_type,
        "donor_obs_column": manifest.donor_obs_column,
        "pairing": manifest.pairing,
        "train_donors": list(manifest.train_donors),
        "held_out_donors": list(manifest.held_out_donors),
        "formal_plan_complete": all(candidate.formal_plan_complete for candidate in manifest.candidates),
        "formal_plan_issues": [
            f"{candidate.ensembl_id}:{candidate.intervention_type} mode plan is not formally complete"
            for candidate in manifest.candidates
            if not candidate.formal_plan_complete
        ],
        "n_runs": len(entries),
        "runs": entries,
    }


def _validate_frozen_cohort_asset(
    manifest: FrozenCohortManifest,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Validate the manifest cohort and its fixed state/donor partition.

    The donor/state coverage rule follows the frozen ``pairing``: under
    ``within_donor`` every frozen donor must appear in both states (the
    perturbation-paired design Gate-0 accepted); under ``between_donor`` each
    donor belongs to exactly one state, so the frozen donors only need to be
    present in the cohort while per-state donor counts and disjointness are
    enforced by ``prepare_perturbgen_anndata`` with the same pairing.
    """

    details: dict[str, Any] = {
        "path": manifest.cohort_h5ad,
        "cell_type": manifest.cell_type,
        "donor_obs_column": manifest.donor_obs_column,
        "pairing": manifest.pairing,
    }
    adata = None
    try:
        import anndata as ad
        from .contracts import PerturbGenDataSpec
        from .data_prep import prepare_perturbgen_anndata

        adata = ad.read_h5ad(manifest.cohort_h5ad)
        spec = PerturbGenDataSpec(donor_col=manifest.donor_obs_column, pairing=manifest.pairing)
        prepared = prepare_perturbgen_anndata(adata, cell_type=manifest.cell_type, spec=spec)
        target = prepared.adata.obs.loc[
            prepared.adata.obs[spec.cell_type_col].astype(str).str.strip() == manifest.cell_type
        ]
        expected_donors = set(manifest.train_donors) | set(manifest.held_out_donors)
        state_donors = {
            state: sorted(
                set(
                    target.loc[
                        target[spec.state_col].astype(str).str.strip() == state,
                        spec.donor_col,
                    ]
                    .astype(str)
                    .str.strip()
                )
            )
            for state in (spec.normal_state, spec.disease_state)
        }
        actual_donors = set().union(*state_donors.values())
        if manifest.pairing == "within_donor":
            missing = {
                state: sorted(expected_donors - set(donors))
                for state, donors in state_donors.items()
                if expected_donors - set(donors)
            }
        else:
            absent = sorted(expected_donors - actual_donors)
            missing = {"both_states": absent} if absent else {}
        extra = sorted(actual_donors - expected_donors)
        details.update(
            {
                "n_cells": int(target.shape[0]),
                "n_genes": int(prepared.adata.n_vars),
                "state_donors": state_donors,
                "target_donors": sorted(actual_donors),
            }
        )
        issues: list[str] = []
        if missing:
            issues.append(f"fixed donors are absent from the cohort states under pairing={manifest.pairing}: {missing}")
        if extra:
            issues.append(f"cohort contains donors outside the frozen split: {extra}")
        return not issues, details, issues
    except ImportError as exc:
        return False, details, [f"anndata/data-prep unavailable: {exc}"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return False, details, [f"cohort contract failed: {exc}"]
    finally:
        if adata is not None and getattr(adata, "file", None) is not None:
            adata.file.close()


def _validate_declared_cohort_reference(
    manifest: FrozenCohortManifest,
    source: Mapping[str, Any],
) -> list[str]:
    """Compare optional cohort declarations with the frozen manifest."""

    path_value = source.get("cohort_h5ad")
    hash_value = source.get("cohort_sha256")
    if path_value is None and hash_value is None:
        return []
    issues: list[str] = []
    if path_value is None:
        issues.append("cohort_sha256 is declared without cohort_h5ad")
    else:
        try:
            declared_path = Path(str(path_value)).expanduser().resolve(strict=True)
            frozen_path = Path(manifest.cohort_h5ad).expanduser().resolve(strict=True)
            if declared_path != frozen_path:
                issues.append(f"cohort_h5ad does not match frozen cohort: {declared_path}")
        except (OSError, ValueError) as exc:
            issues.append(f"declared cohort_h5ad is not readable: {exc}")
    if hash_value is not None and str(hash_value).strip() != manifest.cohort_sha256:
        issues.append("declared cohort_sha256 does not match frozen cohort")
    return issues


def _validate_declared_pairing(
    manifest: FrozenCohortManifest,
    source: Mapping[str, Any],
) -> list[str]:
    """Compare an optional cohort_pairing declaration with the frozen manifest."""

    declared = source.get("cohort_pairing")
    if declared is None:
        return []
    if str(declared).strip() != manifest.pairing:
        return [f"declared cohort_pairing {declared!r} does not match frozen pairing {manifest.pairing!r}"]
    return []


def _evaluation_mode(eval_input: Mapping[str, Any]) -> str:
    raw_mode = eval_input.get("evaluation_mode")
    return "engineering" if raw_mode is None else str(raw_mode).strip().lower()


def _validate_eval_input_lineage(eval_input: Mapping[str, Any]) -> list[str]:
    """Validate the extra lineage required by formal frozen replay."""

    mode = _evaluation_mode(eval_input)
    if mode not in _VALID_EVALUATION_MODES:
        return [f"eval input evaluation_mode must be one of {sorted(_VALID_EVALUATION_MODES)}"]
    if mode != "formal":
        return []

    issues: list[str] = []
    if eval_input.get("evidence_class") != "empirical_null":
        issues.append("formal eval input requires evidence_class='empirical_null'")
    raw_source = eval_input.get("pvalue_source")
    pvalue_source = str(raw_source).strip() if raw_source is not None else ""
    if not pvalue_source or pvalue_source in _SYNTHETIC_PVALUE_SOURCES or not pvalue_source.startswith("empirical_"):
        issues.append("formal eval input requires an empirical pvalue_source")

    raw_input_json = eval_input.get("input_json")
    input_path: Path | None = None
    if not isinstance(raw_input_json, str) or not raw_input_json.strip():
        issues.append("formal eval input requires input_json lineage")
    else:
        try:
            input_path = Path(raw_input_json).expanduser().resolve(strict=True)
        except (OSError, ValueError) as exc:
            issues.append(f"formal eval input input_json is not readable: {exc}")

    raw_hash = eval_input.get("input_sha256")
    input_hash = str(raw_hash).strip() if raw_hash is not None else ""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", input_hash):
        issues.append("formal eval input requires a 64-character input_sha256")
    elif input_path is not None:
        try:
            actual_hash = sha256_file(input_path)
        except OSError as exc:
            issues.append(f"formal eval input input_json cannot be hashed: {exc}")
        else:
            if actual_hash != input_hash:
                issues.append("formal eval input input_sha256 does not match input_json")

    contract = eval_input.get("contract")
    if not isinstance(contract, Mapping):
        issues.append("formal eval input requires contract lineage")
    else:
        if contract.get("schema_version") != _EVAL_INPUT_SCHEMA_VERSION:
            issues.append("formal eval input contract.schema_version is invalid")
        for field_name in ("evaluation_mode", "evidence_class", "pvalue_source"):
            if contract.get(field_name) != eval_input.get(field_name):
                issues.append(f"formal eval input contract.{field_name} does not match eval input")

    candidates = eval_input.get("candidates")
    if isinstance(candidates, list):
        for index, entry in enumerate(candidates):
            if not isinstance(entry, Mapping):
                continue
            if "candidate_pvalue" in entry:
                issues.append(f"formal eval input candidate[{index}] must not declare candidate_pvalue")
            candidate_source = entry.get("pvalue_source")
            if candidate_source is not None:
                candidate_source = str(candidate_source).strip()
                if not candidate_source.startswith("empirical_") or candidate_source in _SYNTHETIC_PVALUE_SOURCES:
                    issues.append(f"formal eval input candidate[{index}] requires an empirical pvalue_source")
                elif candidate_source != pvalue_source:
                    issues.append(f"formal eval input candidate[{index}] pvalue_source does not match eval input")
            quality = entry.get("unperturbed_quality")
            if not isinstance(quality, Mapping) or quality.get("source") != "extract_unperturbed_quality_from_h5ad":
                issues.append(f"formal eval input candidate[{index}] requires unperturbed_quality extracted from h5ad")
            elif quality.get("status") != entry.get("unperturbed_quality_status"):
                issues.append(f"formal eval input candidate[{index}] quality status does not match its declaration")
    return issues


def _same_path(left: Any, right: Any) -> bool:
    try:
        return Path(str(left)).expanduser().resolve() == Path(str(right)).expanduser().resolve()
    except (OSError, ValueError):
        return False


def _validate_report_declaration(
    report_manifest: Mapping[str, Any],
    eval_input: Mapping[str, Any],
) -> list[str]:
    """Bind report-level declarations to the supplied evaluation input."""

    if _evaluation_mode(eval_input) != "formal":
        return []

    issues: list[str] = []
    for field_name in ("input_json", "input_sha256", "contract"):
        if field_name not in report_manifest:
            issues.append(f"report manifest is missing {field_name} declaration")
            continue
        expected = eval_input.get(field_name)
        actual = report_manifest[field_name]
        if field_name == "input_json":
            if not _same_path(actual, expected):
                issues.append("report input_json does not match eval input")
        elif actual != expected:
            issues.append(f"report {field_name} does not match eval input")

    for field_name in ("evaluation_mode", "evidence_class", "pvalue_source"):
        if report_manifest.get(field_name) != eval_input.get(field_name):
            issues.append(f"report {field_name} does not match eval input")
    return issues


def _run_identity(run: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(run.get("path", "")), str(run.get("mode", "")), str(run.get("seed", "")))


def _index_eval_candidates(
    eval_candidates: list[Any],
) -> tuple[dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] | None, list[str]]:
    """Index eval-input candidates by ensembl_id; a fatal shape error returns None."""

    expected_by_id: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for index, raw_entry in enumerate(eval_candidates):
        if not isinstance(raw_entry, Mapping):
            return None, [f"eval input candidate[{index}] must be an object"]
        raw_candidate = raw_entry.get("candidate")
        if not isinstance(raw_candidate, Mapping):
            return None, [f"eval input candidate[{index}] must declare a candidate object"]
        ensembl_id = str(raw_candidate.get("ensembl_id", "")).strip()
        if not ensembl_id:
            return None, [f"eval input candidate[{index}] is missing ensembl_id"]
        if ensembl_id in expected_by_id:
            return None, [f"eval input contains duplicate candidate {ensembl_id}"]
        expected_by_id[ensembl_id] = (raw_entry, raw_candidate)
    return expected_by_id, []


def _index_report_candidates(
    report_candidates: list[Any],
) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    """Index report replay candidates by ensembl_id, collecting shape issues."""

    actual_by_id: dict[str, Mapping[str, Any]] = {}
    issues: list[str] = []
    for index, raw_entry in enumerate(report_candidates):
        if not isinstance(raw_entry, Mapping):
            issues.append(f"report candidate[{index}] must be an object")
            continue
        replay = raw_entry.get("replay")
        raw_candidate = replay.get("candidate") if isinstance(replay, Mapping) else None
        if not isinstance(raw_candidate, Mapping):
            issues.append(f"report candidate[{index}] is missing replay.candidate")
            continue
        ensembl_id = str(raw_candidate.get("ensembl_id", "")).strip()
        if not ensembl_id:
            issues.append(f"report candidate[{index}] is missing replay candidate ensembl_id")
        elif ensembl_id in actual_by_id:
            issues.append(f"report contains duplicate candidate {ensembl_id}")
        else:
            actual_by_id[ensembl_id] = raw_entry
    return actual_by_id, issues


def _compare_replay_runs(ensembl_id: str, expected_runs: Any, actual_runs: Any) -> list[str]:
    """Compare per-run fields between the eval input and the report replay."""

    issues: list[str] = []
    if not isinstance(expected_runs, list) or not isinstance(actual_runs, list):
        return [f"{ensembl_id}: report/eval input runs must both be lists"]
    actual_by_identity: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for raw_run in actual_runs:
        if isinstance(raw_run, Mapping):
            actual_by_identity[_run_identity(raw_run)] = raw_run
    for index, raw_run in enumerate(expected_runs):
        if not isinstance(raw_run, Mapping):
            issues.append(f"{ensembl_id}: eval input run[{index}] must be an object")
            continue
        identity = _run_identity(raw_run)
        actual_run = actual_by_identity.get(identity)
        if actual_run is None:
            issues.append(f"{ensembl_id}: report is missing eval input run {identity}")
            continue
        for field_name, expected_value in raw_run.items():
            if field_name not in actual_run:
                issues.append(f"{ensembl_id} run {identity}: report is missing {field_name}")
            else:
                issues.extend(
                    _compare_replay_values(expected_value, actual_run[field_name], f"{ensembl_id}.run.{field_name}")
                )
    if len(expected_runs) != len(actual_runs):
        issues.append(f"{ensembl_id}: report/eval input run counts differ")
    return issues


def _compare_candidate_replay(
    ensembl_id: str,
    expected_entry: Mapping[str, Any],
    expected_candidate: Mapping[str, Any],
    actual_entry: Mapping[str, Any],
    eval_mode: str,
) -> list[str]:
    """Compare one candidate's replay fields against its eval-input entry."""

    issues: list[str] = []
    replay = actual_entry.get("replay")
    if not isinstance(replay, Mapping):
        return issues
    actual_candidate = replay.get("candidate")
    if not isinstance(actual_candidate, Mapping):
        return issues
    for field_name, expected_value in expected_candidate.items():
        if field_name not in actual_candidate:
            issues.append(f"{ensembl_id}: report replay candidate is missing {field_name}")
        else:
            issues.extend(
                _compare_replay_values(
                    expected_value, actual_candidate[field_name], f"{ensembl_id}.candidate.{field_name}"
                )
            )
    for field_name in ("observed_direction", "unperturbed_quality_status"):
        if expected_entry.get(field_name) != replay.get(field_name):
            issues.append(f"{ensembl_id}: report replay {field_name} does not match eval input")

    if eval_mode != "formal" and "candidate_pvalue" in expected_entry:
        if actual_entry.get("candidate_pvalue") != expected_entry.get("candidate_pvalue"):
            issues.append(f"{ensembl_id}: report candidate_pvalue does not match eval input")

    expected_runs = expected_entry.get("runs")
    actual_runs = replay.get("runs")
    issues.extend(_compare_replay_runs(ensembl_id, expected_runs, actual_runs))
    return issues


def _validate_report_against_eval_input(
    report_manifest: Mapping[str, Any],
    eval_input: Mapping[str, Any],
) -> list[str]:
    """Ensure report replay inputs are the ones supplied in ``eval_input``."""

    report_candidates = report_manifest.get("candidates")
    eval_candidates = eval_input.get("candidates")
    if not isinstance(report_candidates, list) or not isinstance(eval_candidates, list):
        return ["report and eval input candidates must both be lists"]

    expected_by_id, fatal = _index_eval_candidates(eval_candidates)
    if expected_by_id is None:
        return fatal
    actual_by_id, issues = _index_report_candidates(report_candidates)

    if set(expected_by_id) != set(actual_by_id):
        issues.append(
            "report/eval input candidate set differs: "
            f"missing={sorted(set(expected_by_id) - set(actual_by_id))}, "
            f"extra={sorted(set(actual_by_id) - set(expected_by_id))}"
        )

    eval_mode = _evaluation_mode(eval_input)
    for ensembl_id, (expected_entry, expected_candidate) in expected_by_id.items():
        actual_entry = actual_by_id.get(ensembl_id)
        if actual_entry is None:
            continue
        issues.extend(
            _compare_candidate_replay(ensembl_id, expected_entry, expected_candidate, actual_entry, eval_mode)
        )
    return issues


def _read_deg_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, Mapping) and isinstance(payload.get("records"), list):
            records = payload["records"]
        else:
            raise ValueError("DEG JSON must be a record list or {'records': [...]}")
        if not records or any(not isinstance(record, Mapping) for record in records):
            raise ValueError("DEG table must contain non-empty record mappings")
        table = pd.DataFrame(records)
    else:
        raise ValueError("DEG table must end with .csv or .json")
    if table.empty:
        raise ValueError("DEG table must contain non-empty records")
    return table


def _read_records(path: Path) -> list[Mapping[str, Any]]:
    table = _read_deg_table(path)
    records = table.to_dict(orient="records")
    if not records or any(not isinstance(record, Mapping) for record in records):
        raise ValueError("DEG table must contain non-empty record mappings")
    return cast(list[Mapping[str, Any]], records)


def _validate_deg_training_donors(
    manifest: FrozenCohortManifest,
    run: Mapping[str, Any],
) -> list[str]:
    raw_path = run.get("deg_table_path")
    raw_column = run.get("deg_donor_column")
    if raw_path is None or not str(raw_path).strip():
        return ["run is missing deg_table_path"]
    if raw_column is None or not str(raw_column).strip():
        return ["run is missing deg_donor_column"]
    try:
        records = _read_records(Path(str(raw_path)).expanduser().resolve(strict=True))
    except (OSError, ValueError, TypeError) as exc:
        return [f"DEG table cannot be read: {exc}"]
    column = str(raw_column).strip()
    donors: set[str] = set()
    for record in records:
        value = record.get(column)
        if value is None or not str(value).strip():
            return [f"DEG table column {column!r} contains an empty donor"]
        donors.add(str(value).strip())
    expected = set(manifest.train_donors)
    if donors != expected:
        return [f"DEG donors {sorted(donors)!r} do not equal frozen training donors {sorted(expected)!r}"]
    return []


def _validate_output_donors(
    manifest: FrozenCohortManifest,
    output_h5ad: Path,
) -> list[str]:
    adata = None
    try:
        import anndata as ad

        adata = ad.read_h5ad(output_h5ad, backed="r")
        if manifest.donor_obs_column not in adata.obs.columns:
            return [f"output h5ad is missing donor column {manifest.donor_obs_column!r}"]
        donor_values = adata.obs[manifest.donor_obs_column]
        if donor_values.isna().any():
            return [f"output h5ad donor column {manifest.donor_obs_column!r} contains missing values"]
        donors = set(donor_values.astype(str).str.strip())
        if "" in donors:
            return [f"output h5ad donor column {manifest.donor_obs_column!r} contains empty values"]
        expected = set(manifest.held_out_donors)
        if donors != expected:
            return [f"output h5ad donors {sorted(donors)!r} do not equal frozen held-out donors {sorted(expected)!r}"]
        return []
    except ImportError as exc:
        return [f"anndata unavailable for output donor validation: {exc}"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return [f"output h5ad donor validation failed: {exc}"]
    finally:
        if adata is not None and getattr(adata, "file", None) is not None:
            adata.file.close()


def _validate_stage_lineage(
    manifest: FrozenCohortManifest,
    stage_manifest_path: Path | None,
    stage_payload: Mapping[str, Any] | None,
    run: Mapping[str, Any],
    provenance: Mapping[str, Any] | None,
) -> list[str]:
    """Require an explicit perturb→tokenise artifact chain for the cohort."""

    if stage_manifest_path is None or stage_payload is None:
        return ["perturb stage manifest is required for cohort lineage"]

    tokenise_reference = run.get("tokenise_stage_manifest")
    if tokenise_reference is None and provenance is not None:
        tokenise_reference = provenance.get("tokenise_stage_manifest")
    if tokenise_reference is None or not str(tokenise_reference).strip():
        return ["run is missing explicit tokenise_stage_manifest binding"]

    try:
        tokenise_path = Path(str(tokenise_reference)).expanduser().resolve(strict=True)
        tokenise_payload = json.loads(tokenise_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return [f"tokenise stage manifest cannot be read: {exc}"]
    if not isinstance(tokenise_payload, Mapping):
        return ["tokenise stage manifest root must be an object"]
    if tokenise_payload.get("status") != "success" or tokenise_payload.get("stage") != "tokenise":
        return ["tokenise stage manifest must be a successful tokenise stage"]

    tokenise_artifacts = tokenise_payload.get("artifacts")
    if not isinstance(tokenise_artifacts, Mapping):
        return ["tokenise stage manifest is missing artifacts"]
    material = stage_payload.get("fingerprint_material")
    fingerprint_config = material.get("fingerprint_config") if isinstance(material, Mapping) else None
    stage_config = fingerprint_config.get("stage_config") if isinstance(fingerprint_config, Mapping) else None
    perturb_config = stage_config.get("perturb_config") if isinstance(stage_config, Mapping) else None
    data = perturb_config.get("data") if isinstance(perturb_config, Mapping) else None
    trainer = perturb_config.get("trainer") if isinstance(perturb_config, Mapping) else None
    if not isinstance(data, Mapping) or not isinstance(trainer, Mapping):
        return ["perturb stage manifest is missing resolved perturb data/trainer configuration"]

    references = {
        "src_dataset_file": (data.get("src_dataset_file"), "src_dataset"),
        "tgt_dataset_folder": (data.get("tgt_dataset_folder"), "tgt_dataset_folder"),
        "src_adata": (data.get("src_adata"), "src_h5ad"),
        "tgt_adata_folder": (data.get("tgt_adata_folder"), "tgt_h5ad_folder"),
        "mapping_dict_path": (trainer.get("mapping_dict_path"), "rowid_to_gene_name"),
        "tokenid_to_rowid_path": (trainer.get("tokenid_to_rowid_path"), "tokenid_to_rowid"),
    }
    issues: list[str] = []
    for field_name, (raw_actual, artifact_name) in references.items():
        raw_expected = tokenise_artifacts.get(artifact_name)
        if raw_actual is None or raw_expected is None:
            issues.append(f"tokenise artifact binding is missing {field_name}/{artifact_name}")
            continue
        try:
            actual = Path(str(raw_actual)).expanduser().resolve(strict=True)
            expected = Path(str(raw_expected)).expanduser().resolve(strict=True)
        except (OSError, ValueError) as exc:
            issues.append(f"tokenise artifact binding {field_name} is not readable: {exc}")
            continue
        if actual != expected:
            issues.append(f"perturb {field_name} does not match tokenise artifact {artifact_name}")

    tokenise_material = tokenise_payload.get("fingerprint_material")
    fingerprint_files = tokenise_material.get("fingerprint_files") if isinstance(tokenise_material, Mapping) else None
    if not isinstance(fingerprint_files, Mapping):
        issues.append("tokenise stage manifest is missing fingerprint_files")
    else:
        frozen_path = Path(manifest.cohort_h5ad).expanduser().resolve()
        matches = []
        for raw_path, raw_hash in fingerprint_files.items():
            try:
                path = Path(str(raw_path)).expanduser().resolve()
            except (OSError, ValueError):
                continue
            if path == frozen_path:
                matches.append(str(raw_hash).strip())
        if matches != [manifest.cohort_sha256]:
            issues.append("tokenise fingerprint_files do not bind the frozen cohort and hash")
    return issues


def _independent_replay_options(run: Mapping[str, Any]) -> dict[str, Any]:
    missing = [
        field_name for field_name in _REPLAY_PARAMETER_FIELDS if field_name not in run or run[field_name] is None
    ]
    if missing:
        raise FrozenCohortError(f"run is missing independent replay parameters: {', '.join(missing)}")

    text_values: dict[str, str] = {}
    for field_name in (
        "target_gene",
        "donor_obs_column",
        "var_gene_column",
        "deg_donor_column",
        "deg_gene_column",
        "deg_effect_column",
        "deg_fdr_column",
    ):
        value = run[field_name]
        if not isinstance(value, str) or not value.strip():
            raise FrozenCohortError(f"run independent replay parameter {field_name!r} must be non-empty")
        text_values[field_name] = value.strip()

    raw_fdr = run["fdr_threshold"]
    if isinstance(raw_fdr, bool):
        raise FrozenCohortError("run fdr_threshold must be numeric")
    try:
        fdr_threshold = float(raw_fdr)
    except (TypeError, ValueError) as exc:
        raise FrozenCohortError("run fdr_threshold must be numeric") from exc
    if not math.isfinite(fdr_threshold) or not 0.0 <= fdr_threshold <= 1.0:
        raise FrozenCohortError("run fdr_threshold must be within [0, 1]")

    integer_values: dict[str, int] = {}
    for field_name, minimum in (
        ("min_training_donors", 2),
        ("min_evaluable_donors", 3),
        ("top_k", 1),
        ("bootstrap_iterations", 1),
        ("bootstrap_seed", 0),
    ):
        value = run[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise FrozenCohortError(f"run {field_name} must be an integer >= {minimum}")
        integer_values[field_name] = value
    return {**text_values, "fdr_threshold": fdr_threshold, **integer_values}


def _independent_replay_parameter_issues(run: Mapping[str, Any]) -> list[str]:
    try:
        _independent_replay_options(run)
    except FrozenCohortError as exc:
        return [str(exc)]
    return []


def _compare_replay_values(expected: Any, actual: Any, label: str) -> list[str]:
    if isinstance(expected, Mapping) or isinstance(actual, Mapping):
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
            return [f"{label}: published and recomputed values have different types"]
        mismatches: list[str] = []
        for key in sorted(set(expected) | set(actual), key=str):
            if key not in expected:
                mismatches.append(f"{label}.{key}: missing from published value")
            elif key not in actual:
                mismatches.append(f"{label}.{key}: missing from recomputed value")
            else:
                mismatches.extend(_compare_replay_values(expected[key], actual[key], f"{label}.{key}"))
        return mismatches
    if isinstance(expected, (list, tuple)) or isinstance(actual, (list, tuple)):
        if not isinstance(expected, (list, tuple)) or not isinstance(actual, (list, tuple)):
            return [f"{label}: published and recomputed values have different types"]
        mismatches = []
        if len(expected) != len(actual):
            mismatches.append(f"{label}: published length {len(expected)} != recomputed length {len(actual)}")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=False)):
            mismatches.extend(_compare_replay_values(expected_item, actual_item, f"{label}[{index}]"))
        return mismatches
    if (
        isinstance(expected, Real)
        and not isinstance(expected, bool)
        and isinstance(actual, Real)
        and not isinstance(actual, bool)
    ):
        if not math.isclose(float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-12):
            return [f"{label}: published {expected!r} != recomputed {actual!r}"]
        return []
    if expected != actual:
        return [f"{label}: published {expected!r} != recomputed {actual!r}"]
    return []


def _independently_recompute_run(
    run: Mapping[str, Any],
    *,
    candidate: FrozenCandidate,
) -> tuple[Any, list[str]]:
    path_kind = str(run.get("path", ""))
    mode = str(run.get("mode", ""))
    seed = run.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FrozenCohortError("report replay run seed must be a non-negative integer")
    options = _independent_replay_options(run)
    required_published = ("path_result", "bootstrap_ci", "donor_scores", "baseline_matrix", "perturbed_matrix")
    missing = [field_name for field_name in required_published if field_name not in run]
    if missing:
        raise FrozenCohortError(f"report replay run is missing published independent statistics: {', '.join(missing)}")

    try:
        from .null_selection import load_null_distribution_manifest
        from .results import extract_path_result_from_perturbgen_h5ad

        output_h5ad = Path(str(run["output_h5ad"])).expanduser().resolve(strict=True)
        provenance = run["h5ad_provenance"]
        if not isinstance(provenance, Mapping):
            raise ValueError("h5ad_provenance must be a mapping")
        deg_path = Path(str(run["deg_table_path"])).expanduser().resolve(strict=True)
        deg_table = _read_deg_table(deg_path)
        null_path = Path(str(run["null_distribution_manifest_path"])).expanduser().resolve(strict=True)
        null_distribution = load_null_distribution_manifest(
            null_path,
            candidate_ensembl_id=candidate.ensembl_id,
            path_name=path_kind,
            mode=mode,
            seed=seed,
            required_count=candidate.matched_nulls,
        )
        inline_nulls = run["null_distribution"]
        if not isinstance(inline_nulls, list):
            raise ValueError("null_distribution must be a list")
        inline_values = [float(value) for value in inline_nulls]
        bound_values = [float(value) for value in null_distribution["values"]]
        if inline_values != bound_values:
            raise ValueError("inline null_distribution does not equal the bound null manifest")
        extraction = extract_path_result_from_perturbgen_h5ad(
            output_h5ad=output_h5ad,
            h5ad_provenance=provenance,
            deg_table=deg_table,
            null_distribution=bound_values,
            path=cast(Literal["source_intervention", "within_state"], path_kind),
            mode=cast(Literal["mask", "pad", "delete", "overexpress"], mode),
            seed=seed,
            donor_obs_column=options["donor_obs_column"],
            var_gene_column=options["var_gene_column"],
            target_gene=options["target_gene"],
            donor_column=options["deg_donor_column"],
            gene_column=options["deg_gene_column"],
            effect_column=options["deg_effect_column"],
            fdr_column=options["deg_fdr_column"],
            fdr_threshold=options["fdr_threshold"],
            min_training_donors=options["min_training_donors"],
            min_evaluable_donors=options["min_evaluable_donors"],
            top_k=options["top_k"],
            bootstrap_iterations=options["bootstrap_iterations"],
            bootstrap_seed=options["bootstrap_seed"],
        )
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        raise FrozenCohortError(f"independent H5AD/rescue replay failed: {exc}") from exc

    recomputed = {
        "path_result": asdict(extraction.path_result),
        "bootstrap_ci": list(extraction.bootstrap_ci) if extraction.bootstrap_ci is not None else None,
        "donor_scores": [asdict(score) for score in extraction.donor_scores],
        "baseline_matrix": extraction.baseline_matrix,
        "perturbed_matrix": extraction.perturbed_matrix,
    }
    published = {field_name: run[field_name] for field_name in required_published}
    mismatches: list[str] = []
    for field_name in required_published:
        mismatches.extend(_compare_replay_values(published[field_name], recomputed[field_name], field_name))
    return extraction, mismatches


def _validate_stage_manifest_binding(
    path_kind: str,
    mode: str,
    seed: int,
    output_h5ad: Path | None,
    stage_manifest_path: Path | None,
    stage_payload: Mapping[str, Any] | None,
) -> list[str]:
    """Cross-check the loaded stage manifest against the run declaration."""

    issues: list[str] = []
    if stage_payload is None:
        return issues
    if stage_payload.get("status") != "success":
        issues.append("stage_manifest status is not success")
    if stage_payload.get("stage") != path_kind:
        issues.append("stage_manifest stage does not match run path")
    artifacts = stage_payload.get("artifacts")
    outputs = stage_payload.get("outputs")
    if not isinstance(artifacts, Mapping) or not isinstance(outputs, Mapping):
        issues.append("stage_manifest must contain artifacts and outputs mappings")
    elif "result_h5ad" not in artifacts:
        issues.append("stage_manifest is missing result_h5ad")
    elif output_h5ad is not None:
        try:
            artifact_path = Path(str(artifacts["result_h5ad"])).expanduser()
            if not artifact_path.is_absolute():
                artifact_path = stage_manifest_path.parent / artifact_path  # type: ignore[union-attr]
            if artifact_path.resolve(strict=True) != output_h5ad:
                issues.append("stage_manifest result_h5ad does not match output_h5ad")
        except (OSError, ValueError) as exc:
            issues.append(f"stage_manifest result_h5ad is not readable: {exc}")

    material = stage_payload.get("fingerprint_material")
    fingerprint_config = material.get("fingerprint_config") if isinstance(material, Mapping) else None
    raw_stage_seed = material.get("random_seed") if isinstance(material, Mapping) else None
    if raw_stage_seed is None and isinstance(fingerprint_config, Mapping):
        raw_stage_seed = fingerprint_config.get("random_seed")
    try:
        stage_seed = int(raw_stage_seed) if raw_stage_seed is not None else -1
    except (TypeError, ValueError):
        stage_seed = -1
    if stage_seed != seed:
        issues.append("stage_manifest random_seed does not match run seed")

    stage_config = fingerprint_config.get("stage_config") if isinstance(fingerprint_config, Mapping) else None
    perturb_config = stage_config.get("perturb_config") if isinstance(stage_config, Mapping) else None
    trainer = perturb_config.get("trainer") if isinstance(perturb_config, Mapping) else None
    if not isinstance(trainer, Mapping):
        issues.append("stage_manifest is missing perturb trainer configuration")
    else:
        if str(trainer.get("perturbation_mode", "")).strip().lower() != mode:
            issues.append("stage trainer perturbation_mode does not match run mode")
        sequence = trainer.get("perturbation_sequence")
        expected_sequence = _PATH_TO_SEQUENCE[path_kind]
        if sequence != [expected_sequence]:
            issues.append("stage trainer perturbation_sequence does not match run path")
    return issues


def _validate_output_binding(
    candidate: FrozenCandidate,
    path_kind: str,
    mode: str,
    output_h5ad: Path | None,
    manifest: FrozenCohortManifest,
) -> list[str]:
    """Check the output h5ad filename binding and its donor coverage."""

    issues: list[str] = []
    if output_h5ad is None:
        return issues
    match = _H5AD_NAME_PATTERN.search(output_h5ad.name)
    expected_sequence = _PATH_TO_SEQUENCE[path_kind]
    if match is None:
        issues.append("output_h5ad name does not carry gene/sequence/mode binding")
    else:
        if match.group("gene").upper() != candidate.gene_symbol.upper():
            issues.append("output_h5ad gene does not match frozen candidate")
        if match.group("sequence") != expected_sequence:
            issues.append("output_h5ad sequence does not match frozen path")
        if match.group("mode").lower() != mode:
            issues.append("output_h5ad mode does not match run mode")
    issues.extend(_validate_output_donors(manifest, output_h5ad))
    return issues


def _validate_run_cohort_reference(
    manifest: FrozenCohortManifest,
    run: Mapping[str, Any],
    provenance: Any,
    eval_input: Mapping[str, Any],
) -> list[str]:
    """Validate the declared cohort reference, wherever the run declares it."""

    cohort_source = run if run.get("cohort_h5ad") is not None or run.get("cohort_sha256") is not None else None
    provenance_source = provenance if isinstance(provenance, Mapping) else None
    if cohort_source is not None:
        return _validate_declared_cohort_reference(manifest, cohort_source)
    if provenance_source is not None and (
        provenance_source.get("cohort_h5ad") is not None or provenance_source.get("cohort_sha256") is not None
    ):
        return _validate_declared_cohort_reference(manifest, provenance_source)
    if eval_input.get("cohort_h5ad") is not None or eval_input.get("cohort_sha256") is not None:
        return _validate_declared_cohort_reference(manifest, eval_input)
    return []


def _validate_bound_null_distribution(
    candidate: FrozenCandidate,
    path_kind: str,
    mode: str,
    seed: int,
    run: Mapping[str, Any],
) -> list[str]:
    """Verify inline nulls equal the manifest-bound null distribution."""

    issues: list[str] = []
    if run.get("null_distribution_path") is not None:
        issues.append("legacy unbound null_distribution_path cannot satisfy frozen verification")
    null_path = run.get("null_distribution_manifest_path")
    inline_nulls = run.get("null_distribution")
    if null_path is None or not str(null_path).strip():
        issues.append("run is missing null_distribution_manifest_path")
        return issues
    if not isinstance(inline_nulls, list):
        issues.append("run is missing manifest-bound null_distribution values")
        return issues
    try:
        from .null_selection import load_null_distribution_manifest

        distribution = load_null_distribution_manifest(
            str(null_path),
            candidate_ensembl_id=candidate.ensembl_id,
            path_name=path_kind,
            mode=mode,
            seed=seed,
            required_count=candidate.matched_nulls,
        )
        actual_values = [float(value) for value in inline_nulls]
        expected_values = [float(value) for value in distribution["values"]]
        if actual_values != expected_values:
            issues.append("inline null_distribution does not equal the bound null manifest")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        issues.append(f"bound null distribution is invalid: {exc}")
    return issues


def _validate_frozen_run(
    manifest: FrozenCohortManifest,
    candidate: FrozenCandidate,
    run: Mapping[str, Any],
    *,
    eval_input: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []
    path_kind = str(run.get("path", ""))
    mode = str(run.get("mode", ""))
    try:
        seed = int(run.get("seed", -1))
    except (TypeError, ValueError):
        seed = -1

    declared_route = run.get("intervention_type")
    if declared_route is not None and str(declared_route).strip().upper() != candidate.intervention_type:
        issues.append("run intervention_type does not match frozen DAVF route")
    if str(run.get("target_gene", "")).strip() != candidate.gene_symbol:
        issues.append("run target_gene does not match frozen candidate")
    if str(run.get("donor_obs_column", "")).strip() != manifest.donor_obs_column:
        issues.append("run donor_obs_column does not match frozen cohort")

    raw_output = run.get("output_h5ad")
    output_h5ad: Path | None = None
    if raw_output is None or not str(raw_output).strip():
        issues.append("run is missing output_h5ad")
    else:
        try:
            output_h5ad = Path(str(raw_output)).expanduser().resolve(strict=True)
        except (OSError, ValueError) as exc:
            issues.append(f"output_h5ad is not readable: {exc}")

    provenance = run.get("h5ad_provenance")
    stage_manifest_path: Path | None = None
    stage_payload: Mapping[str, Any] | None = None
    if not isinstance(provenance, Mapping):
        issues.append("run is missing h5ad_provenance")
    else:
        if "sha256" not in provenance or not str(provenance.get("sha256", "")).strip():
            issues.append("h5ad_provenance is missing sha256 for independent replay")
        raw_stage_manifest = provenance.get("stage_manifest")
        if raw_stage_manifest is None or not str(raw_stage_manifest).strip():
            issues.append("h5ad_provenance is missing stage_manifest")
        else:
            try:
                stage_manifest_path = Path(str(raw_stage_manifest)).expanduser().resolve(strict=True)
                loaded = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
                if not isinstance(loaded, Mapping):
                    raise ValueError("stage manifest root must be an object")
                stage_payload = loaded
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                issues.append(f"stage_manifest cannot be read: {exc}")

    issues.extend(
        _validate_stage_manifest_binding(path_kind, mode, seed, output_h5ad, stage_manifest_path, stage_payload)
    )
    issues.extend(_validate_output_binding(candidate, path_kind, mode, output_h5ad, manifest))
    issues.extend(_validate_run_cohort_reference(manifest, run, provenance, eval_input))
    issues.extend(_validate_stage_lineage(manifest, stage_manifest_path, stage_payload, run, provenance))
    issues.extend(_independent_replay_parameter_issues(run))
    issues.extend(_validate_deg_training_donors(manifest, run))
    issues.extend(_validate_bound_null_distribution(candidate, path_kind, mode, seed, run))
    return issues


def verify_eval_input_against_manifest(
    manifest: FrozenCohortManifest,
    eval_input: Mapping[str, Any],
    *,
    require_formal: bool = False,
) -> dict[str, Any]:
    """Check coverage and bind every formal evaluation artifact to the freeze.

    The evaluator's signature function intentionally uses all donors present in
    its DEG table except the current held-out donor.  A frozen run therefore
    must provide DEG rows from the fixed training donors only and h5ad rows from
    the fixed held-out donors only; otherwise another held-out donor can leak
    into the signature before the numerical verdict is replayed.
    """

    if eval_input.get("schema_version") != _EVAL_INPUT_SCHEMA_VERSION:
        raise FrozenCohortError(
            f"unsupported eval input schema_version {eval_input.get('schema_version')!r}; "
            f"expected {_EVAL_INPUT_SCHEMA_VERSION!r}"
        )
    candidates_field = eval_input.get("candidates")
    if not isinstance(candidates_field, list) or not candidates_field:
        raise FrozenCohortError("eval input has no candidates")

    evaluation_mode = _evaluation_mode(eval_input)
    eval_input_issues = _validate_eval_input_lineage(eval_input)
    if require_formal and evaluation_mode != "formal":
        eval_input_issues.insert(0, "formal frozen verification requires evaluation_mode='formal'")
    cohort_ok, cohort_details, cohort_issues = _validate_frozen_cohort_asset(manifest)
    top_cohort_issues = _validate_declared_cohort_reference(manifest, eval_input)
    pairing_issues = _validate_declared_pairing(manifest, eval_input)
    cohort_issues.extend(top_cohort_issues)
    cohort_issues.extend(pairing_issues)
    cohort_issues.extend(eval_input_issues)
    cohort_ok = cohort_ok and not top_cohort_issues and not pairing_issues and not eval_input_issues

    manifest_candidates = {candidate.ensembl_id: candidate for candidate in manifest.candidates}
    by_ensembl: dict[str, dict[str, set[tuple[str, int]]]] = {}
    invalid_runs: list[dict[str, Any]] = []
    duplicate_candidates: list[str] = []
    seen_candidates: set[str] = set()
    for entry in candidates_field:
        if not isinstance(entry, Mapping):
            raise FrozenCohortError("eval input candidate entries must be objects")
        candidate = entry.get("candidate") or {}
        if not isinstance(candidate, Mapping):
            raise FrozenCohortError("eval input candidate must be an object")
        ensembl_id = str(candidate.get("ensembl_id", "")).strip()
        if not ensembl_id:
            raise FrozenCohortError("eval input candidate is missing ensembl_id")
        if ensembl_id in seen_candidates:
            duplicate_candidates.append(ensembl_id)
        seen_candidates.add(ensembl_id)
        runs = entry.get("runs")
        if not isinstance(runs, list) or not runs:
            raise FrozenCohortError(f"eval input candidate {ensembl_id} has no runs")
        frozen_candidate = manifest_candidates.get(ensembl_id)
        candidate_issues: list[str] = []
        if frozen_candidate is None:
            candidate_issues.append("candidate_not_in_frozen_manifest")
        else:
            if str(candidate.get("gene_symbol", "")).strip() != frozen_candidate.gene_symbol:
                candidate_issues.append("candidate_gene_symbol_mismatch")
            if str(candidate.get("intervention_type", "")).strip().upper() != frozen_candidate.intervention_type:
                candidate_issues.append("candidate_route_mismatch")
        combos: dict[str, set[tuple[str, int]]] = {}
        seen_runs: set[tuple[str, str, int]] = set()
        for run in runs:
            if not isinstance(run, Mapping):
                raise FrozenCohortError("eval input runs must be objects")
            path_kind = str(run.get("path", ""))
            mode = str(run.get("mode", ""))
            try:
                seed = int(run.get("seed", -1))
            except (TypeError, ValueError) as exc:
                raise FrozenCohortError("eval input run seed must be an integer") from exc
            if path_kind not in _VALID_PATHS:
                raise FrozenCohortError(f"eval input run has unknown path {path_kind!r}")
            if mode not in _VALID_MODES:
                raise FrozenCohortError(f"eval input run has unknown mode {mode!r}")
            if seed < 0:
                raise FrozenCohortError("eval input run seed must be >= 0")
            identity = (path_kind, mode, seed)
            if identity in seen_runs:
                candidate_issues.append(f"duplicate_run:{path_kind}:{mode}:seed{seed}")
            seen_runs.add(identity)
            combos.setdefault(path_kind, set()).add((mode, seed))
            if frozen_candidate is not None:
                run_issues = _validate_frozen_run(
                    manifest,
                    frozen_candidate,
                    run,
                    eval_input=eval_input,
                )
                if run_issues:
                    invalid_runs.append(
                        {
                            "ensembl_id": ensembl_id,
                            "path": path_kind,
                            "mode": mode,
                            "seed": seed,
                            "issues": run_issues,
                        }
                    )
        if candidate_issues:
            invalid_runs.append({"ensembl_id": ensembl_id, "issues": candidate_issues})
        by_ensembl[ensembl_id] = combos

    missing: list[str] = []
    for candidate in manifest.candidates:
        seen = by_ensembl.get(candidate.ensembl_id)
        if seen is None:
            missing.append(f"{candidate.ensembl_id}:candidate_absent")
            continue
        for path_kind in _VALID_PATHS:
            planned = {(mode, seed) for mode in candidate.modes for seed in candidate.seeds}
            actual = seen.get(path_kind, set())
            gap = planned - actual
            for mode, seed in sorted(gap):
                missing.append(f"{candidate.ensembl_id}:{path_kind}:{mode}:seed{seed}")
    extra_candidates = sorted(set(by_ensembl) - set(manifest_candidates))
    extra_runs: list[str] = []
    for ensembl_id, combos in by_ensembl.items():
        candidate = manifest_candidates.get(ensembl_id)
        if candidate is None:
            continue
        planned_runs = {
            (path_kind, mode, seed)
            for path_kind in _VALID_PATHS
            for mode in candidate.modes
            for seed in candidate.seeds
        }
        actual_runs = {(path_kind, mode, seed) for path_kind, values in combos.items() for mode, seed in values}
        for path_kind, mode, seed in sorted(actual_runs - planned_runs):
            extra_runs.append(f"{ensembl_id}:{path_kind}:{mode}:seed{seed}")
    missing.extend(f"{ensembl_id}:extra_candidate" for ensembl_id in extra_candidates)
    missing.extend(f"{identity}:extra_run" for identity in extra_runs)
    formal_plan_issues = [
        f"{candidate.ensembl_id}:{candidate.intervention_type} mode plan is not formally complete"
        for candidate in manifest.candidates
        if not candidate.formal_plan_complete
    ]
    covered = cohort_ok and not missing and not invalid_runs and not duplicate_candidates
    formal_plan_complete = not formal_plan_issues
    return {
        "schema_version": "ptm2cellnet.frozen-acceptance-verification/v1",
        "cohort_h5ad": manifest.cohort_h5ad,
        "cohort_sha256": manifest.cohort_sha256,
        "pairing": manifest.pairing,
        "donor_leakage_audited": True,
        "cohort_valid": cohort_ok,
        "cohort_details": cohort_details,
        "cohort_issues": cohort_issues,
        "eval_input_issues": eval_input_issues,
        "evaluation_mode": evaluation_mode,
        "evidence_class": eval_input.get("evidence_class"),
        "pvalue_source": eval_input.get("pvalue_source"),
        "n_manifest_candidates": len(manifest.candidates),
        "n_eval_candidates": len(by_ensembl),
        "missing_runs": missing,
        "invalid_runs": invalid_runs,
        "duplicate_candidates": duplicate_candidates,
        "formal_plan_complete": formal_plan_complete,
        "formal_plan_issues": formal_plan_issues,
        "covered": covered,
        "contract_eligible": covered and formal_plan_complete,
    }


def replay_verdicts(
    report_manifest: Mapping[str, Any],
    *,
    manifest: FrozenCohortManifest,
    eval_input: Mapping[str, Any],
    require_formal: bool = False,
) -> dict[str, Any]:
    """Independently recompute the dual-path verdicts of a produced report.

    The report manifest (written by ``evaluate_perturbgen_dual_path.py``)
    stores published statistics.  The supplied evaluation input is the only
    source for replay extraction inputs; report replay blocks are declarations
    checked against it.  Formal candidate p-values and BH q-values are rebuilt
    from independently extracted path results.
    """

    from .results import benjamini_hochberg

    candidates_field = report_manifest.get("candidates")
    if not isinstance(candidates_field, list) or not candidates_field:
        raise FrozenCohortError("report manifest has no candidates")

    if not isinstance(eval_input, Mapping):
        raise FrozenCohortError("eval_input must be a mapping")
    require_formal = require_formal or str(report_manifest.get("evaluation_mode", "")).strip().lower() == "formal"
    binding = verify_eval_input_against_manifest(manifest, eval_input, require_formal=require_formal)
    declaration_mismatches = _validate_report_declaration(report_manifest, eval_input)
    input_mismatches = _validate_report_against_eval_input(report_manifest, eval_input)
    if not binding["contract_eligible"] or declaration_mismatches or input_mismatches:
        binding_mismatches = [*declaration_mismatches, *input_mismatches]
        if not binding["contract_eligible"]:
            binding_mismatches.append("report evidence is not formally bound to the frozen cohort")
        return {
            "schema_version": "ptm2cellnet.frozen-acceptance-replay/v1",
            "n_candidates": len(candidates_field),
            "mismatches": binding_mismatches,
            "reproduced": False,
            "independent_h5ad_recomputed": False,
            "binding": binding,
            "candidates": [],
        }

    manifest_candidates = {candidate.ensembl_id: candidate for candidate in manifest.candidates}
    evaluation_mode = _evaluation_mode(eval_input)
    evidence_class = str(eval_input.get("evidence_class") or "unspecified").strip()
    pvalue_source = str(eval_input.get("pvalue_source") or "unspecified").strip()
    eval_candidates_by_id: dict[str, Mapping[str, Any]] = {}
    for raw_entry in eval_input["candidates"]:
        if not isinstance(raw_entry, Mapping):
            raise FrozenCohortError("eval input candidate entries must be objects")
        raw_candidate = raw_entry.get("candidate")
        if not isinstance(raw_candidate, Mapping):
            raise FrozenCohortError("eval input candidate must be an object")
        ensembl_id = str(raw_candidate.get("ensembl_id", "")).strip()
        if not ensembl_id:
            raise FrozenCohortError("eval input candidate is missing ensembl_id")
        eval_candidates_by_id[ensembl_id] = raw_entry

    mismatches: list[str] = []
    recomputed: list[dict[str, Any]] = []
    pvalues: list[float] = []
    for entry in candidates_field:
        if not isinstance(entry, Mapping):
            raise FrozenCohortError("report manifest candidate entries must be objects")
        replay = entry.get("replay")
        if not isinstance(replay, Mapping):
            raise FrozenCohortError("report manifest entry is missing its replay block")
        runs = replay.get("runs")
        if not isinstance(runs, list) or not runs:
            raise FrozenCohortError("report manifest replay block has no runs")
        replay_candidate = replay.get("candidate")
        if not isinstance(replay_candidate, Mapping):
            raise FrozenCohortError("report manifest replay is missing its candidate route")
        route = replay_candidate.get("intervention_type")
        if route is None or not str(route).strip():
            raise FrozenCohortError("report manifest candidate is missing intervention_type")
        candidate_gene = str(entry.get("candidate_gene", "")).strip()
        replay_gene = str(replay_candidate.get("gene_symbol", "")).strip()
        if not candidate_gene or not replay_gene or candidate_gene != replay_gene:
            raise FrozenCohortError("report manifest candidate_gene does not match replay candidate")
        candidate_ensembl = str(replay_candidate.get("ensembl_id", "")).strip()
        if not candidate_ensembl:
            raise FrozenCohortError("report manifest replay candidate is missing ensembl_id")
        eval_entry = eval_candidates_by_id.get(candidate_ensembl)
        if eval_entry is None:
            raise FrozenCohortError("report manifest replay candidate is absent from the eval input")
        frozen_candidate = manifest_candidates.get(candidate_ensembl)
        if frozen_candidate is None:
            raise FrozenCohortError("report manifest replay candidate is absent from the frozen manifest")
        eval_runs = eval_entry.get("runs")
        if not isinstance(eval_runs, list) or not eval_runs:
            raise FrozenCohortError("eval input replay block has no runs")
        report_runs_by_identity = {_run_identity(run): run for run in runs if isinstance(run, Mapping)}
        recomputed_path_results: list[Mapping[str, Any]] = []
        run_mismatches: list[str] = []
        for index, eval_run in enumerate(eval_runs):
            if not isinstance(eval_run, Mapping):
                raise FrozenCohortError("eval input runs must be objects")
            report_run = report_runs_by_identity.get(_run_identity(eval_run))
            if report_run is None:
                raise FrozenCohortError(
                    f"report manifest is missing eval input run {_run_identity(eval_run)} for {candidate_ensembl}"
                )
            replay_run = dict(eval_run)
            for field_name in ("path_result", "bootstrap_ci", "donor_scores", "baseline_matrix", "perturbed_matrix"):
                if field_name not in report_run:
                    raise FrozenCohortError(
                        f"report replay run is missing published independent statistic {field_name}"
                    )
                replay_run[field_name] = report_run[field_name]
            extraction, statistics_mismatches = _independently_recompute_run(
                replay_run,
                candidate=frozen_candidate,
            )
            recomputed_path_results.append(asdict(extraction.path_result))
            run_mismatches.extend(f"run[{index}] {mismatch}" for mismatch in statistics_mismatches)
        mismatches.extend(f"{candidate_gene}: {mismatch}" for mismatch in run_mismatches)

        observed_direction = str(eval_entry.get("observed_direction", "")).strip()
        intervention_type = str(replay_candidate.get("intervention_type", "")).strip().upper()
        if evaluation_mode == "formal":
            try:
                aggregation = aggregate_candidate_empirical_pvalues(
                    recomputed_path_results,
                    intervention_type,
                    observed_direction,
                    ensembl_id=candidate_ensembl,
                )
            except EmpiricalPvalueError as exc:
                raise FrozenCohortError(f"formal empirical p aggregation failed: {exc}") from exc
            candidate_pvalue = float(aggregation["pvalue"])
        else:
            if "candidate_pvalue" not in entry:
                raise FrozenCohortError("report manifest candidate is missing candidate_pvalue")
            candidate_pvalue = float(entry["candidate_pvalue"])
        pvalues.append(candidate_pvalue)
        recomputed.append(
            {
                "entry": entry,
                "replay": replay,
                "candidate_gene": candidate_gene,
                "candidate_ensembl": candidate_ensembl,
                "intervention_type": intervention_type,
                "observed_direction": observed_direction,
                "unperturbed_quality_status": str(eval_entry.get("unperturbed_quality_status", "")),
                "path_results": recomputed_path_results,
                "run_mismatches": run_mismatches,
                "candidate_pvalue": candidate_pvalue,
            }
        )

    q_values = benjamini_hochberg(pvalues)
    replayed: list[dict[str, Any]] = []
    for record, q_value in zip(recomputed, q_values, strict=True):
        entry = record["entry"]
        candidate_gene = record["candidate_gene"]
        if "candidate_pvalue" not in entry or "q_value" not in entry or "verdict" not in entry:
            raise FrozenCohortError("report manifest candidate is missing candidate_pvalue, q_value or verdict")
        mismatches.extend(
            _compare_replay_values(
                entry["candidate_pvalue"], record["candidate_pvalue"], f"{candidate_gene}.candidate_pvalue"
            )
        )
        mismatches.extend(_compare_replay_values(entry["q_value"], q_value, f"{candidate_gene}.q_value"))
        decision = evaluate_dual_path_candidate(
            record["path_results"],
            observed_direction=record["observed_direction"],
            q_value=q_value,
            candidate_gene=candidate_gene,
            intervention_type=record["intervention_type"],
            unperturbed_quality_status=record["unperturbed_quality_status"],
            evaluation_mode=evaluation_mode,
            evidence_class=evidence_class,
            pvalue_source=pvalue_source,
        )
        replayed.append(
            {
                "candidate_gene": candidate_gene,
                "published_verdict": entry["verdict"],
                "replayed_verdict": decision.verdict,
                "published_candidate_pvalue": entry["candidate_pvalue"],
                "replayed_candidate_pvalue": record["candidate_pvalue"],
                "replayed_q_value": q_value,
                "replayed_reasons": list(decision.reasons),
                "statistics_mismatches": record["run_mismatches"],
            }
        )
        if decision.verdict != entry["verdict"]:
            mismatches.append(f"{candidate_gene}: published {entry['verdict']!r} vs replayed {decision.verdict!r}")
    return {
        "schema_version": "ptm2cellnet.frozen-acceptance-replay/v1",
        "n_candidates": len(candidates_field),
        "mismatches": mismatches,
        "reproduced": not mismatches,
        "independent_h5ad_recomputed": True,
        "candidates": replayed,
        "binding": binding,
        "report_declaration": {
            "input_json": report_manifest.get("input_json"),
            "input_sha256": report_manifest.get("input_sha256"),
            "contract": report_manifest.get("contract"),
        },
    }


def _load_candidates_csv(path: str | Path) -> list[dict[str, str]]:
    resolved = Path(path).expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ("gene_symbol", "ensembl_id", "intervention_type")
        missing = [column for column in required if column not in (reader.fieldnames or [])]
        if missing:
            raise FrozenCohortError(f"candidates CSV is missing columns: {', '.join(missing)}")
        rows = [{key: str(value or "").strip() for key, value in row.items() if key} for row in reader]
    if not rows:
        raise FrozenCohortError("candidates CSV must contain at least one row")
    return rows
