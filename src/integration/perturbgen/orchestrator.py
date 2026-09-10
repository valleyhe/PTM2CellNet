"""Strict serial DAVF -> PerturbGen orchestration.

The two models have different jobs and different coordinate systems:

* DAVF receives a verified PerturbGen token and an explicit KO/KD route, then
  predicts the sign of the target gene expression change.
* the direction gate decides whether the candidate is allowed to enter
  PerturbGen;
* PerturbGen receives the gene symbol as its own external-package target and
  runs the two utility paths in isolated output directories.

This module only builds the bridge and stage plans.  It never imports the
external ``perturbgen`` package and never turns an incomplete run into a
biological PASS.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
import json
from numbers import Integral
from pathlib import Path
from typing import Any, Literal

from src.models.gene_vocabulary import normalize_ensembl_id, normalize_gene_symbol
from src.models.ptm_direction_mapper import PTMDirectionMapper, PTMDirectionMapperOutput

from .config_builder import StagePlan, build_stage_plans, load_pipeline_config
from .contracts import (
    CandidateEvidence,
    DAVFDirectionEvidence,
    DirectionGateResult,
    DirectionGateStatus,
    DavfAction,
    ObservedDirection,
    PTMSiteDirectionProposal,
    PathKind,
    PerturbationMode,
)
from .direction_gate import build_direction_gated_candidate


InterventionType = Literal["KO", "KD"]
_INTERVENTION_TYPES = {"KO", "KD"}
_PATHS: tuple[PathKind, PathKind] = ("source_intervention", "within_state")
_ACTION_TO_MODE: dict[DavfAction, PerturbationMode] = {
    "ko": "mask",
    "oe": "overexpress",
}


class DAVFPerturbGenE2EError(RuntimeError):
    """Raised when the serial DAVF/PerturbGen contract is incomplete."""


@dataclass(frozen=True)
class PerturbGenInvocation:
    """A fully validated candidate allowed to enter PerturbGen."""

    intervention_type: InterventionType
    gene_symbol: str
    ensembl_id: str
    target_token_id: int
    perturbation_mode: PerturbationMode
    paths: tuple[PathKind, ...]
    candidate: CandidateEvidence
    davf_evidence: DAVFDirectionEvidence
    perturbgen_config_path: Path | None = None
    output_root: Path | None = None

    def __post_init__(self) -> None:
        route = str(self.intervention_type).strip().upper()
        if route not in _INTERVENTION_TYPES:
            raise ValueError("intervention_type must be KO or KD")
        object.__setattr__(self, "intervention_type", route)
        object.__setattr__(self, "gene_symbol", normalize_gene_symbol(self.gene_symbol))
        object.__setattr__(self, "ensembl_id", normalize_ensembl_id(self.ensembl_id))
        if (
            isinstance(self.target_token_id, bool)
            or not isinstance(self.target_token_id, Integral)
            or self.target_token_id < 0
        ):
            raise ValueError("target_token_id must be a non-negative integer")
        if self.perturbation_mode not in {"mask", "overexpress"}:
            raise ValueError("formal DAVF/PerturbGen bridge supports mask or overexpress actions")
        paths = tuple(self.paths)
        if not paths:
            raise ValueError("at least one PerturbGen path is required")
        if len(set(paths)) != len(paths) or any(path not in _PATHS for path in paths):
            raise ValueError("paths must be unique source_intervention/within_state values")
        object.__setattr__(self, "paths", paths)
        if self.candidate.gene_symbol != self.gene_symbol:
            raise ValueError("candidate gene_symbol does not match invocation")
        if self.candidate.ensembl_id != self.ensembl_id:
            raise ValueError("candidate ensembl_id does not match invocation")
        if self.davf_evidence.gene_symbol != self.gene_symbol:
            raise ValueError("DAVF evidence gene_symbol does not match invocation")
        if self.davf_evidence.ensembl_id != self.ensembl_id:
            raise ValueError("DAVF evidence ensembl_id does not match invocation")
        if self.perturbgen_config_path is not None:
            object.__setattr__(
                self,
                "perturbgen_config_path",
                Path(self.perturbgen_config_path).expanduser().resolve(),
            )
        if self.output_root is not None:
            object.__setattr__(self, "output_root", Path(self.output_root).expanduser().resolve())

    def to_dict(self) -> dict[str, Any]:
        """Serialize the invocation without tensors or external objects."""

        return {
            "intervention_type": self.intervention_type,
            "gene_symbol": self.gene_symbol,
            "ensembl_id": self.ensembl_id,
            "target_token_id": self.target_token_id,
            "perturbation_mode": self.perturbation_mode,
            "paths": list(self.paths),
            "candidate": _to_plain(self.candidate),
            "davf_evidence": _to_plain(self.davf_evidence),
            "perturbgen_config_path": (
                str(self.perturbgen_config_path)
                if self.perturbgen_config_path is not None
                else None
            ),
            "output_root": str(self.output_root) if self.output_root is not None else None,
        }


@dataclass(frozen=True)
class DAVFPerturbGenPreparation:
    """Result of DAVF inference and direction gating for one candidate."""

    intervention_type: InterventionType
    mapper_output: PTMDirectionMapperOutput
    davf_evidence: DAVFDirectionEvidence
    direction_gate: DirectionGateResult
    candidate: CandidateEvidence | None
    invocation: PerturbGenInvocation | None

    @property
    def status(self) -> DirectionGateStatus:
        return self.direction_gate.status

    def to_dict(self) -> dict[str, Any]:
        return {
            "intervention_type": self.intervention_type,
            "status": self.status,
            "davf_evidence": _to_plain(self.davf_evidence),
            "direction_gate": _to_plain(self.direction_gate),
            "candidate": _to_plain(self.candidate),
            "invocation": self.invocation.to_dict() if self.invocation is not None else None,
        }


class DAVFPerturbGenOrchestrator:
    """Run the strict DAVF direction gate and build downstream invocations."""

    def __init__(
        self,
        *,
        davf_module: Any,
        mapper: PTMDirectionMapper | None = None,
    ) -> None:
        config = getattr(davf_module, "config", None)
        configured_route = getattr(config, "intervention_type", None)
        route = str(configured_route or "").strip().upper()
        if route not in _INTERVENTION_TYPES:
            raise DAVFPerturbGenE2EError(
                "KO/KD E2E orchestration requires DAVF config.intervention_type="
                "'KO' or 'KD'"
            )
        self.davf_module = davf_module
        self.intervention_type: InterventionType = route  # type: ignore[assignment]
        self.mapper = mapper or davf_module.build_perturbgen_direction_mapper()
        if not isinstance(self.mapper, PTMDirectionMapper):
            raise TypeError("mapper must be a PTMDirectionMapper")
        if self.mapper.gene_to_idx is None:
            raise DAVFPerturbGenE2EError(
                "formal DAVF/PerturbGen orchestration requires the verified "
                "PerturbGen embedding vocabulary"
            )

    def prepare_candidates(
        self,
        proposals: Sequence[PTMSiteDirectionProposal],
        z_0: Any,
        *,
        cell_type: str | Sequence[str],
        ptm_context: str | Sequence[str],
        observed_log2fc: float | Sequence[float],
        observed_fdr: float | Sequence[float] | None,
        observed_direction: ObservedDirection | Sequence[ObservedDirection | None] | None,
        scvi_adapter: Any | None = None,
        scvi_context: Any | None = None,
        library_size: float | None = None,
        direction_epsilon: float = 0.0,
        n_samples: int = 1,
        perturbgen_config_path: str | Path | None = None,
        output_root: str | Path | None = None,
    ) -> tuple[DAVFPerturbGenPreparation, ...]:
        """Infer DAVF directions and gate a batch of PTM candidates.

        ``z_0`` and the optional scVI context must already be aligned row by
        row with ``proposals``.  No target decoder indices are accepted here;
        :meth:`DAVFInferenceModule.predict_expression_direction` resolves
        them from the live adapter's ordered ``gene_names``.
        """

        proposal_batch = tuple(proposals)
        if not proposal_batch:
            raise ValueError("proposals must not be empty")
        if any(not isinstance(item, PTMSiteDirectionProposal) for item in proposal_batch):
            raise TypeError("proposals must contain PTMSiteDirectionProposal objects")
        batch_size = len(proposal_batch)
        cell_types = _as_batch(cell_type, batch_size, "cell_type")
        contexts = _as_batch(ptm_context, batch_size, "ptm_context")
        log2fcs = _as_batch(observed_log2fc, batch_size, "observed_log2fc")
        fdrs = _as_batch(observed_fdr, batch_size, "observed_fdr")
        directions = _as_batch(observed_direction, batch_size, "observed_direction")

        for proposal in proposal_batch:
            self._validate_target_identity(proposal)

        mapper_output = self.mapper.map_intervention_batch(
            [[proposal.gene_symbol] for proposal in proposal_batch],
            self.intervention_type,
        )
        evidence = self.davf_module.predict_expression_direction(
            mapper_output,
            z_0,
            scvi_adapter,
            target_gene_symbols=[proposal.gene_symbol for proposal in proposal_batch],
            target_ensembl_ids=[proposal.ensembl_id for proposal in proposal_batch],
            library_size=library_size,
            scvi_context=scvi_context,
            direction_epsilon=direction_epsilon,
            n_samples=n_samples,
        )
        if len(evidence) != batch_size:
            raise DAVFPerturbGenE2EError(
                "DAVF returned a different number of evidence rows than proposals"
            )

        preparations: list[DAVFPerturbGenPreparation] = []
        for row, (proposal, davf_evidence) in enumerate(
            zip(proposal_batch, evidence, strict=True)
        ):
            if not isinstance(davf_evidence, DAVFDirectionEvidence):
                raise TypeError("DAVF direction output must contain DAVFDirectionEvidence")
            gate, candidate = build_direction_gated_candidate(
                proposal,
                davf_evidence,
                cell_type=cell_types[row],
                ptm_context=contexts[row],
                observed_log2fc=log2fcs[row],
                observed_fdr=fdrs[row],
                observed_direction=directions[row],
            )
            invocation = None
            if gate.status == "pass":
                if candidate is None or candidate.davf_action is None:
                    raise DAVFPerturbGenE2EError(
                        "passing direction gate did not produce a downstream action"
                    )
                try:
                    perturbation_mode = _ACTION_TO_MODE[candidate.davf_action]
                except KeyError as exc:
                    raise DAVFPerturbGenE2EError(
                        f"unsupported downstream PerturbGen action: {candidate.davf_action!r}"
                    ) from exc
                invocation = PerturbGenInvocation(
                    intervention_type=self.intervention_type,
                    gene_symbol=proposal.gene_symbol,
                    ensembl_id=proposal.ensembl_id,
                    target_token_id=int(mapper_output.gene_ids[row, 0].item()),
                    perturbation_mode=perturbation_mode,
                    paths=_PATHS,
                    candidate=candidate,
                    davf_evidence=davf_evidence,
                    perturbgen_config_path=(
                        Path(perturbgen_config_path)
                        if perturbgen_config_path is not None
                        else None
                    ),
                    output_root=Path(output_root) if output_root is not None else None,
                )
            preparations.append(
                DAVFPerturbGenPreparation(
                    intervention_type=self.intervention_type,
                    mapper_output=_select_mapper_row(mapper_output, row),
                    davf_evidence=davf_evidence,
                    direction_gate=gate,
                    candidate=candidate,
                    invocation=invocation,
                )
            )
        return tuple(preparations)

    def prepare_candidate(
        self,
        proposal: PTMSiteDirectionProposal,
        z_0: Any,
        **kwargs: Any,
    ) -> DAVFPerturbGenPreparation:
        """Single-candidate convenience wrapper around ``prepare_candidates``."""

        preparations = self.prepare_candidates([proposal], z_0, **kwargs)
        if len(preparations) != 1:
            raise DAVFPerturbGenE2EError("single-candidate preparation returned an invalid result count")
        return preparations[0]

    def run_perturbgen(
        self,
        invocation: PerturbGenInvocation,
        config_or_path: str | Path | Mapping[str, Any],
        *,
        runner: Any,
        output_root: str | Path | None = None,
        resume: bool = False,
        dry_run: bool = False,
        project_root: str | Path | None = None,
    ) -> list[Any]:
        """Execute the isolated PerturbGen stages for one gated candidate.

        The runner still owns environment checks, GPU locking, manifests and
        resume fingerprints.  This method only materializes the candidate
        target and the two path-specific perturb plans.
        """

        if invocation.intervention_type != self.intervention_type:
            raise DAVFPerturbGenE2EError(
                "invocation route does not match the loaded DAVF route: "
                f"{invocation.intervention_type} != {self.intervention_type}"
            )
        plans = build_candidate_stage_plans(
            config_or_path,
            invocation,
            output_root=output_root,
            project_root=project_root,
        )
        return runner.run_pipeline(plans, resume=resume, dry_run=dry_run)

    def _validate_target_identity(self, proposal: PTMSiteDirectionProposal) -> None:
        """Reject a symbol/Ensembl pair that disagrees with the verified asset."""

        aliases = getattr(self.davf_module, "_embedding_symbol_to_ensembl", {})
        if not isinstance(aliases, Mapping) or not aliases:
            raise DAVFPerturbGenE2EError(
                "formal DAVF target identity requires the configured Ensembl-to-symbol "
                "alias asset"
            )
        expected = aliases.get(proposal.gene_symbol)
        if expected is None:
            raise KeyError(
                f"gene symbol {proposal.gene_symbol!r} is absent from the verified DAVF alias asset"
            )
        if normalize_ensembl_id(str(expected)) != proposal.ensembl_id:
            raise ValueError(
                "PTM proposal symbol/Ensembl pair does not match the verified DAVF alias asset: "
                f"{proposal.gene_symbol} -> {expected}, not {proposal.ensembl_id}"
            )


def materialize_candidate_config(
    config: Mapping[str, Any],
    invocation: PerturbGenInvocation,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a candidate-specific copy of a resolved six-stage config."""

    materialized = deepcopy(dict(config))
    try:
        pipeline = materialized["pipeline"]
        stages = materialized["stages"]
        perturb_stage = stages["perturb"]
        perturb_config = perturb_stage["perturb_config"]
        trainer = perturb_config["trainer"]
    except (KeyError, TypeError) as exc:
        raise ValueError("PerturbGen config is missing stages.perturb.perturb_config.trainer") from exc
    if not isinstance(pipeline, dict) or not isinstance(stages, dict) or not isinstance(trainer, dict):
        raise TypeError("PerturbGen config sections must be mappings")
    if not isinstance(perturb_stage, dict) or not isinstance(perturb_config, dict):
        raise TypeError("stages.perturb.perturb_config must be mappings")
    old_targets = trainer.get("genes_to_perturb")
    if old_targets is None:
        old_target = None
    elif (
        not isinstance(old_targets, Sequence)
        or isinstance(old_targets, (str, bytes))
        or len(old_targets) != 1
    ):
        raise ValueError("candidate materialization requires at most one base genes_to_perturb target")
    else:
        old_target = str(old_targets[0])
        if not old_target:
            raise ValueError("base genes_to_perturb target must not be empty")
    trainer["genes_to_perturb"] = [invocation.gene_symbol]
    trainer["perturbation_mode"] = invocation.perturbation_mode
    old_root_value = pipeline.get("output_root")
    root_value = output_root if output_root is not None else old_root_value
    if not isinstance(root_value, (str, Path)) or not str(root_value):
        raise ValueError("a concrete PerturbGen output_root is required")
    root = Path(root_value).expanduser().resolve()
    if old_root_value is not None:
        old_root = str(old_root_value)
        replacements = {old_root}
        if not Path(old_root).is_absolute():
            replacements.add(str(Path(old_root).expanduser().resolve()))
        for old_value in sorted(replacements, key=len, reverse=True):
            stages = _replace_target(stages, old_value, str(root))
        materialized["stages"] = stages
        perturb_stage = stages["perturb"]
        perturb_config = perturb_stage["perturb_config"]
        trainer = perturb_config["trainer"]
    pipeline["output_root"] = str(root)
    pipeline["intervention_type"] = invocation.intervention_type
    pipeline["candidate_gene"] = invocation.gene_symbol
    pipeline["candidate_ensembl_id"] = invocation.ensembl_id
    expected_outputs = perturb_stage.get("expected_outputs")
    if expected_outputs is not None and old_target is not None:
        perturb_stage["expected_outputs"] = _replace_target(expected_outputs, old_target, invocation.gene_symbol)
    return materialized


def build_candidate_stage_plans(
    config_or_path: str | Path | Mapping[str, Any],
    invocation: PerturbGenInvocation,
    *,
    output_root: str | Path | None = None,
    project_root: str | Path | None = None,
    paths: Sequence[PathKind] | None = None,
) -> tuple[StagePlan, ...]:
    """Build common training plans plus isolated source/within-state plans."""

    if isinstance(config_or_path, Mapping):
        config = deepcopy(dict(config_or_path))
    else:
        config = load_pipeline_config(config_or_path)
    selected_paths = invocation.paths if paths is None else tuple(paths)
    if not selected_paths:
        raise ValueError("paths must not be empty")
    if len(set(selected_paths)) != len(selected_paths) or any(path not in _PATHS for path in selected_paths):
        raise ValueError("paths must contain unique source_intervention/within_state values")
    config = materialize_candidate_config(config, invocation, output_root=output_root)
    base_plans = build_stage_plans(config, project_root=project_root)
    by_name = {plan.name: plan for plan in base_plans}
    common_names = ("tokenise", "train_mask", "train_decoder")
    missing = [name for name in (*common_names, "export_gene_embeddings", "report") if name not in by_name]
    if missing:
        raise ValueError(f"six-stage PerturbGen config is missing plans: {', '.join(missing)}")

    plans: list[StagePlan] = [by_name[name] for name in common_names]
    for path in selected_paths:
        path_config = deepcopy(config)
        _apply_path(path_config, path)
        stage = path_config["stages"]["perturb"]
        stage["output_subdir"] = f"perturb/{path}"
        trainer = stage["perturb_config"]["trainer"]
        trainer["output_dir"] = str(
            Path(path_config["pipeline"]["output_root"]) / "perturb" / path / "results"
        )
        perturb_plan = next(
            plan
            for plan in build_stage_plans(path_config, project_root=project_root)
            if plan.name == "perturb"
        )
        plans.append(replace(perturb_plan, name=path))
    plans.extend(by_name[name] for name in ("export_gene_embeddings", "report"))
    return tuple(plans)


def merge_route_preparations(
    preparations: Sequence[DAVFPerturbGenPreparation],
) -> list[dict[str, Any]]:
    """Merge KO/KD preparation records by canonical Ensembl ID only."""

    merged: dict[str, dict[str, Any]] = {}
    for preparation in preparations:
        if preparation.status != "pass" or preparation.invocation is None:
            continue
        invocation = preparation.invocation
        row = merged.setdefault(
            invocation.ensembl_id,
            {
                "ensembl_id": invocation.ensembl_id,
                "gene_symbol": invocation.gene_symbol,
                "routes": {},
            },
        )
        if row["gene_symbol"] != invocation.gene_symbol:
            raise ValueError(
                "the same Ensembl ID maps to different gene symbols across DAVF routes: "
                f"{invocation.ensembl_id}"
            )
        route = invocation.intervention_type
        if route in row["routes"]:
            raise ValueError(f"duplicate {route} preparation for {invocation.ensembl_id}")
        row["routes"][route] = preparation.to_dict()
    return [merged[key] for key in sorted(merged)]


def merge_route_reports(
    reports: Sequence[str | Path | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge separate KO/KD E2E reports by canonical Ensembl ID.

    Reports are merged only after each route has completed its own DAVF/scVI/
    PerturbGen coordinate system. Duplicate routes and symbol conflicts are
    errors rather than reasons to choose one record silently.
    """

    if not reports:
        raise ValueError("reports must not be empty")
    merged: dict[str, dict[str, Any]] = {}
    seen_routes: set[str] = set()
    for report in reports:
        payload = _load_report(report)
        route = str(payload.get("intervention_type", "")).strip().upper()
        if route not in _INTERVENTION_TYPES:
            raise ValueError("each E2E report must declare intervention_type KO or KD")
        seen_routes.add(route)
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("each E2E report must contain a candidates list")
        raw_runs = payload.get("perturbgen_runs", [])
        if not isinstance(raw_runs, list):
            raise ValueError("perturbgen_runs must be a list")
        runs_by_ensembl: dict[str, list[dict[str, Any]]] = {}
        for run in raw_runs:
            if not isinstance(run, Mapping):
                raise ValueError("each PerturbGen run record must be an object")
            run_ensembl = normalize_ensembl_id(str(run.get("ensembl_id", "")))
            runs_by_ensembl.setdefault(run_ensembl, []).append(dict(run))

        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError("each E2E candidate record must be an object")
            invocation = candidate.get("invocation")
            if not isinstance(invocation, Mapping):
                continue
            invocation_route = str(invocation.get("intervention_type", "")).strip().upper()
            if invocation_route != route:
                raise ValueError("candidate invocation route does not match report route")
            ensembl_id = normalize_ensembl_id(str(invocation.get("ensembl_id", "")))
            gene_symbol = normalize_gene_symbol(str(invocation.get("gene_symbol", "")))
            row = merged.setdefault(
                ensembl_id,
                {"ensembl_id": ensembl_id, "gene_symbol": gene_symbol, "routes": {}},
            )
            if row["gene_symbol"] != gene_symbol:
                raise ValueError(
                    "the same Ensembl ID maps to different symbols across E2E reports: "
                    f"{ensembl_id}"
                )
            if route in row["routes"]:
                raise ValueError(f"duplicate {route} report for {ensembl_id}")
            row["routes"][route] = {
                "report": dict(candidate),
                "perturbgen_runs": runs_by_ensembl.get(ensembl_id, []),
                "davf_config": payload.get("davf_config"),
                "context_h5ad": payload.get("context_h5ad"),
            }
    if seen_routes != _INTERVENTION_TYPES:
        missing = sorted(_INTERVENTION_TYPES - seen_routes)
        raise ValueError(
            "final DAVF × PerturbGen merge requires exactly one KO and one KD report; "
            f"missing routes: {', '.join(missing) if missing else 'none'}"
        )
    return [merged[key] for key in sorted(merged)]


def _load_report(report: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(report, Mapping):
        return report
    path = Path(report).expanduser().resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"E2E report root must be an object: {path}")
    return payload


def _apply_path(config: dict[str, Any], path: PathKind) -> None:
    perturb = config["stages"]["perturb"]["perturb_config"]
    trainer = perturb["trainer"]
    datamodule = perturb["datamodule"]
    if path == "source_intervention":
        trainer["perturbation_sequence"] = ["src"]
        trainer.pop("pert_tps", None)
        datamodule.pop("pert_tps", None)
    elif path == "within_state":
        trainer["perturbation_sequence"] = ["tgt"]
        if not trainer.get("pert_tps") or not datamodule.get("pert_tps"):
            raise ValueError("within_state requires trainer/datamodule pert_tps")
    else:  # pragma: no cover - validated by the public function first.
        raise ValueError(f"unknown PerturbGen path: {path}")


def _replace_target(value: Any, old_target: str, new_target: str) -> Any:
    if isinstance(value, str):
        return value.replace(old_target, new_target)
    if isinstance(value, list):
        return [_replace_target(item, old_target, new_target) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_target(item, old_target, new_target) for item in value)
    if isinstance(value, dict):
        return {
            key: _replace_target(item, old_target, new_target)
            for key, item in value.items()
        }
    return value


def _as_batch(value: Any, batch_size: int, field_name: str) -> list[Any]:
    if isinstance(value, str) or value is None:
        values = [value] * batch_size
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        values = [value] * batch_size
    if len(values) != batch_size:
        raise ValueError(f"{field_name} must have the same batch length as proposals")
    return values


def _select_mapper_row(output: PTMDirectionMapperOutput, row: int) -> PTMDirectionMapperOutput:
    return PTMDirectionMapperOutput(
        gene_ids=output.gene_ids[row : row + 1].clone(),
        directions=output.directions[row : row + 1].clone(),
        attention_mask=output.attention_mask[row : row + 1].clone(),
    )


def _to_plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {
            field_name: _to_plain(getattr(value, field_name))
            for field_name in value.__dataclass_fields__
        }
    return str(value)
