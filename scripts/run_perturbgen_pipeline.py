#!/usr/bin/env python3
"""Run or inspect the isolated PerturbGen six-stage pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import fields, replace
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.perturbgen.config_builder import (  # noqa: E402
    STAGE_ORDER,
    build_stage_plans,
    load_pipeline_config,
)
from src.integration.perturbgen.contracts import (  # noqa: E402
    CandidateEvidence,
    DAVFDirectionEvidence,
)
from src.integration.perturbgen.orchestrator import PerturbGenInvocation  # noqa: E402
from src.integration.perturbgen.runner import PerturbGenRunner  # noqa: E402
from src.models.gene_vocabulary import normalize_ensembl_id  # noqa: E402


def _apply_path(config: dict[str, Any], path: str | None) -> None:
    if path is None:
        return
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
    else:  # argparse choices make this defensive branch unreachable from CLI.
        raise ValueError(f"unknown path: {path}")


def _serialize_result(value: Any) -> Any:
    if hasattr(value, "__dict__"):
        return {
            key: str(item) if isinstance(item, Path) else item
            for key, item in value.__dict__.items()
        }
    return value


def _dataclass_kwargs(dataclass_type: type[Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        item.name: payload[item.name]
        for item in fields(dataclass_type)
        if item.name in payload
    }


def _validated_report_invocation(
    record: Mapping[str, Any],
) -> PerturbGenInvocation:
    invocation_payload = record.get("invocation")
    outer_candidate_payload = record.get("candidate")
    if not isinstance(invocation_payload, Mapping) or not isinstance(outer_candidate_payload, Mapping):
        raise ValueError("E2E gate record must contain complete invocation and candidate mappings")
    nested_candidate_payload = invocation_payload.get("candidate")
    nested_evidence_payload = invocation_payload.get("davf_evidence")
    if not isinstance(nested_candidate_payload, Mapping) or not isinstance(nested_evidence_payload, Mapping):
        raise ValueError("E2E gate invocation must contain candidate and davf_evidence mappings")
    try:
        candidate = CandidateEvidence(
            **_dataclass_kwargs(CandidateEvidence, nested_candidate_payload)
        )
        davf_evidence = DAVFDirectionEvidence(
            **_dataclass_kwargs(DAVFDirectionEvidence, nested_evidence_payload)
        )
        invocation = PerturbGenInvocation(
            intervention_type=invocation_payload["intervention_type"],
            gene_symbol=invocation_payload["gene_symbol"],
            ensembl_id=invocation_payload["ensembl_id"],
            target_token_id=invocation_payload["target_token_id"],
            perturbation_mode=invocation_payload["perturbation_mode"],
            paths=invocation_payload["paths"],
            candidate=candidate,
            davf_evidence=davf_evidence,
            perturbgen_config_path=invocation_payload.get("perturbgen_config_path"),
            output_root=invocation_payload.get("output_root"),
            seed=invocation_payload["seed"],
            is_sensitivity=invocation_payload.get("is_sensitivity", False),
        )
        outer_candidate = CandidateEvidence(
            **_dataclass_kwargs(CandidateEvidence, outer_candidate_payload)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"E2E gate invocation violates PerturbGenInvocation contract: {exc}") from exc

    if outer_candidate != invocation.candidate:
        raise ValueError("E2E gate outer candidate does not match invocation candidate")
    outer_evidence_payload = record.get("davf_evidence")
    if outer_evidence_payload is not None:
        if not isinstance(outer_evidence_payload, Mapping):
            raise ValueError("E2E gate outer davf_evidence must be a mapping")
        try:
            outer_evidence = DAVFDirectionEvidence(
                **_dataclass_kwargs(DAVFDirectionEvidence, outer_evidence_payload)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"E2E gate outer DAVF evidence violates its contract: {exc}") from exc
        if outer_evidence != invocation.davf_evidence:
            raise ValueError("E2E gate outer DAVF evidence does not match invocation evidence")
    return invocation


def _config_gate_binding(
    config: Mapping[str, Any],
    *,
    path: str | None,
    config_path: Path,
) -> dict[str, Any]:
    """Extract the fields owned by the base YAML for gate-report binding."""

    stages = config.get("stages")
    pipeline = config.get("pipeline")
    if not isinstance(stages, Mapping) or not isinstance(pipeline, Mapping):
        raise ValueError("PerturbGen config must declare stages and pipeline for E2E gate binding")
    perturb_stage = stages.get("perturb")
    if not isinstance(perturb_stage, Mapping):
        raise ValueError("PerturbGen config must declare stages.perturb for E2E gate binding")
    perturb_config = perturb_stage.get("perturb_config")
    if not isinstance(perturb_config, Mapping):
        raise ValueError("PerturbGen config stages.perturb.perturb_config must be a mapping")
    trainer = perturb_config.get("trainer")
    datamodule = perturb_config.get("datamodule")
    if not isinstance(trainer, Mapping) or not isinstance(datamodule, Mapping):
        raise ValueError("PerturbGen perturb config must declare trainer and datamodule mappings")

    raw_genes = trainer.get("genes_to_perturb")
    if not isinstance(raw_genes, Sequence) or isinstance(raw_genes, (str, bytes)) or len(raw_genes) != 1:
        raise ValueError("PerturbGen config trainer.genes_to_perturb must contain exactly one gene")
    gene_symbol = str(raw_genes[0]).strip().upper()
    if not gene_symbol:
        raise ValueError("PerturbGen config trainer.genes_to_perturb must contain a non-empty gene")

    perturbation_mode = str(trainer.get("perturbation_mode", "")).strip().lower()
    if perturbation_mode not in {"mask", "pad", "delete", "overexpress"}:
        raise ValueError("PerturbGen config trainer.perturbation_mode must be an explicit supported mode")

    raw_seed = pipeline.get("random_seed")
    if isinstance(raw_seed, bool) or raw_seed is None:
        raise ValueError("PerturbGen config pipeline.random_seed must be explicit and non-negative")
    try:
        pipeline_seed = int(raw_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("PerturbGen config pipeline.random_seed must be an integer") from exc
    if pipeline_seed < 0 or (isinstance(raw_seed, float) and raw_seed != pipeline_seed):
        raise ValueError("PerturbGen config pipeline.random_seed must be a non-negative integer")

    route_value = pipeline.get("intervention_type")
    route = None if route_value is None else str(route_value).strip().upper()
    if route is not None and route not in {"KO", "KD"}:
        raise ValueError("PerturbGen config pipeline.intervention_type must be KO or KD when declared")
    ensembl_value = pipeline.get("candidate_ensembl_id")
    ensembl_id = None if ensembl_value is None else normalize_ensembl_id(str(ensembl_value))

    has_pert_tps = bool(trainer.get("pert_tps")) and bool(datamodule.get("pert_tps"))
    if path is None:
        sequence = trainer.get("perturbation_sequence")
        if not isinstance(sequence, Sequence) or isinstance(sequence, (str, bytes)) or len(sequence) != 1:
            raise ValueError("PerturbGen config perturbation_sequence must identify one route")
        sequence_value = str(sequence[0]).strip().lower()
        if sequence_value == "src":
            expected_paths = {"source_intervention"}
        elif sequence_value == "tgt" and has_pert_tps:
            expected_paths = {"within_state"}
        else:
            raise ValueError("PerturbGen config cannot resolve an explicit PerturbGen path")
    elif path == "both":
        if not has_pert_tps:
            raise ValueError("both path binding requires trainer/datamodule pert_tps")
        expected_paths = {"source_intervention", "within_state"}
    elif path == "source_intervention":
        expected_paths = {path}
    elif path == "within_state":
        if not has_pert_tps:
            raise ValueError("within_state path binding requires trainer/datamodule pert_tps")
        expected_paths = {path}
    else:
        raise ValueError(f"unknown path for E2E gate binding: {path!r}")

    return {
        "gene_symbol": gene_symbol,
        "perturbation_mode": perturbation_mode,
        "intervention_type": route,
        "ensembl_id": ensembl_id,
        "seed": pipeline_seed,
        "paths": expected_paths,
        "config_path": str(config_path.expanduser().resolve()),
    }


def _validate_invocation_binding(
    invocation: PerturbGenInvocation,
    expected: Mapping[str, Any],
) -> None:
    if invocation.gene_symbol != expected["gene_symbol"]:
        raise ValueError(
            f"E2E gate gene does not match config: {invocation.gene_symbol} != {expected['gene_symbol']}"
        )
    if invocation.perturbation_mode != expected["perturbation_mode"]:
        raise ValueError(
            "E2E gate perturbation mode does not match config: "
            f"{invocation.perturbation_mode} != {expected['perturbation_mode']}"
        )
    expected_ensembl_id = expected.get("ensembl_id")
    if expected_ensembl_id is not None and invocation.ensembl_id != expected_ensembl_id:
        raise ValueError(
            "E2E gate Ensembl ID does not match config: "
            f"{invocation.ensembl_id} != {expected_ensembl_id}"
        )
    expected_route = expected.get("intervention_type")
    if expected_route is not None and invocation.intervention_type != expected_route:
        raise ValueError(
            f"E2E gate route does not match config: {invocation.intervention_type} != {expected_route}"
        )
    if invocation.seed != expected["seed"]:
        raise ValueError(
            f"E2E gate seed does not match config pipeline.random_seed: "
            f"{invocation.seed} != {expected['seed']}"
        )
    expected_paths = set(expected["paths"])
    actual_paths = set(invocation.paths)
    if not expected_paths.issubset(actual_paths):
        raise ValueError(
            f"E2E gate paths do not authorize the configured path(s): "
            f"required={sorted(expected_paths)}, report={sorted(actual_paths)}"
        )
    if invocation.perturbgen_config_path is not None:
        actual_config_path = str(invocation.perturbgen_config_path.expanduser().resolve())
        if actual_config_path != expected["config_path"]:
            raise ValueError(
                "E2E gate perturbgen_config_path must refer to the current base config: "
                f"{actual_config_path} != {expected['config_path']}"
            )


def _validate_e2e_gate_report(
    report_path: Path,
    *,
    expected_binding: Mapping[str, Any],
) -> None:
    path = report_path.expanduser()
    if not path.is_file():
        raise ValueError(f"E2E gate report does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid E2E gate report JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("E2E gate report must be a JSON mapping")

    records: list[Mapping[str, Any]] = []
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        records.extend(item for item in candidates if isinstance(item, Mapping))
    elif (
        all(key in payload for key in ("candidate", "preparation", "invocation"))
        and isinstance(payload.get("preparation"), Mapping)
    ):
        records.append(payload)
    elif isinstance(payload.get("preparation"), Mapping):
        records.append(payload["preparation"])

    binding_errors: list[str] = []
    for record in records:
        candidate = record.get("candidate")
        invocation = record.get("invocation")
        if (
            isinstance(candidate, Mapping)
            and isinstance(invocation, Mapping)
            and (record.get("status") is None or record.get("status") == "pass")
            and candidate.get("direction_gate_status") == "pass"
        ):
            try:
                invocation_object = _validated_report_invocation(record)
                _validate_invocation_binding(invocation_object, expected_binding)
            except ValueError as exc:
                binding_errors.append(str(exc))
                continue
            return
    if binding_errors:
        raise ValueError(
            "E2E gate report has no candidate bound to the current PerturbGen config: "
            + binding_errors[0]
        )
    raise ValueError(
        "E2E gate report must contain a candidate/preparation/invocation record "
        "with candidate.direction_gate_status='pass'"
    )


def _build_selected_plans(
    config: dict[str, Any], selected: set[str], path: str | None
):
    if path != "both":
        _apply_path(config, path)
        return tuple(
            plan for plan in build_stage_plans(config, project_root=PROJECT_ROOT)
            if plan.name in selected
        )

    base_plans = build_stage_plans(config, project_root=PROJECT_ROOT)
    path_plans = []
    for path_name in ("source_intervention", "within_state"):
        path_config = deepcopy(config)
        _apply_path(path_config, path_name)
        stage = path_config["stages"]["perturb"]
        stage["output_subdir"] = f"perturb/{path_name}"
        stage["perturb_config"]["trainer"]["output_dir"] = str(
            Path(path_config["pipeline"]["output_root"]) / "perturb" / path_name / "results"
        )
        perturb_plan = build_stage_plans(path_config, project_root=PROJECT_ROOT)[3]
        path_plans.append(replace(perturb_plan, name=path_name))

    output = []
    for plan in base_plans:
        if plan.name not in selected:
            continue
        if plan.name == "perturb":
            output.extend(path_plans)
        else:
            output.append(plan)
    return tuple(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=STAGE_ORDER, default=list(STAGE_ORDER))
    parser.add_argument(
        "--path",
        choices=("source_intervention", "within_state", "both"),
        help="Materialize the perturb stage as PerturbGen src or tgt+pert_tps",
    )
    parser.add_argument(
        "--e2e-gate-report",
        type=Path,
        help="Required JSON report from the explicit DAVF/PerturbGen E2E gate",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu-lock-file", type=Path, default=Path("outputs/perturbgen/.gpu.lock"))
    args = parser.parse_args(argv)

    selected = set(args.stages)
    requires_gate_report = "perturb" in selected or args.path is not None
    if requires_gate_report:
        if args.e2e_gate_report is None:
            parser.error("perturb stages require --e2e-gate-report")

    config = load_pipeline_config(args.config)
    if requires_gate_report:
        try:
            expected_binding = _config_gate_binding(
                config,
                path=args.path,
                config_path=args.config,
            )
            _validate_e2e_gate_report(
                args.e2e_gate_report,
                expected_binding=expected_binding,
            )
        except ValueError as exc:
            parser.error(str(exc))
    plans = _build_selected_plans(config, selected, args.path)
    runner = PerturbGenRunner(gpu_lock_file=args.gpu_lock_file)
    results = runner.run_pipeline(plans, resume=args.resume, dry_run=args.dry_run)
    print(json.dumps([_serialize_result(item) for item in results], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
