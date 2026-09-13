"""Batch matched-null PerturbGen stage generation.

``select_matched_nulls`` only writes a selection manifest.  Formal ≥99 null
rescue scores need a generator that:

1. binds each selected null to the candidate's path / mode / seed;
2. reuses the existing six-stage runner for *perturb-only* execution;
3. collects finite ``rescue_excl_target``, stage manifest, and the runner's
   ``result_h5ad.sha256`` into ``perturbgen_null_distribution/v1``.

Null genes are controls, not direction-gated candidates: this module never
constructs a fake passing ``PerturbGenInvocation``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from src.models.gene_vocabulary import normalize_ensembl_id

from .config_builder import StagePlan, build_stage_plans, load_pipeline_config
from .null_selection import (  # noqa: SLF001
    _mode,
    _path_kind,
    _seed,
    _validate_selection_manifest,
    summarize_null_distribution,
)
from .orchestrator import _apply_mode_and_seed, _apply_path, _replace_target  # noqa: SLF001

NULL_STAGE_PLAN_SCHEMA_VERSION = "perturbgen_null_stage_plan/v1"


class NullGenerationError(ValueError):
    """Raised when matched-null stages cannot be planned, executed, or collected."""


@dataclass(frozen=True)
class NullStageRecord:
    """One completed null perturb stage."""

    null_ensembl_id: str
    null_gene_symbol: str
    candidate_ensembl_id: str
    path: str
    mode: str
    seed: int
    rescue_excl_target: float
    stage_manifest: str
    result_h5ad: str
    result_h5ad_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "null_ensembl_id", normalize_ensembl_id(self.null_ensembl_id))
        object.__setattr__(self, "candidate_ensembl_id", normalize_ensembl_id(self.candidate_ensembl_id))
        object.__setattr__(self, "path", _path_kind(self.path))
        object.__setattr__(self, "mode", _mode(self.mode))
        object.__setattr__(self, "seed", _seed(self.seed))


NullStageExecutor = Callable[[Mapping[str, Any]], NullStageRecord]


def _validate_record_binding(
    record: NullStageRecord,
    request_or_expected: Mapping[str, Any],
) -> None:
    expected = {
        "candidate_ensembl_id": normalize_ensembl_id(request_or_expected["candidate_ensembl_id"]),
        "path": _path_kind(request_or_expected["path"]),
        "mode": _mode(request_or_expected["mode"]),
        "seed": _seed(request_or_expected["seed"]),
    }
    for field, value in expected.items():
        if getattr(record, field) != value:
            raise NullGenerationError(f"null stage record has mismatched {field}")


def _load_json_mapping(value: Mapping[str, Any] | str | Path, *, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value).expanduser().resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise NullGenerationError(f"{name} must be a JSON object")
    return dict(payload)


def candidate_identity_from_e2e_report(report: Mapping[str, Any]) -> dict[str, str]:
    if report.get("schema_version") not in {None, "davf_perturbgen_e2e/v1"}:
        raise NullGenerationError(f"unsupported E2E schema_version {report.get('schema_version')!r}")
    for preparation in report.get("candidates", []):
        if not isinstance(preparation, Mapping) or preparation.get("status") != "pass":
            continue
        invocation = preparation.get("invocation")
        if not isinstance(invocation, Mapping):
            continue
        gene_symbol = str(invocation.get("gene_symbol", "")).strip()
        ensembl_id = str(invocation.get("ensembl_id", "")).strip()
        if gene_symbol and ensembl_id:
            return {
                "gene_symbol": gene_symbol,
                "ensembl_id": normalize_ensembl_id(ensembl_id),
            }
    raise NullGenerationError("E2E gate report has no passing invocation to bind null stages")


def _resolve_null_symbol(
    item: Mapping[str, Any],
    *,
    ensembl_to_symbol: Mapping[str, str] | None,
) -> str:
    null_id = normalize_ensembl_id(item["ensembl_id"])
    symbol = item.get("gene_symbol")
    if symbol is not None and str(symbol).strip():
        return str(symbol).strip()
    if ensembl_to_symbol is not None and null_id in ensembl_to_symbol:
        mapped = str(ensembl_to_symbol[null_id]).strip()
        if mapped:
            return mapped
    raise NullGenerationError(
        f"null {null_id} needs gene_symbol on the selection entry or ensembl_to_symbol mapping"
    )


def _stage_output_record(manifest: Mapping[str, Any], *, result_h5ad: Path) -> dict[str, Any]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise NullGenerationError("null stage_manifest.outputs must be a mapping")
    resolved_h5ad = result_h5ad.expanduser().resolve(strict=True)
    matches = []
    for key, record in outputs.items():
        try:
            path = Path(str(key)).expanduser().resolve(strict=True)
        except OSError:
            continue
        if path == resolved_h5ad:
            matches.append(record)
    if len(matches) != 1:
        raise NullGenerationError("null stage_manifest.outputs must contain exactly one result_h5ad record")
    record = matches[0]
    if not isinstance(record, Mapping) or not str(record.get("sha256", "")).strip():
        raise NullGenerationError("null stage_manifest output record must include sha256")
    return dict(record)


def plan_matched_null_stages(
    selection_manifest: Mapping[str, Any] | str | Path,
    base_config: Mapping[str, Any] | str | Path,
    e2e_gate_report: Mapping[str, Any] | str | Path,
    path: str,
    mode: str,
    seed: int,
    output_root: str | Path,
    *,
    ensembl_to_symbol: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> tuple[dict[str, Any], tuple[StagePlan, ...]]:
    """Build perturb-only stage plans for every selected null gene."""

    selection = _validate_selection_manifest(_load_json_mapping(selection_manifest, name="selection_manifest"))
    report = _load_json_mapping(e2e_gate_report, name="e2e_gate_report")
    candidate = candidate_identity_from_e2e_report(report)
    expected_path = _path_kind(path)
    expected_mode = _mode(mode)
    expected_seed = _seed(seed)
    root = Path(output_root).expanduser().resolve()
    if isinstance(base_config, Mapping):
        config = deepcopy(dict(base_config))
    else:
        config = load_pipeline_config(base_config)

    requests: list[dict[str, Any]] = []
    plans: list[StagePlan] = []
    candidate_symbol = candidate["gene_symbol"]
    for item in selection["selected_nulls"]:
        if not isinstance(item, Mapping):
            raise NullGenerationError("selected_nulls entries must be mappings")
        null_id = normalize_ensembl_id(item["ensembl_id"])
        if null_id == candidate["ensembl_id"]:
            raise NullGenerationError("selected nulls must exclude the candidate gene")
        symbol = _resolve_null_symbol(item, ensembl_to_symbol=ensembl_to_symbol)
        null_root = root / null_id
        request = {
            "null_ensembl_id": null_id,
            "null_gene_symbol": symbol,
            "candidate_ensembl_id": candidate["ensembl_id"],
            "candidate_gene_symbol": candidate_symbol,
            "path": expected_path,
            "mode": expected_mode,
            "seed": expected_seed,
            "output_root": str(null_root),
        }
        requests.append(request)
        path_config = deepcopy(config)
        trainer = path_config["stages"]["perturb"]["perturb_config"]["trainer"]
        old_targets = trainer.get("genes_to_perturb")
        old_target = candidate_symbol
        if (
            isinstance(old_targets, Sequence)
            and not isinstance(old_targets, (str, bytes))
            and len(old_targets) == 1
            and str(old_targets[0])
        ):
            old_target = str(old_targets[0])
        trainer["genes_to_perturb"] = [symbol]
        trainer["perturbation_mode"] = expected_mode
        path_config["pipeline"]["random_seed"] = expected_seed
        path_config["pipeline"]["output_root"] = str(null_root)
        perturb_stage = path_config["stages"]["perturb"]
        perturb_stage["output_subdir"] = f"perturb/{expected_path}/null_{null_id}"
        expected_outputs = perturb_stage.get("expected_outputs")
        if expected_outputs is not None:
            perturb_stage["expected_outputs"] = _replace_target(expected_outputs, old_target, symbol)
        _apply_path(path_config, expected_path)  # type: ignore[arg-type]
        _apply_mode_and_seed(path_config, expected_mode, expected_seed)  # type: ignore[arg-type]
        perturb_plan = next(
            plan
            for plan in build_stage_plans(path_config, project_root=project_root)
            if plan.name == "perturb"
        )
        plans.append(perturb_plan)

    payload = {
        "schema_version": NULL_STAGE_PLAN_SCHEMA_VERSION,
        "candidate_ensembl_id": candidate["ensembl_id"],
        "candidate_gene_symbol": candidate_symbol,
        "path": expected_path,
        "mode": expected_mode,
        "seed": expected_seed,
        "required_count": int(selection["required_count"]),
        "selection_schema_version": selection["schema_version"],
        "planned_stages": requests,
    }
    return payload, tuple(plans)


def collect_null_stage_records(
    records: Sequence[NullStageRecord | Mapping[str, Any]],
    *,
    candidate_ensembl_id: str,
    path: str,
    mode: str,
    seed: int,
    required_count: int = 99,
    selection_manifest_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Summarise completed null stages into ``perturbgen_null_distribution/v1``."""

    expected = {
        "candidate_ensembl_id": normalize_ensembl_id(candidate_ensembl_id),
        "path": _path_kind(path),
        "mode": _mode(mode),
        "seed": _seed(seed),
    }
    parsed: list[NullStageRecord] = []
    for record in records:
        if isinstance(record, NullStageRecord):
            parsed_record = record
        else:
            if not isinstance(record, Mapping):
                raise NullGenerationError("null stage records must be mappings")
            parsed_record = NullStageRecord(
                null_ensembl_id=record["null_ensembl_id"],
                null_gene_symbol=str(record["null_gene_symbol"]).strip(),
                candidate_ensembl_id=record["candidate_ensembl_id"],
                path=record["path"],
                mode=record["mode"],
                seed=record["seed"],
                rescue_excl_target=float(record["rescue_excl_target"]),
                stage_manifest=str(record["stage_manifest"]),
                result_h5ad=str(record["result_h5ad"]),
                result_h5ad_sha256=str(record["result_h5ad_sha256"]).strip(),
            )
        _validate_record_binding(parsed_record, expected)
        parsed.append(parsed_record)
    summary_records = [
        {
            "candidate_ensembl_id": item.candidate_ensembl_id,
            "null_ensembl_id": item.null_ensembl_id,
            "path": item.path,
            "mode": item.mode,
            "seed": item.seed,
            "rescue_excl_target": item.rescue_excl_target,
        }
        for item in parsed
    ]
    payload = summarize_null_distribution(
        summary_records,
        candidate_ensembl_id=candidate_ensembl_id,
        path=path,
        mode=mode,
        seed=seed,
        required_count=required_count,
        selection_manifest_path=selection_manifest_path,
        output_path=output_path,
    )
    by_id = {item.null_ensembl_id: item for item in parsed}
    payload["stage_manifests"] = [by_id[null_id].stage_manifest for null_id in payload["null_ensembl_ids"]]
    payload["result_h5ad"] = [by_id[null_id].result_h5ad for null_id in payload["null_ensembl_ids"]]
    payload["result_h5ad_sha256"] = [by_id[null_id].result_h5ad_sha256 for null_id in payload["null_ensembl_ids"]]
    if any(not digest for digest in payload["result_h5ad_sha256"]):
        raise NullGenerationError("every null result_h5ad.sha256 must be non-empty")
    return payload


def run_matched_null_stages(
    selection_manifest: Mapping[str, Any] | str | Path,
    base_config: Mapping[str, Any] | str | Path | None,
    e2e_gate_report: Mapping[str, Any] | str | Path,
    path: str,
    mode: str,
    seed: int,
    output_root: str | Path,
    *,
    runner: Any | None = None,
    execute: bool = True,
    dry_run: bool = False,
    stage_executor: NullStageExecutor | None = None,
    rescue_extractor: Callable[[Mapping[str, Any], Any], NullStageRecord] | None = None,
    ensembl_to_symbol: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
    selection_manifest_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Plan, optionally execute, and collect ≥99 matched-null perturb stages.

    ``stage_executor`` is the unit-test seam: it must still return finite
    rescue scores and runner-style hashes.  GPU execution uses
    ``PerturbGenRunner.run_pipeline`` on the perturb-only plans.
    """

    selection = _validate_selection_manifest(_load_json_mapping(selection_manifest, name="selection_manifest"))
    report = _load_json_mapping(e2e_gate_report, name="e2e_gate_report")
    candidate = candidate_identity_from_e2e_report(report)
    expected_path = _path_kind(path)
    expected_mode = _mode(mode)
    expected_seed = _seed(seed)
    required_count = int(selection["required_count"])
    if base_config is None and stage_executor is None and execute and not dry_run:
        raise NullGenerationError("base_config is required unless a stage_executor is supplied")

    planned_stages: list[dict[str, Any]] = []
    plans: tuple[StagePlan, ...] = ()
    if base_config is not None:
        plan_payload, plans = plan_matched_null_stages(
            selection,
            base_config,
            report,
            expected_path,
            expected_mode,
            expected_seed,
            output_root,
            ensembl_to_symbol=ensembl_to_symbol,
            project_root=project_root,
        )
        planned_stages = list(plan_payload["planned_stages"])
    else:
        root = Path(output_root).expanduser().resolve()
        for item in selection["selected_nulls"]:
            null_id = normalize_ensembl_id(item["ensembl_id"])
            symbol = str(item.get("gene_symbol") or null_id)
            if ensembl_to_symbol and not item.get("gene_symbol"):
                symbol = ensembl_to_symbol.get(null_id, symbol)
            planned_stages.append(
                {
                    "null_ensembl_id": null_id,
                    "null_gene_symbol": symbol,
                    "candidate_ensembl_id": candidate["ensembl_id"],
                    "candidate_gene_symbol": candidate["gene_symbol"],
                    "path": expected_path,
                    "mode": expected_mode,
                    "seed": expected_seed,
                    "output_root": str(root / null_id),
                }
            )

    if dry_run or not execute:
        return {
            "schema_version": NULL_STAGE_PLAN_SCHEMA_VERSION,
            "candidate_ensembl_id": candidate["ensembl_id"],
            "path": expected_path,
            "mode": expected_mode,
            "seed": expected_seed,
            "required_count": required_count,
            "planned_stages": planned_stages,
            "execute": False,
        }

    records: list[NullStageRecord] = []
    if stage_executor is not None:
        for request in planned_stages:
            record = stage_executor(request)
            if not isinstance(record, NullStageRecord):
                raise NullGenerationError("stage_executor must return NullStageRecord")
            if record.null_ensembl_id != request["null_ensembl_id"]:
                raise NullGenerationError("stage_executor returned a mismatched null_ensembl_id")
            _validate_record_binding(record, request)
            records.append(record)
    else:
        if runner is None:
            raise NullGenerationError("PerturbGenRunner is required when execute=True and no stage_executor is set")
        if rescue_extractor is None:
            raise NullGenerationError(
                "GPU null execution requires rescue_extractor(request, stage_result) -> NullStageRecord; "
                "this module does not guess DEG columns or h5ad provenance"
            )
        if not plans:
            raise NullGenerationError("null perturb plans are empty; supply base_config")
        results = runner.run_pipeline(plans, dry_run=False)
        if len(results) != len(planned_stages):
            raise NullGenerationError("runner result count must match planned null stages")
        for request, stage_result in zip(planned_stages, results, strict=True):
            record = rescue_extractor(request, stage_result)
            if not isinstance(record, NullStageRecord):
                raise NullGenerationError("rescue_extractor must return NullStageRecord")
            if record.null_ensembl_id != request["null_ensembl_id"]:
                raise NullGenerationError("rescue_extractor returned a mismatched null_ensembl_id")
            _validate_record_binding(record, request)
            records.append(record)

    return collect_null_stage_records(
        records,
        candidate_ensembl_id=candidate["ensembl_id"],
        path=expected_path,
        mode=expected_mode,
        seed=expected_seed,
        required_count=required_count,
        selection_manifest_path=selection_manifest_path,
        output_path=output_path,
    )
