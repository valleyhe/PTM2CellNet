#!/usr/bin/env python3
"""Run the strict DAVF -> PerturbGen serial candidate workflow.

The candidate specification is JSON and must contain one explicit scVI
context cell per candidate.  Example::

    {
      "context_h5ad": "data/processed/context.h5ad",
      "candidates": [
        {
          "context_cell_index": 0,
          "gene_symbol": "STAT3",
          "ensembl_id": "ENSG00000168610",
          "position": 12,
          "ptm_type": "phosphorylation",
          "proposed_direction": "down",
          "site_probability": 0.95,
          "provenance": "ptm-site-model/run-1",
          "cell_type": "K562",
          "ptm_context": "STAT3:S12",
          "observed_log2fc": -1.0,
          "observed_fdr": 0.01,
          "observed_direction": "down",
          "semantic_context": {
            "context": "disease",
            "intervention": "KO",
            "comparison_baseline": "normal",
            "reference_axis": "disease-minus-normal",
            "research_objective": "replication",
            "evidence_source": "donor_level_expression+davf_decode",
            "cohort": "formal_normal_disease_cohort"
          }
        }
      ]
    }

By default this command only performs real DAVF inference and writes the
direction-gated invocation manifest.  ``--run-perturbgen`` additionally runs
the isolated six-stage external pipeline for every gated candidate.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import fields
import json
from pathlib import Path
import sys
from typing import Any, cast

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.perturbgen.config_builder import load_pipeline_config  # noqa: E402
from src.integration.perturbgen.orchestrator import (  # noqa: E402
    DAVFPerturbGenOrchestrator,
    PerturbGenInvocation,
    build_shared_prepare_plans,
    merge_route_preparations,
)
from src.integration.perturbgen.runner import PerturbGenRunner, StageExecutionResult  # noqa: E402
from src.integration.perturbgen.contracts import (  # noqa: E402
    PTMSiteDirectionProposal,
    PerturbGenDataSpec,
    PerturbationMode,
    SemanticContext,
)
from src.integration.perturbgen.data_prep import (  # noqa: E402
    prepare_perturbgen_anndata,
)
from src.integration.perturbgen.donor_split import (  # noqa: E402
    DonorSplitError,
    bind_frozen_donor_split,
    optional_donor_split_from_args,
)
from src.integration.perturbgen.reports import to_plain_object  # noqa: E402
from src.models.davf_inference import DAVFInferenceConfig, DAVFInferenceModule  # noqa: E402
from src.models.gene_vocabulary import normalize_ensembl_id  # noqa: E402


def _load_davf_config(path: str | Path) -> DAVFInferenceConfig:
    """Load a DAVF YAML section and resolve project-relative asset paths."""

    config_path = Path(path).expanduser().resolve(strict=True)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DAVF config root must be a mapping")
    section = payload.get("davf", payload)
    if not isinstance(section, dict):
        raise ValueError("DAVF config must contain a mapping under 'davf'")

    allowed = {item.name for item in fields(DAVFInferenceConfig)}
    values = {key: value for key, value in section.items() if key in allowed}
    for key in ("checkpoint_path", "scvi_model_path", "embedding_asset_path", "gene_names_path"):
        raw = values.get(key)
        if raw is None:
            continue
        resolved = Path(str(raw)).expanduser()
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
        values[key] = str(resolved.resolve())
    return DAVFInferenceConfig(**values)


def _load_candidate_spec(path: str | Path) -> tuple[Path, tuple[dict[str, Any], ...]]:
    """Load and validate the explicit candidate/context specification."""

    spec_path = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("candidate spec root must be a JSON object")
    raw_context = payload.get("context_h5ad")
    if not isinstance(raw_context, str) or not raw_context.strip():
        raise ValueError("candidate spec requires a non-empty context_h5ad")
    context_path = Path(raw_context).expanduser()
    if not context_path.is_absolute():
        context_path = PROJECT_ROOT / context_path
    context_path = context_path.resolve(strict=True)
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("candidate spec requires a non-empty candidates list")
    candidates: list[dict[str, Any]] = []
    for row, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate {row} must be a JSON object")
        index = candidate.get("context_cell_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"candidate {row} context_cell_index must be a non-negative integer")
        candidates.append(dict(candidate))
    return context_path, tuple(candidates)


def _build_proposal(candidate: dict[str, Any], row: int) -> PTMSiteDirectionProposal:
    required = (
        "gene_symbol",
        "ensembl_id",
        "position",
        "ptm_type",
        "proposed_direction",
        "site_probability",
        "provenance",
    )
    missing = [key for key in required if key not in candidate]
    if missing:
        raise ValueError(f"candidate {row} is missing fields: {', '.join(missing)}")
    return PTMSiteDirectionProposal(
        gene_symbol=candidate["gene_symbol"],
        ensembl_id=candidate["ensembl_id"],
        position=candidate["position"],
        ptm_type=candidate["ptm_type"],
        proposed_direction=candidate["proposed_direction"],
        site_probability=candidate["site_probability"],
        provenance=candidate["provenance"],
    )


def _validate_perturbgen_tokenise_input(
    config: Mapping[str, Any],
    context_path: Path,
    *,
    cohort_pairing: str = "within_donor",
) -> tuple[PerturbGenDataSpec, Path]:
    """Bind the PerturbGen tokenise input and metadata to the E2E context.

    ``cohort_pairing`` freezes the donor/state design of the cohort:
    ``within_donor`` requires shared donors across the two states
    (perturbation cohorts), ``between_donor`` accepts case-control cohorts
    with donor-disjoint state groups.
    """

    try:
        stages = config["stages"]
        tokenise = stages["tokenise"]
        tokenise_args = tokenise["args"]
    except (KeyError, TypeError) as exc:
        raise ValueError("PerturbGen config must declare stages.tokenise.args") from exc
    if not isinstance(stages, Mapping) or not isinstance(tokenise, Mapping) or not isinstance(tokenise_args, Mapping):
        raise ValueError("PerturbGen stages.tokenise.args must be mappings")

    raw_input = tokenise_args.get("h5ad_path")
    if not isinstance(raw_input, str) or not raw_input.strip():
        raise ValueError("PerturbGen tokenise.args.h5ad_path must be a non-empty path")
    tokenise_path = Path(raw_input).expanduser()
    if not tokenise_path.is_absolute():
        raise ValueError("PerturbGen tokenise.args.h5ad_path must be absolute")
    tokenise_path = tokenise_path.resolve(strict=True)
    if tokenise_path != context_path:
        raise ValueError(
            f"PerturbGen tokenise input must be the exact candidate context_h5ad: {tokenise_path} != {context_path}"
        )

    raw_var_list = tokenise_args.get("var_list")
    if not isinstance(raw_var_list, Sequence) or isinstance(raw_var_list, (str, bytes)):
        raise ValueError("PerturbGen tokenise.args.var_list must list cell_type, state, and donor columns")
    if len(raw_var_list) != 3 or any(not isinstance(value, str) or not value.strip() for value in raw_var_list):
        raise ValueError("PerturbGen tokenise.args.var_list must contain exactly three non-empty columns")
    cell_type_col, state_col, donor_col = (value.strip() for value in raw_var_list)
    if len({cell_type_col, state_col, donor_col}) != 3:
        raise ValueError("PerturbGen tokenise.args.var_list columns must be distinct")
    if tokenise_args.get("main_pairing_obs") != cell_type_col:
        raise ValueError("PerturbGen tokenise main_pairing_obs must match var_list cell_type column")
    if tokenise_args.get("time_obs") != state_col:
        raise ValueError("PerturbGen tokenise time_obs must match var_list state column")

    reference_state = tokenise_args.get("reference_time")
    time_point_order = tokenise_args.get("time_point_order")
    if not isinstance(reference_state, str) or not reference_state.strip():
        raise ValueError("PerturbGen tokenise reference_time must be explicit")
    if not isinstance(time_point_order, Sequence) or isinstance(time_point_order, (str, bytes)):
        raise ValueError("PerturbGen tokenise time_point_order must be an explicit two-state sequence")
    if (
        len(time_point_order) != 2
        or any(not isinstance(value, str) or not value.strip() for value in time_point_order)
        or time_point_order[0].strip() != reference_state.strip()
        or time_point_order[0].strip() == time_point_order[1].strip()
    ):
        raise ValueError("PerturbGen tokenise time_point_order must be [reference_time, disease_state]")

    return (
        PerturbGenDataSpec(
            cell_type_col=cell_type_col,
            state_col=state_col,
            donor_col=donor_col,
            normal_state=reference_state.strip(),
            disease_state=time_point_order[1].strip(),
            pairing=cohort_pairing,
        ),
        tokenise_path,
    )


def _preflight_perturbgen_context(
    context_adata: Any,
    context_path: Path,
    candidates: Sequence[dict[str, Any]],
    *,
    spec: PerturbGenDataSpec,
    tokenise_path: Path,
) -> dict[str, Any]:
    """Run Gate-0 on the exact cohort path declared for tokenisation.

    The prepared AnnData is a validation copy.  The registered external
    tokeniser reads ``tokenise_path`` itself, so versioned Ensembl IDs in the
    original context are rejected instead of silently relying on the copy's
    normalisation.
    """

    _validate_original_tokenise_ensembl(context_adata, spec)
    reports: dict[str, Any] = {}
    for row, candidate in enumerate(candidates):
        cell_type = candidate.get("cell_type")
        if not isinstance(cell_type, str) or not cell_type.strip():
            raise ValueError(f"candidate {row} cell_type must be a non-empty string")
        cell_type = cell_type.strip()
        if cell_type not in reports:
            prepared = prepare_perturbgen_anndata(context_adata, cell_type=cell_type, spec=spec)
            reports[cell_type] = prepared.report

        context_index = int(candidate["context_cell_index"])
        observed_cell_type = str(context_adata.obs.iloc[context_index][spec.cell_type_col]).strip()
        if observed_cell_type != cell_type:
            raise ValueError(
                f"candidate {row} context cell {context_index} has cell_type {observed_cell_type!r}, not {cell_type!r}"
            )

    return {
        "status": "pass",
        "context_h5ad": str(context_path),
        "tokenise_h5ad": str(tokenise_path),
        "data_spec": to_plain_object(spec),
        "cell_types": {cell_type: to_plain_object(report) for cell_type, report in reports.items()},
    }


def _validate_original_tokenise_ensembl(context_adata: Any, spec: PerturbGenDataSpec) -> None:
    """Require original tokeniser input IDs to already be canonical ENSG values."""

    var = getattr(context_adata, "var", None)
    columns = getattr(var, "columns", None)
    if var is None or columns is None or spec.ensembl_id_col not in columns:
        return
    for value in var[spec.ensembl_id_col]:
        raw = str(value).strip()
        canonical = normalize_ensembl_id(raw)
        if raw != canonical:
            raise ValueError(
                "Gate-0 tokenise input must already contain canonical Ensembl IDs in the original "
                f"context_h5ad; {raw!r} would only be normalized on the prepared copy"
            )


def _observed_direction_for_run(payload: Mapping[str, Any], ensembl_id: str) -> str:
    for preparation in payload.get("candidates", []):
        if not isinstance(preparation, Mapping) or preparation.get("status") != "pass":
            continue
        invocation = preparation.get("invocation")
        if not isinstance(invocation, Mapping):
            continue
        if str(invocation.get("ensembl_id", "")) != ensembl_id:
            continue
        direction = invocation.get("candidate", {}).get("observed_direction")
        if direction not in ("up", "down"):
            raise ValueError(f"passing invocation for {ensembl_id} has no recorded observed_direction")
        return str(direction)
    raise ValueError(f"E2E report has no passing invocation for {ensembl_id}")


def _assemble_statistical_evidence(
    payload: dict[str, Any],
    *,
    deg_table_path: Path,
    null_distribution_manifest_path: Path,
    output_dir: Path,
    donor_obs_column: str,
    cohort_pairing: str | None = None,
    deg_donor_column: str = "donor",
    deg_gene_column: str = "gene_symbol",
    deg_effect_column: str = "log2fc",
    deg_fdr_column: str = "fdr",
) -> dict[str, Any]:
    """Assemble formal dual-path statistical evidence from the E2E runs (F-01).

    Chains the existing interfaces end to end: per-candidate unperturbed
    quality extraction from the primary-mode within_state h5ad files, formal
    eval-input assembly against the manifest-bound null distributions,
    empirical candidate p aggregation, BH-FDR q-values and the dual-path AND
    decision.  ``cohort_pairing`` (from the Gate-0 data spec) and the DEG
    column names are recorded into the eval-input and report lineage so the
    donor design of the statistics stays provable.  Nothing is fabricated: a
    missing null/quality/seed coverage is a hard error and stays
    ``INCONCLUSIVE`` in the report, never a PASS.
    """

    from dataclasses import asdict

    from src.integration.perturbgen.empirical_pvalue import primary_mode_for_direction
    from src.integration.perturbgen.eval_assembly import (
        build_eval_input_payload,
        resolve_run_artifacts,
        write_eval_input,
    )
    from src.integration.perturbgen.replay_evaluation import (
        load_deg_table,
        replay_dual_path_evaluation,
    )
    from src.integration.perturbgen.results import extract_unperturbed_quality_from_h5ad

    runs = payload.get("perturbgen_runs") or []
    if not runs:
        raise ValueError("statistical evidence assembly requires completed --run-perturbgen stages")

    deg_table = load_deg_table(deg_table_path)
    candidate_quality: dict[str, dict[str, Any]] = {}
    for run in runs:
        ensembl_id = str(run["ensembl_id"])
        gene_symbol = str(run["gene_symbol"])
        observed_direction = _observed_direction_for_run(payload, ensembl_id)
        primary_mode = primary_mode_for_direction(observed_direction)
        artifacts = resolve_run_artifacts(
            run["output_root"],
            gene_symbol,
            prepare_root=run.get("prepare_root"),
        )
        within_primary = {
            artifact.seed: artifact.path for artifact in artifacts["within_state"] if artifact.mode == primary_mode
        }
        if not within_primary:
            raise ValueError(
                f"candidate {ensembl_id} has no within_state {primary_mode} perturb artifacts "
                "for unperturbed quality extraction"
            )
        quality = extract_unperturbed_quality_from_h5ad(
            within_primary,
            deg_table,
            donor_obs_column=donor_obs_column,
            var_gene_column="__index__",
            target_gene=gene_symbol,
            donor_column=deg_donor_column,
            gene_column=deg_gene_column,
            effect_column=deg_effect_column,
            fdr_column=deg_fdr_column,
        )
        candidate_quality[ensembl_id] = asdict(quality)

    eval_payload = build_eval_input_payload(
        payload,
        deg_table_path=deg_table_path,
        null_distribution_manifest_path=null_distribution_manifest_path,
        unperturbed_quality_status="inconclusive",
        candidate_unperturbed_quality=candidate_quality,
        evaluation_mode="formal",
        donor_obs_column=donor_obs_column,
        cohort_pairing=cohort_pairing,
        deg_donor_column=deg_donor_column,
        deg_gene_column=deg_gene_column,
        deg_effect_column=deg_effect_column,
        deg_fdr_column=deg_fdr_column,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_input_path = write_eval_input(
        eval_payload,
        output_dir / "dual_path_eval_input.json",
    )
    report_dir = output_dir / "dual_path_reports"
    if report_dir.exists() and any(report_dir.iterdir()):
        raise ValueError(f"statistical report dir already exists and is not empty: {report_dir}")
    manifest = replay_dual_path_evaluation(
        eval_payload,
        input_path=eval_input_path,
        output_dir=report_dir,
    )
    candidates = [
        {
            "ensembl_id": entry.get("replay", {}).get("candidate", {}).get("ensembl_id"),
            "candidate_index": entry.get("candidate_index"),
            "candidate_gene": entry.get("candidate_gene"),
            "intervention_type": entry.get("intervention_type"),
            "pvalue": entry.get("candidate_pvalue"),
            "q_value": entry.get("q_value"),
            "verdict": entry.get("verdict"),
            "reasons": entry.get("reasons"),
            "scientific_acceptance": entry.get("scientific_acceptance"),
            "report_dir": entry.get("report_dir"),
        }
        for entry in manifest.get("candidates", [])
    ]
    return {
        "status": "assembled",
        "evaluation_mode": "formal",
        "scientific_acceptance": bool(candidates) and all(bool(entry["scientific_acceptance"]) for entry in candidates),
        "deg_table": str(Path(deg_table_path).expanduser().resolve(strict=True)),
        "null_distribution_manifest": str(Path(null_distribution_manifest_path).expanduser().resolve(strict=True)),
        "donor_obs_column": donor_obs_column,
        "pairing": cohort_pairing,
        "deg_columns": {
            "donor": deg_donor_column,
            "gene": deg_gene_column,
            "effect": deg_effect_column,
            "fdr": deg_fdr_column,
        },
        "eval_input": str(eval_input_path),
        "report_manifest": str(Path(report_dir) / "manifest.json"),
        "candidates": candidates,
    }


def _resolve_perturbgen_contract(
    args: argparse.Namespace,
    context_path: Path,
) -> tuple[
    dict[str, Any],
    tuple[PerturbGenDataSpec, Path],
    tuple[int, ...],
    tuple[PerturbationMode, ...],
    int,
]:
    """Validate the --run-perturbgen arguments and bind the Gate-0 contract."""

    if args.perturbgen_config is None:
        raise ValueError("--run-perturbgen requires --perturbgen-config")
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("--seeds must be unique non-negative integers")
    raw_sensitivity_modes = tuple(item.strip() for item in args.sensitivity_modes.split(",") if item.strip())
    if any(mode not in ("pad", "delete") for mode in raw_sensitivity_modes):
        raise ValueError("--sensitivity-modes only accepts pad/delete")
    sensitivity_modes = cast(tuple[PerturbationMode, ...], raw_sensitivity_modes)
    perturbgen_config = load_pipeline_config(args.perturbgen_config)
    pipeline = perturbgen_config.get("pipeline")
    if not isinstance(pipeline, Mapping):
        raise ValueError("PerturbGen config must declare pipeline.random_seed")
    raw_pipeline_seed = pipeline.get("random_seed")
    if isinstance(raw_pipeline_seed, bool) or not isinstance(raw_pipeline_seed, int) or raw_pipeline_seed < 0:
        raise ValueError("PerturbGen config pipeline.random_seed must be a non-negative integer")
    gate0_contract = _validate_perturbgen_tokenise_input(
        perturbgen_config,
        context_path,
        cohort_pairing=args.perturbgen_cohort_pairing,
    )
    return perturbgen_config, gate0_contract, seeds, sensitivity_modes, int(raw_pipeline_seed)


def _resolve_statistical_evidence_inputs(
    args: argparse.Namespace,
    gate0_contract: tuple[PerturbGenDataSpec, Path] | None,
) -> tuple[Path, Path, Path]:
    """Validate and resolve the --assemble-statistical-evidence inputs."""

    if not args.run_perturbgen or args.dry_run:
        raise ValueError("--assemble-statistical-evidence requires a real --run-perturbgen execution")
    if args.deg_table is None:
        raise ValueError("--assemble-statistical-evidence requires --deg-table")
    if args.null_distribution_manifest is None:
        raise ValueError("--assemble-statistical-evidence requires --null-distribution-manifest")
    if gate0_contract is None:
        raise ValueError("statistical assembly requires the Gate-0 data spec of this run")
    statistical_dir = (
        Path(args.statistical_output_dir).expanduser().resolve()
        if args.statistical_output_dir is not None
        else (Path(args.output).expanduser().resolve().parent / "statistical_evidence")
    )
    deg_table_path = Path(args.deg_table).expanduser().resolve(strict=True)
    null_distribution_manifest_path = Path(args.null_distribution_manifest).expanduser().resolve(strict=True)
    return deg_table_path, null_distribution_manifest_path, statistical_dir


def _downstream_delta_frame(output_h5ad: str | Path, *, donor_obs_column: str) -> Any:
    """Build a gene-by-donor PerturbGen delta frame from one result h5ad."""

    import pandas as pd

    from src.integration.perturbgen.results import _aggregate_donor_expression_from_h5ad

    baseline_by_donor, perturbed_by_donor = _aggregate_donor_expression_from_h5ad(
        output_h5ad=output_h5ad,
        donor_obs_column=donor_obs_column,
        var_gene_column="__index__",
    )
    if set(baseline_by_donor) != set(perturbed_by_donor):
        raise ValueError(f"result h5ad donor sets differ for downstream delta evaluation: {output_h5ad}")

    delta_by_donor: dict[str, dict[str, float]] = {}
    for donor in sorted(baseline_by_donor):
        baseline = baseline_by_donor[donor]
        perturbed = perturbed_by_donor[donor]
        if set(baseline) != set(perturbed):
            raise ValueError(f"result h5ad gene sets differ between pred_counts and X for donor {donor!r}")
        delta_by_donor[donor] = {gene: float(perturbed[gene]) - float(baseline[gene]) for gene in baseline}
    if not delta_by_donor:
        raise ValueError(f"result h5ad has no donor rows for downstream delta evaluation: {output_h5ad}")
    return pd.DataFrame(delta_by_donor)


def _assemble_downstream_target_evaluation(
    payload: Mapping[str, Any],
    *,
    sidecar_path: str | Path,
    raw_candidates: Sequence[dict[str, Any]],
    donor_obs_column: str,
) -> dict[str, Any]:
    """Evaluate the sidecar targets against every gated candidate's result h5ad."""

    import pandas as pd

    from src.integration.perturbgen.downstream_target_evaluation import (
        DownstreamTargetSidecar,
        evaluate_target_set_deltas,
        evaluation_to_payload,
        load_downstream_target_sidecar,
    )
    from src.integration.perturbgen.eval_assembly import resolve_run_artifacts

    resolved_sidecar_path = Path(sidecar_path).expanduser().resolve(strict=True)
    sidecar = load_downstream_target_sidecar(resolved_sidecar_path)

    candidate_records: dict[str, tuple[int, dict[str, Any], str]] = {}
    for row, candidate in enumerate(raw_candidates):
        raw_cell_type = candidate.get("cell_type")
        cell_type = str(raw_cell_type).strip() if raw_cell_type is not None else ""
        if not cell_type:
            raise ValueError(f"candidate {row} has no cell_type for downstream sidecar binding")
        if cell_type != sidecar.cell_type:
            raise ValueError(
                f"downstream sidecar cell_type {sidecar.cell_type!r} does not match candidate {row} cell_type {cell_type!r}"
            )
        raw_ensembl_id = candidate.get("ensembl_id")
        if not isinstance(raw_ensembl_id, str) or not raw_ensembl_id.strip():
            raise ValueError(f"candidate {row} has no ensembl_id for downstream sidecar binding")
        ensembl_id = normalize_ensembl_id(raw_ensembl_id)
        if ensembl_id in candidate_records:
            raise ValueError(f"candidate spec contains duplicated canonical Ensembl ID {ensembl_id}")
        source_activity_id = candidate.get("source_activity_id")
        if not isinstance(source_activity_id, str) or not source_activity_id.strip():
            raise ValueError(f"candidate {row} requires source_activity_id when using a downstream sidecar")
        source = sidecar.source_by_ensembl(ensembl_id)
        if source.source_activity_id != source_activity_id.strip():
            raise ValueError(
                f"downstream sidecar source relation for {ensembl_id} does not match candidate {row}: "
                f"{source.source_activity_id!r} != {source_activity_id.strip()!r}"
            )
        gene_symbol = str(candidate.get("gene_symbol") or "").strip().upper()
        if not gene_symbol or source.gene_symbol != gene_symbol:
            raise ValueError(
                f"downstream sidecar source gene for {ensembl_id} does not match candidate {row}: "
                f"{source.gene_symbol!r} != {gene_symbol!r}"
            )
        candidate_records[ensembl_id] = (row, candidate, source_activity_id.strip())

    sidecar_ids = [source.ensembl_id for source in sidecar.sources]
    if len(sidecar_ids) != len(set(sidecar_ids)):
        raise ValueError("downstream sidecar contains duplicated canonical source Ensembl IDs")
    candidate_ids = set(candidate_records)
    sidecar_id_set = set(sidecar_ids)
    if candidate_ids != sidecar_id_set:
        missing = sorted(candidate_ids - sidecar_id_set)
        extra = sorted(sidecar_id_set - candidate_ids)
        raise ValueError(f"downstream sidecar/spec source mismatch: missing={missing}, extra={extra}")

    preparations = payload.get("candidates")
    if not isinstance(preparations, list):
        raise ValueError("E2E report candidates must be a list for downstream target evaluation")
    gated: list[tuple[str, int, dict[str, Any], str]] = []
    gated_ids: set[str] = set()
    for preparation in preparations:
        if not isinstance(preparation, Mapping):
            raise ValueError("E2E report candidate preparation must be a mapping")
        if preparation.get("status") != "pass":
            continue
        invocation = preparation.get("invocation")
        if not isinstance(invocation, Mapping):
            raise ValueError("passing E2E candidate has no invocation for downstream target evaluation")
        raw_ensembl_id = invocation.get("ensembl_id")
        if not isinstance(raw_ensembl_id, str) or not raw_ensembl_id.strip():
            raise ValueError("passing E2E invocation has no ensembl_id for downstream target evaluation")
        ensembl_id = normalize_ensembl_id(raw_ensembl_id)
        if ensembl_id in gated_ids:
            raise ValueError(f"E2E report contains duplicated gated candidate {ensembl_id}")
        if ensembl_id not in candidate_records:
            raise ValueError(f"E2E gated candidate {ensembl_id} is absent from the candidate spec/sidecar")
        row, candidate, source_activity_id = candidate_records[ensembl_id]
        invocation_gene = str(invocation.get("gene_symbol") or "").strip().upper()
        expected_gene = str(candidate.get("gene_symbol") or "").strip().upper()
        if invocation_gene != expected_gene:
            raise ValueError(
                f"E2E invocation gene for {ensembl_id} does not match candidate spec: "
                f"{invocation_gene!r} != {expected_gene!r}"
            )
        gated.append((ensembl_id, row, candidate, source_activity_id))
        gated_ids.add(ensembl_id)
    if not gated:
        raise ValueError("downstream target evaluation requires at least one gated candidate")

    raw_runs = payload.get("perturbgen_runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("downstream target evaluation requires completed --run-perturbgen result h5ad files")
    runs_by_ensembl: dict[str, Mapping[str, Any]] = {}
    for run in raw_runs:
        if not isinstance(run, Mapping):
            raise ValueError("E2E perturbgen run must be a mapping for downstream target evaluation")
        raw_ensembl_id = run.get("ensembl_id")
        if not isinstance(raw_ensembl_id, str) or not raw_ensembl_id.strip():
            raise ValueError("E2E perturbgen run has no ensembl_id for downstream target evaluation")
        ensembl_id = normalize_ensembl_id(raw_ensembl_id)
        if ensembl_id not in gated_ids:
            raise ValueError(f"E2E perturbgen run {ensembl_id} has no gated candidate")
        if ensembl_id in runs_by_ensembl:
            raise ValueError(f"E2E report contains duplicated perturbgen run for {ensembl_id}")
        expected_gene = str(candidate_records[ensembl_id][1].get("gene_symbol") or "").strip().upper()
        run_gene = str(run.get("gene_symbol") or "").strip().upper()
        if run_gene != expected_gene:
            raise ValueError(
                f"E2E perturbgen run gene for {ensembl_id} does not match candidate spec: "
                f"{run_gene!r} != {expected_gene!r}"
            )
        runs_by_ensembl[ensembl_id] = run
    if set(runs_by_ensembl) != gated_ids:
        raise ValueError(
            "E2E perturbgen runs do not cover every gated candidate: "
            f"missing={sorted(gated_ids - set(runs_by_ensembl))}"
        )

    evaluation_payload: dict[str, Any] | None = None
    all_result_paths: list[str] = []
    candidate_lineage: list[dict[str, Any]] = []
    for ensembl_id, row, candidate, source_activity_id in gated:
        run = runs_by_ensembl[ensembl_id]
        output_root = run.get("output_root")
        if not isinstance(output_root, (str, Path)) or not str(output_root).strip():
            raise ValueError(f"E2E perturbgen run for {ensembl_id} has no output_root")
        prepare_root = run.get("prepare_root")
        if not isinstance(prepare_root, (str, Path)) or not str(prepare_root).strip():
            raise ValueError(f"E2E perturbgen run for {ensembl_id} has no prepare_root")
        gene_symbol = str(candidate.get("gene_symbol") or "").strip().upper()
        artifacts = resolve_run_artifacts(
            output_root,
            gene_symbol,
            prepare_root=prepare_root,
        )
        frames: list[Any] = []
        artifact_lineage: list[dict[str, Any]] = []
        gene_index: tuple[object, ...] | None = None
        for path_kind in sorted(artifacts):
            for artifact in sorted(
                artifacts[path_kind],
                key=lambda item: (str(item.mode), int(item.seed), str(item.path)),
            ):
                result_path = Path(artifact.path).expanduser().resolve(strict=True)
                try:
                    frame = _downstream_delta_frame(result_path, donor_obs_column=donor_obs_column)
                except (KeyError, OSError, ValueError) as exc:
                    raise ValueError(
                        f"downstream result {result_path} must provide readable donor obs, pred_counts layer and X: {exc}"
                    ) from exc
                current_index = tuple(frame.index.tolist())
                if gene_index is None:
                    gene_index = current_index
                elif set(current_index) != set(gene_index):
                    raise ValueError(f"resolved result h5ad gene sets differ for candidate {ensembl_id}")
                frame = frame.reindex(gene_index)
                frame.columns = [
                    f"{path_kind}|mode={artifact.mode}|seed={artifact.seed}|donor={donor}" for donor in frame.columns
                ]
                frames.append(frame)
                result_path_text = str(result_path)
                all_result_paths.append(result_path_text)
                artifact_lineage.append(
                    {
                        "path": path_kind,
                        "mode": str(artifact.mode),
                        "seed": int(artifact.seed),
                        "result_h5ad": result_path_text,
                        "stage_manifest": str(Path(artifact.stage_manifest).expanduser().resolve(strict=True)),
                        "tokenise_stage_manifest": str(
                            Path(artifact.tokenise_stage_manifest).expanduser().resolve(strict=True)
                        ),
                    }
                )
        if not frames:
            raise ValueError(f"no resolved result h5ad artifacts for gated candidate {ensembl_id}")
        delta_frame = pd.concat(frames, axis=1)
        source = sidecar.source_by_ensembl(ensembl_id)
        candidate_sidecar = DownstreamTargetSidecar(
            cell_type=sidecar.cell_type,
            sources=(source,),
            lineage=sidecar.lineage,
        )
        serialized = evaluation_to_payload(
            evaluate_target_set_deltas(delta_frame, candidate_sidecar),
            cell_type=sidecar.cell_type,
            delta_matrix_source=";".join(item["result_h5ad"] for item in artifact_lineage),
        )
        if evaluation_payload is None:
            evaluation_payload = serialized
        else:
            evaluation_payload["sources"].update(serialized["sources"])
        candidate_lineage.append(
            {
                "candidate_index": row,
                "ensembl_id": ensembl_id,
                "gene_symbol": gene_symbol,
                "source_activity_id": source_activity_id,
                "result_h5ad": artifact_lineage,
            }
        )

    if evaluation_payload is None:
        raise ValueError("downstream target evaluation produced no library payload")
    evaluation_payload["delta_matrix_source"] = ";".join(all_result_paths)
    return {
        "payload": evaluation_payload,
        "lineage": {
            "sidecar_path": str(resolved_sidecar_path),
            "sidecar_lineage": to_plain_object(sidecar.lineage),
            "cell_type": sidecar.cell_type,
            "donor_obs_column": donor_obs_column,
            "baseline_layer": "pred_counts",
            "perturbed_layer": "X",
            "candidates": candidate_lineage,
            "gate_boundary": "source three-way direction gate remains the only pass/fail decision",
        },
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    davf_config = _load_davf_config(args.davf_config)
    context_path, raw_candidates = _load_candidate_spec(args.candidate_spec)
    proposals = tuple(_build_proposal(candidate, row) for row, candidate in enumerate(raw_candidates))
    downstream_required = (
        "cell_type",
        "ptm_context",
        "observed_log2fc",
        "observed_fdr",
        "observed_direction",
        "semantic_context",
    )
    for row, candidate in enumerate(raw_candidates):
        missing = [key for key in downstream_required if key not in candidate]
        if missing:
            raise ValueError(f"candidate {row} is missing direction-gate fields: {', '.join(missing)}")
        if not isinstance(candidate["semantic_context"], Mapping):
            raise ValueError(f"candidate {row} semantic_context must be a mapping")
        SemanticContext.from_mapping(candidate["semantic_context"])
    perturbgen_config: dict[str, Any] | None = None
    perturbgen_gate0_contract: tuple[PerturbGenDataSpec, Path] | None = None
    perturbgen_pipeline_seed = 0
    seeds: tuple[int, ...] = ()
    sensitivity_modes: tuple[PerturbationMode, ...] = ()
    if args.run_perturbgen:
        (
            perturbgen_config,
            perturbgen_gate0_contract,
            seeds,
            sensitivity_modes,
            perturbgen_pipeline_seed,
        ) = _resolve_perturbgen_contract(args, context_path)
    elif args.perturbgen_config is not None:
        raise ValueError("--perturbgen-config requires --run-perturbgen")
    statistical_inputs: tuple[Path, Path, Path] | None = None
    if args.assemble_statistical_evidence:
        statistical_inputs = _resolve_statistical_evidence_inputs(args, perturbgen_gate0_contract)
    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError("anndata is required to run real DAVF/scVI inference") from exc

    context_adata = ad.read_h5ad(context_path)
    context_indices = [int(candidate["context_cell_index"]) for candidate in raw_candidates]
    if any(index >= context_adata.n_obs for index in context_indices):
        raise IndexError(
            "candidate context_cell_index is outside the context AnnData row range: "
            f"n_obs={context_adata.n_obs}, indices={context_indices}"
        )
    perturbgen_gate0 = None
    if perturbgen_gate0_contract is not None:
        spec, tokenise_path = perturbgen_gate0_contract
        perturbgen_gate0 = _preflight_perturbgen_context(
            context_adata,
            context_path,
            raw_candidates,
            spec=spec,
            tokenise_path=tokenise_path,
        )
    # Copying the selected rows makes the row alignment explicit and avoids
    # relying on anndata backed slicing internals during scVI decoding.
    selected_context = context_adata[context_indices].copy()

    davf = DAVFInferenceModule(davf_config)
    scvi_adapter = davf.load_scvi_adapter(selected_context)
    z_0 = scvi_adapter.encode(selected_context)
    orchestrator = DAVFPerturbGenOrchestrator(davf_module=davf)

    preparations = orchestrator.prepare_candidates(
        proposals,
        z_0,
        cell_type=[candidate.get("cell_type", "") for candidate in raw_candidates],
        ptm_context=[candidate.get("ptm_context", "") for candidate in raw_candidates],
        observed_log2fc=[float(candidate["observed_log2fc"]) for candidate in raw_candidates],
        observed_fdr=[float(candidate["observed_fdr"]) for candidate in raw_candidates],
        observed_direction=[candidate.get("observed_direction") for candidate in raw_candidates],
        semantic_context=[candidate["semantic_context"] for candidate in raw_candidates],
        scvi_adapter=scvi_adapter,
        scvi_context=selected_context,
        perturbgen_config_path=args.perturbgen_config,
        seed=perturbgen_pipeline_seed,
    )

    payload: dict[str, Any] = {
        "schema_version": "davf_perturbgen_e2e/v1",
        "davf_config": str(Path(args.davf_config).expanduser().resolve()),
        "intervention_type": davf_config.intervention_type,
        "context_h5ad": str(context_path),
        "candidates": [preparation.to_dict() for preparation in preparations],
        "merged_gated_routes": merge_route_preparations(preparations),
        "perturbgen_runs": [],
    }
    payload["statistical_evidence"] = {
        "status": "inconclusive",
        "scientific_acceptance": False,
        "reason": "statistical_evidence_not_assembled" if args.run_perturbgen else "perturbgen_not_requested",
    }
    if args.run_perturbgen:
        payload["statistical_evidence"]["interfaces"] = [
            "src.integration.perturbgen.null_generation",
            "src.integration.perturbgen.reports",
            "src.integration.perturbgen.empirical_pvalue",
            "src.integration.perturbgen.dual_path",
        ]
    if perturbgen_gate0 is not None:
        payload["perturbgen_gate0"] = perturbgen_gate0
    try:
        donor_split = optional_donor_split_from_args(
            train_donors=args.train_donors,
            held_out_donors=args.held_out_donors,
            require=bool(args.require_donor_split or args.frozen_cohort_manifest),
        )
    except DonorSplitError as exc:
        raise ValueError(str(exc)) from exc
    if donor_split is not None and args.frozen_cohort_manifest is not None:
        from src.integration.perturbgen.frozen_cohort import load_frozen_manifest

        frozen = load_frozen_manifest(args.frozen_cohort_manifest)
        if frozen.pairing != args.perturbgen_cohort_pairing:
            raise ValueError(
                "--perturbgen-cohort-pairing "
                f"{args.perturbgen_cohort_pairing!r} does not match the frozen manifest pairing {frozen.pairing!r}"
            )
        bind_frozen_donor_split(donor_split, frozen)
    if donor_split is not None:
        payload["donor_split"] = donor_split.to_payload()
        if perturbgen_config is not None:
            pipeline_section = perturbgen_config.setdefault("pipeline", {})
            if not isinstance(pipeline_section, dict):
                raise ValueError("PerturbGen config pipeline must be a mapping")
            pipeline_section["donor_split"] = donor_split.to_payload()
            pipeline_section["train_donors"] = list(donor_split.train_donors)
            pipeline_section["held_out_donors"] = list(donor_split.held_out_donors)

    if args.run_perturbgen:
        runner = PerturbGenRunner(gpu_lock_file=args.gpu_lock_file)
        base_output = (
            Path(args.perturbgen_output_root).expanduser().resolve()
            if args.perturbgen_output_root is not None
            else (Path(args.output).expanduser().resolve().parent / "perturbgen")
        )
        gated: list[tuple[Any, PerturbGenInvocation]] = []
        for preparation in preparations:
            invocation = preparation.invocation
            if invocation is not None:
                gated.append((preparation, invocation))
        prepare_registry: dict[tuple[str, str], str] = {}
        prepare_root: Path | None = None
        if gated:
            prepare_root = base_output / orchestrator.intervention_type / "_prepare"
            prepare_plans = build_shared_prepare_plans(
                cast(dict[str, Any], perturbgen_config),
                output_root=prepare_root,
                project_root=PROJECT_ROOT,
            )
            prepare_results = runner.run_pipeline(prepare_plans, resume=args.resume, dry_run=args.dry_run)
            payload["perturbgen_prepare"] = {
                "output_root": str(prepare_root),
                "stage_names": [plan.name for plan in prepare_plans],
                "results": to_plain_object(prepare_results),
            }
            if not args.dry_run:
                for result in prepare_results:
                    if not isinstance(result, StageExecutionResult):
                        raise TypeError("non-dry-run prepare stages must return StageExecutionResult objects")
                    for name, path in result.artifacts.items():
                        prepare_registry[(result.stage, name)] = str(path)
        for _, invocation in gated:
            candidate_root = base_output / orchestrator.intervention_type / invocation.ensembl_id
            stage_results = orchestrator.run_perturbgen(
                invocation,
                cast(dict[str, Any], perturbgen_config),
                runner=runner,
                output_root=candidate_root,
                resume=args.resume,
                dry_run=args.dry_run,
                project_root=PROJECT_ROOT,
                seeds=seeds,
                sensitivity_modes=sensitivity_modes,
                skip_prepare_stages=True,
                prepare_artifact_paths=prepare_registry if not args.dry_run else None,
            )
            payload["perturbgen_runs"].append(
                {
                    "intervention_type": invocation.intervention_type,
                    "gene_symbol": invocation.gene_symbol,
                    "ensembl_id": invocation.ensembl_id,
                    "output_root": str(candidate_root),
                    "prepare_root": str(prepare_root) if prepare_root is not None else None,
                    "stages": to_plain_object(stage_results),
                }
            )
    elif args.dry_run or args.resume:
        raise ValueError("--dry-run/--resume require --run-perturbgen")

    downstream_sidecar_path = getattr(args, "downstream_target_sidecar", None)
    if downstream_sidecar_path is not None:
        if not args.run_perturbgen:
            raise ValueError("--downstream-target-sidecar requires --run-perturbgen")
        if args.dry_run:
            raise ValueError(
                "--downstream-target-sidecar requires real PerturbGen result h5ad files; --dry-run cannot evaluate"
            )
        if perturbgen_gate0_contract is None:
            raise ValueError("downstream target evaluation requires the Gate-0 data spec of this run")
        payload["downstream_target_evaluation"] = _assemble_downstream_target_evaluation(
            payload,
            sidecar_path=downstream_sidecar_path,
            raw_candidates=raw_candidates,
            donor_obs_column=perturbgen_gate0_contract[0].donor_col,
        )

    if statistical_inputs is not None:
        assert perturbgen_gate0_contract is not None
        spec, _ = perturbgen_gate0_contract
        deg_table_path, null_distribution_manifest_path, statistical_dir = statistical_inputs
        payload["statistical_evidence"] = _assemble_statistical_evidence(
            payload,
            deg_table_path=deg_table_path,
            null_distribution_manifest_path=null_distribution_manifest_path,
            output_dir=statistical_dir,
            donor_obs_column=spec.donor_col,
            cohort_pairing=spec.pairing,
            deg_donor_column=args.deg_donor_column,
            deg_gene_column=args.deg_gene_column,
            deg_effect_column=args.deg_effect_column,
            deg_fdr_column=args.deg_fdr_column,
        )

    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--davf-config", type=Path, required=True)
    parser.add_argument("--candidate-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--perturbgen-config", type=Path)
    parser.add_argument(
        "--perturbgen-cohort-pairing",
        choices=("within_donor", "between_donor"),
        default="within_donor",
        help=(
            "cohort donor/state design for Gate-0: within_donor requires >=3 donors shared "
            "across the two states (perturbation cohorts); between_donor requires donor-disjoint "
            "state groups with >=3 donors each (case-control cohorts)"
        ),
    )
    parser.add_argument("--perturbgen-output-root", type=Path)
    parser.add_argument("--gpu-lock-file", type=Path, default=Path("outputs/perturbgen/.gpu.lock"))
    parser.add_argument("--run-perturbgen", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--seeds",
        default="0",
        help="comma-separated random seeds for the perturb stages (formal verdicts need >=3)",
    )
    parser.add_argument(
        "--sensitivity-modes",
        default="",
        help="comma-separated KO sensitivity modes (pad,delete) analysed besides the primary mask mode",
    )
    parser.add_argument(
        "--train-donors",
        default=None,
        help="comma-separated training donor IDs bound into the E2E report and pipeline fingerprint",
    )
    parser.add_argument(
        "--held-out-donors",
        default=None,
        help="comma-separated held-out donor IDs; must be disjoint from --train-donors",
    )
    parser.add_argument(
        "--frozen-cohort-manifest",
        type=Path,
        default=None,
        help="optional frozen M6 manifest; donor lists must match exactly including SHA",
    )
    parser.add_argument(
        "--require-donor-split",
        action="store_true",
        help="fail if train/held-out donor lists are omitted",
    )
    parser.add_argument(
        "--assemble-statistical-evidence",
        action="store_true",
        help="assemble formal null/quality/p/q/dual-path evidence after the perturbgen stages (F-01)",
    )
    parser.add_argument(
        "--deg-table",
        type=Path,
        default=None,
        help="donor-level DEG table (csv/json) required by --assemble-statistical-evidence",
    )
    parser.add_argument(
        "--deg-donor-column",
        default="donor",
        help="DEG table column carrying the donor id (default: donor)",
    )
    parser.add_argument(
        "--deg-gene-column",
        default="gene_symbol",
        help="DEG table column carrying the gene identifier (default: gene_symbol)",
    )
    parser.add_argument(
        "--deg-effect-column",
        default="log2fc",
        help="DEG table column carrying the effect size (default: log2fc)",
    )
    parser.add_argument(
        "--deg-fdr-column",
        default="fdr",
        help="DEG table column carrying the FDR (default: fdr)",
    )
    parser.add_argument(
        "--null-distribution-manifest",
        type=Path,
        default=None,
        help="manifest-bound matched-null distribution index required by --assemble-statistical-evidence",
    )
    parser.add_argument(
        "--statistical-output-dir",
        type=Path,
        default=None,
        help="output directory for the assembled statistical evidence",
    )
    parser.add_argument(
        "--downstream-target-sidecar",
        type=Path,
        default=None,
        help="explicit downstream target-set sidecar to evaluate against gated perturb result h5ad files",
    )
    args = parser.parse_args(argv)

    payload = _run(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(to_plain_object(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(to_plain_object(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
