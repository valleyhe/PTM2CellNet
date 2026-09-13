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
from dataclasses import fields, is_dataclass
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
    merge_route_preparations,
)
from src.integration.perturbgen.runner import PerturbGenRunner  # noqa: E402
from src.integration.perturbgen.contracts import (  # noqa: E402
    PTMSiteDirectionProposal,
    PerturbGenDataSpec,
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


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in value.__dict__.items()}
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _validate_perturbgen_tokenise_input(
    config: Mapping[str, Any],
    context_path: Path,
) -> tuple[PerturbGenDataSpec, Path]:
    """Bind the PerturbGen tokenise input and metadata to the E2E context."""

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
        "data_spec": _serialize(spec),
        "cell_types": {cell_type: _serialize(report) for cell_type, report in reports.items()},
    }


def _validate_original_tokenise_ensembl(context_adata: Any, spec: PerturbGenDataSpec) -> None:
    """Require original tokeniser input IDs to already be canonical ENSG values."""

    var = getattr(context_adata, "var", None)
    columns = getattr(var, "columns", None)
    if columns is None or spec.ensembl_id_col not in columns:
        return
    for value in var[spec.ensembl_id_col]:
        raw = str(value).strip()
        canonical = normalize_ensembl_id(raw)
        if raw != canonical:
            raise ValueError(
                "Gate-0 tokenise input must already contain canonical Ensembl IDs in the original "
                f"context_h5ad; {raw!r} would only be normalized on the prepared copy"
            )


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
    if args.run_perturbgen:
        if args.perturbgen_config is None:
            raise ValueError("--run-perturbgen requires --perturbgen-config")
        seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
        if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
            raise ValueError("--seeds must be unique non-negative integers")
        sensitivity_modes = tuple(item.strip() for item in args.sensitivity_modes.split(",") if item.strip())
        if any(mode not in ("pad", "delete") for mode in sensitivity_modes):
            raise ValueError("--sensitivity-modes only accepts pad/delete")
        perturbgen_config = load_pipeline_config(args.perturbgen_config)
        pipeline = perturbgen_config.get("pipeline")
        if not isinstance(pipeline, Mapping):
            raise ValueError("PerturbGen config must declare pipeline.random_seed")
        raw_pipeline_seed = pipeline.get("random_seed")
        if isinstance(raw_pipeline_seed, bool) or not isinstance(raw_pipeline_seed, int) or raw_pipeline_seed < 0:
            raise ValueError("PerturbGen config pipeline.random_seed must be a non-negative integer")
        perturbgen_pipeline_seed = int(raw_pipeline_seed)
        perturbgen_gate0_contract = _validate_perturbgen_tokenise_input(
            perturbgen_config,
            context_path,
        )
    elif args.perturbgen_config is not None:
        raise ValueError("--perturbgen-config requires --run-perturbgen")
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
        for preparation in preparations:
            if preparation.invocation is None:
                continue
            invocation = preparation.invocation
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
            )
            payload["perturbgen_runs"].append(
                {
                    "intervention_type": invocation.intervention_type,
                    "gene_symbol": invocation.gene_symbol,
                    "ensembl_id": invocation.ensembl_id,
                    "output_root": str(candidate_root),
                    "stages": _serialize(stage_results),
                }
            )
    elif args.dry_run or args.resume:
        raise ValueError("--dry-run/--resume require --run-perturbgen")

    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--davf-config", type=Path, required=True)
    parser.add_argument("--candidate-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--perturbgen-config", type=Path)
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
    args = parser.parse_args(argv)

    payload = _run(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_serialize(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_serialize(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
