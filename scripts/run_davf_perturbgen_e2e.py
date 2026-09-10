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
          "observed_direction": "down"
        }
      ]
    }

By default this command only performs real DAVF inference and writes the
direction-gated invocation manifest.  ``--run-perturbgen`` additionally runs
the isolated six-stage external pipeline for every gated candidate.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import sys
from typing import Any

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
from src.integration.perturbgen.contracts import PTMSiteDirectionProposal  # noqa: E402
from src.models.davf_inference import DAVFInferenceConfig, DAVFInferenceModule  # noqa: E402


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
    )
    for row, candidate in enumerate(raw_candidates):
        missing = [key for key in downstream_required if key not in candidate]
        if missing:
            raise ValueError(
                f"candidate {row} is missing direction-gate fields: {', '.join(missing)}"
            )
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
        observed_log2fc=[candidate.get("observed_log2fc") for candidate in raw_candidates],
        observed_fdr=[candidate.get("observed_fdr") for candidate in raw_candidates],
        observed_direction=[candidate.get("observed_direction") for candidate in raw_candidates],
        scvi_adapter=scvi_adapter,
        scvi_context=selected_context,
        perturbgen_config_path=args.perturbgen_config,
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

    if args.run_perturbgen:
        if args.perturbgen_config is None:
            raise ValueError("--run-perturbgen requires --perturbgen-config")
        load_pipeline_config(args.perturbgen_config)
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
            candidate_root = base_output / davf_config.intervention_type / invocation.ensembl_id
            stage_results = orchestrator.run_perturbgen(
                invocation,
                args.perturbgen_config,
                runner=runner,
                output_root=candidate_root,
                resume=args.resume,
                dry_run=args.dry_run,
                project_root=PROJECT_ROOT,
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
    args = parser.parse_args(argv)

    payload = _run(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_serialize(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_serialize(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
