#!/usr/bin/env python3
"""从版本化 JSON 重放 PerturbGen 双路径评估并产出报告。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.perturbgen.dual_path import evaluate_dual_path_candidate  # noqa: E402
from src.integration.perturbgen.reports import (  # noqa: E402
    build_candidate_report_payload,
    build_candidate_summary_dataframe,
    save_report_artifacts,
)
from src.integration.perturbgen.results import (  # noqa: E402
    benjamini_hochberg,
    extract_path_result_from_perturbgen_h5ad,
)

INPUT_SCHEMA_VERSION = "perturbgen_dual_path_eval/v1"
MANIFEST_SCHEMA_VERSION = "perturbgen_dual_path_manifest/v1"
_VALID_PATHS = {"source_intervention", "within_state"}
_VALID_MODES = {"mask", "pad", "delete", "overexpress"}
_VALID_QUALITY = {"pass", "fail", "inconclusive"}
_VALID_DIRECTIONS = {"up", "down"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    input_path = args.input_json.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output_dir already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = json.loads(input_path.read_text(encoding="utf-8"))
    _validate_top_level_spec(spec)

    candidates = list(spec["candidates"])
    q_values = benjamini_hochberg(
        [_coerce_probability(item.get("candidate_pvalue"), name="candidate_pvalue") for item in candidates]
    )

    payloads: list[dict[str, Any]] = []
    candidate_entries: list[dict[str, Any]] = []
    run_id = _require_non_empty_string(spec.get("run_id", input_path.stem), name="run_id")

    for index, (candidate_spec, q_value) in enumerate(zip(candidates, q_values, strict=True), start=1):
        payload, manifest_entry = _evaluate_candidate(
            candidate_spec=candidate_spec,
            q_value=q_value,
            candidate_index=index,
            input_path=input_path,
            input_base_dir=input_path.parent,
            output_dir=output_dir,
            run_id=run_id,
        )
        payloads.append(payload)
        candidate_entries.append(manifest_entry)

    summary = build_candidate_summary_dataframe(payloads)
    summary_path = output_dir / "candidate_summary.csv"
    summary.to_csv(summary_path, index=False)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "input_json": str(input_path),
        "input_sha256": _sha256_file(input_path),
        "summary_csv": str(summary_path),
        "candidate_count": len(candidate_entries),
        "candidates": candidate_entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(_to_plain_object(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


def _evaluate_candidate(
    *,
    candidate_spec: Mapping[str, Any],
    q_value: float,
    candidate_index: int,
    input_path: Path,
    input_base_dir: Path,
    output_dir: Path,
    run_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = _require_mapping(candidate_spec.get("candidate"), name="candidate")
    candidate_gene = _require_non_empty_string(
        candidate.get("gene_symbol") or candidate.get("candidate_gene"),
        name="candidate.gene_symbol",
    )
    observed_direction = _require_choice(
        candidate_spec.get("observed_direction"),
        valid_values=_VALID_DIRECTIONS,
        name="observed_direction",
    )
    unperturbed_quality_status = _require_choice(
        candidate_spec.get("unperturbed_quality_status"),
        valid_values=_VALID_QUALITY,
        name="unperturbed_quality_status",
    )
    candidate_pvalue = _coerce_probability(
        candidate_spec.get("candidate_pvalue"),
        name="candidate_pvalue",
    )
    runs = candidate_spec.get("runs")
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)) or not runs:
        raise ValueError("candidate runs must be a non-empty sequence")

    extracted_runs = [_extract_run(run_spec, input_base_dir=input_base_dir) for run_spec in runs]
    decision = evaluate_dual_path_candidate(
        [item["path_result"] for item in extracted_runs],
        observed_direction=observed_direction,
        q_value=q_value,
        candidate_gene=candidate_gene,
        unperturbed_quality_status=unperturbed_quality_status,
    )

    candidate_dir = output_dir / f"{candidate_index:03d}_{_slug(candidate_gene)}"
    manifest = {
        "run_id": run_id,
        "input_json": str(input_path),
        "candidate_index": candidate_index,
        "candidate_pvalue": candidate_pvalue,
        "candidate_qvalue": q_value,
        "replay": {
            "candidate": _to_plain_object(candidate),
            "observed_direction": observed_direction,
            "unperturbed_quality_status": unperturbed_quality_status,
            "runs": extracted_runs,
        },
    }
    payload = build_candidate_report_payload(
        decision,
        manifest=manifest,
        candidate=candidate,
    )
    artifacts = save_report_artifacts(payload, candidate_dir)
    return payload, {
        "candidate_gene": candidate_gene,
        "candidate_index": candidate_index,
        "candidate_pvalue": candidate_pvalue,
        "q_value": q_value,
        "verdict": decision.verdict,
        "reasons": list(decision.reasons),
        "report_dir": str(candidate_dir),
        "artifacts": artifacts,
        "input_runs": extracted_runs,
    }


def _extract_run(run_spec: Any, *, input_base_dir: Path) -> dict[str, Any]:
    run = _require_mapping(run_spec, name="run")
    path = _require_choice(run.get("path"), valid_values=_VALID_PATHS, name="path")
    mode = _require_choice(run.get("mode"), valid_values=_VALID_MODES, name="mode")
    seed = _require_non_negative_int(run.get("seed"), name="seed")
    output_h5ad = _resolve_input_path(
        input_base_dir,
        _require_non_empty_string(run.get("output_h5ad"), name="output_h5ad"),
    )
    raw_provenance = _require_mapping(run.get("h5ad_provenance"), name="h5ad_provenance")
    h5ad_provenance = dict(raw_provenance)
    h5ad_provenance["stage_manifest"] = str(
        _resolve_input_path(
            input_base_dir,
            _require_non_empty_string(raw_provenance.get("stage_manifest"), name="h5ad_provenance.stage_manifest"),
        )
    )
    donor_obs_column = _require_non_empty_string(run.get("donor_obs_column"), name="donor_obs_column")
    var_gene_column = _require_non_empty_string(run.get("var_gene_column"), name="var_gene_column")
    deg_table_path = _resolve_input_path(
        input_base_dir,
        _require_non_empty_string(run.get("deg_table_path"), name="deg_table_path"),
    )
    null_distribution_path = _resolve_input_path(
        input_base_dir,
        _require_non_empty_string(run.get("null_distribution_path"), name="null_distribution_path"),
    )

    extraction = extract_path_result_from_perturbgen_h5ad(
        output_h5ad=output_h5ad,
        h5ad_provenance=h5ad_provenance,
        deg_table=_load_deg_table(deg_table_path),
        null_distribution=_load_null_distribution(null_distribution_path),
        path=path,
        mode=mode,
        seed=seed,
        donor_obs_column=donor_obs_column,
        var_gene_column=var_gene_column,
        target_gene=_optional_non_empty_string(run.get("target_gene")),
        donor_column=_require_non_empty_string(run.get("deg_donor_column"), name="deg_donor_column"),
        gene_column=_require_non_empty_string(run.get("deg_gene_column"), name="deg_gene_column"),
        effect_column=_require_non_empty_string(run.get("deg_effect_column"), name="deg_effect_column"),
        fdr_column=_require_non_empty_string(run.get("deg_fdr_column"), name="deg_fdr_column"),
        fdr_threshold=float(run.get("fdr_threshold", 0.05)),
        min_training_donors=int(run.get("min_training_donors", 2)),
        min_evaluable_donors=int(run.get("min_evaluable_donors", 3)),
        top_k=int(run.get("top_k", 50)),
        bootstrap_iterations=int(run.get("bootstrap_iterations", 1000)),
        bootstrap_seed=None if run.get("bootstrap_seed") is None else int(run["bootstrap_seed"]),
    )
    return {
        "path": path,
        "mode": mode,
        "seed": seed,
        "output_h5ad": str(output_h5ad.expanduser().resolve(strict=True)),
        "deg_table_path": str(deg_table_path.expanduser().resolve(strict=True)),
        "null_distribution_path": str(null_distribution_path.expanduser().resolve(strict=True)),
        "h5ad_provenance": _to_plain_object(h5ad_provenance),
        "baseline_matrix": extraction.baseline_matrix,
        "perturbed_matrix": extraction.perturbed_matrix,
        "bootstrap_ci": list(extraction.bootstrap_ci) if extraction.bootstrap_ci is not None else None,
        "donor_scores": _to_plain_object(extraction.donor_scores),
        "path_result": _to_plain_object(extraction.path_result),
    }


def _validate_top_level_spec(spec: Any) -> None:
    mapping = _require_mapping(spec, name="input_json")
    version = _require_non_empty_string(mapping.get("schema_version"), name="schema_version")
    if version != INPUT_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {version!r}; expected {INPUT_SCHEMA_VERSION!r}")
    candidates = mapping.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or not candidates:
        raise ValueError("candidates must be a non-empty sequence")


def _load_deg_table(path: Path) -> pd.DataFrame:
    resolved = path.expanduser().resolve(strict=True)
    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(resolved)
    if suffix == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, Mapping):
            records = payload.get("records")
            if isinstance(records, list):
                return pd.DataFrame(records)
        raise ValueError("deg_table_path JSON must be a record list or {'records': [...]}")
    raise ValueError("deg_table_path must end with .csv or .json")


def _load_null_distribution(path: Path) -> list[float]:
    resolved = path.expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    values = payload.get("values") if isinstance(payload, Mapping) else payload
    if not isinstance(values, list) or not values:
        raise ValueError("null_distribution_path must contain a non-empty JSON array")
    return [float(item) for item in values]


def _resolve_input_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip()).strip("_")
    return text or "candidate"


def _to_plain_object(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _to_plain_object(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_object(item) for item in value]
    return value


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    return value


def _require_non_empty_string(value: Any, *, name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{name} must be a non-empty string")
    return text


def _optional_non_empty_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_choice(value: Any, *, valid_values: set[str], name: str) -> str:
    text = _require_non_empty_string(value, name=name)
    if text not in valid_values:
        raise ValueError(f"{name} must be one of {sorted(valid_values)}, got {text!r}")
    return text


def _coerce_probability(value: Any, *, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} must be provided explicitly")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return number


def _require_non_negative_int(value: Any, *, name: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be >= 0")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
