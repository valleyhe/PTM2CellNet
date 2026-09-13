#!/usr/bin/env python3
"""从版本化 JSON 重放 PerturbGen 双路径评估并产出报告。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Literal, Mapping, Sequence, cast

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.perturbgen.dual_path import evaluate_dual_path_candidate  # noqa: E402
from src.integration.perturbgen.empirical_pvalue import (  # noqa: E402
    EmpiricalPvalueError,
    aggregate_candidate_empirical_pvalues,
)
from src.integration.perturbgen.reports import (  # noqa: E402
    build_candidate_report_payload,
    build_candidate_summary_dataframe,
    save_report_artifacts,
)
from src.integration.perturbgen.results import (  # noqa: E402
    benjamini_hochberg,
    extract_path_result_from_perturbgen_h5ad,
)
from src.integration.perturbgen.null_selection import load_null_distribution_manifest  # noqa: E402

INPUT_SCHEMA_VERSION = "perturbgen_dual_path_eval/v1"
MANIFEST_SCHEMA_VERSION = "perturbgen_dual_path_manifest/v1"
_VALID_PATHS = {"source_intervention", "within_state"}
_VALID_MODES = {"mask", "pad", "delete", "overexpress"}
_VALID_QUALITY = {"pass", "fail", "inconclusive"}
_VALID_DIRECTIONS = {"up", "down"}
_VALID_INTERVENTION_TYPES = {"KO", "KD"}
_VALID_EVALUATION_MODES = {"engineering", "formal"}
_SYNTHETIC_PVALUE_SOURCES = {"uniform", "hand_filled", "external_table"}
LiteralPath = Literal["source_intervention", "within_state"]
LiteralMode = Literal["mask", "pad", "delete", "overexpress"]


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
    evaluation_mode = str(spec.get("evaluation_mode", "engineering")).strip().lower()
    if evaluation_mode not in _VALID_EVALUATION_MODES:
        raise ValueError(f"evaluation_mode must be one of {sorted(_VALID_EVALUATION_MODES)}")

    candidates = list(spec["candidates"])
    extracted: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]] = []
    pvalues: list[float] = []
    for candidate_spec in candidates:
        candidate_mapping = _require_mapping(candidate_spec, name="candidate spec")
        pvalue_kind = str(
            candidate_mapping.get("pvalue_source") or spec.get("pvalue_source") or "unspecified"
        ).strip()
        if evaluation_mode == "formal":
            if pvalue_kind in _SYNTHETIC_PVALUE_SOURCES or "candidate_pvalue" in candidate_mapping:
                raise ValueError(
                    "formal evaluation rejects uniform/hand-filled/external candidate_pvalue; "
                    "q_value must come from aggregate_candidate_empirical_pvalues"
                )
            quality = candidate_mapping.get("unperturbed_quality")
            if (
                not isinstance(quality, Mapping)
                or quality.get("source") != "extract_unperturbed_quality_from_h5ad"
            ):
                raise ValueError(
                    "formal evaluation requires unperturbed_quality extracted from h5ad "
                    "(source=extract_unperturbed_quality_from_h5ad)"
                )
        candidate = _require_mapping(candidate_mapping.get("candidate"), name="candidate")
        runs = candidate_mapping.get("runs")
        if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)) or not runs:
            raise ValueError("candidate runs must be a non-empty sequence")
        extracted_runs = [
            _extract_run(
                run_spec,
                input_base_dir=input_path.parent,
                candidate_ensembl_id=candidate.get("ensembl_id"),
            )
            for run_spec in runs
        ]
        aggregation = None
        if evaluation_mode == "formal":
            try:
                aggregation = aggregate_candidate_empirical_pvalues(
                    extracted_runs,
                    _require_intervention_type(
                        candidate.get("intervention_type"),
                        name="candidate.intervention_type",
                    ),
                    _require_choice(
                        candidate_mapping.get("observed_direction"),
                        valid_values=_VALID_DIRECTIONS,
                        name="observed_direction",
                    ),
                    ensembl_id=str(candidate.get("ensembl_id", "")),
                )
            except EmpiricalPvalueError as exc:
                raise ValueError(f"formal empirical p aggregation failed: {exc}") from exc
            pvalues.append(float(aggregation["pvalue"]))
        else:
            pvalues.append(
                _coerce_probability(candidate_mapping.get("candidate_pvalue"), name="candidate_pvalue")
            )
        extracted.append((dict(candidate_mapping), extracted_runs, aggregation))

    q_values = benjamini_hochberg(pvalues)

    payloads: list[dict[str, Any]] = []
    candidate_entries: list[dict[str, Any]] = []
    run_id = _require_non_empty_string(spec.get("run_id", input_path.stem), name="run_id")

    for index, ((candidate_spec, extracted_runs, aggregation), q_value) in enumerate(
        zip(extracted, q_values, strict=True), start=1
    ):
        payload, manifest_entry = _evaluate_candidate(
            candidate_spec=candidate_spec,
            q_value=q_value,
            candidate_index=index,
            input_path=input_path,
            output_dir=output_dir,
            run_id=run_id,
            extracted_runs=extracted_runs,
            evaluation_mode=evaluation_mode,
            empirical_aggregation=aggregation,
            spec_pvalue_source=str(spec.get("pvalue_source", "")).strip() or None,
            spec_evidence_class=str(spec.get("evidence_class", "")).strip() or None,
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
    output_dir: Path,
    run_id: str,
    extracted_runs: Sequence[Mapping[str, Any]],
    evaluation_mode: str,
    empirical_aggregation: Mapping[str, Any] | None,
    spec_pvalue_source: str | None,
    spec_evidence_class: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = _require_mapping(candidate_spec.get("candidate"), name="candidate")
    candidate_gene = _require_non_empty_string(
        candidate.get("gene_symbol") or candidate.get("candidate_gene"),
        name="candidate.gene_symbol",
    )
    intervention_type = _require_intervention_type(
        candidate.get("intervention_type"),
        name="candidate.intervention_type",
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
    if not extracted_runs:
        raise ValueError("candidate runs must be a non-empty sequence")
    pvalue_kind = str(candidate_spec.get("pvalue_source") or spec_pvalue_source or "unspecified").strip()
    if evaluation_mode == "formal":
        if empirical_aggregation is None:
            raise ValueError("formal evaluation requires empirical p aggregation provenance")
        candidate_pvalue = float(empirical_aggregation["pvalue"])
        evidence_class = "empirical_null"
        pvalue_kind = "empirical_aggregated"
    else:
        candidate_pvalue = _coerce_probability(
            candidate_spec.get("candidate_pvalue"),
            name="candidate_pvalue",
        )
        evidence_class = spec_evidence_class or (
            "synthetic" if pvalue_kind in _SYNTHETIC_PVALUE_SOURCES else "unspecified"
        )
    decision = evaluate_dual_path_candidate(
        [item["path_result"] for item in extracted_runs],
        observed_direction=observed_direction,
        intervention_type=intervention_type,
        q_value=q_value,
        candidate_gene=candidate_gene,
        unperturbed_quality_status=unperturbed_quality_status,
        evaluation_mode=evaluation_mode,
        evidence_class=evidence_class,
        pvalue_source=pvalue_kind,
    )

    candidate_dir = output_dir / f"{candidate_index:03d}_{_slug(candidate_gene)}"
    manifest = {
        "run_id": run_id,
        "input_json": str(input_path),
        "candidate_index": candidate_index,
        "candidate_pvalue": candidate_pvalue,
        "candidate_qvalue": q_value,
        "evaluation_mode": evaluation_mode,
        "evidence_class": evidence_class,
        "pvalue_source": pvalue_kind,
        "empirical_aggregation": _to_plain_object(empirical_aggregation) if empirical_aggregation else None,
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
        "intervention_type": intervention_type,
        "candidate_index": candidate_index,
        "candidate_pvalue": candidate_pvalue,
        "q_value": q_value,
        "verdict": decision.verdict,
        "reasons": list(decision.reasons),
        "scientific_acceptance": decision.scientific_acceptance,
        "evaluation_mode": decision.evaluation_mode,
        "evidence_class": decision.evidence_class,
        "report_dir": str(candidate_dir),
        "artifacts": artifacts,
        "input_runs": extracted_runs,
        "replay": manifest["replay"],
    }


def _extract_run(
    run_spec: Any,
    *,
    input_base_dir: Path,
    candidate_ensembl_id: Any = None,
) -> dict[str, Any]:
    run = _require_mapping(run_spec, name="run")
    path = cast(
        LiteralPath,
        _require_choice(run.get("path"), valid_values=_VALID_PATHS, name="path"),
    )
    mode = cast(
        LiteralMode,
        _require_choice(run.get("mode"), valid_values=_VALID_MODES, name="mode"),
    )
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
    h5ad_provenance["tokenise_stage_manifest"] = str(
        _resolve_input_path(
            input_base_dir,
            _require_non_empty_string(
                raw_provenance.get("tokenise_stage_manifest"),
                name="h5ad_provenance.tokenise_stage_manifest",
            ),
        )
    )
    donor_obs_column = _require_non_empty_string(run.get("donor_obs_column"), name="donor_obs_column")
    var_gene_column = _require_non_empty_string(run.get("var_gene_column"), name="var_gene_column")
    deg_table_path = _resolve_input_path(
        input_base_dir,
        _require_non_empty_string(run.get("deg_table_path"), name="deg_table_path"),
    )
    inline_null_distribution = run.get("null_distribution")
    null_manifest_value = run.get("null_distribution_manifest_path")
    if inline_null_distribution is not None:
        if run.get("null_distribution_path") is not None:
            raise ValueError("provide either null_distribution or null_distribution_path, not both")
        if null_manifest_value is None:
            raise ValueError(
                "inline null_distribution must include null_distribution_manifest_path from the N-05 assembler"
            )
        candidate_id = _require_non_empty_string(
            candidate_ensembl_id,
            name="candidate.ensembl_id for inline null_distribution",
        )
        null_manifest_path = _resolve_input_path(
            input_base_dir,
            _require_non_empty_string(null_manifest_value, name="null_distribution_manifest_path"),
        )
        try:
            distribution = load_null_distribution_manifest(
                null_manifest_path,
                candidate_ensembl_id=candidate_id,
                path_name=path,
                mode=mode,
                seed=seed,
            )
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"inline null_distribution binding does not match {candidate_id}/{path}/{mode}/{seed}: {exc}"
            ) from exc
        null_distribution = _validate_inline_null_distribution(inline_null_distribution)
        if null_distribution != distribution["values"]:
            raise ValueError("inline null_distribution does not equal the N-05 manifest-bound values")
        null_distribution_manifest_path = null_manifest_path
        null_distribution_path = None
    else:
        if null_manifest_value is not None:
            raise ValueError(
                "null_distribution_manifest_path is only valid with N-05 assembler embedded null_distribution"
            )
        null_distribution_path = _resolve_input_path(
            input_base_dir,
            _require_non_empty_string(run.get("null_distribution_path"), name="null_distribution_path"),
        )
        null_distribution = _load_null_distribution(null_distribution_path)
        null_distribution_manifest_path = None

    target_gene = _optional_non_empty_string(run.get("target_gene"))
    deg_donor_column = _require_non_empty_string(run.get("deg_donor_column"), name="deg_donor_column")
    deg_gene_column = _require_non_empty_string(run.get("deg_gene_column"), name="deg_gene_column")
    deg_effect_column = _require_non_empty_string(run.get("deg_effect_column"), name="deg_effect_column")
    deg_fdr_column = _require_non_empty_string(run.get("deg_fdr_column"), name="deg_fdr_column")
    fdr_threshold = float(run.get("fdr_threshold", 0.05))
    min_training_donors = int(run.get("min_training_donors", 2))
    min_evaluable_donors = int(run.get("min_evaluable_donors", 3))
    top_k = int(run.get("top_k", 50))
    bootstrap_iterations = int(run.get("bootstrap_iterations", 1000))
    if run.get("bootstrap_seed") is None:
        raise ValueError("bootstrap_seed must be explicitly recorded for deterministic formal replay")
    bootstrap_seed = _require_non_negative_int(run.get("bootstrap_seed"), name="bootstrap_seed")

    extraction = extract_path_result_from_perturbgen_h5ad(
        output_h5ad=output_h5ad,
        h5ad_provenance=h5ad_provenance,
        deg_table=_load_deg_table(deg_table_path),
        null_distribution=null_distribution,
        path=path,
        mode=mode,
        seed=seed,
        donor_obs_column=donor_obs_column,
        var_gene_column=var_gene_column,
        target_gene=target_gene,
        donor_column=deg_donor_column,
        gene_column=deg_gene_column,
        effect_column=deg_effect_column,
        fdr_column=deg_fdr_column,
        fdr_threshold=fdr_threshold,
        min_training_donors=min_training_donors,
        min_evaluable_donors=min_evaluable_donors,
        top_k=top_k,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    result = {
        "path": path,
        "mode": mode,
        "seed": seed,
        "output_h5ad": str(output_h5ad.expanduser().resolve(strict=True)),
        "deg_table_path": str(deg_table_path.expanduser().resolve(strict=True)),
        "h5ad_provenance": _to_plain_object(h5ad_provenance),
        "target_gene": target_gene,
        "donor_obs_column": donor_obs_column,
        "var_gene_column": var_gene_column,
        "deg_donor_column": deg_donor_column,
        "deg_gene_column": deg_gene_column,
        "deg_effect_column": deg_effect_column,
        "deg_fdr_column": deg_fdr_column,
        "fdr_threshold": fdr_threshold,
        "min_training_donors": min_training_donors,
        "min_evaluable_donors": min_evaluable_donors,
        "top_k": top_k,
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
        "empirical_pvalue": extraction.path_result.empirical_pvalue,
        "baseline_matrix": extraction.baseline_matrix,
        "perturbed_matrix": extraction.perturbed_matrix,
        "bootstrap_ci": list(extraction.bootstrap_ci) if extraction.bootstrap_ci is not None else None,
        "donor_scores": _to_plain_object(extraction.donor_scores),
        "path_result": _to_plain_object(extraction.path_result),
    }
    if null_distribution_path is not None:
        result["null_distribution_path"] = str(null_distribution_path.expanduser().resolve(strict=True))
    else:
        result["null_distribution"] = null_distribution
        result["null_distribution_manifest_path"] = str(null_distribution_manifest_path)
    return result


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
    try:
        payload = load_null_distribution_manifest(path, required_count=1)
    except (OSError, ValueError) as exc:
        raise ValueError(
            "null_distribution_path must contain non-empty finite values: "
            f"{path}: {exc}"
        ) from exc
    return payload["values"]


def _validate_inline_null_distribution(values: Any) -> list[float]:
    if not isinstance(values, list):
        raise ValueError("null_distribution must be a list")
    if len(values) < 99:
        raise ValueError("inline null_distribution must contain at least 99 values")
    converted: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("inline null_distribution values must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("inline null_distribution values must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError("inline null_distribution values must be finite")
        converted.append(number)
    return converted


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
        return asdict(cast(Any, value))
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


def _require_intervention_type(value: Any, *, name: str) -> str:
    text = _require_non_empty_string(value, name=name).upper()
    if text not in _VALID_INTERVENTION_TYPES:
        raise ValueError(f"{name} must be explicitly KO or KD, got {text!r}")
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
