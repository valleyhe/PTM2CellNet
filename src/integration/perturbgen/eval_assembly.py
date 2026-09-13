"""Assemble dual-path evaluation inputs from a DAVF→PerturbGen E2E report.

Closes gap N-5 of ``project_analysis_20260910.md`` §4.1: the E2E CLI emits
``davf_perturbgen_e2e/v1`` JSON while ``evaluate_perturbgen_dual_path.py``
expects ``perturbgen_dual_path_eval/v1``; until now the two schemas were
bridged by hand.

The assembler never invents evidence: h5ad paths come exclusively from the
successful perturb stage manifests bound to each E2E run, and the DEG table,
null distribution, candidate p-value and unperturbed quality status must be
supplied explicitly because the E2E report does not carry them.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .null_selection import load_null_distribution_manifest

E2E_SCHEMA_VERSION = "davf_perturbgen_e2e/v1"
EVAL_INPUT_SCHEMA_VERSION = "perturbgen_dual_path_eval/v1"

_VALID_PATHS = ("source_intervention", "within_state")
_VALID_QUALITY = ("pass", "fail", "inconclusive")
_VALID_INTERVENTION_TYPES = {"KO", "KD"}
_PERTURB_TO_TOKENISE_ARTIFACTS = {
    "src_dataset_file": "src_dataset",
    "src_adata": "src_h5ad",
    "tgt_dataset_folder": "tgt_dataset_folder",
    "tgt_adata_folder": "tgt_h5ad_folder",
}

# Upstream perturb outputs ``*_minference_adata_g{GENE}_s{src|tgt}_t{MODE}.h5ad``:
# g is the perturbed gene, s is the perturbation sequence (src/tgt), t is the
# token perturbation mode.  The random seed is not part of the filename; it is
# read from the stage manifest's fingerprint material.
_H5AD_NAME_PATTERN = re.compile(r"_g(?P<gene>[^_]+)_s(?P<sequence>src|tgt)_t(?P<mode>[A-Za-z]+)\.h5ad$")
_PATH_TO_SEQUENCE = {"source_intervention": "src", "within_state": "tgt"}


class EvalAssemblyError(ValueError):
    """Raised when E2E outputs cannot be assembled into evaluation inputs."""


@dataclass(frozen=True)
class RunArtifact:
    """One perturb-stage artifact bound to a path and seed."""

    path: str
    mode: str
    seed: int
    stage_manifest: str
    tokenise_stage_manifest: str
    sha256: str
    fingerprint: str | None


def load_e2e_report(path: str | Path) -> dict[str, Any]:
    """Load and validate a ``davf_perturbgen_e2e/v1`` report."""

    resolved = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvalAssemblyError(f"E2E report root must be an object: {resolved}")
    version = payload.get("schema_version")
    if version != E2E_SCHEMA_VERSION:
        raise EvalAssemblyError(f"unsupported E2E schema_version {version!r}; expected {E2E_SCHEMA_VERSION!r}")
    runs = payload.get("perturbgen_runs")
    if not isinstance(runs, list) or not runs:
        raise EvalAssemblyError("E2E report has no perturbgen_runs; rerun with --run-perturbgen before assembling")
    return payload


def _resolve_stage_artifact_path(value: Any, *, manifest_path: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip() or value.startswith("@artifact:"):
        raise EvalAssemblyError(
            f"{manifest_path} fingerprint config has no resolved {name} artifact path"
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise EvalAssemblyError(f"{manifest_path} {name} artifact is not readable: {path}") from exc


def _tokenise_stage_manifest_for_perturb(
    output_root: Path,
    perturb_manifest: Mapping[str, Any],
    perturb_manifest_path: Path,
) -> str:
    """Bind a perturb run to the one tokenise manifest that supplied its inputs."""

    material = perturb_manifest.get("fingerprint_material")
    fingerprint_config = material.get("fingerprint_config") if isinstance(material, Mapping) else None
    stage_config = fingerprint_config.get("stage_config") if isinstance(fingerprint_config, Mapping) else None
    perturb_config = stage_config.get("perturb_config") if isinstance(stage_config, Mapping) else None
    data = perturb_config.get("data") if isinstance(perturb_config, Mapping) else None
    if not isinstance(data, Mapping):
        raise EvalAssemblyError(
            f"perturb manifest {perturb_manifest_path} has no resolved perturb_config.data lineage"
        )

    expected = {
        artifact_name: _resolve_stage_artifact_path(
            data.get(config_key),
            manifest_path=perturb_manifest_path,
            name=artifact_name,
        )
        for config_key, artifact_name in _PERTURB_TO_TOKENISE_ARTIFACTS.items()
    }
    matches: list[Path] = []
    for candidate_manifest_path in sorted(output_root.rglob("stage_manifest.json")):
        if candidate_manifest_path.resolve() == perturb_manifest_path.resolve():
            continue
        try:
            candidate_payload = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvalAssemblyError(
                f"cannot read stage manifest while resolving tokenise lineage: {candidate_manifest_path}"
            ) from exc
        if not isinstance(candidate_payload, Mapping) or candidate_payload.get("stage") != "tokenise":
            continue
        if candidate_payload.get("status") != "success":
            continue
        artifacts = candidate_payload.get("artifacts")
        if not isinstance(artifacts, Mapping):
            continue
        try:
            actual = {
                name: _resolve_stage_artifact_path(
                    artifacts.get(name),
                    manifest_path=candidate_manifest_path,
                    name=name,
                )
                for name in expected
            }
        except EvalAssemblyError:
            continue
        if actual == expected:
            matches.append(candidate_manifest_path.resolve())

    if len(matches) != 1:
        raise EvalAssemblyError(
            f"perturb manifest {perturb_manifest_path} must match exactly one successful tokenise "
            f"stage manifest by consumed src/tgt artifacts; found {len(matches)}"
        )
    return str(matches[0])


def resolve_run_artifacts(
    output_root: str | Path,
    gene_symbol: str,
    *,
    paths: Sequence[str] = _VALID_PATHS,
) -> dict[str, list[RunArtifact]]:
    """Bind every successful perturb artifact under ``output_root`` to its manifest.

    Perturb stages live in per-path directories (optionally further split by
    mode/seed); each successful ``stage_manifest.json`` whose stage name is a
    dual-path kind contributes one artifact, so multi-seed and sensitivity
    runs are all discovered.
    """

    root = Path(output_root).expanduser().resolve(strict=True)
    allowed_paths = tuple(paths)
    for path_kind in allowed_paths:
        if path_kind not in _VALID_PATHS:
            raise EvalAssemblyError(f"unknown PerturbGen path: {path_kind!r}")
    artifacts: dict[str, list[RunArtifact]] = {path_kind: [] for path_kind in allowed_paths}
    for manifest_path in sorted(root.rglob("stage_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        path_kind = manifest.get("stage")
        if path_kind not in allowed_paths:
            continue
        if manifest.get("status") != "success":
            raise EvalAssemblyError(
                f"perturb stage {path_kind!r} did not succeed "
                f"(status={manifest.get('status')!r}); cannot assemble evaluation input"
            )
        artifacts_map = manifest.get("artifacts")
        if not isinstance(artifacts_map, dict) or "result_h5ad" not in artifacts_map:
            raise EvalAssemblyError(f"perturb stage manifest for {path_kind!r} has no result_h5ad artifact")
        h5ad_path = Path(str(artifacts_map["result_h5ad"])).expanduser().resolve(strict=True)
        match = _H5AD_NAME_PATTERN.search(h5ad_path.name)
        if match is None:
            raise EvalAssemblyError(f"cannot parse gene/sequence/mode from perturb artifact name: {h5ad_path.name}")
        parsed_gene = match.group("gene")
        if parsed_gene.upper() != gene_symbol.upper():
            raise EvalAssemblyError(
                f"perturb artifact {h5ad_path.name} targets gene {parsed_gene!r}, expected {gene_symbol!r}"
            )
        sequence = match.group("sequence")
        if sequence != _PATH_TO_SEQUENCE[path_kind]:
            raise EvalAssemblyError(
                f"perturb artifact {h5ad_path.name} carries sequence {sequence!r}, "
                f"which does not match path {path_kind!r}"
            )
        seed = _seed_from_manifest(manifest, stage_dir=str(manifest_path.parent))
        tokenise_stage_manifest = _tokenise_stage_manifest_for_perturb(
            root,
            manifest,
            manifest_path,
        )
        sha256 = _result_h5ad_sha256(
            manifest,
            manifest_path=manifest_path,
            output_h5ad=h5ad_path,
        )
        artifacts[path_kind].append(
            RunArtifact(
                path=str(h5ad_path),
                mode=match.group("mode").lower(),
                seed=seed,
                stage_manifest=str(manifest_path),
                tokenise_stage_manifest=tokenise_stage_manifest,
                sha256=sha256,
                fingerprint=manifest.get("fingerprint"),
            )
        )
    for path_kind, path_artifacts in artifacts.items():
        identities = [(artifact.mode, artifact.seed) for artifact in path_artifacts]
        if len(identities) != len(set(identities)):
            raise EvalAssemblyError(f"duplicate PerturbGen artifact binding for path {path_kind!r}")
    missing = [path_kind for path_kind, found in artifacts.items() if not found]
    if missing:
        raise EvalAssemblyError(f"no successful perturb artifacts found for paths: {', '.join(missing)} under {root}")
    return artifacts


def _result_h5ad_sha256(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    output_h5ad: Path,
) -> str:
    """Copy the runner-recorded hash for ``artifacts.result_h5ad``.

    The stage runner already records output hashes in its ``outputs`` mapping.
    The assembler only resolves that exact output record; it does not compute
    a new digest or identify an artifact by filename.
    """

    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise EvalAssemblyError(f"stage manifest {manifest_path} has no runner outputs mapping")
    matches: list[Mapping[str, Any]] = []
    for raw_path, raw_record in outputs.items():
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise EvalAssemblyError(f"stage manifest {manifest_path} has an invalid runner output path")
        candidate_path = Path(raw_path).expanduser()
        if not candidate_path.is_absolute():
            candidate_path = manifest_path.parent / candidate_path
        try:
            resolved_path = candidate_path.resolve(strict=True)
        except OSError as exc:
            raise EvalAssemblyError(
                f"stage manifest {manifest_path} runner output is not readable: {candidate_path}"
            ) from exc
        if resolved_path == output_h5ad:
            if not isinstance(raw_record, Mapping):
                raise EvalAssemblyError(f"runner output record for result_h5ad is not a mapping: {manifest_path}")
            matches.append(raw_record)
    if len(matches) != 1:
        raise EvalAssemblyError(
            f"stage manifest {manifest_path} must contain exactly one runner output record for result_h5ad; "
            f"found {len(matches)}"
        )
    raw_sha256 = matches[0].get("sha256")
    if not isinstance(raw_sha256, str) or not raw_sha256.strip():
        raise EvalAssemblyError(
            f"runner output record for result_h5ad has no sha256: {manifest_path}"
        )
    return raw_sha256.strip()


def _seed_from_manifest(manifest: Mapping[str, Any], *, stage_dir: str) -> int:
    """Read the run's random seed from the stage manifest fingerprint material."""

    material = manifest.get("fingerprint_material")
    if isinstance(material, Mapping):
        raw_seed = material.get("random_seed")
        if raw_seed is not None:
            try:
                seed = int(raw_seed)
            except (TypeError, ValueError) as exc:
                raise EvalAssemblyError(
                    f"stage manifest for {stage_dir!r} has a non-integer random_seed: {raw_seed!r}"
                ) from exc
            if seed >= 0:
                return seed
    raise EvalAssemblyError(
        f"stage manifest for {stage_dir!r} has no usable fingerprint_material.random_seed; "
        "the evaluation input requires an explicit seed per run"
    )


def load_candidate_pvalues(path: str | Path) -> dict[str, float]:
    """Load a CSV of ``ensembl_id,pvalue`` rows for candidate-level p-values."""

    import csv as _csv

    resolved = Path(path).expanduser().resolve(strict=True)
    values: dict[str, float] = {}
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = _csv.DictReader(handle)
        missing = [column for column in ("ensembl_id", "pvalue") if column not in (reader.fieldnames or [])]
        if missing:
            raise EvalAssemblyError(f"candidate p-value table is missing columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            ensembl_id = (row.get("ensembl_id") or "").strip()
            raw = (row.get("pvalue") or "").strip()
            if not ensembl_id or not raw:
                raise EvalAssemblyError(f"p-value row {row_number} has empty fields")
            try:
                pvalue = float(raw)
            except ValueError as exc:
                raise EvalAssemblyError(f"p-value row {row_number} is not numeric: {raw!r}") from exc
            if not 0.0 <= pvalue <= 1.0:
                raise EvalAssemblyError(f"p-value row {row_number} must be within [0, 1]")
            if ensembl_id in values and values[ensembl_id] != pvalue:
                raise EvalAssemblyError(f"conflicting p-values for {ensembl_id}")
            values[ensembl_id] = pvalue
    return values


def build_eval_input_payload(
    e2e_report: Mapping[str, Any],
    *,
    deg_table_path: str | Path,
    null_distribution_path: str | Path | None = None,
    null_distribution_manifest_path: str | Path | None = None,
    unperturbed_quality_status: str,
    candidate_pvalues: Mapping[str, float] | None = None,
    uniform_candidate_pvalue: float | None = None,
    run_id: str | None = None,
    donor_obs_column: str = "donor",
    var_gene_column: str = "__index__",
    deg_donor_column: str = "donor",
    deg_gene_column: str = "gene_symbol",
    deg_effect_column: str = "log2fc",
    deg_fdr_column: str = "fdr",
    fdr_threshold: float = 0.05,
    min_training_donors: int = 2,
    min_evaluable_donors: int = 3,
    top_k: int = 50,
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int | None = None,
) -> dict[str, Any]:
    """Build a ``perturbgen_dual_path_eval/v1`` payload from an E2E report."""

    if null_distribution_path is None and null_distribution_manifest_path is None:
        raise EvalAssemblyError("null_distribution_path or null_distribution_manifest_path must be supplied")
    if null_distribution_path is not None and null_distribution_manifest_path is not None:
        raise EvalAssemblyError("provide only one null distribution source")
    if unperturbed_quality_status not in _VALID_QUALITY:
        raise EvalAssemblyError(f"unperturbed_quality_status must be one of {_VALID_QUALITY}")
    pvalue_source: Mapping[str, float] | None = candidate_pvalues
    if pvalue_source is None:
        if uniform_candidate_pvalue is None:
            raise EvalAssemblyError(
                "candidate p-values must be provided (table or uniform value); "
                "the E2E report does not carry empirical null calibration"
            )
        if not math.isfinite(uniform_candidate_pvalue) or not 0.0 <= uniform_candidate_pvalue <= 1.0:
            raise EvalAssemblyError("uniform candidate p-value must be within [0, 1]")

    deg_path = str(Path(deg_table_path).expanduser().resolve(strict=True))
    null_path = (
        None if null_distribution_path is None else str(Path(null_distribution_path).expanduser().resolve(strict=True))
    )
    null_manifest_path = (
        None
        if null_distribution_manifest_path is None
        else str(Path(null_distribution_manifest_path).expanduser().resolve(strict=True))
    )
    if null_path is not None:
        try:
            load_null_distribution_manifest(null_path, required_count=1)
        except (OSError, ValueError) as exc:
            raise EvalAssemblyError(
                "legacy null_distribution_path must contain non-empty finite values: "
                f"{null_path}: {exc}"
            ) from exc

    candidates: list[dict[str, Any]] = []
    for run in e2e_report.get("perturbgen_runs", []):
        gene_symbol = str(run.get("gene_symbol", "")).strip()
        ensembl_id = str(run.get("ensembl_id", "")).strip()
        output_root = run.get("output_root")
        if not gene_symbol or not ensembl_id or not output_root:
            raise EvalAssemblyError("each perturbgen run needs gene_symbol, ensembl_id and output_root")
        artifacts = resolve_run_artifacts(output_root, gene_symbol)
        # Recover the observed direction recorded by the passing direction gate.
        invocation = _passing_invocation(e2e_report, ensembl_id)
        observed_direction = invocation.get("candidate", {}).get("observed_direction")
        if observed_direction not in ("up", "down"):
            raise EvalAssemblyError(f"invocation for {ensembl_id} has no recorded observed_direction")
        invocation_route = str(invocation.get("intervention_type", "")).strip().upper()
        if invocation_route not in _VALID_INTERVENTION_TYPES:
            raise EvalAssemblyError(
                f"invocation for {ensembl_id} must declare intervention_type KO or KD"
            )
        run_route = str(run.get("intervention_type", "")).strip().upper()
        if run_route not in _VALID_INTERVENTION_TYPES:
            raise EvalAssemblyError(
                f"PerturbGen run for {ensembl_id} must declare intervention_type KO or KD"
            )
        if run_route != invocation_route:
            raise EvalAssemblyError(
                f"PerturbGen run route does not match passing invocation for {ensembl_id}: "
                f"{run_route} != {invocation_route}"
            )
        if pvalue_source is not None:
            if ensembl_id not in pvalue_source:
                raise EvalAssemblyError(f"no candidate p-value provided for {ensembl_id} ({gene_symbol})")
            candidate_pvalue = float(pvalue_source[ensembl_id])
        else:
            assert uniform_candidate_pvalue is not None
            candidate_pvalue = float(uniform_candidate_pvalue)

        run_records: list[dict[str, Any]] = []
        for path_kind, path_artifacts in artifacts.items():
            for artifact in path_artifacts:
                run_records.append(
                    {
                        "path": path_kind,
                        "mode": artifact.mode,
                        "seed": artifact.seed,
                        "output_h5ad": artifact.path,
                        "h5ad_provenance": {
                            "stage_manifest": artifact.stage_manifest,
                            "tokenise_stage_manifest": artifact.tokenise_stage_manifest,
                            "sha256": artifact.sha256,
                            "stage": path_kind,
                            "fingerprint": artifact.fingerprint,
                        },
                        "donor_obs_column": donor_obs_column,
                        "var_gene_column": var_gene_column,
                        "deg_table_path": deg_path,
                        "target_gene": gene_symbol,
                        "deg_donor_column": deg_donor_column,
                        "deg_gene_column": deg_gene_column,
                        "deg_effect_column": deg_effect_column,
                        "deg_fdr_column": deg_fdr_column,
                        "fdr_threshold": fdr_threshold,
                        "min_training_donors": min_training_donors,
                        "min_evaluable_donors": min_evaluable_donors,
                        "top_k": top_k,
                        "bootstrap_iterations": bootstrap_iterations,
                        # The perturb artifact seed is the only reproducible
                        # default available from N-05's manifest-bound run;
                        # an explicit caller override remains deterministic
                        # and is recorded as the actual extractor input.
                        "bootstrap_seed": artifact.seed if bootstrap_seed is None else bootstrap_seed,
                    }
                )
                if null_path is not None:
                    run_records[-1]["null_distribution_path"] = null_path
                else:
                    if null_manifest_path is None:
                        raise EvalAssemblyError("null distribution manifest path is missing")
                    try:
                        distribution = load_null_distribution_manifest(
                            null_manifest_path,
                            candidate_ensembl_id=ensembl_id,
                            path_name=path_kind,
                            mode=artifact.mode,
                            seed=artifact.seed,
                        )
                    except (OSError, ValueError) as exc:
                        raise EvalAssemblyError(
                            f"null distribution manifest does not match {ensembl_id}/{path_kind}/"
                            f"{artifact.mode}/{artifact.seed}: {exc}"
                        ) from exc
                    run_records[-1]["null_distribution"] = list(distribution["values"])
                    run_records[-1]["null_distribution_manifest_path"] = null_manifest_path
        candidates.append(
            {
                "candidate": {
                    "gene_symbol": gene_symbol,
                    "ensembl_id": ensembl_id,
                    "intervention_type": invocation_route,
                    "davf_provenance": _davf_provenance_from_run(e2e_report, ensembl_id),
                },
                "observed_direction": observed_direction,
                "unperturbed_quality_status": unperturbed_quality_status,
                "candidate_pvalue": candidate_pvalue,
                "runs": run_records,
            }
        )

    if not candidates:
        raise EvalAssemblyError("no evaluable perturbgen runs found in the E2E report")
    return {
        "schema_version": EVAL_INPUT_SCHEMA_VERSION,
        "run_id": run_id or f"e2e-{len(candidates)}-candidates",
        "candidates": candidates,
    }


def write_eval_input(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write the evaluation input JSON and return the resolved path."""

    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved


def _observed_direction_from_run(
    e2e_report: Mapping[str, Any],
    ensembl_id: str,
) -> str:
    invocation = _passing_invocation(e2e_report, ensembl_id)
    direction = invocation.get("candidate", {}).get("observed_direction")
    if direction not in ("up", "down"):
        raise EvalAssemblyError(f"invocation for {ensembl_id} has no recorded observed_direction")
    return str(direction)


def _passing_invocation(e2e_report: Mapping[str, Any], ensembl_id: str) -> Mapping[str, Any]:
    for preparation in e2e_report.get("candidates", []):
        if not isinstance(preparation, Mapping) or preparation.get("status") != "pass":
            continue
        invocation = preparation.get("invocation")
        candidate = preparation.get("candidate")
        if not isinstance(invocation, Mapping):
            continue
        invocation_candidate = invocation.get("candidate")
        if not isinstance(invocation_candidate, Mapping):
            continue
        if not isinstance(candidate, Mapping):
            candidate = invocation_candidate
        if candidate.get("direction_gate_status") != "pass":
            continue
        if invocation_candidate.get("direction_gate_status") != "pass":
            continue
        if str(invocation.get("ensembl_id", "")) == ensembl_id:
            return invocation
    raise EvalAssemblyError(
        f"E2E report has no passing invocation for {ensembl_id}; "
        "only gated candidates may be assembled into evaluation inputs"
    )


def _davf_provenance_from_run(
    e2e_report: Mapping[str, Any],
    ensembl_id: str,
) -> str | None:
    invocation = _passing_invocation(e2e_report, ensembl_id)
    provenance = invocation.get("candidate", {}).get("davf_provenance")
    return str(provenance) if provenance else None
