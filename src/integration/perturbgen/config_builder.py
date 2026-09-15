"""Build strict, resumable PerturbGen stage plans from YAML config."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence, cast

import yaml

from .env_guard import ProjectRoots, validate_repo_roots


STAGE_ORDER = (
    "tokenise",
    "train_mask",
    "train_decoder",
    "perturb",
    "export_gene_embeddings",
    "report",
)
_UNRESOLVED_ENV_PATTERN = re.compile(r"\$\{[^}]+\}")
_ARTIFACT_REF_PATTERN = re.compile(r"^@artifact:([a-zA-Z0-9_.-]+):([a-zA-Z0-9_.-]+)$")
_SAFE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class PerturbGenConfigError(ValueError):
    """Raised when the PerturbGen integration config is incomplete or unsafe."""


@dataclass(frozen=True)
class GeneratedFileSpec:
    """A file materialized by the runner right before executing a stage."""

    path: Path
    format: str
    payload: Any


@dataclass(frozen=True)
class OutputCheck:
    """Expected stage output plus a lightweight schema check."""

    path: Path
    kind: str = "file"
    schema: Mapping[str, Any] = field(default_factory=dict)
    name: str | None = None
    discover_glob: str | None = None


@dataclass(frozen=True)
class StagePlan:
    """Resolved execution plan for one stage."""

    name: str
    driver: str
    argv: tuple[str, ...]
    cwd: Path
    output_dir: Path
    timeout_seconds: int
    uses_gpu: bool
    output_root: Path
    external_python: Path
    expected_outputs: tuple[OutputCheck, ...]
    generated_files: tuple[GeneratedFileSpec, ...] = ()
    fingerprint_paths: tuple[Path | str, ...] = ()
    dependency_files: tuple[Path, ...] = ()
    asset_paths: tuple[Path, ...] = ()
    resource_estimate: Mapping[str, Any] = field(default_factory=dict)
    fingerprint_config: Mapping[str, Any] = field(default_factory=dict)
    roots: ProjectRoots | None = None


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if _UNRESOLVED_ENV_PATTERN.search(expanded):
            raise PerturbGenConfigError(f"unresolved environment placeholder: {value}")
        return expanded
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _expand_env(item) for key, item in value.items()}
    return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PerturbGenConfigError("config root must be a mapping")
    return payload


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve(strict=True)
    payload = _expand_env(_load_yaml(config_path))
    if payload.get("schema_version") != 1:
        raise PerturbGenConfigError("schema_version must be 1")
    for key in ("repo", "environment", "pipeline", "stages"):
        if key not in payload:
            raise PerturbGenConfigError(f"config missing required section: {key}")
    stages = payload["stages"]
    if not isinstance(stages, dict):
        raise PerturbGenConfigError("stages must be a mapping")
    missing = [stage for stage in STAGE_ORDER if stage not in stages]
    if missing:
        raise PerturbGenConfigError(f"config missing required stages: {', '.join(missing)}")
    return cast(dict[str, Any], payload)


def _ensure_under(path: Path, root: Path, *, label: str) -> Path:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PerturbGenConfigError(f"{label} escapes output root: {path}") from exc
    return path


def _resolve_path(
    raw_path: str | Path,
    *,
    project_root: Path,
    perturbgen_repo_root: Path,
    prefer_repo: bool = False,
) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = perturbgen_repo_root if prefer_repo else project_root
    return (base / path).resolve()


def _cli_args_to_argv(arguments: Mapping[str, Any], *, hyphenate_underscores: bool = False) -> list[str]:
    argv: list[str] = []
    for key, value in arguments.items():
        if value is None:
            continue
        flag_name = key.replace("_", "-") if hyphenate_underscores else key
        flag = f"--{flag_name}"
        argv.append(flag)
        if isinstance(value, bool):
            argv.append("True" if value else "False")
        elif isinstance(value, (list, tuple)):
            if not value:
                raise PerturbGenConfigError(f"{flag} cannot be an empty list")
            argv.extend(str(item) for item in value)
        else:
            argv.append(str(value))
    return argv


def _output_checks(
    stage_name: str,
    stage_output_dir: Path,
    raw_outputs: Sequence[Mapping[str, Any]],
    output_root: Path,
    project_root: Path,
    perturbgen_repo_root: Path,
) -> tuple[OutputCheck, ...]:
    checks: list[OutputCheck] = []
    for raw in raw_outputs:
        if not isinstance(raw, Mapping):
            raise PerturbGenConfigError(f"{stage_name}.expected_outputs entries must be mappings")
        relative_path = raw.get("path")
        if not isinstance(relative_path, str) or not relative_path:
            raise PerturbGenConfigError(f"{stage_name}.expected_outputs.path must be a non-empty string")
        root_name = str(raw.get("root", "stage"))
        if root_name == "stage":
            base = stage_output_dir
            allowed_root = output_root
        elif root_name == "project":
            base = project_root
            allowed_root = project_root
        elif root_name == "perturbgen":
            base = perturbgen_repo_root
            allowed_root = perturbgen_repo_root
        else:
            raise PerturbGenConfigError(f"{stage_name}.expected_outputs.root must be stage, project, or perturbgen")
        candidate = _ensure_under(
            (base / relative_path).resolve(),
            allowed_root,
            label=f"{stage_name}.expected_outputs",
        )
        discover_glob = raw.get("discover_glob")
        if discover_glob is not None:
            if not isinstance(discover_glob, str) or not discover_glob:
                raise PerturbGenConfigError(f"{stage_name}.expected_outputs.discover_glob must be a non-empty string")
            glob_path = PurePosixPath(discover_glob)
            if glob_path.is_absolute() or ".." in glob_path.parts:
                raise PerturbGenConfigError(
                    f"{stage_name}.expected_outputs.discover_glob must stay below its path root"
                )
        name = raw.get("name")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise PerturbGenConfigError(f"{stage_name}.expected_outputs.name must be a non-empty string")
        if discover_glob is not None and name is None:
            raise PerturbGenConfigError(f"{stage_name}.expected_outputs with discover_glob requires name")
        schema = raw.get("schema", {})
        if not isinstance(schema, Mapping):
            raise PerturbGenConfigError(f"{stage_name}.expected_outputs.schema must be a mapping")
        checks.append(
            OutputCheck(
                path=candidate,
                kind=str(raw.get("kind", "file")),
                schema=dict(schema),
                name=name.strip() if isinstance(name, str) else None,
                discover_glob=discover_glob,
            )
        )
    names = [check.name for check in checks if check.name is not None]
    if len(names) != len(set(names)):
        raise PerturbGenConfigError(f"{stage_name}.expected_outputs names must be unique")
    return tuple(checks)


def _fingerprint_path(
    raw: Any,
    *,
    project_root: Path,
    perturbgen_repo_root: Path,
) -> Path | str:
    if isinstance(raw, str) and _ARTIFACT_REF_PATTERN.fullmatch(raw):
        return raw
    if not isinstance(raw, (str, Path)):
        raise PerturbGenConfigError("fingerprint_paths entries must be paths or artifact references")
    return _resolve_path(
        raw,
        project_root=project_root,
        perturbgen_repo_root=perturbgen_repo_root,
    )


def _generated_perturb_config(
    stage_name: str,
    stage_output_dir: Path,
    output_root: Path,
    config_relpath: str,
    payload: Mapping[str, Any],
) -> GeneratedFileSpec:
    target = _ensure_under(
        (stage_output_dir / config_relpath).resolve(),
        output_root,
        label=f"{stage_name}.generated_config",
    )
    return GeneratedFileSpec(path=target, format="yaml", payload=dict(payload))


def _validate_tokenise_output_contract(
    stage_config: Mapping[str, Any],
    outputs: Sequence[OutputCheck],
    *,
    perturbgen_repo_root: Path,
) -> None:
    by_name = {output.name: output for output in outputs if output.name is not None}
    required = {
        "src_dataset",
        "tgt_dataset_folder",
        "src_h5ad",
        "tgt_h5ad_folder",
        "rowid_to_gene_name",
        "tokenid_to_rowid",
    }
    if not required.intersection(by_name):
        return
    if set(by_name) != required:
        missing = sorted(required - set(by_name))
        raise PerturbGenConfigError(f"tokenise named output contract is incomplete: {', '.join(missing)}")
    args = stage_config.get("args", {})
    if not isinstance(args, Mapping):
        raise PerturbGenConfigError("tokenise.args must be a mapping")
    dataset = str(args.get("dataset", ""))
    reference_time = str(args.get("reference_time", ""))
    filtering_mode = str(args.get("gene_filtering_mode", ""))
    if not dataset or not reference_time or not filtering_mode:
        raise PerturbGenConfigError(
            "tokenise output contract requires dataset, reference_time, and gene_filtering_mode"
        )
    if _SAFE_IDENTIFIER_PATTERN.fullmatch(dataset) is None:
        raise PerturbGenConfigError(
            "tokenise dataset must be one safe path segment containing only letters, digits, dot, underscore, or hyphen"
        )
    time_points = args.get("time_point_order", ())
    if not isinstance(time_points, Sequence) or isinstance(time_points, (str, bytes)):
        raise PerturbGenConfigError("tokenise time_point_order must be a sequence")
    for value in (reference_time, *(str(item) for item in time_points)):
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise PerturbGenConfigError("tokenise time points must be non-empty safe path segments")
    if str(args.get("src_mode")) != "Geneformer":
        raise PerturbGenConfigError("tokenise output contract requires explicit src_mode=Geneformer")
    suffix = filtering_mode
    if filtering_mode != "all":
        try:
            suffix = f"{int(args['n_hvg'])}_{filtering_mode}"
        except (KeyError, TypeError, ValueError) as exc:
            raise PerturbGenConfigError("tokenise output contract requires integer n_hvg") from exc
    base = perturbgen_repo_root.parent / "T_perturb" / "tokenized_data" / dataset
    expected = {
        "src_dataset": base / f"dataset_{suffix}_src" / f"{reference_time}.dataset",
        "tgt_dataset_folder": base / f"dataset_{suffix}_tgt",
        "src_h5ad": base / f"h5ad_pairing_{suffix}_src" / f"{reference_time}.h5ad",
        "tgt_h5ad_folder": base / f"h5ad_pairing_{suffix}_tgt",
        "rowid_to_gene_name": base / f"token_id_to_genename_{suffix}.pkl",
        "tokenid_to_rowid": base / f"tokenid_to_rowid_{suffix}.pkl",
    }
    for name, expected_path in expected.items():
        if by_name[name].path != expected_path.resolve():
            raise PerturbGenConfigError(
                f"tokenise output {name} does not match upstream path formula: "
                f"expected {expected_path.resolve()}, got {by_name[name].path}"
            )


def _validate_perturb_candidate_contract(stage_config: Mapping[str, Any]) -> None:
    """Validate the upstream val.py contract for the perturb stage.

    Three upstream pitfalls (verified against ref/Perturbgen-src val.py) are
    rejected here instead of silently corrupting runs:

    - val.py:253 skips inference entirely (exit 0) when
      ``model.ckpt_masking_path`` is None — the generated config must always
      carry a downstream checkpoint.
    - val.py:53 only takes the explicit branch when BOTH ``tgt_vocab_size``
      and ``max_seq_length`` are present in ``trainer``; otherwise val.py:63
      derives them via ``max(max(input_id))``, which compares ragged lists
      lexicographically. Both fields are therefore mandatory.
    - val.py:216-218 adds a +100/+50 runtime buffer to
      ``trainer.max_seq_length``/``tgt_vocab_size`` and rewrites
      ``datamodule.max_len`` to the unbuffered base value. The config must
      carry consistent BASE values (datamodule.max_len == trainer.max_seq_length
      when provided).
    """
    perturb_config = stage_config.get("perturb_config", {})
    if not isinstance(perturb_config, Mapping):
        return
    trainer = perturb_config.get("trainer", {})
    if not isinstance(trainer, Mapping):
        raise PerturbGenConfigError("perturb.perturb_config.trainer must be a mapping")
    model = perturb_config.get("model", {})
    if not isinstance(model, Mapping):
        raise PerturbGenConfigError("perturb.perturb_config.model must be a mapping")
    ckpt_masking_path = model.get("ckpt_masking_path")
    if not isinstance(ckpt_masking_path, str) or not ckpt_masking_path.strip():
        raise PerturbGenConfigError(
            "perturb model.ckpt_masking_path must be a non-empty string: "
            "upstream val.py silently skips inference when it is None"
        )
    explicit_dims: dict[str, int | str] = {}
    for field_name in ("tgt_vocab_size", "max_seq_length"):
        value = trainer.get(field_name)
        if value == "auto":
            explicit_dims[field_name] = value
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise PerturbGenConfigError(
                f"perturb trainer.{field_name} must be an explicit positive integer: "
                "upstream val.py falls back to lexicographic max(input_id) over "
                "ragged lists unless BOTH tgt_vocab_size and max_seq_length are set"
            )
        explicit_dims[field_name] = value
    if len({value == "auto" for value in explicit_dims.values()}) > 1:
        raise PerturbGenConfigError(
            "perturb trainer.tgt_vocab_size and trainer.max_seq_length must both be positive integers or both be 'auto'"
        )
    datamodule = perturb_config.get("datamodule", {})
    if isinstance(datamodule, Mapping) and "max_len" in datamodule:
        max_len = datamodule["max_len"]
        if max_len == "auto":
            if any(value != "auto" for value in explicit_dims.values()):
                raise PerturbGenConfigError(
                    "perturb datamodule.max_len='auto' requires both trainer dimensions to be 'auto'"
                )
        elif not isinstance(max_len, int) or isinstance(max_len, bool):
            raise PerturbGenConfigError("perturb datamodule.max_len must be an integer or 'auto'")
        elif max_len != explicit_dims["max_seq_length"]:
            raise PerturbGenConfigError(
                "perturb datamodule.max_len must equal trainer.max_seq_length "
                "(base value; val.py adds the +100/+50 runtime buffer itself)"
            )
    if "genes_to_perturb" not in trainer:
        return
    genes = trainer["genes_to_perturb"]
    if not isinstance(genes, Sequence) or isinstance(genes, (str, bytes)):
        raise PerturbGenConfigError("perturb genes_to_perturb must be a sequence")
    if len(genes) != 1:
        raise PerturbGenConfigError("strict candidate pipeline requires exactly one gene_to_perturb per stage")
    gene = str(genes[0])
    if _SAFE_IDENTIFIER_PATTERN.fullmatch(gene) is None:
        raise PerturbGenConfigError("perturb target gene must contain only letters, digits, dot, underscore, or hyphen")


def _validate_stage_dimension_consistency(stages: Mapping[str, Any]) -> None:
    """Reject tgt_vocab_size drift across train_mask/train_decoder/perturb.

    A checkpoint restored with a different tgt_vocab_size than it was trained
    with fails deep inside val.py with a tensor size mismatch (observed in the
    2026-08-23 M0 smoke iterations); when more than one stage declares the
    dimension explicitly they must agree.
    """
    declared: dict[str, int] = {}
    for stage_name in ("train_mask", "train_decoder"):
        stage_config = stages.get(stage_name)
        if not isinstance(stage_config, Mapping):
            continue
        args = stage_config.get("args", {})
        if isinstance(args, Mapping) and isinstance(args.get("tgt_vocab_size"), int):
            declared[stage_name] = args["tgt_vocab_size"]
    perturb_trainer = (
        stages.get("perturb", {}).get("perturb_config", {}).get("trainer", {})
        if isinstance(stages.get("perturb"), Mapping)
        else {}
    )
    if isinstance(perturb_trainer, Mapping) and isinstance(perturb_trainer.get("tgt_vocab_size"), int):
        declared["perturb"] = perturb_trainer["tgt_vocab_size"]
    values = set(declared.values())
    if len(values) > 1:
        detail = ", ".join(f"{name}={value}" for name, value in sorted(declared.items()))
        raise PerturbGenConfigError(
            f"tgt_vocab_size drift across stages ({detail}); restored checkpoints "
            "fail with tensor size mismatches when dimensions disagree"
        )


def build_stage_plans(
    config_or_path: str | Path | Mapping[str, Any],
    *,
    project_root: str | Path | None = None,
) -> tuple[StagePlan, ...]:
    """Build the fixed six-stage execution plan."""

    if isinstance(config_or_path, Mapping):
        config = dict(config_or_path)
    else:
        config = load_pipeline_config(config_or_path)

    repo_config = config["repo"]
    if not isinstance(repo_config, Mapping):
        raise PerturbGenConfigError("repo must be a mapping")
    roots = validate_repo_roots(
        project_root=project_root,
        perturbgen_repo_root=repo_config.get("perturbgen_repo"),
    )

    env_config = config["environment"]
    pipeline_config = config["pipeline"]
    if not isinstance(env_config, Mapping) or not isinstance(pipeline_config, Mapping):
        raise PerturbGenConfigError("environment and pipeline must be mappings")

    python_raw = env_config.get("python")
    if not isinstance(python_raw, str) or not python_raw:
        raise PerturbGenConfigError("environment.python must be a non-empty string")
    external_python = _resolve_path(
        python_raw,
        project_root=roots.project_root,
        perturbgen_repo_root=roots.perturbgen_repo_root,
    )

    output_root_raw = pipeline_config.get("output_root")
    if not isinstance(output_root_raw, str) or not output_root_raw:
        raise PerturbGenConfigError("pipeline.output_root must be a non-empty string")
    output_root = _resolve_path(
        output_root_raw,
        project_root=roots.project_root,
        perturbgen_repo_root=roots.perturbgen_repo_root,
    )

    perturbgen_commit = str(pipeline_config.get("perturbgen_commit", "")).strip().lower()
    if re.fullmatch(r"[0-9a-f]{7,40}", perturbgen_commit) is None:
        raise PerturbGenConfigError("pipeline.perturbgen_commit must be a 7-40 character hexadecimal git commit")
    try:
        estimated_total_output_bytes = int(pipeline_config["estimated_total_output_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PerturbGenConfigError("pipeline.estimated_total_output_bytes must be a positive integer") from exc
    if estimated_total_output_bytes <= 0:
        raise PerturbGenConfigError("pipeline.estimated_total_output_bytes must be a positive integer")

    stage_versions = pipeline_config.get("stage_versions", {})
    dry_run_estimates = pipeline_config.get("dry_run_estimates", {})
    dependency_files = tuple(
        _resolve_path(
            raw,
            project_root=roots.project_root,
            perturbgen_repo_root=roots.perturbgen_repo_root,
            prefer_repo=True,
        )
        for raw in env_config.get("dependency_files", [])
    )
    asset_paths = tuple(
        _resolve_path(
            raw,
            project_root=roots.project_root,
            perturbgen_repo_root=roots.perturbgen_repo_root,
        )
        for raw in env_config.get("asset_paths", [])
    )

    stages = config["stages"]
    _validate_stage_dimension_consistency(stages)
    plans: list[StagePlan] = []
    for stage_name in STAGE_ORDER:
        stage_config = stages[stage_name]
        if not isinstance(stage_config, Mapping):
            raise PerturbGenConfigError(f"stage {stage_name} must be a mapping")
        driver = str(stage_config.get("driver"))
        timeout_seconds = int(stage_config.get("timeout_seconds", 0))
        if timeout_seconds <= 0:
            raise PerturbGenConfigError(f"{stage_name}.timeout_seconds must be > 0")
        stage_output_dir = _ensure_under(
            (output_root / str(stage_config.get("output_subdir", stage_name))).resolve(),
            output_root,
            label=f"{stage_name}.output_subdir",
        )
        expected_outputs = _output_checks(
            stage_name,
            stage_output_dir,
            raw_outputs=stage_config.get("expected_outputs", ()),
            output_root=output_root,
            project_root=roots.project_root,
            perturbgen_repo_root=roots.perturbgen_repo_root,
        )
        if stage_name == "tokenise":
            _validate_tokenise_output_contract(
                stage_config,
                expected_outputs,
                perturbgen_repo_root=roots.perturbgen_repo_root,
            )
        elif stage_name == "perturb":
            _validate_perturb_candidate_contract(stage_config)
        uses_gpu = bool(stage_config.get("uses_gpu", False))
        generated_files: tuple[GeneratedFileSpec, ...] = ()
        cwd = roots.perturbgen_repo_root

        if driver == "module":
            module = str(stage_config.get("module", "")).strip()
            subcommand = str(stage_config.get("subcommand", "")).strip()
            if not module or not subcommand:
                raise PerturbGenConfigError(f"{stage_name} module driver requires module and subcommand")
            argv = (
                str(external_python),
                "-m",
                module,
                subcommand,
                *_cli_args_to_argv(stage_config.get("args", {})),
            )
        elif driver == "perturb_script":
            config_relpath = str(stage_config.get("generated_config_path", "generated/perturbation.yaml"))
            perturb_config = stage_config.get("perturb_config")
            if not isinstance(perturb_config, Mapping):
                raise PerturbGenConfigError(f"{stage_name}.perturb_config must be a mapping")
            generated_files = (
                _generated_perturb_config(
                    stage_name,
                    stage_output_dir,
                    output_root,
                    config_relpath,
                    perturb_config,
                ),
            )
            argv = (
                str(external_python),
                str((roots.perturbgen_repo_root / "perturbgen/Perturb/val.py").resolve()),
                "--config",
                str(generated_files[0].path),
            )
            raw_seed = stage_config.get("seed")
            if raw_seed is not None:
                seed_value = int(raw_seed)
                if seed_value < 0:
                    raise PerturbGenConfigError(f"{stage_name}.seed must be >= 0")
                # Upstream val.py natively supports --seed; multi-seed formal
                # verdicts (proposal §4.7) rely on it.
                argv = (*argv, "--seed", str(seed_value))
        elif driver == "script":
            script_root = str(stage_config.get("script_root", "project"))
            prefer_repo = script_root == "perturbgen"
            script_path = stage_config.get("script_path")
            if not isinstance(script_path, str) or not script_path:
                raise PerturbGenConfigError(f"{stage_name}.script_path must be a non-empty string")
            resolved_script = _resolve_path(
                script_path,
                project_root=roots.project_root,
                perturbgen_repo_root=roots.perturbgen_repo_root,
                prefer_repo=prefer_repo,
            )
            allowed_script_root = roots.perturbgen_repo_root if prefer_repo else roots.project_root
            try:
                resolved_script.relative_to(allowed_script_root)
            except ValueError as exc:
                raise PerturbGenConfigError(
                    f"{stage_name}.script_path escapes declared script_root: {resolved_script}"
                ) from exc
            if not resolved_script.is_file():
                raise PerturbGenConfigError(f"{stage_name}.script_path is not a file: {resolved_script}")
            cwd = roots.perturbgen_repo_root if prefer_repo else roots.project_root
            argv = (
                str(external_python),
                str(resolved_script),
                *_cli_args_to_argv(
                    stage_config.get("args", {}),
                    hyphenate_underscores=str(stage_config.get("flag_style", "underscore")) == "hyphen",
                ),
            )
        elif driver == "internal_report":
            cwd = roots.project_root
            argv = ()
        else:
            raise PerturbGenConfigError(f"unsupported driver for stage {stage_name}: {driver}")

        fingerprint_paths = tuple(
            _fingerprint_path(
                raw,
                project_root=roots.project_root,
                perturbgen_repo_root=roots.perturbgen_repo_root,
            )
            for raw in stage_config.get("fingerprint_paths", [])
        )
        stage_version = stage_versions.get(stage_name, 1)
        fingerprint_config = {
            "pipeline_commit": perturbgen_commit,
            "random_seed": pipeline_config.get("random_seed"),
            "stage_version": stage_version,
            "estimated_total_output_bytes": estimated_total_output_bytes,
            "stage_config": json.loads(json.dumps(stage_config)),
        }
        plans.append(
            StagePlan(
                name=stage_name,
                driver=driver,
                argv=tuple(argv),
                cwd=cwd,
                output_dir=stage_output_dir,
                timeout_seconds=timeout_seconds,
                uses_gpu=uses_gpu,
                output_root=output_root,
                external_python=external_python,
                expected_outputs=expected_outputs,
                generated_files=generated_files,
                fingerprint_paths=fingerprint_paths,
                dependency_files=dependency_files,
                asset_paths=asset_paths,
                resource_estimate=dry_run_estimates.get(stage_name, {}),
                fingerprint_config=fingerprint_config,
                roots=roots,
            )
        )
    return tuple(plans)
