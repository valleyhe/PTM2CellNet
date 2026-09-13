"""Strict stage runner for the external PerturbGen pipeline."""

from __future__ import annotations

import fcntl
import json
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

import yaml

from .config_builder import GeneratedFileSpec, OutputCheck, StagePlan
from .dimensions import PerturbGenDimensionError, derive_perturbgen_dimensions
from .env_guard import probe_external_environment, sanitize_text, sha256_file, sha256_path


class PerturbGenStageError(RuntimeError):
    """Raised when a stage exits unsuccessfully or produces invalid outputs."""


class PerturbGenResumeError(PerturbGenStageError):
    """Raised when resume/reuse is requested but the fingerprint does not match."""


STAGE_MANIFEST = "stage_manifest.json"
_ARTIFACT_REF_PATTERN = re.compile(r"^@artifact:([a-zA-Z0-9_.-]+):([a-zA-Z0-9_.-]+)$")


@dataclass(frozen=True)
class StageExecutionResult:
    """Runtime result for one stage."""

    stage: str
    status: str
    output_dir: Path
    manifest_path: Path
    reused: bool
    fingerprint: str
    artifacts: Mapping[str, str]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_digest(payload: Mapping[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    import hashlib

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _sha256_outputs(outputs: Iterable[OutputCheck]) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for output in outputs:
        record = {"kind": output.kind}
        if output.path.exists() and output.path.is_file():
            record["sha256"] = sha256_file(output.path)
        elif output.path.exists() and output.path.is_dir():
            files = sorted(path for path in output.path.rglob("*") if path.is_file())
            record["files"] = str(len(files))
            record["tree_sha256"] = _json_digest(
                {str(path.relative_to(output.path)): sha256_file(path) for path in files}
            )
        records[str(output.path)] = record
    return records


def _artifact_paths(outputs: Iterable[OutputCheck]) -> dict[str, str]:
    return {
        output.name: str(output.path)
        for output in outputs
        if output.name is not None
    }


@contextmanager
def _gpu_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class PerturbGenRunner:
    """Execute the fixed six-stage PerturbGen bridge with strict resume rules."""

    def __init__(self, *, gpu_lock_file: str | Path):
        self._gpu_lock_file = Path(gpu_lock_file).expanduser().resolve()

    def dry_run(self, plans: Iterable[StagePlan]) -> list[dict[str, Any]]:
        """Return the command plan and resource estimate without writing files."""

        payload: list[dict[str, Any]] = []
        for plan in plans:
            payload.append(
                {
                    "stage": plan.name,
                    "driver": plan.driver,
                    "argv": list(plan.argv),
                    "cwd": str(plan.cwd),
                    "output_dir": str(plan.output_dir),
                    "uses_gpu": plan.uses_gpu,
                    "timeout_seconds": plan.timeout_seconds,
                    "expected_outputs": [str(output.path) for output in plan.expected_outputs],
                    "output_discovery": {
                        output.name: output.discover_glob
                        for output in plan.expected_outputs
                        if output.discover_glob is not None and output.name is not None
                    },
                    "generated_files": [str(generated.path) for generated in plan.generated_files],
                    "resource_estimate": dict(plan.resource_estimate),
                }
            )
        return payload

    def run_pipeline(
        self,
        plans: Iterable[StagePlan],
        *,
        resume: bool = False,
        dry_run: bool = False,
    ) -> list[StageExecutionResult] | list[dict[str, Any]]:
        plans = list(plans)
        if dry_run:
            return self.dry_run(plans)

        if not plans:
            return []

        first = plans[0]
        self._ensure_disk_budget(first)
        env_report = probe_external_environment(
            first.external_python,
            project_root=first.roots.project_root if first.roots is not None else None,
            perturbgen_repo_root=first.roots.perturbgen_repo_root if first.roots is not None else None,
            dependency_files=first.dependency_files,
            asset_paths=first.asset_paths,
            expected_perturbgen_commit=str(
                first.fingerprint_config.get("pipeline_commit", "")
            ) or None,
        )
        env_report.ensure_ready()

        artifacts = self._load_artifact_registry(first.output_root)
        results: list[StageExecutionResult] = []
        for plan in plans:
            result = self.run_stage(
                plan,
                env_report=env_report,
                resume=resume,
                artifact_registry=artifacts,
            )
            results.append(result)
            artifacts.update(
                {(result.stage, name): Path(path) for name, path in result.artifacts.items()}
            )
        return results

    def _load_artifact_registry(self, output_root: Path) -> dict[tuple[str, str], Path]:
        registry: dict[tuple[str, str], Path] = {}
        if not output_root.exists():
            return registry
        for manifest_path in sorted(output_root.rglob(STAGE_MANIFEST)):
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if payload.get("status") != "success":
                continue
            stage = payload.get("stage")
            artifact_paths = payload.get("artifacts", {})
            if not isinstance(stage, str) or not isinstance(artifact_paths, Mapping):
                continue
            output_records = payload.get("outputs", {})
            if not isinstance(output_records, Mapping):
                raise PerturbGenResumeError(
                    f"manifest has invalid output records: {manifest_path}"
                )
            for name, path in artifact_paths.items():
                if isinstance(name, str) and isinstance(path, str):
                    artifact_path = Path(path)
                    previous_record = output_records.get(path)
                    if not isinstance(previous_record, Mapping):
                        raise PerturbGenResumeError(
                            f"manifest artifact {stage}:{name} has no output hash record"
                        )
                    current_record = _sha256_outputs(
                        (OutputCheck(artifact_path, str(previous_record.get("kind", "file"))),)
                    ).get(path)
                    if current_record != dict(previous_record):
                        raise PerturbGenResumeError(
                            f"upstream artifact {stage}:{name} changed; refusing downstream reuse"
                        )
                    registry[(stage, name)] = artifact_path
        return registry

    def _resolve_value(
        self,
        value: Any,
        artifact_registry: Mapping[tuple[str, str], Path],
    ) -> Any:
        if isinstance(value, str):
            match = _ARTIFACT_REF_PATTERN.fullmatch(value)
            if match is None:
                return value
            key = (match.group(1), match.group(2))
            if key not in artifact_registry:
                raise PerturbGenStageError(
                    f"unresolved upstream artifact reference: {value}"
                )
            return str(artifact_registry[key].resolve())
        if isinstance(value, list):
            return [self._resolve_value(item, artifact_registry) for item in value]
        if isinstance(value, tuple):
            return tuple(self._resolve_value(item, artifact_registry) for item in value)
        if isinstance(value, dict):
            return {
                key: self._resolve_value(item, artifact_registry)
                for key, item in value.items()
            }
        return value

    def _resolve_plan_artifact_refs(
        self,
        plan: StagePlan,
        artifact_registry: Mapping[tuple[str, str], Path],
    ) -> StagePlan:
        generated_files = tuple(
            replace(
                generated,
                payload=self._resolve_value(generated.payload, artifact_registry),
            )
            for generated in plan.generated_files
        )
        fingerprint_paths = tuple(
            Path(self._resolve_value(path, artifact_registry)).resolve()
            if isinstance(path, str)
            else path
            for path in plan.fingerprint_paths
        )
        return replace(
            plan,
            argv=tuple(str(self._resolve_value(item, artifact_registry)) for item in plan.argv),
            generated_files=generated_files,
            fingerprint_paths=fingerprint_paths,
            fingerprint_config=self._resolve_value(
                dict(plan.fingerprint_config), artifact_registry
            ),
        )

    def _resolve_auto_perturb_dimensions(self, plan: StagePlan) -> StagePlan:
        """Resolve token dimensions after upstream tokenization succeeds.

        train.py derives these values from the actual tokenized datasets.
        The perturbation script must receive the identical BASE values because
        it adds its own +50 vocabulary and +100 sequence buffers. Computing
        them from the artifacts here keeps train and inference architectures
        exactly aligned without importing the external package in PTM2CellNet.
        """

        if plan.driver != "perturb_script" or not plan.generated_files:
            return plan
        generated = plan.generated_files[0]
        payload = generated.payload
        if not isinstance(payload, Mapping):
            return plan
        trainer = payload.get("trainer")
        datamodule = payload.get("datamodule")
        data = payload.get("data")
        if not isinstance(trainer, Mapping) or not isinstance(datamodule, Mapping):
            return plan
        if not isinstance(data, Mapping):
            return plan
        dimensions = (trainer.get("tgt_vocab_size"), trainer.get("max_seq_length"))
        if "auto" not in dimensions and datamodule.get("max_len") != "auto":
            return plan
        if dimensions != ("auto", "auto") or datamodule.get("max_len") != "auto":
            raise PerturbGenStageError(
                "perturb auto dimensions require trainer.tgt_vocab_size, "
                "trainer.max_seq_length, and datamodule.max_len all to be 'auto'"
            )
        src_dataset = data.get("src_dataset_file")
        tgt_dataset_folder = data.get("tgt_dataset_folder")
        if not isinstance(src_dataset, str) or not isinstance(tgt_dataset_folder, str):
            raise PerturbGenStageError(
                "perturb auto dimensions require resolved src_dataset_file and tgt_dataset_folder"
            )
        try:
            resolved = derive_perturbgen_dimensions(
                plan.external_python,
                src_dataset=src_dataset,
                tgt_dataset_folder=tgt_dataset_folder,
                cwd=plan.cwd,
            )
        except PerturbGenDimensionError as exc:
            raise PerturbGenStageError(str(exc)) from exc

        updated_payload = {
            key: dict(value) if isinstance(value, Mapping) else value
            for key, value in payload.items()
        }
        updated_payload["trainer"]["tgt_vocab_size"] = resolved.tgt_vocab_size
        updated_payload["trainer"]["max_seq_length"] = resolved.max_seq_length
        updated_payload["datamodule"]["max_len"] = resolved.max_seq_length
        updated_generated = replace(generated, payload=updated_payload)
        return replace(plan, generated_files=(updated_generated,))

    def _resolve_discovered_outputs(
        self, outputs: Iterable[OutputCheck]
    ) -> tuple[OutputCheck, ...]:
        resolved: list[OutputCheck] = []
        for output in outputs:
            if output.discover_glob is None:
                resolved.append(output)
                continue
            if not output.path.is_dir():
                raise PerturbGenStageError(
                    f"artifact discovery root missing: {output.path}"
                )
            matches = sorted(path.resolve() for path in output.path.glob(output.discover_glob))
            if output.kind == "directory":
                matches = [path for path in matches if path.is_dir()]
            else:
                matches = [path for path in matches if path.is_file()]
            if len(matches) != 1:
                raise PerturbGenStageError(
                    f"artifact discovery for {output.name!r} expected exactly one match "
                    f"below {output.path} with {output.discover_glob!r}, found {len(matches)}"
                )
            resolved.append(replace(output, path=matches[0], discover_glob=None))
        return tuple(resolved)

    def _ensure_external_output_targets_clean(self, plan: StagePlan) -> None:
        for output in plan.expected_outputs:
            try:
                output.path.relative_to(plan.output_dir)
                continue
            except ValueError:
                pass
            if output.discover_glob is not None:
                if output.path.exists() and any(output.path.glob(output.discover_glob)):
                    raise PerturbGenStageError(
                        f"external artifact discovery root already contains matches: {output.path}"
                    )
            elif output.path.exists():
                raise PerturbGenStageError(
                    f"external expected output already exists before stage start: {output.path}"
                )

    def _ensure_disk_budget(self, plan: StagePlan) -> None:
        estimated = int(plan.fingerprint_config.get("estimated_total_output_bytes", 0))
        if estimated <= 0:
            raise PerturbGenStageError("missing positive estimated_total_output_bytes")
        probe = plan.output_root
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        free = shutil.disk_usage(probe).free
        required = estimated * 2 + 10 * 1024**3
        if free < required:
            raise PerturbGenStageError(
                "insufficient disk space for PerturbGen pipeline: "
                f"free={free}, required={required}"
            )

    def run_stage(
        self,
        plan: StagePlan,
        *,
        env_report,
        resume: bool = False,
        artifact_registry: Mapping[tuple[str, str], Path] | None = None,
    ) -> StageExecutionResult:
        plan = self._resolve_plan_artifact_refs(plan, artifact_registry or {})
        plan = self._resolve_auto_perturb_dimensions(plan)
        manifest_path = plan.output_dir / STAGE_MANIFEST
        fingerprint_material = self._build_fingerprint_material(plan, env_report)
        fingerprint = _json_digest(fingerprint_material)

        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous_fingerprint = previous.get("fingerprint")
            if previous_fingerprint != fingerprint:
                raise PerturbGenResumeError(
                    f"stage {plan.name} fingerprint changed; refusing to reuse {plan.output_dir}"
                )
            if not resume:
                raise PerturbGenResumeError(
                    f"stage {plan.name} already completed at {plan.output_dir}; pass resume=True to reuse"
                )
            if previous.get("status") != "success":
                raise PerturbGenResumeError(
                    f"stage {plan.name} cannot resume because prior status is {previous.get('status')!r}"
                )
            resolved_outputs = self._resolve_discovered_outputs(plan.expected_outputs)
            self._validate_outputs(resolved_outputs)
            current_outputs = _sha256_outputs(resolved_outputs)
            if current_outputs != previous.get("outputs"):
                raise PerturbGenResumeError(
                    f"stage {plan.name} outputs changed; refusing to reuse {plan.output_dir}"
                )
            return StageExecutionResult(
                stage=plan.name,
                status="success",
                output_dir=plan.output_dir,
                manifest_path=manifest_path,
                reused=True,
                fingerprint=fingerprint,
                artifacts=_artifact_paths(resolved_outputs),
            )

        if plan.output_dir.exists() and any(plan.output_dir.iterdir()):
            raise PerturbGenStageError(f"stage output directory is not empty: {plan.output_dir}")
        self._ensure_external_output_targets_clean(plan)
        plan.output_dir.mkdir(parents=True, exist_ok=True)

        for generated in plan.generated_files:
            self._materialize_generated_file(generated)

        started_at = _now_iso()
        start = time.monotonic()
        try:
            if plan.driver == "internal_report":
                stdout, stderr = self._write_report(plan)
                returncode = 0
                # widened by the except branch below
            else:
                stdout, stderr, returncode = self._execute_subprocess(plan, env_report)
            resolved_outputs = self._resolve_discovered_outputs(plan.expected_outputs)
            self._validate_outputs(resolved_outputs)
            plan = replace(plan, expected_outputs=resolved_outputs)
            status = "success"
            error_payload = None
        except Exception as exc:
            stdout = ""
            stderr = str(exc)
            fallback_returncode = cast(int | None, getattr(exc, "returncode", None))
            status = "failed"
            returncode = fallback_returncode if fallback_returncode is not None else -1
            error_payload = {"type": exc.__class__.__name__, "message": str(exc)}
            self._write_manifest(
                manifest_path,
                self._manifest_payload(
                    plan=plan,
                    env_report=env_report,
                    fingerprint=fingerprint,
                    fingerprint_material=fingerprint_material,
                    started_at=started_at,
                    duration_seconds=time.monotonic() - start,
                    status=status,
                    stdout=stdout,
                    stderr=stderr,
                    returncode=returncode,
                    error=error_payload,
                    outputs=(),
                ),
            )
            raise

        self._write_manifest(
            manifest_path,
            self._manifest_payload(
                plan=plan,
                env_report=env_report,
                fingerprint=fingerprint,
                fingerprint_material=fingerprint_material,
                started_at=started_at,
                duration_seconds=time.monotonic() - start,
                status=status,
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                error=error_payload,
                outputs=plan.expected_outputs,
            ),
        )
        return StageExecutionResult(
            stage=plan.name,
            status=status,
            output_dir=plan.output_dir,
            manifest_path=manifest_path,
            reused=False,
            fingerprint=fingerprint,
            artifacts=_artifact_paths(plan.expected_outputs),
        )

    def _build_fingerprint_material(self, plan: StagePlan, env_report) -> dict[str, Any]:
        file_hashes = {str(path): sha256_path(path) for path in plan.fingerprint_paths}
        return {
            "stage": plan.name,
            "driver": plan.driver,
            "argv": list(plan.argv),
            "cwd": str(plan.cwd),
            "timeout_seconds": plan.timeout_seconds,
            "uses_gpu": plan.uses_gpu,
            "external_python": str(env_report.python_path),
            "external_python_version": env_report.python_version,
            "dependency_hashes": dict(env_report.dependency_hashes),
            "asset_hashes": dict(env_report.asset_hashes),
            "fingerprint_files": file_hashes,
            "fingerprint_config": dict(plan.fingerprint_config),
        }

    def _materialize_generated_file(self, generated: GeneratedFileSpec) -> None:
        generated.path.parent.mkdir(parents=True, exist_ok=True)
        if generated.format == "yaml":
            generated.path.write_text(
                yaml.safe_dump(generated.payload, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            return
        if generated.format == "json":
            generated.path.write_text(
                json.dumps(generated.payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return
        if generated.format == "text":
            generated.path.write_text(str(generated.payload), encoding="utf-8")
            return
        raise PerturbGenStageError(f"unsupported generated file format: {generated.format}")

    def _execute_subprocess(self, plan: StagePlan, env_report) -> tuple[str, str, int]:
        kwargs = {
            "cwd": plan.cwd,
            "text": True,
            "capture_output": True,
            "check": False,
            "timeout": plan.timeout_seconds,
            "shell": False,
        }
        run_kwargs = cast(Mapping[str, Any], kwargs)
        if plan.uses_gpu:
            with _gpu_lock(self._gpu_lock_file):
                completed = subprocess.run(list(plan.argv), **run_kwargs)
        else:
            completed = subprocess.run(list(plan.argv), **run_kwargs)

        stdout = sanitize_text(completed.stdout, env_report.roots)
        stderr = sanitize_text(completed.stderr, env_report.roots)
        if completed.returncode != 0:
            raise PerturbGenStageError(
                f"stage {plan.name} failed with return code {completed.returncode}: {stderr or stdout}"
            )
        return stdout, stderr, completed.returncode

    def _validate_outputs(self, outputs: Iterable[OutputCheck]) -> None:
        for output in outputs:
            if output.kind == "directory":
                if not output.path.is_dir() or not any(output.path.iterdir()):
                    raise PerturbGenStageError(
                        f"expected non-empty output directory missing: {output.path}"
                    )
                continue
            if output.kind == "perturbgen_embedding_asset":
                from src.models.perturbgen_embedding import load_perturbgen_embedding_asset

                try:
                    load_perturbgen_embedding_asset(output.path)
                except (ImportError, OSError, TypeError, ValueError) as exc:
                    raise PerturbGenStageError(
                        f"invalid PerturbGen embedding asset {output.path}: {exc}"
                    ) from exc
                continue
            if not output.path.is_file():
                raise PerturbGenStageError(f"expected output missing: {output.path}")
            if output.kind == "json":
                try:
                    payload = json.loads(output.path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise PerturbGenStageError(f"invalid JSON output: {output.path}") from exc
                if not isinstance(payload, dict):
                    raise PerturbGenStageError(f"JSON output must contain an object root: {output.path}")
            elif output.kind == "yaml":
                payload = yaml.safe_load(output.path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise PerturbGenStageError(f"YAML output must contain a mapping root: {output.path}")
            elif output.kind == "h5ad":
                from .results import validate_perturbgen_h5ad_schema

                try:
                    validate_perturbgen_h5ad_schema(output.path, **dict(output.schema))
                except (ImportError, KeyError, TypeError, ValueError, OSError) as exc:
                    raise PerturbGenStageError(
                        f"invalid PerturbGen h5ad output {output.path}: {exc}"
                    ) from exc
            elif output.kind != "file":
                raise PerturbGenStageError(f"unsupported output check kind: {output.kind}")

    def _write_report(self, plan: StagePlan) -> tuple[str, str]:
        prior_manifests = {}
        for manifest_path in sorted(plan.output_root.rglob(STAGE_MANIFEST)):
            if manifest_path.parent == plan.output_dir:
                continue
            key = str(manifest_path.parent.relative_to(plan.output_root))
            prior_manifests[key] = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = {
            "schema_version": 1,
            "generated_at": _now_iso(),
            "stages": prior_manifests,
        }
        target = plan.expected_outputs[0].path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return json.dumps({"report": str(target)}), ""

    def _manifest_payload(
        self,
        *,
        plan: StagePlan,
        env_report,
        fingerprint: str,
        fingerprint_material: Mapping[str, Any],
        started_at: str,
        duration_seconds: float,
        status: str,
        stdout: str,
        stderr: str,
        returncode: int | None,
        error: Mapping[str, Any] | None,
        outputs: Iterable[OutputCheck],
    ) -> dict[str, Any]:
        stdout_path = plan.output_dir / "stdout.log"
        stderr_path = plan.output_dir / "stderr.log"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        return {
            "schema_version": 1,
            "stage": plan.name,
            "status": status,
            "fingerprint": fingerprint,
            "fingerprint_material": fingerprint_material,
            "started_at": started_at,
            "finished_at": _now_iso(),
            "duration_seconds": round(duration_seconds, 6),
            "command": list(plan.argv),
            "cwd": str(plan.cwd),
            "returncode": returncode,
            "uses_gpu": plan.uses_gpu,
            "gpu_lock_file": str(self._gpu_lock_file) if plan.uses_gpu else None,
            "outputs": _sha256_outputs(outputs),
            "artifacts": _artifact_paths(outputs),
            "logs": {
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            },
            "environment": {
                "python": str(env_report.python_path),
                "python_version": env_report.python_version,
                "dependency_hashes": dict(env_report.dependency_hashes),
                "asset_hashes": dict(env_report.asset_hashes),
                "perturbgen_commit": env_report.perturbgen_commit,
                "warnings": list(env_report.warnings),
            },
            "generated_files": [str(item.path) for item in plan.generated_files],
            "error": error,
        }

    def _write_manifest(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
