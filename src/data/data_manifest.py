"""Versioned data inventory and local snapshot validation.

The project needs to distinguish three different facts that were previously
mixed together: a dataset is named in the scientific design, a loader exists,
and a particular local release is available.  This module makes those facts
machine-readable without downloading or redistributing controlled data.

The manifest format is intentionally YAML and human-reviewable.  A dataset may
therefore be ``controlled`` or ``planned`` with a null local snapshot while
still documenting its source, expected format and quality gates.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import yaml

from ..utils.logging import setup_logger

logger = setup_logger(__name__)

ManifestValue = Union[str, int, float, bool, None, List[Any], Dict[str, Any]]
Manifest = Dict[str, Any]
ValidationReport = Dict[str, Any]

MANIFEST_REQUIRED_KEYS = ("manifest_version", "project", "datasets")
DATASET_REQUIRED_KEYS = (
    "id",
    "name",
    "category",
    "status",
    "source",
    "formats",
    "canonical_schema",
    "quality_requirements",
)
VALID_STATUSES = {"implemented", "partial", "controlled", "planned", "local"}


class DataManifestError(ValueError):
    """Raised when a manifest cannot be parsed or violates its schema."""


def sha256_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Manifest snapshot file does not exist: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Return a stable digest for a parsed manifest."""

    payload = json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_mapping(value: Any, context: str, errors: List[str]) -> Optional[Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        errors.append(f"{context} must be a mapping")
        return None
    return value


def _resolve_snapshot_path(
    manifest_path: Optional[Union[str, Path]],
    root_dir: Optional[Union[str, Path]],
    relative_path: str,
) -> Path:
    if root_dir is not None:
        base = Path(root_dir)
    elif manifest_path is not None:
        # data/manifests/datasets.yaml -> repository root by default.
        base = Path(manifest_path).resolve().parent.parent.parent
    else:
        base = Path.cwd()
    candidate = Path(relative_path)
    return candidate if candidate.is_absolute() else base / candidate


def _iter_files(dataset: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    files = dataset.get("files", [])
    if isinstance(files, Mapping):
        yield files
    elif isinstance(files, Sequence) and not isinstance(files, (str, bytes)):
        for item in files:
            if isinstance(item, Mapping):
                yield item


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Optional[Union[str, Path]] = None,
    root_dir: Optional[Union[str, Path]] = None,
    check_files: bool = False,
    verify_hashes: bool = False,
    strict_warnings: bool = False,
    profile: Optional[str] = None,
) -> ValidationReport:
    """Validate manifest structure and, optionally, local snapshots.

    ``check_files`` only turns a missing file into an error when that file is
    marked ``required: true``.  Controlled/planned entries are expected to have
    no local path.  ``verify_hashes`` requires a non-null ``sha256`` for every
    existing snapshot and reports drift as an error.

    ``profile`` activates a named asset profile declared under the manifest's
    top-level ``profiles`` key (P1-04).  When a profile is active, every
    dataset listed in its ``required_datasets`` is treated as mandatory for
    that release: missing dataset, null snapshot path, absent file (with
    ``check_files``) or missing/mismatched hash (with ``verify_hashes``)
    all become hard errors — regardless of the per-file ``required`` flag.
    An unknown profile name is itself an error, so typos fail fast instead of
    silently validating nothing.
    """

    errors: List[str] = []
    warnings: List[str] = []

    missing_root = [key for key in MANIFEST_REQUIRED_KEYS if key not in manifest]
    if missing_root:
        errors.append("manifest missing required keys: " + ", ".join(missing_root))

    datasets_raw = manifest.get("datasets")
    if not isinstance(datasets_raw, list):
        errors.append("manifest.datasets must be a list")
        datasets: List[Any] = []
    else:
        datasets = datasets_raw

    # --- P1-04: profile activation -----------------------------------------
    profiles_raw = manifest.get("profiles")
    profile_spec: Optional[Mapping[str, Any]] = None
    profile_required_ids: set = set()
    if profile is not None:
        if not isinstance(profiles_raw, Mapping):
            errors.append(
                f"manifest has no profiles declared; cannot activate profile {profile!r}"
            )
        else:
            profile_spec = profiles_raw.get(profile)
            if not isinstance(profile_spec, Mapping):
                errors.append(
                    f"profile {profile!r} is not declared in manifest.profiles; "
                    f"available: {sorted(str(k) for k in profiles_raw.keys())}"
                )
            else:
                required_ids = profile_spec.get("required_datasets")
                if not isinstance(required_ids, list) or not all(
                    isinstance(item, str) for item in required_ids
                ):
                    errors.append(
                        f"profile {profile!r}.required_datasets must be a non-empty list of dataset ids"
                    )
                else:
                    profile_required_ids = set(required_ids)
                optional_ids = profile_spec.get("optional_datasets", [])
                if not isinstance(optional_ids, list) or not all(
                    isinstance(item, str) for item in optional_ids
                ):
                    errors.append(
                        f"profile {profile!r}.optional_datasets must be a list of dataset ids"
                    )
                else:
                    overlap = profile_required_ids.intersection(optional_ids)
                    if overlap:
                        errors.append(
                            f"profile {profile!r} lists the same dataset as required and optional: "
                            f"{sorted(overlap)}"
                        )
    # -----------------------------------------------------------------------

    seen_ids = set()
    dataset_reports: List[Dict[str, Any]] = []
    datasets_by_id: Dict[str, Mapping[str, Any]] = {}
    for index, raw_dataset in enumerate(datasets):
        context = f"datasets[{index}]"
        dataset = _as_mapping(raw_dataset, context, errors)
        if dataset is None:
            continue

        missing = [key for key in DATASET_REQUIRED_KEYS if key not in dataset]
        if missing:
            errors.append(f"{context} missing required keys: {', '.join(missing)}")

        dataset_id = dataset.get("id")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            errors.append(f"{context}.id must be a non-empty string")
            dataset_id = f"<index:{index}>"
        elif dataset_id in seen_ids:
            errors.append(f"duplicate dataset id: {dataset_id}")
        seen_ids.add(dataset_id)
        datasets_by_id[dataset_id] = dataset

        status = dataset.get("status")
        if status not in VALID_STATUSES:
            errors.append(
                f"{context}.status={status!r} is invalid; expected one of {sorted(VALID_STATUSES)}"
            )

        source = _as_mapping(dataset.get("source"), f"{context}.source", errors)
        if source is not None:
            for key in ("provider", "access", "license"):
                if key not in source or not str(source.get(key, "")).strip():
                    errors.append(f"{context}.source.{key} must be documented")

        formats = dataset.get("formats")
        if not isinstance(formats, list) or not formats or not all(isinstance(item, str) for item in formats):
            errors.append(f"{context}.formats must be a non-empty list of strings")

        schema = _as_mapping(dataset.get("canonical_schema"), f"{context}.canonical_schema", errors)
        if schema is not None:
            required_columns = schema.get("required_columns")
            if not isinstance(required_columns, list) or not required_columns:
                errors.append(f"{context}.canonical_schema.required_columns must be a non-empty list")

        quality = _as_mapping(
            dataset.get("quality_requirements"),
            f"{context}.quality_requirements",
            errors,
        )
        if quality is not None:
            if "checks" not in quality or not isinstance(quality.get("checks"), list):
                errors.append(f"{context}.quality_requirements.checks must be a list")

        file_report: List[Dict[str, Any]] = []
        # P1-04: profile 激活时，required dataset 的每个文件按 required 语义强制，
        # 无论 manifest 中的 per-file required 标志如何。
        profile_forced = dataset_id in profile_required_ids
        for file_index, file_entry in enumerate(_iter_files(dataset)):
            file_context = f"{context}.files[{file_index}]"
            relative_path = file_entry.get("path")
            required = bool(file_entry.get("required", False)) or profile_forced
            entry_report: Dict[str, Any] = {
                "path": relative_path,
                "required": required,
                "status": "not_registered",
            }
            if relative_path in (None, ""):
                if required and check_files:
                    errors.append(f"{file_context} is required but has no path")
                    entry_report["status"] = "missing_path"
                elif profile_forced:
                    # P1-04: profile 激活时要求本地快照路径存在——无路径即失败，
                    # 不依赖 --check-files（这是"资产齐备"的机器可判定条件）。
                    errors.append(
                        f"{file_context} is required by profile {profile!r} but has no registered path"
                    )
                    entry_report["status"] = "missing_path"
                else:
                    entry_report["status"] = "not_available"
                file_report.append(entry_report)
                continue
            if not isinstance(relative_path, str):
                errors.append(f"{file_context}.path must be a string or null")
                entry_report["status"] = "invalid_path"
                file_report.append(entry_report)
                continue

            resolved = _resolve_snapshot_path(manifest_path, root_dir, relative_path)
            entry_report["resolved_path"] = str(resolved)
            if not check_files:
                entry_report["status"] = "registered"
                file_report.append(entry_report)
                continue
            if not resolved.is_file():
                entry_report["status"] = "missing"
                message = f"{file_context} snapshot not found: {relative_path}"
                (errors if required else warnings).append(message)
                file_report.append(entry_report)
                continue

            actual_hash = sha256_file(resolved)
            entry_report["status"] = "present"
            entry_report["sha256_actual"] = actual_hash
            expected_hash = file_entry.get("sha256")
            if verify_hashes:
                if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                    errors.append(f"{file_context}.sha256 must be a 64-character hash when verifying")
                    entry_report["status"] = "missing_hash"
                elif expected_hash.lower() != actual_hash:
                    errors.append(
                        f"{file_context} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
                    )
                    entry_report["status"] = "hash_mismatch"
            elif expected_hash and str(expected_hash).lower() != actual_hash:
                warnings.append(f"{file_context} SHA-256 differs from the recorded snapshot")
            file_report.append(entry_report)

        dataset_reports.append(
            {
                "id": dataset_id,
                "status": status,
                "category": dataset.get("category"),
                "file_checks": file_report,
            }
        )

    # P1-04: profile 引用的 dataset 必须真实存在，避免"无声空校验"。
    if profile is not None:
        unknown_ids = sorted(profile_required_ids - seen_ids)
        if unknown_ids:
            errors.append(
                f"profile {profile!r} references unknown dataset ids: {unknown_ids}"
            )

    if strict_warnings and warnings:
        errors.extend(f"warning promoted to error: {warning}" for warning in warnings)

    report: ValidationReport = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "dataset_count": len(datasets),
        "dataset_reports": dataset_reports,
        "manifest_digest": manifest_digest(manifest),
    }
    if profile is not None:
        report["profile"] = profile
        report["profile_required_datasets"] = sorted(profile_required_ids)
    return report


def load_manifest(
    path: Union[str, Path],
    *,
    check_files: bool = False,
    verify_hashes: bool = False,
    strict_warnings: bool = False,
    profile: Optional[str] = None,
) -> Manifest:
    """Load and structurally validate a YAML manifest.

    ``profile`` activates the named asset profile (see :func:`validate_manifest`);
    when the profile's required assets are missing, loading fails fast.
    """

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Data manifest does not exist: {manifest_path}")
    try:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DataManifestError(f"Invalid YAML in {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataManifestError(f"Manifest root must be a mapping: {manifest_path}")

    report = validate_manifest(
        payload,
        manifest_path=manifest_path,
        check_files=check_files,
        verify_hashes=verify_hashes,
        strict_warnings=strict_warnings,
        profile=profile,
    )
    if not report["ok"]:
        raise DataManifestError(
            f"Invalid data manifest {manifest_path}: " + "; ".join(report["errors"])
        )
    return payload


def get_dataset(manifest: Mapping[str, Any], dataset_id: str) -> Dict[str, Any]:
    """Return one dataset entry or raise an actionable ``KeyError``."""

    for dataset in manifest.get("datasets", []):
        if isinstance(dataset, Mapping) and dataset.get("id") == dataset_id:
            return dict(dataset)
    raise KeyError(f"Dataset id not found in manifest: {dataset_id}")


__all__ = [
    "DataManifestError",
    "MANIFEST_REQUIRED_KEYS",
    "DATASET_REQUIRED_KEYS",
    "load_manifest",
    "validate_manifest",
    "sha256_file",
    "manifest_digest",
    "get_dataset",
]
