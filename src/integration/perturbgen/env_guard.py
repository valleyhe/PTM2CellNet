"""Strict environment and path checks for the PerturbGen bridge.

The PTM2CellNet main process must never import ``perturbgen`` directly.
This module only probes paths and the external Python executable.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_SENTINELS = ("AGENTS.md", "src", "configs")
PERTURBGEN_SENTINELS = ("pyproject.toml", "perturbgen/__main__.py", "perturbgen/Perturb/val.py")


class PerturbGenEnvError(ValueError):
    """Raised when the external PerturbGen environment is unsafe or incomplete."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of *path*."""

    digest = hashlib.sha256()
    file_path = Path(path).expanduser().resolve(strict=True)
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    """Hash one file or a directory tree including relative file names."""

    resolved = Path(path).expanduser().resolve(strict=True)
    if resolved.is_file():
        return sha256_file(resolved)
    if not resolved.is_dir():
        raise ValueError(f"path is neither a regular file nor directory: {resolved}")
    files = sorted(item for item in resolved.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"directory fingerprint input is empty: {resolved}")
    digest = hashlib.sha256()
    for item in files:
        relative = str(item.relative_to(resolved)).encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class ProjectRoots:
    """Validated PTM2CellNet and external PerturbGen repository roots."""

    project_root: Path
    perturbgen_repo_root: Path


@dataclass(frozen=True)
class ExternalEnvironmentReport:
    """Probe result for the external PerturbGen environment."""

    python_path: Path
    python_version: str | None
    roots: ProjectRoots
    dependency_hashes: Mapping[str, str]
    asset_hashes: Mapping[str, str]
    perturbgen_commit: str | None = None
    issues: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.issues

    def ensure_ready(self) -> None:
        if self.issues:
            raise PerturbGenEnvError("; ".join(self.issues))


def validate_repo_roots(
    project_root: str | Path | None = None,
    perturbgen_repo_root: str | Path | None = None,
) -> ProjectRoots:
    """Resolve and validate the PTM2CellNet root and PerturbGen repo root."""

    resolved_project_root = Path(project_root or PROJECT_ROOT).expanduser().resolve(strict=True)
    missing_project = [
        sentinel for sentinel in PROJECT_SENTINELS if not (resolved_project_root / sentinel).exists()
    ]
    if missing_project:
        raise PerturbGenEnvError(
            "invalid PTM2CellNet project root "
            f"{resolved_project_root}: missing {', '.join(missing_project)}"
        )

    candidate_repo = Path(perturbgen_repo_root or resolved_project_root / "ref/Perturbgen-src")
    if not candidate_repo.is_absolute():
        candidate_repo = resolved_project_root / candidate_repo
    resolved_repo_root = candidate_repo.expanduser().resolve(strict=True)
    missing_repo = [
        sentinel for sentinel in PERTURBGEN_SENTINELS if not (resolved_repo_root / sentinel).exists()
    ]
    if missing_repo:
        raise PerturbGenEnvError(
            "invalid PerturbGen repo root "
            f"{resolved_repo_root}: missing {', '.join(missing_repo)}"
        )

    return ProjectRoots(
        project_root=resolved_project_root,
        perturbgen_repo_root=resolved_repo_root,
    )


def sanitize_text(text: str, roots: ProjectRoots) -> str:
    """Remove absolute repository paths from logs."""

    sanitized = text.replace(str(roots.project_root), "<PROJECT_ROOT>")
    sanitized = sanitized.replace(str(roots.perturbgen_repo_root), "<PERTURBGEN_REPO>")
    return sanitized


def _probe_python_version(external_python: Path, cwd: Path) -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            [
                str(external_python),
                "-c",
                "import json, sys; print(json.dumps({'version': sys.version.split()[0]}))",
            ],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"failed to execute external python {external_python}: {exc}"

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        return None, f"external python probe failed: {stderr or f'return code {completed.returncode}'}"

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return None, f"external python probe returned invalid JSON: {exc}"

    version = payload.get("version")
    if not isinstance(version, str) or not version:
        return None, "external python probe did not return a usable version string"
    return version, None


def _probe_repo_commit(repo_root: Path) -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"failed to inspect PerturbGen commit: {exc}"
    commit = completed.stdout.strip()
    if completed.returncode != 0 or len(commit) != 40:
        return None, "failed to resolve PerturbGen repository commit"
    return commit, None


def _resolve_files(
    paths: Iterable[str | Path],
    *,
    project_root: Path,
    perturbgen_repo_root: Path,
) -> tuple[dict[str, str], list[str]]:
    hashes: dict[str, str] = {}
    issues: list[str] = []
    for raw_path in paths:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            project_candidate = (project_root / candidate).resolve()
            repo_candidate = (perturbgen_repo_root / candidate).resolve()
            if project_candidate.exists():
                candidate = project_candidate
            elif repo_candidate.exists():
                candidate = repo_candidate
            else:
                candidate = project_candidate
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            issues.append(f"required file missing: {raw_path}")
            continue
        if not resolved.is_file():
            issues.append(f"required file is not a regular file: {raw_path}")
            continue
        hashes[_relative_to(resolved, project_root)] = sha256_file(resolved)
    return hashes, issues


def probe_external_environment(
    external_python: str | Path,
    *,
    project_root: str | Path | None = None,
    perturbgen_repo_root: str | Path | None = None,
    dependency_files: Sequence[str | Path] = (),
    asset_paths: Sequence[str | Path] = (),
    expected_perturbgen_commit: str | None = None,
) -> ExternalEnvironmentReport:
    """Probe the external PerturbGen environment without importing perturbgen."""

    roots = validate_repo_roots(project_root=project_root, perturbgen_repo_root=perturbgen_repo_root)
    issues: list[str] = []
    warnings: list[str] = []

    python_path = Path(external_python).expanduser()
    if not python_path.is_absolute():
        python_path = (roots.project_root / python_path).resolve()
    else:
        python_path = python_path.resolve()
    if not python_path.exists():
        issues.append(f"external python not found: {python_path}")
        return ExternalEnvironmentReport(
            python_path=python_path,
            python_version=None,
            roots=roots,
            dependency_hashes={},
            asset_hashes={},
            perturbgen_commit=None,
            issues=tuple(issues),
            warnings=tuple(warnings),
        )

    python_version, python_issue = _probe_python_version(
        external_python=python_path,
        cwd=roots.perturbgen_repo_root,
    )
    if python_issue is not None:
        issues.append(python_issue)
    elif python_version is not None and not python_version.startswith("3.11"):
        issues.append(f"external python version is {python_version}, expected 3.11.x")

    perturbgen_commit, commit_issue = _probe_repo_commit(roots.perturbgen_repo_root)
    if commit_issue is not None:
        issues.append(commit_issue)
    elif expected_perturbgen_commit is not None and perturbgen_commit is not None:
        expected = expected_perturbgen_commit.strip().lower()
        if not perturbgen_commit.lower().startswith(expected):
            issues.append(
                "PerturbGen commit mismatch: "
                f"expected {expected_perturbgen_commit}, got {perturbgen_commit}"
            )

    dependency_hashes, dependency_issues = _resolve_files(
        dependency_files,
        project_root=roots.project_root,
        perturbgen_repo_root=roots.perturbgen_repo_root,
    )
    issues.extend(dependency_issues)

    asset_hashes, asset_issues = _resolve_files(
        asset_paths,
        project_root=roots.project_root,
        perturbgen_repo_root=roots.perturbgen_repo_root,
    )
    issues.extend(asset_issues)

    return ExternalEnvironmentReport(
        python_path=python_path,
        python_version=python_version,
        roots=roots,
        dependency_hashes=dependency_hashes,
        asset_hashes=asset_hashes,
        perturbgen_commit=perturbgen_commit,
        issues=tuple(issues),
        warnings=tuple(warnings),
    )
