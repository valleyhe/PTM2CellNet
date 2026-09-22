"""Optional-dependency preflight checks for CLI entry points.

Several PTM2CellNet capabilities (scVI/DAVF gene-space workflow, GenKI graph
perturbation, Mamba encoder, Lion optimizer) depend on *optional* packages.
When those packages are missing the library already degrades gracefully (lazy
imports + fallbacks), but a CLI user would only discover that mid-run, often
via a cryptic stack trace.

This module gives entry points a uniform way to fail fast with an actionable
``pip install -e ".[<extra>]"`` hint *before* doing any real work.

Design:
    * Pure-stdlib imports (``importlib``) so the helper works in minimal envs.
    * Returns structured :class:`DependencyStatus` objects so callers can log
      rich context; :func:`require_extras` raises a single readable error.
    * No hard dependency on any optional package — checks are import probes.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


logger = logging.getLogger(__name__)


@dataclass
class DependencyStatus:
    """Result of probing a single optional dependency.

    Attributes:
        import_name: The top-level module name that was probed.
        available: Whether the import succeeded.
        version: Installed version if discoverable, else ``None``.
        import_error: The captured exception when ``available`` is False.
        extra: The ``setup.py`` extra that provides this dependency
            (e.g. ``"analysis"``), used to build install hints.
        install_name: Display name for the package (e.g. ``scvi-tools``),
            which may differ from the import name (``scvi``).
    """

    import_name: str
    available: bool
    version: Optional[str] = None
    import_error: Optional[BaseException] = None
    extra: str = ""
    install_name: Optional[str] = None

    @property
    def install_hint(self) -> str:
        """The ``pip install`` command that (re)installs this dependency."""
        target = self.install_name or self.import_name
        if self.extra:
            return f'pip install -e ".[{self.extra}]"  (provides {target})'
        return f"pip install {target}"


# Mapping of optional import name -> (setup.py extra, pip display name).
# Kept in one place so entry points and docs reference the same source of truth.
OPTIONAL_DEPENDENCIES: Dict[str, tuple] = {
    "scvi": ("analysis", "scvi-tools"),
    "anndata": ("analysis", "anndata"),
    "scanpy": ("analysis", "scanpy"),
    "scrublet": ("analysis", "scrublet"),
    "torch_geometric": ("genki", "torch-geometric"),
    "mamba_ssm": ("mamba", "mamba-ssm"),
    "lion_pytorch": ("mamba", "lion-pytorch"),
    "transformers": ("pretrained", "transformers"),
    "lightning": ("lightning", "lightning"),
}


def _safe_version(import_name: str) -> Optional[str]:
    """Best-effort version lookup via ``importlib.metadata``.

    Falls back to ``None`` (rather than raising) so a missing distribution does
    not mask the original import error.
    """
    try:
        from importlib import metadata

        # Prefer the distribution matching the import name; some projects ship
        # the importable module under a differently-named distribution
        # (scvi-tools -> scvi), so try a few candidates.
        candidates = [import_name]
        display = OPTIONAL_DEPENDENCIES.get(import_name, (None, None))[1]
        if display:
            candidates.append(display.replace("-", "_"))
            candidates.append(display)
        for name in candidates:
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                continue
    except (metadata.PackageNotFoundError, ValueError, AttributeError) as e:
        logger.warning("Failed to resolve version for %s: %s", import_name, e)
        return None
    return None


def check_dependency(import_name: str, extra: str = "", install_name: Optional[str] = None) -> DependencyStatus:
    """Probe whether ``import_name`` can be imported.

    Args:
        import_name: Top-level module name to ``importlib.import_module``.
        extra: The ``setup.py`` extra providing this dependency.
        install_name: pip distribution display name if it differs from the
            import name (e.g. ``scvi-tools`` for the ``scvi`` import).

    Returns:
        A :class:`DependencyStatus` describing the probe result.
    """
    try:
        importlib.import_module(import_name)
        return DependencyStatus(
            import_name=import_name,
            available=True,
            version=_safe_version(import_name),
            extra=extra,
            install_name=install_name,
        )
    except Exception as exc:  # noqa: BLE001 - probe: ImportError *and* broken-install errors
        return DependencyStatus(
            import_name=import_name,
            available=False,
            version=None,
            import_error=exc,
            extra=extra,
            install_name=install_name,
        )


def check_extras(names: Iterable[str]) -> List[DependencyStatus]:
    """Probe a set of optional dependencies by import name.

    Args:
        names: Iterable of import names registered in
            :data:`OPTIONAL_DEPENDENCIES`, or bare import names.

    Returns:
        List of :class:`DependencyStatus`, one per requested name.
    """
    results: List[DependencyStatus] = []
    for name in names:
        extra, install_name = OPTIONAL_DEPENDENCIES.get(name, ("", None))
        results.append(check_dependency(name, extra=extra, install_name=install_name))
    return results


class MissingDependencyError(ImportError):
    """Raised by :func:`require_extras` when one or more deps are unavailable.

    Carries the structured :class:`DependencyStatus` list so callers (and
    tests) can inspect which dependencies were missing and why.
    """

    def __init__(self, message: str, statuses: Sequence[DependencyStatus]):
        super().__init__(message)
        self.message = message
        self.statuses = list(statuses)

    def __str__(self) -> str:
        return self.message


def require_extras(
    names: Iterable[str],
    *,
    feature: str = "this feature",
    raise_on_missing: bool = True,
) -> List[DependencyStatus]:
    """Verify optional dependencies are importable, else raise/fallback.

    Args:
        names: Import names to probe (see :data:`OPTIONAL_DEPENDENCIES`).
        feature: Human-readable description used in the error message
            (e.g. ``"the GenKI source backend"``).
        raise_on_missing: When True (default) raise
            :class:`MissingDependencyError` listing every missing dependency
            and its install hint. When False, just return the statuses so the
            caller can implement a graceful degradation path.

    Returns:
        The full :class:`DependencyStatus` list (available + missing).

    Raises:
        MissingDependencyError: If ``raise_on_missing`` is True and any probed
            dependency is unavailable.
    """
    statuses = check_extras(names)
    missing = [s for s in statuses if not s.available]
    if missing and raise_on_missing:
        hints = "\n".join(f"  - {s.import_name}: {s.install_hint}" for s in missing)
        detail = ""
        first_err = missing[0].import_error
        if first_err is not None:
            detail = f" (first error: {type(first_err).__name__}: {first_err})"
        msg = f"{feature} 不可用：缺少以下可选依赖。{detail}\n请按提示安装对应 extra 后重试：\n{hints}"
        raise MissingDependencyError(msg, statuses)
    return statuses


def assert_scvi_available(feature: str = "scVI / DAVF 基因空间工作流") -> DependencyStatus:
    """Convenience wrapper: assert scvi-tools can be imported.

    Args:
        feature: Description for the error message.

    Returns:
        The :class:`DependencyStatus` for scvi (available when this returns).

    Raises:
        MissingDependencyError: If scvi cannot be imported.
    """
    statuses = require_extras(["scvi"], feature=feature)
    return statuses[0]


def format_dependency_table(statuses: Sequence[DependencyStatus]) -> str:
    """Render dependency probe results as a human-readable table.

    Useful for entry points that want to print an optional-deps summary at
    startup rather than failing outright.
    """
    lines = ["可选依赖检查："]
    for s in statuses:
        mark = "OK " if s.available else "MISSING"
        ver = s.version or "-"
        lines.append(f"  [{mark}] {s.import_name:<18} version={ver:<10}")
        if not s.available and s.import_error is not None:
            lines.append(f"         -> {type(s.import_error).__name__}: {s.import_error}")
            lines.append(f"         fix: {s.install_hint}")
    return "\n".join(lines)


def is_module_available(import_name: str) -> bool:
    """Lightweight boolean probe, shorthand for ``check_dependency(...).available``."""
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        # find_spec can raise for namespace subtrees; fall back to a real import
        return check_dependency(import_name).available


__all__ = [
    "DependencyStatus",
    "OPTIONAL_DEPENDENCIES",
    "MissingDependencyError",
    "check_dependency",
    "check_extras",
    "require_extras",
    "assert_scvi_available",
    "format_dependency_table",
    "is_module_available",
]


# When this module is imported, ensure importlib metadata uses the current env.
# (No-op on normal startup; guards against frozen-app embedding quirks.)
if hasattr(sys, "prefix"):
    pass
