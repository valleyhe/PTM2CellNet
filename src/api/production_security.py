"""Production deployment security guards (TD-M3 / TD-M4).

These helpers close two operational foot-guns that the optional auth/download
configuration introduced:

* **TD-M3** — ``_OptionalAuthMiddleware`` skips auth entirely when
  ``PTM2CELLNET_API_KEY`` is unset. That is intentional for local/dev
  back-compat, but in production a forgotten env var silently exposes the
  prediction surface. We add a startup check that fails fast (or at least
  marks readiness degraded) when ``PTM2CELLNET_ENV=production`` and no key
  is configured.

* **TD-M4** — ``PTM2CELLNET_DOWNLOAD_ALLOWLIST`` accepts an empty string to
  mean "allow any host" (dev back-compat). In production that defeats the
  whole supply-chain boundary. We refuse to start in production with an
  empty allowlist.

The policy is env-driven so it stays explicit per deployment:

* ``PTM2CELLNET_ENV=production``            — enable production mode
* ``PTM2CELLNET_ALLOW_INSECURE_PROD=1``     — opt out of fail-fast (still logs)
* ``PTM2CELLNET_ALLOW_UNAUTHED_PROD=1``     — opt out of the API-key fail-fast
* ``PTM2CELLNET_ALLOW_UNRESTRICTED_DOWNLOADS=1`` — opt out of the allowlist fail-fast

Fail-fast mode raises ``ProductionSecurityError`` at app construction time so
the bad config never reaches a request. Non-fail-fast mode still logs a
prominent ``SECURITY`` warning so the issue is visible in logs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class ProductionSecurityError(RuntimeError):
    """Raised when a production deployment ships with an insecure configuration.

    Surfacing as a dedicated exception (rather than ``RuntimeError``) lets
    operators and tests distinguish a deliberate startup refusal from other
    init errors.
    """


@dataclass(frozen=True)
class ProductionSecurityReport:
    """Result of :func:`audit_production_security`.

    ``degraded`` is true when the deployment is allowed to start (no
    fail-fast) but one or more security checks did not pass — the readiness
    probe should reflect this so orchestrators can drain traffic.
    """

    is_production: bool
    api_key_configured: bool
    download_allowlist_restricted: bool
    fail_fast: bool
    degraded: bool
    reasons: tuple[str, ...]


def _is_production() -> bool:
    """Return True when running with ``PTM2CELLNET_ENV=production``."""
    return os.environ.get("PTM2CELLNET_ENV", "").strip().lower() == "production"


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def audit_production_security() -> ProductionSecurityReport:
    """Inspect production-relevant security configuration.

    Pure function: reads env vars only, performs no I/O. Safe to call from
    tests and from app construction.
    """
    is_production = _is_production()
    api_key = os.environ.get("PTM2CELLNET_API_KEY", "").strip()
    api_key_configured = bool(api_key)

    # An "unrestricted" download policy is one where the operator explicitly
    # set the allowlist env var to an empty/whitespace-only string. The
    # default (env var unset) is the curated production allowlist and is
    # therefore considered restricted.
    allowlist_env = os.environ.get("PTM2CELLNET_DOWNLOAD_ALLOWLIST", None)
    download_allowlist_restricted = not (allowlist_env is not None and allowlist_env.strip() == "")

    if not is_production:
        return ProductionSecurityReport(
            is_production=False,
            api_key_configured=api_key_configured,
            download_allowlist_restricted=download_allowlist_restricted,
            fail_fast=False,
            degraded=False,
            reasons=(),
        )

    fail_fast = not _truthy("PTM2CELLNET_ALLOW_INSECURE_PROD")
    reasons: list[str] = []

    if not api_key_configured and not _truthy("PTM2CELLNET_ALLOW_UNAUTHED_PROD"):
        reasons.append(
            "PTM2CELLNET_ENV=production but PTM2CELLNET_API_KEY is unset — "
            "prediction endpoints would be unauthenticated. Set the key, or "
            "explicitly opt out with PTM2CELLNET_ALLOW_UNAUTHED_PROD=1 "
            "(not recommended)."
        )
    if not download_allowlist_restricted and not _truthy("PTM2CELLNET_ALLOW_UNRESTRICTED_DOWNLOADS"):
        reasons.append(
            "PTM2CELLNET_ENV=production but PTM2CELLNET_DOWNLOAD_ALLOWLIST is "
            "empty — any host could be downloaded from. Set a comma-separated "
            "allowlist, leave the env var unset to use the default production "
            "allowlist, or explicitly opt out with "
            "PTM2CELLNET_ALLOW_UNRESTRICTED_DOWNLOADS=1 (not recommended)."
        )

    degraded = bool(reasons) and not fail_fast
    return ProductionSecurityReport(
        is_production=True,
        api_key_configured=api_key_configured,
        download_allowlist_restricted=download_allowlist_restricted,
        fail_fast=fail_fast,
        degraded=degraded,
        reasons=tuple(reasons),
    )


def enforce_production_security() -> ProductionSecurityReport:
    """Audit configuration and apply the configured policy.

    In fail-fast mode (default for production) this raises
    :class:`ProductionSecurityError` listing every offending check. In
    opt-out mode it logs a prominent warning per reason and returns the
    report so callers (e.g. readiness) can mark themselves degraded.

    No-op outside production.
    """
    report = audit_production_security()
    if not report.is_production or not report.reasons:
        return report

    for reason in report.reasons:
        logger.warning("⚠️  SECURITY (production): %s", reason)

    if report.fail_fast:
        # Join all reasons so operators see every problem in one refusal
        # rather than fixing them one restart at a time.
        joined = " || ".join(report.reasons)
        raise ProductionSecurityError(
            f"Refusing to start in production with insecure configuration: {joined} "
            "(set PTM2CELLNET_ALLOW_INSECURE_PROD=1 to start anyway with warnings)"
        )
    return report


# Readiness hook: tests / probes can read this to decide whether the
# instance should receive traffic. Updated by enforce_production_security().
_LAST_REPORT: Optional[ProductionSecurityReport] = None


def get_last_security_report() -> Optional[ProductionSecurityReport]:
    """Return the most recent :class:`ProductionSecurityReport`, if any."""
    return _LAST_REPORT


def _record_report(report: ProductionSecurityReport) -> ProductionSecurityReport:
    global _LAST_REPORT
    _LAST_REPORT = report
    return report


def configure_production_security() -> ProductionSecurityReport:
    """Top-level entry called from ``create_app``.

    Equivalent to :func:`enforce_production_security` but also stashes the
    report so readiness can consult :func:`get_last_security_report`.
    """
    return _record_report(enforce_production_security())


__all__ = [
    "ProductionSecurityError",
    "ProductionSecurityReport",
    "audit_production_security",
    "configure_production_security",
    "enforce_production_security",
    "get_last_security_report",
]
