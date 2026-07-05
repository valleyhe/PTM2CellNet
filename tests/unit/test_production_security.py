"""Tests for production deployment security guards (TD-M3 / TD-M4).

These tests cover the fail-fast / degrade policies implemented in
``src/api/production_security.py``:

* ``PTM2CELLNET_ENV=production`` with no API key  → fail-fast by default
* ``PTM2CELLNET_ENV=production`` with empty download allowlist → fail-fast
* Opt-out env vars (``PTM2CELLNET_ALLOW_INSECURE_PROD`` etc.) flip the policy
  to "log warning + mark readiness degraded" instead of raising.
* Outside production the audit is a no-op.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from src.api import production_security
from src.api.production_security import (
    audit_production_security,
    enforce_production_security,
)


PROD_SECURITY_ENVS = (
    "PTM2CELLNET_ENV",
    "PTM2CELLNET_API_KEY",
    "PTM2CELLNET_DOWNLOAD_ALLOWLIST",
    "PTM2CELLNET_ALLOW_INSECURE_PROD",
    "PTM2CELLNET_ALLOW_UNAUTHED_PROD",
    "PTM2CELLNET_ALLOW_UNRESTRICTED_DOWNLOADS",
)


@pytest.fixture(autouse=True)
def _isolate_security_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip every security-related env var before each test.

    Tests then set only the vars they care about, so ordering across the
    suite can never leak state. We do NOT ``importlib.reload`` the module
    because that would rebind ``ProductionSecurityError`` to a new class
    object and break ``pytest.raises`` identity checks in tests that grab
    the class at import time.
    """
    for var in PROD_SECURITY_ENVS:
        monkeypatch.delenv(var, raising=False)
    yield


def _prod_security_error_type():
    """Always fetch the live exception class from the module.

    Avoids stale class identity if anything reloads the module elsewhere.
    """
    return production_security.ProductionSecurityError


def test_audit_outside_production_is_noop():
    report = audit_production_security()
    assert report.is_production is False
    assert report.degraded is False
    assert report.reasons == ()
    assert report.fail_fast is False


def test_audit_production_with_key_and_allowlist_is_clean(monkeypatch):
    monkeypatch.setenv("PTM2CELLNET_ENV", "production")
    monkeypatch.setenv("PTM2CELLNET_API_KEY", "real-key-123")
    monkeypatch.setenv("PTM2CELLNET_DOWNLOAD_ALLOWLIST", "rest.uniprot.org,www.ebi.ac.uk")
    report = audit_production_security()
    assert report.is_production is True
    assert report.api_key_configured is True
    assert report.download_allowlist_restricted is True
    assert report.degraded is False
    assert report.reasons == ()


def test_enforce_fail_fast_on_missing_api_key(monkeypatch):
    monkeypatch.setenv("PTM2CELLNET_ENV", "production")
    # No PTM2CELLNET_API_KEY set
    with pytest.raises(_prod_security_error_type()) as excinfo:
        enforce_production_security()
    assert "PTM2CELLNET_API_KEY" in str(excinfo.value)


def test_enforce_fail_fast_on_empty_allowlist(monkeypatch):
    monkeypatch.setenv("PTM2CELLNET_ENV", "production")
    monkeypatch.setenv("PTM2CELLNET_API_KEY", "real-key-123")
    monkeypatch.setenv("PTM2CELLNET_DOWNLOAD_ALLOWLIST", "   ")
    with pytest.raises(_prod_security_error_type()) as excinfo:
        enforce_production_security()
    assert "PTM2CELLNET_DOWNLOAD_ALLOWLIST" in str(excinfo.value)


def test_enforce_fail_fast_reports_both_problems_at_once(monkeypatch):
    """When multiple checks fail, fail-fast lists every reason in one refusal."""
    monkeypatch.setenv("PTM2CELLNET_ENV", "production")
    monkeypatch.setenv("PTM2CELLNET_DOWNLOAD_ALLOWLIST", "")
    with pytest.raises(_prod_security_error_type()) as excinfo:
        enforce_production_security()
    msg = str(excinfo.value)
    assert "PTM2CELLNET_API_KEY" in msg
    assert "PTM2CELLNET_DOWNLOAD_ALLOWLIST" in msg


def test_allow_insecure_prod_flips_to_degraded(monkeypatch):
    monkeypatch.setenv("PTM2CELLNET_ENV", "production")
    monkeypatch.setenv("PTM2CELLNET_ALLOW_INSECURE_PROD", "1")
    # No API key, empty allowlist — both would normally fail-fast.
    monkeypatch.setenv("PTM2CELLNET_DOWNLOAD_ALLOWLIST", "")
    report = enforce_production_security()
    assert report.is_production is True
    assert report.fail_fast is False
    assert report.degraded is True
    assert len(report.reasons) == 2


def test_unauthed_opt_out_clears_only_api_key_reason(monkeypatch):
    monkeypatch.setenv("PTM2CELLNET_ENV", "production")
    monkeypatch.setenv("PTM2CELLNET_ALLOW_UNAUTHED_PROD", "1")
    monkeypatch.setenv("PTM2CELLNET_DOWNLOAD_ALLOWLIST", "rest.uniprot.org")
    report = enforce_production_security()
    assert report.degraded is False
    assert report.reasons == ()


def test_unrestricted_downloads_opt_out(monkeypatch):
    monkeypatch.setenv("PTM2CELLNET_ENV", "production")
    monkeypatch.setenv("PTM2CELLNET_API_KEY", "k")
    monkeypatch.setenv("PTM2CELLNET_ALLOW_UNRESTRICTED_DOWNLOADS", "1")
    monkeypatch.setenv("PTM2CELLNET_DOWNLOAD_ALLOWLIST", "")
    report = enforce_production_security()
    assert report.degraded is False


def test_configure_records_report_for_readiness(monkeypatch):
    monkeypatch.setenv("PTM2CELLNET_ENV", "production")
    monkeypatch.setenv("PTM2CELLNET_API_KEY", "k")
    monkeypatch.setenv("PTM2CELLNET_DOWNLOAD_ALLOWLIST", "rest.uniprot.org")
    production_security.configure_production_security()
    last = production_security.get_last_security_report()
    assert last is not None
    assert last.is_production is True
    assert last.degraded is False
