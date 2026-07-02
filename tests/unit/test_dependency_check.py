"""Unit tests for ``src.utils.dependency_check``.

These cover the audit P1-1 / P2-2 preflight helper that gives CLI entry points
a uniform way to fail fast with an actionable ``pip install -e ".[<extra>]"``
hint when an optional dependency is missing.
"""

from __future__ import annotations

import builtins
import logging

import pytest

from src.utils import dependency_check as dc
from src.utils.dependency_check import (
    DependencyStatus,
    MissingDependencyError,
    assert_scvi_available,
    check_dependency,
    check_extras,
    format_dependency_table,
    is_module_available,
    require_extras,
)


# ---------------------------------------------------------------------------
# DependencyStatus
# ---------------------------------------------------------------------------

class TestDependencyStatus:
    def test_available_status_has_no_error(self):
        s = DependencyStatus(import_name="foo", available=True, version="1.2.3")
        assert s.available is True
        assert s.import_error is None
        assert s.version == "1.2.3"

    def test_install_hint_with_extra(self):
        s = DependencyStatus(
            import_name="scvi",
            available=False,
            extra="analysis",
            install_name="scvi-tools",
        )
        hint = s.install_hint
        assert "scvi-tools" in hint
        assert "[analysis]" in hint

    def test_install_hint_without_extra(self):
        s = DependencyStatus(import_name="foo", available=False)
        assert "pip install foo" in s.install_hint


# ---------------------------------------------------------------------------
# check_dependency / check_extras / is_module_available
# ---------------------------------------------------------------------------

class TestCheckDependency:
    def test_real_available_module(self):
        # json is stdlib, always present.
        s = check_dependency("json")
        assert s.available is True
        assert s.import_error is None

    def test_missing_module(self):
        s = check_dependency("definitely_not_a_real_module_xyz123")
        assert s.available is False
        assert s.import_error is not None

    def test_is_module_available_boolean(self):
        assert is_module_available("json") is True
        assert is_module_available("definitely_not_a_real_module_xyz123") is False

    def test_check_extras_returns_known_metadata(self):
        # scvi is registered in OPTIONAL_DEPENDENCIES with extra="analysis".
        statuses = check_extras(["scvi"])
        assert len(statuses) == 1
        assert statuses[0].import_name == "scvi"
        assert statuses[0].extra == "analysis"
        assert statuses[0].install_name == "scvi-tools"


# ---------------------------------------------------------------------------
# require_extras / MissingDependencyError
# ---------------------------------------------------------------------------

class TestRequireExtras:
    def test_raises_when_missing(self):
        with pytest.raises(MissingDependencyError) as excinfo:
            require_extras(
                ["definitely_not_a_real_module_xyz123"],
                feature="the test backend",
            )
        msg = str(excinfo.value)
        assert "test backend" in msg
        assert "pip install" in msg  # actionable hint present
        # The structured status list is attached for callers/tests to inspect.
        assert len(excinfo.value.statuses) == 1
        assert excinfo.value.statuses[0].available is False

    def test_returns_statuses_when_not_raising(self):
        statuses = require_extras(
            ["json", "definitely_not_a_real_module_xyz123"],
            raise_on_missing=False,
        )
        assert len(statuses) == 2
        assert statuses[0].available is True
        assert statuses[1].available is False

    def test_passes_when_all_available(self):
        statuses = require_extras(["json"])
        assert statuses[0].available is True


# ---------------------------------------------------------------------------
# assert_scvi_available
# ---------------------------------------------------------------------------

class TestAssertScviAvailable:
    def test_returns_status_when_scvi_present(self, monkeypatch):
        # Force scvi to look available regardless of the real environment.
        monkeypatch.setattr(
            dc, "check_extras",
            lambda names: [DependencyStatus(import_name="scvi", available=True, version="9.9.9")],
        )
        status = assert_scvi_available()
        assert status.available is True

    def test_raises_when_scvi_missing(self, monkeypatch):
        monkeypatch.setattr(
            dc, "check_extras",
            lambda names: [DependencyStatus(
                import_name="scvi",
                available=False,
                import_error=ModuleNotFoundError("No module named 'scvi'"),
                extra="analysis",
                install_name="scvi-tools",
            )],
        )
        with pytest.raises(MissingDependencyError, match="scvi"):
            assert_scvi_available()


# ---------------------------------------------------------------------------
# format_dependency_table
# ---------------------------------------------------------------------------

class TestFormatDependencyTable:
    def test_table_marks_missing_with_hint(self, caplog):
        statuses = [
            DependencyStatus(import_name="json", available=True, version="x"),
            DependencyStatus(
                import_name="scvi",
                available=False,
                import_error=ModuleNotFoundError("No module named 'scvi'"),
                extra="analysis",
                install_name="scvi-tools",
            ),
        ]
        with caplog.at_level(logging.INFO):
            text = format_dependency_table(statuses)
        assert "[OK ]" in text
        assert "[MISSING]" in text
        assert "scvi-tools" in text
        assert "ModuleNotFoundError" in text


# ---------------------------------------------------------------------------
# GenKI / Mamba preflight (P1-2)
# ---------------------------------------------------------------------------

class TestOptionalBackendPreflight:
    """The genki_source / Mamba backends must fail fast with install hints."""

    def test_genki_preflight_raises_with_genki_extra_hint(self, monkeypatch):
        """A missing genki-extra dependency surfaces a ``pip install -e .[genki]`` hint.

        The probe uses a fictitious module name that is guaranteed to be absent
        from every environment, then registers it against the ``genki`` extra so
        the install hint is deterministic. This keeps the test environment-
        agnostic: it does not depend on whether ``torch_geometric`` is actually
        installed (the extra may legitimately be present in CI).
        """
        fake_missing = "definitely_not_a_real_genki_module_xyz123"
        monkeypatch.setitem(
            dc.OPTIONAL_DEPENDENCIES,
            fake_missing,
            ("genki", "torch-geometric"),
        )
        with pytest.raises(MissingDependencyError) as exc_info:
            require_extras([fake_missing], feature="GenKI source backend")
        msg = str(exc_info.value)
        assert "GenKI source backend" in msg
        assert "pip install -e \".[genki]\"" in msg

    def test_mamba_preflight_hint_references_mamba_extra(self):
        """The OPTIONAL_DEPENDENCIES table maps mamba_ssm to the mamba extra."""
        assert dc.OPTIONAL_DEPENDENCIES["mamba_ssm"][0] == "mamba"
        assert dc.OPTIONAL_DEPENDENCIES["lion_pytorch"][0] == "mamba"

    def test_genki_extra_registered_in_optional_dependencies(self):
        """torch_geometric / scanpy map to the genki extra (install hint source)."""
        assert dc.OPTIONAL_DEPENDENCIES["torch_geometric"][0] == "genki"
        assert dc.OPTIONAL_DEPENDENCIES["scanpy"][0] == "genki"
