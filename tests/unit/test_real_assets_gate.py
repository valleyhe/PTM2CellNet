"""Unit test for the real-assets gate logic itself.

The actual real-assets tests skip by default; this test verifies the gate
helper behaves correctly so a future refactor of the opt-in flag does not
silently enable heavy/networked tests in default CI.
"""

from __future__ import annotations

import importlib

import pytest


def test_gate_disabled_by_default(monkeypatch):
    monkeypatch.delenv("PTM2CELLNET_RUN_REAL_ASSET_TESTS", raising=False)
    from tests import real_assets as ra

    importlib.reload(ra)
    assert ra.real_assets_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on", "On"])
def test_gate_accepts_truthy_values(monkeypatch, val):
    monkeypatch.setenv("PTM2CELLNET_RUN_REAL_ASSET_TESTS", val)
    from tests import real_assets as ra

    importlib.reload(ra)
    assert ra.real_assets_enabled() is True


@pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "garbage"])
def test_gate_rejects_non_truthy_values(monkeypatch, val):
    monkeypatch.setenv("PTM2CELLNET_RUN_REAL_ASSET_TESTS", val)
    from tests import real_assets as ra

    importlib.reload(ra)
    assert ra.real_assets_enabled() is False


def test_record_evidence_writes_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tests import real_assets as ra

    importlib.reload(ra)
    ra.record_evidence(
        "test_unit_record",
        asset="asset-x",
        outcome="pass",
        duration_s=1.23,
        extra={"k": "v"},
    )
    target = tmp_path / "outputs" / "real_assets" / "test_unit_record.json"
    assert target.is_file()
    import json

    payload = json.loads(target.read_text())
    assert payload["test"] == "test_unit_record"
    assert payload["asset"] == "asset-x"
    assert payload["outcome"] == "pass"
    assert payload["extra"] == {"k": "v"}
