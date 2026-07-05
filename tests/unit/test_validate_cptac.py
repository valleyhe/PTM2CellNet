"""Unit tests for the refactored validate_cptac script (F-03 / TD-H3).

Verifies that:

* the experimental guard refuses to run without the env var
* mock mode produces a result file explicitly marked not scientifically valid
* the PDC-backed downloader delegates to :class:`PDCClient` for the manifest
* the validator's compare_with_known_sites parses site IDs correctly
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "validate_cptac.py"


def _import_script():
    """Import scripts/validate_cptac.py as a module (it's not a package)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("validate_cptac", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_guard_raises_without_env(monkeypatch):
    monkeypatch.delenv("PTM2CELLNET_ALLOW_EXPERIMENTAL", raising=False)
    # Force re-import so the module-level guard runs again.
    sys.modules.pop("validate_cptac", None)
    with pytest.raises(RuntimeError, match="EXPERIMENTAL"):
        _import_script()


def test_guard_passes_with_env(monkeypatch):
    monkeypatch.setenv("PTM2CELLNET_ALLOW_EXPERIMENTAL", "1")
    sys.modules.pop("validate_cptac", None)
    module = _import_script()
    assert hasattr(module, "CPTACDataDownloader")
    assert hasattr(module, "CPTACValidator")


def test_mock_backend_produces_synthetic_matrix(monkeypatch, tmp_path):
    monkeypatch.setenv("PTM2CELLNET_ALLOW_EXPERIMENTAL", "1")
    sys.modules.pop("validate_cptac", None)
    module = _import_script()
    downloader = module.CPTACDataDownloader(str(tmp_path), backend="mock")
    df = downloader.download_phosphoproteomics("BRCA")
    assert df.shape[0] > 0
    assert df.shape[1] == 100
    # Cache file written.
    assert (tmp_path / "BRCA_phosphoproteomics.csv").is_file()


def test_pdc_backend_delegates_to_pdc_client(monkeypatch, tmp_path):
    """In PDC backend, the manifest call must route through PDCClient.get_study."""
    monkeypatch.setenv("PTM2CELLNET_ALLOW_EXPERIMENTAL", "1")
    sys.modules.pop("validate_cptac", None)
    module = _import_script()

    # Fake the PDCClient.get_study to return a known PDCStudy.
    from src.analysis.pdc_client import PDCStudy

    captured: dict = {}

    class _FakeClient:
        def __init__(self, *a, **kw):
            captured["init_args"] = (a, kw)

        def get_study(self, study_id):
            captured["study_id"] = study_id
            return PDCStudy(
                study_id=study_id,
                study_submitter_id="CPTAC-BRCA",
                study_name="CPTAC Breast Cancer",
                disease_type="Breast",
                primary_site="Breast",
                files=[{"file_name": "phospho.tsv", "data_category": "Phosphoproteomics"}],
            )

    monkeypatch.setattr(module, "PDCClient", _FakeClient) if hasattr(module, "PDCClient") else None
    # PDCClient is imported lazily inside the method; patch at the source.
    import src.analysis.pdc_client as pdc_mod

    monkeypatch.setattr(pdc_mod, "PDCClient", _FakeClient)

    downloader = module.CPTACDataDownloader(str(tmp_path), backend="pdc")
    manifest = downloader.download_study_manifest("BRCA")
    assert captured["study_id"] == "7c0c6e28-d405-11e8-b853-a005056ab009"
    assert manifest["study_submitter_id"] == "CPTAC-BRCA"
    assert manifest["files"][0]["file_name"] == "phospho.tsv"


def test_compare_with_known_sites_parses_site_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("PTM2CELLNET_ALLOW_EXPERIMENTAL", "1")
    sys.modules.pop("validate_cptac", None)
    module = _import_script()

    phospho = pd.DataFrame(
        {"Sample_0": [1.0, 2.0, 3.0]},
        index=["P12345_100S", "P12345_200T", "P67890_300Y"],
    )
    phospho.index.name = "Site"
    validator = module.CPTACValidator.__new__(module.CPTACValidator)
    validator.phospho_data = phospho
    result = validator.compare_with_known_sites()
    assert result["total_phospho_sites"] == 3
    assert result["unique_proteins"] == 2
    assert set(result["aa_distribution"].keys()) == {"S", "T", "Y"}


def test_main_mock_mode_marks_results_not_valid(monkeypatch, tmp_path):
    """End-to-end: mock mode writes a JSON with scientifically_valid=false."""
    monkeypatch.setenv("PTM2CELLNET_ALLOW_EXPERIMENTAL", "1")
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))
    sys.modules.pop("validate_cptac", None)
    module = _import_script()

    out_dir = tmp_path / "cptac_out"
    out_dir.mkdir()

    # Argv with --mock + --download-only to avoid the model load path
    # (which would require a real checkpoint).
    monkeypatch.setattr(
        sys, "argv",
        ["validate_cptac.py", "--mock", "-o", str(out_dir), "--download-only"],
    )
    module.main()

    # download-only doesn't write validation_results.json; verify data files.
    assert (out_dir / "BRCA_phosphoproteomics.csv").is_file()
    assert (out_dir / "BRCA_mutations.csv").is_file()


def test_main_mock_full_run_marks_results_not_valid(monkeypatch, tmp_path):
    """Full mock run (not download-only) writes scientifically_valid=false."""
    monkeypatch.setenv("PTM2CELLNET_ALLOW_EXPERIMENTAL", "1")
    sys.modules.pop("validate_cptac", None)
    module = _import_script()

    out_dir = tmp_path / "cptac_full"
    out_dir.mkdir()

    monkeypatch.setattr(
        sys, "argv",
        ["validate_cptac.py", "--mock", "-o", str(out_dir),
         "-m", str(tmp_path / "no_models")],
    )
    module.main()

    result_path = out_dir / "validation_results.json"
    assert result_path.is_file()
    payload = json.loads(result_path.read_text())
    assert payload["scientifically_valid"] is False
    assert payload["backend"] == "mock"
