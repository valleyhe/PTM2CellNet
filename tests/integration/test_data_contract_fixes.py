"""Integration tests: data-contract fixes (2026-08-16 round 2).

Covers the manifest/dataset gaps registered in project_analysis_20260816:

* scperturb: 30 h5ad snapshots registered with paths + sha256 (was
  ``controlled`` with ``path: null`` although the files were on disk);
* epsd: reverse gap closed — dataset registered with a working loader;
* GSE90546: probe status semantics fixed from ``parsed`` (too strong)
  to ``probed`` (structure inspection only).

These tests read the real manifest and the real on-disk snapshots; they
skip when the snapshots are absent (e.g. fresh clones without data).
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manifest import load_manifest, validate_manifest  # noqa: E402

MANIFEST = PROJECT_ROOT / "data/manifests/datasets.yaml"
SCPERTURB_DIR = PROJECT_ROOT / "data/raw/scperturb"
EPSD_DIR = PROJECT_ROOT / "data/raw/epsd"
GSE90546_REPORT = PROJECT_ROOT / "data/processed/norman_adamson/GSE90546_structure_report.json"


def _dataset(manifest, dataset_id: str) -> dict:
    for entry in manifest["datasets"]:
        if entry["id"] == dataset_id:
            return entry
    raise AssertionError(f"dataset {dataset_id} not in manifest")


class TestScperturbRegistration:
    @pytest.fixture(autouse=True)
    def _require_snapshots(self):
        if not SCPERTURB_DIR.is_dir():
            pytest.skip("scperturb snapshots not on disk")

    def test_manifest_lists_all_snapshots_with_hashes(self):
        manifest = load_manifest(MANIFEST)
        entry = _dataset(manifest, "scperturb")
        assert entry["status"] == "implemented"
        assert entry["loader"] == "scripts/import_scperturb.py"
        # 4 snapshots were truncated downloads and are intentionally not
        # registered (see the manifest comment); the registered set must be
        # exactly the on-disk files minus those.
        truncated = {
            "GasperiniShendure2019_atscale.h5ad",
            "LaraAstiasoHuntly2023_invivo.h5ad",
            "NadigOConner2024_hepg2.h5ad",
            "SunshineHein2023.h5ad",
        }
        on_disk = {p.name for p in SCPERTURB_DIR.glob("*.h5ad")}
        registered = {Path(f["path"]).name for f in entry["files"] if f.get("path")}
        assert registered == on_disk - truncated
        assert all(f.get("sha256") for f in entry["files"])
        assert all(f.get("required") for f in entry["files"])

    def test_validator_passes_scperturb_files(self):
        manifest = load_manifest(MANIFEST)
        report = validate_manifest(manifest, manifest_path=MANIFEST, check_files=True)
        assert report["ok"] is True

    def test_validation_report_leaves_only_replogle_and_scgenescope(self):
        """After the scperturb registration, the cross_scale_training profile
        gap shrinks to the two datasets that genuinely have no snapshots."""
        manifest = load_manifest(MANIFEST)
        report = validate_manifest(
            manifest, manifest_path=MANIFEST, check_files=True, profile="cross_scale_training"
        )
        errors = report.get("errors") or []
        assert len(errors) == 2
        # Errors reference datasets by index; map them back to ids.
        import re

        indexes = [int(re.search(r"datasets\[(\d+)\]", e).group(1)) for e in errors]
        failed_ids = {manifest["datasets"][i]["id"] for i in indexes}
        assert failed_ids == {"replogle", "scgenescope"}
        assert "scperturb" not in failed_ids


class TestEpsdRegistration:
    @pytest.fixture(autouse=True)
    def _require_snapshots(self):
        if not (EPSD_DIR / "Homo sapiens.txt").is_file():
            pytest.skip("epsd snapshots not on disk")

    def test_manifest_entry_and_loader_contract(self):
        manifest = load_manifest(MANIFEST)
        entry = _dataset(manifest, "epsd")
        assert entry["status"] == "implemented"
        assert entry["loader"].endswith("load_from_epsd")
        txt = next(f for f in entry["files"] if f["path"].endswith(".txt"))
        assert Path(txt["path"]).is_file()
        assert txt["sha256"]

    def test_epsd_loader_parses_real_snapshot(self):
        from src.data.loaders import DataLoader

        df = DataLoader().load_from_epsd(str(EPSD_DIR / "Homo sapiens.txt"), ptm_type="phosphorylation")
        assert len(df) > 1000
        assert set(df.columns) == {"protein_accession", "position", "ptm_type", "amino_acid", "source"}
        assert df["source"].iloc[0] == "EPSD"


class TestGSE90546StatusSemantics:
    def test_probe_status_is_probed_not_parsed(self):
        if not GSE90546_REPORT.is_file():
            pytest.skip("GSE90546 report not regenerated")
        report = json.loads(GSE90546_REPORT.read_text(encoding="utf-8"))
        assert report["status"] == "probed"
        assert report["status"] != "parsed"
