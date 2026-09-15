"""Unit tests for scripts/import_scperturb.py (synthetic h5ad fixtures)."""

import json
from pathlib import Path

import numpy as np
import pytest

anndata = pytest.importorskip("anndata")

import scripts.import_scperturb as scp  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_study(path: Path, *, n_cells: int = 4, n_genes: int = 5, perturbation_values=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    obs = {
        "perturbation": perturbation_values or ["NT", "GENE1", "NT", "GENE2"],
        "cell_line": ["K562"] * n_cells,
    }
    var = {"gene_id": [f"ENSG{i:011d}" for i in range(1, n_genes + 1)]}
    ad = anndata.AnnData(
        X=np.zeros((n_cells, n_genes), dtype=np.float32),
        obs=obs,
        var=var,
    )
    ad.var.index = [f"GENE{i}" for i in range(1, n_genes + 1)]
    ad.write_h5ad(path)
    return path


@pytest.fixture()
def raw_root(tmp_path: Path) -> Path:
    root = tmp_path / "scperturb"
    _make_study(root / "StudyA.h5ad", n_cells=4, n_genes=5, perturbation_values=["NT", "GENE1", "NT", "GENE2"])
    _make_study(
        root / "StudyB.h5ad", n_cells=6, n_genes=8, perturbation_values=["ctrl", "T1", "T2", "ctrl", "T1", "T3"]
    )
    return root


# ---------------------------------------------------------------------------
# probe_study
# ---------------------------------------------------------------------------


class TestProbeStudy:
    def test_probe_metadata(self, raw_root: Path):
        probe = scp.probe_study(raw_root / "StudyA.h5ad")
        assert probe["n_cells"] == 4
        assert probe["n_genes"] == 5
        assert "perturbation" in probe["obs_columns"]
        assert "cell_line" in probe["obs_columns"]
        assert probe["perturbation_columns"] == ["perturbation"]
        assert probe["n_unique_perturbations"] == 3  # NT, GENE1, GENE2
        assert probe["var_index_head"] == ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
        assert len(probe["sha256"]) == 64

    def test_corrupt_file_raises(self, tmp_path: Path):
        p = tmp_path / "broken.h5ad"
        p.write_text("not an h5ad", encoding="utf-8")
        with pytest.raises((OSError, ValueError)):
            scp.probe_study(p)


# ---------------------------------------------------------------------------
# build_registration_report
# ---------------------------------------------------------------------------


class TestBuildRegistrationReport:
    def test_aggregates_studies(self, raw_root: Path):
        studies = list(raw_root.glob("*.h5ad"))
        report = scp.build_registration_report(raw_root, studies)
        assert report["schema_version"] == "ptm2cellnet.scperturb.registration.v1"
        assert report["summary"]["n_studies_registered"] == 2
        assert report["summary"]["n_studies_failed"] == 0
        assert report["summary"]["total_cells"] == 10
        assert report["summary"]["total_genes"] == 13
        assert report["summary"]["perturbation_metadata_present"] == 2
        assert len(report["studies"]) == 2

    def test_fail_soft_on_corrupt_file(self, raw_root: Path):
        bad = raw_root / "Broken.h5ad"
        bad.write_text("garbage", encoding="utf-8")
        studies = list(raw_root.glob("*.h5ad"))
        report = scp.build_registration_report(raw_root, studies)
        assert report["summary"]["n_studies_registered"] == 2
        assert report["summary"]["n_studies_failed"] == 1
        assert report["errors"][0]["file"] == "Broken.h5ad"

    def test_empty_studies(self, tmp_path: Path):
        report = scp.build_registration_report(tmp_path, [])
        assert report["summary"]["n_studies_registered"] == 0
        assert report["summary"]["total_cells"] == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestMain:
    def test_cli_end_to_end(self, raw_root: Path, tmp_path: Path, capsys):
        out = tmp_path / "processed"
        rc = scp.main(["--raw-root", str(raw_root), "--output", str(out)])
        assert rc == 0
        captured = json.loads(capsys.readouterr().out)
        assert captured["ok"] is True
        assert captured["n_studies_registered"] == 2
        assert captured["total_cells"] == 10
        report = json.loads((out / scp.STUDY_MANIFEST_NAME).read_text(encoding="utf-8"))
        assert report["summary"]["n_studies_registered"] == 2

    def test_cli_limit(self, raw_root: Path, tmp_path: Path, capsys):
        rc = scp.main(["--raw-root", str(raw_root), "--output", str(tmp_path / "o"), "--limit", "1"])
        assert rc == 0
        captured = json.loads(capsys.readouterr().out)
        assert captured["n_studies_registered"] == 1

    def test_cli_missing_root(self, tmp_path: Path, capsys):
        rc = scp.main(["--raw-root", str(tmp_path / "nope"), "--output", str(tmp_path / "o")])
        assert rc == 1
        assert "不存在" in capsys.readouterr().err

    def test_cli_empty_root(self, tmp_path: Path, capsys):
        rc = scp.main(["--raw-root", str(tmp_path), "--output", str(tmp_path / "o")])
        assert rc == 1
        assert "无 h5ad" in capsys.readouterr().err
