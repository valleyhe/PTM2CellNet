"""Unit tests for scripts/parse_scperturb.py (F-05, 2026-08-17).

Covers the end-to-end h5ad → NPZ/TSV parser using in-memory AnnData
objects so the tests are deterministic and independent of the real
~21 GB scPerturb download. Tests assert:

* control-cell identification across the perturbation + perturbation_type
  columns (case-insensitive, NaN-safe);
* per-perturbation Δ = mean(target) - mean(control) is numerically correct
  on a small synthetic matrix;
* per-study artifact files are written with the expected schema and
  shape aligned with GSE133344/GSE90546 bundles;
* fail-soft on bad h5ad files (corrupt / missing obs / no perturbations);
* CLI accepts both "Foo" and "Foo.h5ad" for ``--study``.
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scripts import parse_scperturb as ps


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_anndata(
    perturb: list[str],
    perturb_type: list[str] | None = None,
    n_genes: int = 20,
    seed: int = 0,
) -> ad.AnnData:
    """Build a tiny in-memory AnnData with a controllable perturbation matrix."""
    rng = np.random.default_rng(seed)
    n = len(perturb)
    if perturb_type is None:
        perturb_type = [""] * n
    obs = pd.DataFrame({
        "perturbation": perturb,
        "perturbation_type": perturb_type,
    })
    # Distinct mean for control (low) vs each target (offset +3) so Δ is non-trivial.
    X = rng.integers(0, 5, size=(n, n_genes)).astype(np.float32)
    for i, p in enumerate(perturb):
        if not ps._is_control_cell(p, perturb_type[i]):
            X[i] += 3.0
    var = pd.DataFrame(index=[f"G{i}" for i in range(n_genes)])
    return ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)


@pytest.fixture
def tiny_study_path(tmp_path: Path) -> Path:
    """Write a 60-cell synthetic h5ad (2 controls, 3 perturbations × ~20 cells)."""
    perturb: list[str] = []
    perturb_type: list[str] = []
    # 2 controls
    perturb += ["control"] * 2
    perturb_type += ["unedited"] * 2
    # 3 perturbations
    for pid in ("TP53", "MYC", "EGFR"):
        perturb += [pid] * 20
        perturb_type += ["CRISPRi"] * 20
    adata = _make_anndata(perturb, perturb_type, n_genes=10, seed=42)
    p = tmp_path / "tiny.h5ad"
    adata.write_h5ad(p)
    return p


# ---------------------------------------------------------------------------
# Control-cell identification
# ---------------------------------------------------------------------------


def test_is_control_cell_basic() -> None:
    assert ps._is_control_cell("control", "")
    assert ps._is_control_cell("CONTROL", "")
    assert ps._is_control_cell("", "")
    assert ps._is_control_cell("TP53", "control") is True
    assert ps._is_control_cell("non-targeting", "")
    assert ps._is_control_cell("TP53", "CRISPRi") is False


def test_is_control_cell_handles_missing_and_none() -> None:
    assert ps._is_control_cell(None, None) is True
    # 设计决策：缺失扰动列值视为控制池（fail-safe to the control baseline）
    assert ps._is_control_cell(None, None) is True
    assert ps._is_control_cell(None, "CRISPRi") is True  # 缺失扰动即按控制处理


# ---------------------------------------------------------------------------
# parse_study
# ---------------------------------------------------------------------------


def test_parse_study_writes_artifacts(tiny_study_path: Path, tmp_path: Path) -> None:
    summary = ps.parse_study(tiny_study_path, tmp_path)
    assert summary["status"] == "parsed"
    assert summary["n_cells"] == 62  # 2 + 3*20
    assert summary["n_genes"] == 10
    assert summary["n_perturbations"] == 3
    assert summary["n_control_cells"] == 2

    expr_path = Path(summary["expression_npz"])
    delta_path = Path(summary["delta_expression_npz"])
    pert_path = Path(summary["perturbations_tsv"])
    assert expr_path.is_file()
    assert delta_path.is_file()
    assert pert_path.is_file()

    expr = sp.load_npz(expr_path)
    assert expr.shape == (62, 10)

    delta = np.load(delta_path, allow_pickle=True)
    assert delta["delta"].shape == (3, 10)  # 3 perturbations × 10 genes
    assert list(delta["perturbation_ids"]) == ["TP53", "MYC", "EGFR"]
    assert list(delta["n_cells"]) == [20, 20, 20]
    assert delta["control_mean"].shape == (10,)

    tsv = pd.read_csv(pert_path, sep="\t")
    assert list(tsv.columns) == ["perturbation_id", "n_cells"]
    assert tsv["n_cells"].sum() == 60


def test_parse_study_delta_is_target_minus_control(tiny_study_path: Path, tmp_path: Path) -> None:
    """Δ should equal mean(target cells) - mean(control cells) per gene."""
    summary = ps.parse_study(tiny_study_path, tmp_path)
    delta = np.load(summary["delta_expression_npz"], allow_pickle=True)

    expr = sp.load_npz(summary["expression_npz"]).toarray()
    obs = ad.read_h5ad(tiny_study_path).obs
    ctrl_mask = obs["perturbation"].map(ps._is_control_cell).to_numpy() if False else np.array(
        [ps._is_control_cell(p, t) for p, t in zip(obs["perturbation"], obs["perturbation_type"], strict=True)]
    )
    control_mean = expr[ctrl_mask].mean(axis=0)
    for row, pid in enumerate(delta["perturbation_ids"]):
        target_mask = (obs["perturbation"] == pid).to_numpy()
        target_mean = expr[target_mask].mean(axis=0)
        expected = target_mean - control_mean
        np.testing.assert_allclose(delta["delta"][row], expected, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Fail-soft behaviour
# ---------------------------------------------------------------------------


def test_parse_study_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ps.ScPerturbParseError, match="missing"):
        ps.parse_study(tmp_path / "nope.h5ad", tmp_path)


def test_parse_study_no_perturbation_column(tmp_path: Path) -> None:
    adata = _make_anndata(["x", "y"]).copy()
    adata.obs = pd.DataFrame({"other": ["x", "y"]})
    p = tmp_path / "bad.h5ad"
    adata.write_h5ad(p)
    with pytest.raises(ps.ScPerturbParseError, match="perturbation"):
        ps.parse_study(p, tmp_path)


def test_parse_study_all_control(tmp_path: Path) -> None:
    adata = _make_anndata(["control"] * 5)
    p = tmp_path / "control_only.h5ad"
    adata.write_h5ad(p)
    with pytest.raises(ps.ScPerturbParseError, match="no target"):
        ps.parse_study(p, tmp_path)


# ---------------------------------------------------------------------------
# parse_studies (aggregate) — fail-soft
# ---------------------------------------------------------------------------


def test_parse_studies_fail_soft(tmp_path: Path) -> None:
    """One good + one missing file → good parsed, missing reported, no raise."""
    good_perturb = ["control"] * 2 + ["TP53"] * 3
    good_perturb_type = ["unedited"] * 2 + ["CRISPRi"] * 3
    adata = _make_anndata(good_perturb, good_perturb_type, n_genes=4)
    (tmp_path / "good.h5ad").write_bytes(b"")
    adata.write_h5ad(tmp_path / "good.h5ad")
    # Create a fake corrupt h5ad (non-empty but invalid).
    (tmp_path / "corrupt.h5ad").write_bytes(b"NOT_A_VALID_H5AD")

    out = tmp_path / "parsed"
    report = ps.parse_studies(tmp_path, out)
    assert report["n_targets"] == 2
    assert report["n_parsed"] == 1
    assert report["n_errors"] == 1
    assert report["parsed"][0]["study_id"] == "good"
    assert report["errors"][0]["file"] == "corrupt.h5ad"


# ---------------------------------------------------------------------------
# CLI behaviour
# ---------------------------------------------------------------------------


def test_cli_accepts_study_with_and_without_h5ad_suffix(
    tiny_study_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI accepts "tiny" and "tiny.h5ad" interchangeably."""
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    # Copy the fixture into raw/.
    import shutil
    shutil.copy(tiny_study_path, raw / "tiny.h5ad")

    out = tmp_path / "out"
    rc = ps.main(["--raw-root", str(raw), "--output", str(out), "--study", "tiny"])
    assert rc == 0
    assert (out / "tiny_expression.npz").is_file()

    # Now use the with-suffix form — should yield the same set.
    rc = ps.main(["--raw-root", str(raw), "--output", str(out), "--study", "tiny.h5ad"])
    assert rc == 0


def test_cli_missing_root_returns_2(tmp_path: Path, capsys) -> None:
    rc = ps.main(["--raw-root", str(tmp_path / "nope"), "--output", str(tmp_path / "out")])
    assert rc == 2
    captured = capsys.readouterr()
    assert "ok" in captured.out
