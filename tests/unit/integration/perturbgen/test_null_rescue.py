from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.integration.perturbgen.null_generation import NullGenerationError
from src.integration.perturbgen.null_rescue import build_stage_rescue_extractor
from src.integration.perturbgen.runner import StageExecutionResult

AD = pytest.importorskip("anndata")


def _deg_table(genes: tuple[str, str, str]) -> pd.DataFrame:
    rows = []
    for donor in ("D1", "D2", "D3", "D4"):
        rows.append({"donor": donor, "gene": genes[0], "log2fc": 2.0, "fdr": 0.01})
        rows.append({"donor": donor, "gene": genes[1], "log2fc": -2.0, "fdr": 0.01})
        rows.append({"donor": donor, "gene": genes[2], "log2fc": 1.5, "fdr": 0.01})
    return pd.DataFrame(rows)


def _write_result_h5ad(path: Path, *, n_donors: int = 4, var_column: str = "gene_symbol") -> None:
    donors = [f"D{index}" for index in range(1, n_donors + 1)]
    rng = np.random.default_rng(0)
    baseline = np.tile(np.asarray([8.0, 1.0, 5.0], dtype=float), (n_donors, 1)) + rng.normal(0, 0.05, (n_donors, 3))
    perturbed = np.tile(np.asarray([2.0, 7.0, 1.0], dtype=float), (n_donors, 1)) + rng.normal(0, 0.05, (n_donors, 3))
    var = pd.DataFrame({"ensembl_id": [f"ENSG{index}" for index in range(1, 4)]})
    if var_column != "__index__":
        var[var_column] = ["UP_A", "DOWN_A", "NULL_GENE"]
    adata = AD.AnnData(
        X=perturbed,
        obs=pd.DataFrame({"donor_id": donors}, index=[f"c{index}" for index in range(n_donors)]),
        var=var,
    )
    adata.layers["true_counts"] = np.ones_like(perturbed)
    adata.layers["pred_counts"] = baseline
    adata.obsm["true_cls"] = np.ones((n_donors, 2))
    adata.obsm["perturbed_cls"] = np.ones((n_donors, 2))
    adata.obsm["mean_cos_similarity"] = np.ones((n_donors, 1))
    adata.varm["gene_cos_similarity"] = np.ones((3, 1))
    adata.write_h5ad(path)


def _write_stage_manifest(tmp_path: Path, result_h5ad: Path) -> Path:
    digest = hashlib.sha256(result_h5ad.read_bytes()).hexdigest()
    manifest_path = tmp_path / "stage_manifest.json"
    manifest_path.write_text(
        json.dumps({"outputs": {str(result_h5ad): {"kind": "h5ad", "sha256": digest}}}),
        encoding="utf-8",
    )
    return manifest_path


def _request() -> dict[str, str | int]:
    return {
        "null_ensembl_id": "ENSG00000099999",
        "null_gene_symbol": "NULL_GENE",
        "candidate_ensembl_id": "ENSG00000130203",
        "candidate_gene_symbol": "CAND",
        "path": "source_intervention",
        "mode": "mask",
        "seed": 0,
        "output_root": "unused",
    }


def _stage_result(manifest_path: Path, result_h5ad: Path, *, status: str = "success") -> StageExecutionResult:
    return StageExecutionResult(
        stage="perturb",
        status=status,
        output_dir=manifest_path.parent,
        manifest_path=manifest_path,
        reused=False,
        fingerprint="fingerprint",
        artifacts={"result_h5ad": str(result_h5ad)},
    )


def test_extractor_binds_stage_result_to_null_stage_record(tmp_path: Path) -> None:
    result_h5ad = tmp_path / "result.h5ad"
    _write_result_h5ad(result_h5ad)
    manifest_path = _write_stage_manifest(tmp_path, result_h5ad)
    extract = build_stage_rescue_extractor(
        _deg_table(("UP_A", "DOWN_A", "NULL_GENE")),
        donor_obs_column="donor_id",
        var_gene_column="gene_symbol",
    )

    record = extract(_request(), _stage_result(manifest_path, result_h5ad))

    assert record.null_ensembl_id == "ENSG00000099999"
    assert record.candidate_ensembl_id == "ENSG00000130203"
    assert record.path == "source_intervention"
    assert record.mode == "mask"
    assert record.seed == 0
    assert np.isfinite(record.rescue_excl_target)
    assert record.rescue_excl_target > 0
    assert record.result_h5ad == str(result_h5ad.resolve())
    assert record.result_h5ad_sha256 == hashlib.sha256(result_h5ad.read_bytes()).hexdigest()
    assert record.stage_manifest == str(manifest_path.resolve())


def test_extractor_rejects_failed_stage_result(tmp_path: Path) -> None:
    result_h5ad = tmp_path / "result.h5ad"
    _write_result_h5ad(result_h5ad)
    manifest_path = _write_stage_manifest(tmp_path, result_h5ad)
    extract = build_stage_rescue_extractor(
        _deg_table(("UP_A", "DOWN_A", "NULL_GENE")),
        donor_obs_column="donor_id",
        var_gene_column="gene_symbol",
    )

    with pytest.raises(NullGenerationError, match="successful StageExecutionResult"):
        extract(_request(), _stage_result(manifest_path, result_h5ad, status="failed"))


def test_extractor_rejects_missing_result_h5ad_artifact(tmp_path: Path) -> None:
    result_h5ad = tmp_path / "result.h5ad"
    _write_result_h5ad(result_h5ad)
    manifest_path = _write_stage_manifest(tmp_path, result_h5ad)
    extract = build_stage_rescue_extractor(
        _deg_table(("UP_A", "DOWN_A", "NULL_GENE")),
        donor_obs_column="donor_id",
        var_gene_column="gene_symbol",
    )
    stage_result = StageExecutionResult(
        stage="perturb",
        status="success",
        output_dir=tmp_path,
        manifest_path=manifest_path,
        reused=False,
        fingerprint="fingerprint",
        artifacts={"other": str(result_h5ad)},
    )

    with pytest.raises(NullGenerationError, match="result_h5ad"):
        extract(_request(), stage_result)


def test_extractor_rejects_insufficient_evaluable_donors(tmp_path: Path) -> None:
    result_h5ad = tmp_path / "result.h5ad"
    _write_result_h5ad(result_h5ad, n_donors=2)
    manifest_path = _write_stage_manifest(tmp_path, result_h5ad)
    extract = build_stage_rescue_extractor(
        _deg_table(("UP_A", "DOWN_A", "NULL_GENE")),
        donor_obs_column="donor_id",
        var_gene_column="gene_symbol",
    )

    with pytest.raises(NullGenerationError, match="evaluable donors"):
        extract(_request(), _stage_result(manifest_path, result_h5ad))


def test_extractor_rejects_gene_namespace_mismatch(tmp_path: Path) -> None:
    result_h5ad = tmp_path / "result.h5ad"
    _write_result_h5ad(result_h5ad)
    manifest_path = _write_stage_manifest(tmp_path, result_h5ad)
    extract = build_stage_rescue_extractor(
        _deg_table(("MISSING_UP", "MISSING_DOWN", "NULL_GENE")),
        donor_obs_column="donor_id",
        var_gene_column="gene_symbol",
    )

    with pytest.raises(NullGenerationError, match="evaluable donors"):
        extract(_request(), _stage_result(manifest_path, result_h5ad))


def test_builder_rejects_min_evaluable_donors_below_three() -> None:
    with pytest.raises(NullGenerationError, match="min_evaluable_donors"):
        build_stage_rescue_extractor(
            _deg_table(("UP_A", "DOWN_A", "NULL_GENE")),
            donor_obs_column="donor_id",
            var_gene_column="gene_symbol",
            min_evaluable_donors=2,
        )
