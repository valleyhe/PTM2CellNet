import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.integration.perturbgen.results import (
    HeldOutSignature,
    benjamini_hochberg,
    build_held_out_signature,
    compute_rescue_score,
    evaluate_null_calibration,
    evaluate_unperturbed_quality,
    extract_path_result_from_perturbgen_h5ad,
    summarize_rescue_by_donor,
    validate_perturbgen_h5ad_schema,
)


def _deg_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"donor": "D1", "gene": "UP_A", "log2fc": 2.0, "fdr": 0.01},
            {"donor": "D1", "gene": "DOWN_A", "log2fc": -2.0, "fdr": 0.01},
            {"donor": "D1", "gene": "TARGET", "log2fc": 3.0, "fdr": 0.01},
            {"donor": "D2", "gene": "UP_A", "log2fc": 2.1, "fdr": 0.01},
            {"donor": "D2", "gene": "DOWN_A", "log2fc": -1.8, "fdr": 0.01},
            {"donor": "D2", "gene": "TARGET", "log2fc": 2.8, "fdr": 0.01},
            {"donor": "D3", "gene": "UP_A", "log2fc": 1.9, "fdr": 0.01},
            {"donor": "D3", "gene": "DOWN_A", "log2fc": -2.2, "fdr": 0.01},
            {"donor": "D3", "gene": "TARGET", "log2fc": 2.9, "fdr": 0.01},
        ]
    )


def test_build_held_out_signature_excludes_target_gene() -> None:
    signature = build_held_out_signature(
        _deg_table(),
        held_out_donor="D1",
        target_gene="TARGET",
        min_training_donors=2,
        top_k=10,
    )

    assert signature.held_out_donor == "D1"
    assert signature.excluded_target_gene == "TARGET"
    assert signature.up_genes == ("UP_A",)
    assert signature.down_genes == ("DOWN_A",)
    assert "TARGET" not in signature.up_genes


def test_compute_rescue_score_ignores_target_gene_false_positive() -> None:
    signature = HeldOutSignature(
        held_out_donor="D1",
        training_donors=("D2", "D3"),
        up_genes=("UP_A",),
        down_genes=("DOWN_A",),
        excluded_target_gene="TARGET",
    )
    baseline = {"UP_A": 4.0, "DOWN_A": 1.0, "TARGET": 100.0}
    perturbed = {"UP_A": 4.0, "DOWN_A": 1.0, "TARGET": 0.0}

    rescue = compute_rescue_score(baseline, perturbed, signature)

    assert rescue == 0.0


def test_summarize_rescue_by_donor_uses_held_out_signatures() -> None:
    baseline = {
        "D1": {"UP_A": 5.0, "DOWN_A": 1.0, "TARGET": 4.0},
        "D2": {"UP_A": 5.0, "DOWN_A": 1.0, "TARGET": 4.0},
        "D3": {"UP_A": 5.0, "DOWN_A": 1.0, "TARGET": 4.0},
    }
    perturbed = {
        "D1": {"UP_A": 2.0, "DOWN_A": 2.0, "TARGET": 0.0},
        "D2": {"UP_A": 2.0, "DOWN_A": 2.0, "TARGET": 0.0},
        "D3": {"UP_A": 6.0, "DOWN_A": 0.0, "TARGET": 0.0},
    }

    summary = summarize_rescue_by_donor(
        baseline,
        perturbed,
        _deg_table(),
        target_gene="TARGET",
        min_training_donors=2,
        top_k=10,
        bootstrap_iterations=200,
        bootstrap_seed=7,
    )

    assert summary.evaluable_donors == 3
    assert math.isclose(summary.donor_consistency or 0.0, 2 / 3, rel_tol=1e-9)
    assert summary.median_rescue == 4.0
    assert summary.bootstrap_ci is not None
    assert len(summary.donor_scores) == 3


def test_evaluate_null_calibration_marks_20_as_smoke_only() -> None:
    result = evaluate_null_calibration(0.8, [0.1] * 20)

    assert result.matched_null_count == 20
    assert result.smoke_only is True
    assert result.formal_test is False


def test_evaluate_null_calibration_marks_99_as_formal() -> None:
    result = evaluate_null_calibration(0.8, [0.1] * 99)

    assert result.matched_null_count == 99
    assert result.smoke_only is False
    assert result.formal_test is True
    assert 0.0 < result.empirical_pvalue <= 1.0


def test_benjamini_hochberg_returns_monotone_qvalues() -> None:
    q_values = benjamini_hochberg([0.01, 0.04, 0.03, 0.2])

    assert q_values == [0.04, 0.05333333333333334, 0.05333333333333334, 0.2]


def test_validate_perturbgen_h5ad_schema_checks_layers_and_metadata(tmp_path) -> None:
    ad = pytest.importorskip("anndata")
    adata = ad.AnnData(
        X=np.ones((2, 3)),
        obs=pd.DataFrame({"donor": ["D1", "D2"]}, index=["c1", "c2"]),
        var=pd.DataFrame({"ensembl_id": ["ENSG1", "ENSG2", "ENSG3"]}),
    )
    adata.layers["true_counts"] = np.ones((2, 3))
    adata.layers["pred_counts"] = np.ones((2, 3))
    adata.obsm["true_cls"] = np.ones((2, 2))
    adata.obsm["perturbed_cls"] = np.ones((2, 2))
    adata.obsm["mean_cos_similarity"] = np.ones((2, 1))
    adata.varm["gene_cos_similarity"] = np.ones((3, 1))
    path = tmp_path / "result.h5ad"
    adata.write_h5ad(path)

    summary = validate_perturbgen_h5ad_schema(path, required_obs=("donor",), required_var=("ensembl_id",))
    assert (summary.n_obs, summary.n_vars) == (2, 3)

    del adata.layers["pred_counts"]
    adata.write_h5ad(path)
    with pytest.raises(KeyError, match="pred_counts"):
        validate_perturbgen_h5ad_schema(path)


def test_unperturbed_quality_gate_uses_three_seed_median_worst_and_correlation() -> None:
    passed = evaluate_unperturbed_quality([0.6, 0.7, 0.8], [0.1, 0.2, 0.3])
    assert passed.status == "pass"
    failed = evaluate_unperturbed_quality([0.49, 0.7, 0.8], [0.1, 0.2, 0.3])
    assert failed.status == "fail"
    assert "worst_deg_direction_recovery_below_threshold" in failed.reasons
    inconclusive = evaluate_unperturbed_quality([0.9, 0.9], [0.2, 0.2])
    assert inconclusive.status == "inconclusive"


def _write_perturbgen_h5ad(
    tmp_path,
    *,
    donors=("D1", "D2", "D3"),
    baseline=None,
    perturbed=None,
    use_var_index=False,
):
    ad = pytest.importorskip("anndata")
    donor_list = list(donors)
    baseline_array = np.asarray(
        baseline
        if baseline is not None
        else [
            [8.0, 1.0, 100.0],
            [7.5, 1.2, 110.0],
            [9.0, 0.8, 120.0],
        ][: len(donor_list)],
        dtype=float,
    )
    perturbed_array = np.asarray(
        perturbed
        if perturbed is not None
        else [
            [2.0, 2.0, 0.0],
            [2.5, 2.2, 0.0],
            [1.5, 2.1, 0.0],
        ][: len(donor_list)],
        dtype=float,
    )
    var = pd.DataFrame({"ensembl_id": ["ENSG1", "ENSG2", "ENSG3"]})
    if not use_var_index:
        var["gene_symbol"] = ["UP_A", "DOWN_A", "TARGET"]
    adata = ad.AnnData(
        X=perturbed_array,
        obs=pd.DataFrame({"donor_id": donor_list}, index=[f"c{i}" for i in range(len(donor_list))]),
        var=var,
    )
    adata.var_names = ["UP_A", "DOWN_A", "TARGET"]
    adata.layers["true_counts"] = np.ones_like(perturbed_array)
    adata.layers["pred_counts"] = baseline_array
    adata.obsm["true_cls"] = np.ones((len(donor_list), 2))
    adata.obsm["perturbed_cls"] = np.ones((len(donor_list), 2))
    adata.obsm["mean_cos_similarity"] = np.ones((len(donor_list), 1))
    adata.varm["gene_cos_similarity"] = np.ones((3, 1))
    path = tmp_path / "perturbgen_output.h5ad"
    adata.write_h5ad(path)
    return path


def _build_provenance(tmp_path: Path, output_h5ad: Path, *, stage: str = "source_intervention") -> dict[str, str]:
    output_path = output_h5ad.resolve(strict=True)
    sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "stage": stage,
        "status": "success",
        "outputs": {str(output_path): {"kind": "file", "sha256": sha256}},
        "artifacts": {"result_h5ad": str(output_path)},
    }
    stage_manifest = tmp_path / f"{stage}_stage_manifest.json"
    stage_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    return {"stage_manifest": str(stage_manifest), "sha256": sha256}


def test_extract_path_result_from_perturbgen_h5ad_uses_pred_counts_and_x_with_99_null(tmp_path) -> None:
    output_h5ad = _write_perturbgen_h5ad(tmp_path)

    extracted = extract_path_result_from_perturbgen_h5ad(
        output_h5ad=output_h5ad,
        h5ad_provenance=_build_provenance(tmp_path, output_h5ad),
        deg_table=_deg_table(),
        null_distribution=[0.1] * 99,
        path="source_intervention",
        mode="mask",
        seed=7,
        donor_obs_column="donor_id",
        var_gene_column="gene_symbol",
        target_gene="TARGET",
    )

    assert extracted.path_result.status == "evaluable"
    assert extracted.path_result.evaluable_donors == 3
    assert extracted.path_result.matched_null_count == 99
    assert math.isclose(extracted.path_result.empirical_pvalue or 0.0, 0.01, rel_tol=1e-9)
    assert extracted.baseline_matrix == "pred_counts"
    assert extracted.perturbed_matrix == "X"


def test_extract_path_result_from_perturbgen_h5ad_excludes_target_gene_false_positive(tmp_path) -> None:
    output_h5ad = _write_perturbgen_h5ad(
        tmp_path,
        baseline=[[4.0, 1.0, 100.0], [4.0, 1.0, 100.0], [4.0, 1.0, 100.0]],
        perturbed=[[4.0, 1.0, 0.0], [4.0, 1.0, 0.0], [4.0, 1.0, 0.0]],
    )

    extracted = extract_path_result_from_perturbgen_h5ad(
        output_h5ad=output_h5ad,
        h5ad_provenance=_build_provenance(tmp_path, output_h5ad),
        deg_table=_deg_table(),
        null_distribution=[0.0] * 99,
        path="source_intervention",
        mode="mask",
        seed=1,
        donor_obs_column="donor_id",
        var_gene_column="gene_symbol",
        target_gene="TARGET",
    )

    assert extracted.path_result.status == "evaluable"
    assert extracted.path_result.rescue_excl_target == 0.0


def test_extract_path_result_from_perturbgen_h5ad_returns_inconclusive_below_three_evaluable_donors(
    tmp_path,
) -> None:
    output_h5ad = _write_perturbgen_h5ad(tmp_path, donors=("D1", "D2"))

    extracted = extract_path_result_from_perturbgen_h5ad(
        output_h5ad=output_h5ad,
        h5ad_provenance=_build_provenance(tmp_path, output_h5ad),
        deg_table=_deg_table().loc[lambda frame: frame["donor"].isin(["D1", "D2"])],
        null_distribution=[0.1] * 99,
        path="source_intervention",
        mode="mask",
        seed=2,
        donor_obs_column="donor_id",
        var_gene_column="gene_symbol",
        target_gene="TARGET",
    )

    assert extracted.path_result.status == "inconclusive"
    assert extracted.path_result.reason_code == "insufficient_evaluable_donors"


def test_extract_path_result_from_perturbgen_h5ad_requires_explicit_columns_and_provenance(
    tmp_path,
) -> None:
    output_h5ad = _write_perturbgen_h5ad(tmp_path)

    with pytest.raises(ValueError, match="h5ad_provenance"):
        extract_path_result_from_perturbgen_h5ad(
            output_h5ad=output_h5ad,
            h5ad_provenance={},
            deg_table=_deg_table(),
            null_distribution=[0.1] * 99,
            path="source_intervention",
            mode="mask",
            seed=3,
            donor_obs_column="donor_id",
            var_gene_column="gene_symbol",
            target_gene="TARGET",
        )

    with pytest.raises(KeyError, match="missing_obs"):
        extract_path_result_from_perturbgen_h5ad(
            output_h5ad=output_h5ad,
            h5ad_provenance=_build_provenance(tmp_path, output_h5ad),
            deg_table=_deg_table(),
            null_distribution=[0.1] * 99,
            path="source_intervention",
            mode="mask",
            seed=3,
            donor_obs_column="missing_obs",
            var_gene_column="gene_symbol",
            target_gene="TARGET",
        )


def test_extract_path_result_from_perturbgen_h5ad_supports_explicit_var_index_sentinel(tmp_path) -> None:
    output_h5ad = _write_perturbgen_h5ad(tmp_path, use_var_index=True)

    extracted = extract_path_result_from_perturbgen_h5ad(
        output_h5ad=output_h5ad,
        h5ad_provenance=_build_provenance(tmp_path, output_h5ad),
        deg_table=_deg_table(),
        null_distribution=[0.1] * 99,
        path="source_intervention",
        mode="mask",
        seed=9,
        donor_obs_column="donor_id",
        var_gene_column="__index__",
        target_gene="TARGET",
    )

    assert extracted.path_result.status == "evaluable"


def test_extract_path_result_from_perturbgen_h5ad_rejects_tampered_stage_manifest(tmp_path) -> None:
    output_h5ad = _write_perturbgen_h5ad(tmp_path)
    provenance = _build_provenance(tmp_path, output_h5ad)
    manifest_path = Path(provenance["stage_manifest"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifacts"]["result_h5ad"] = str(tmp_path / "other.h5ad")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="stage_manifest path does not exist"):
        extract_path_result_from_perturbgen_h5ad(
            output_h5ad=output_h5ad,
            h5ad_provenance=provenance,
            deg_table=_deg_table(),
            null_distribution=[0.1] * 99,
            path="source_intervention",
            mode="mask",
            seed=4,
            donor_obs_column="donor_id",
            var_gene_column="gene_symbol",
            target_gene="TARGET",
        )


def test_null_and_bh_reject_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        evaluate_null_calibration(0.5, [0.1, float("nan")])
    with pytest.raises(ValueError, match="finite"):
        benjamini_hochberg([0.1, float("inf")])
