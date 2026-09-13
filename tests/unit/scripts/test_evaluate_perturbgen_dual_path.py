import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_perturbgen_dual_path import INPUT_SCHEMA_VERSION, _extract_run, main


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


def _write_perturbgen_h5ad(tmp_path: Path) -> Path:
    ad = pytest.importorskip("anndata")
    baseline = np.asarray(
        [
            [8.0, 1.0, 100.0],
            [7.5, 1.2, 110.0],
            [9.0, 0.8, 120.0],
        ],
        dtype=float,
    )
    perturbed = np.asarray(
        [
            [2.0, 2.0, 0.0],
            [2.5, 2.2, 0.0],
            [1.5, 2.1, 0.0],
        ],
        dtype=float,
    )
    adata = ad.AnnData(
        X=perturbed,
        obs=pd.DataFrame({"donor_id": ["D1", "D2", "D3"]}, index=["c1", "c2", "c3"]),
        var=pd.DataFrame(
            {
                "gene_symbol": ["UP_A", "DOWN_A", "TARGET"],
                "ensembl_id": ["ENSG1", "ENSG2", "ENSG3"],
            }
        ),
    )
    adata.layers["true_counts"] = np.ones_like(perturbed)
    adata.layers["pred_counts"] = baseline
    adata.obsm["true_cls"] = np.ones((3, 2))
    adata.obsm["perturbed_cls"] = np.ones((3, 2))
    adata.obsm["mean_cos_similarity"] = np.ones((3, 1))
    adata.varm["gene_cos_similarity"] = np.ones((3, 1))
    path = tmp_path / "perturbgen_output.h5ad"
    adata.write_h5ad(path)
    return path


def _build_provenance(tmp_path: Path, output_h5ad: Path, *, stage: str) -> dict[str, str]:
    resolved_h5ad = output_h5ad.resolve(strict=True)
    sha256 = hashlib.sha256(resolved_h5ad.read_bytes()).hexdigest()
    stage_manifest = tmp_path / f"{stage}_stage_manifest.json"
    tokenise_stage_manifest = tmp_path / "tokenise_stage_manifest.json"
    stage_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": stage,
                "status": "success",
                "outputs": {str(resolved_h5ad): {"kind": "file", "sha256": sha256}},
                "artifacts": {"result_h5ad": str(resolved_h5ad)},
            }
        ),
        encoding="utf-8",
    )
    tokenise_stage_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "tokenise",
                "status": "success",
            }
        ),
        encoding="utf-8",
    )
    return {
        "stage_manifest": str(stage_manifest),
        "tokenise_stage_manifest": str(tokenise_stage_manifest),
        "sha256": sha256,
    }


def _build_run(
    *,
    output_h5ad: Path,
    deg_table_path: Path,
    null_distribution_path: Path,
    path: str,
    mode: str,
    seed: int,
    h5ad_provenance: dict[str, str] | None = None,
) -> dict:
    return {
        "path": path,
        "mode": mode,
        "seed": seed,
        "output_h5ad": str(output_h5ad),
        "h5ad_provenance": h5ad_provenance or _build_provenance(output_h5ad.parent, output_h5ad, stage=path),
        "donor_obs_column": "donor_id",
        "var_gene_column": "gene_symbol",
        "deg_table_path": str(deg_table_path),
        "null_distribution_path": str(null_distribution_path),
        "target_gene": "TARGET",
        "deg_donor_column": "donor",
        "deg_gene_column": "gene",
        "deg_effect_column": "log2fc",
        "deg_fdr_column": "fdr",
        "bootstrap_iterations": 50,
        "bootstrap_seed": 7,
    }


def _write_input_json(tmp_path: Path, spec: dict) -> Path:
    path = tmp_path / "input.json"
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_bound_null_manifest(
    path: Path,
    *,
    candidate_ensembl_id: str = "ENSG00000168610",
    path_kind: str = "source_intervention",
    mode: str = "mask",
    seed: int = 1,
    values: list[float] | None = None,
) -> Path:
    values = [0.1] * 99 if values is None else values
    path.write_text(
        json.dumps(
            {
                "schema_version": "perturbgen_null_distribution/v1",
                "candidate_ensembl_id": candidate_ensembl_id,
                "path": path_kind,
                "mode": mode,
                "seed": seed,
                "required_count": 99,
                "values": values,
                "null_ensembl_ids": [f"ENSG999{i:08d}" for i in range(len(values))],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_inline_null_distribution_requires_n05_manifest_binding(tmp_path: Path) -> None:
    output_h5ad = _write_perturbgen_h5ad(tmp_path)
    deg_table_path = tmp_path / "deg.csv"
    _deg_table().to_csv(deg_table_path, index=False)
    legacy_null_path = tmp_path / "legacy.json"
    legacy_null_path.write_text(json.dumps([0.1] * 99), encoding="utf-8")
    run = _build_run(
        output_h5ad=output_h5ad,
        deg_table_path=deg_table_path,
        null_distribution_path=legacy_null_path,
        path="source_intervention",
        mode="mask",
        seed=1,
    )
    run.pop("null_distribution_path")
    run["null_distribution"] = [0.1] * 99
    with pytest.raises(ValueError, match="N-05 assembler"):
        _extract_run(run, input_base_dir=tmp_path, candidate_ensembl_id="ENSG00000168610")

    manifest_path = _write_bound_null_manifest(tmp_path / "bound.json")
    run["null_distribution_manifest_path"] = str(manifest_path)
    result = _extract_run(run, input_base_dir=tmp_path, candidate_ensembl_id="ENSG00000168610")
    assert result["null_distribution"] == [0.1] * 99
    assert result["null_distribution_manifest_path"] == str(manifest_path.resolve())

    run["null_distribution"] = [0.2] * 99
    with pytest.raises(ValueError, match="does not equal"):
        _extract_run(run, input_base_dir=tmp_path, candidate_ensembl_id="ENSG00000168610")

    run["null_distribution"] = [0.1] * 99
    mismatch_manifest = _write_bound_null_manifest(tmp_path / "mismatch.json", path_kind="within_state")
    run["null_distribution_manifest_path"] = str(mismatch_manifest)
    with pytest.raises(ValueError, match="binding"):
        _extract_run(run, input_base_dir=tmp_path, candidate_ensembl_id="ENSG00000168610")


def test_cli_computes_bh_qvalues_and_writes_manifest(tmp_path: Path) -> None:
    output_h5ad = _write_perturbgen_h5ad(tmp_path)
    deg_table_path = tmp_path / "deg.csv"
    _deg_table().to_csv(deg_table_path, index=False)
    null_distribution_path = tmp_path / "null_99.json"
    null_distribution_path.write_text(json.dumps([0.1] * 99), encoding="utf-8")

    runs = [
        _build_run(
            output_h5ad=output_h5ad,
            deg_table_path=deg_table_path,
            null_distribution_path=null_distribution_path,
            path=path_name,
            mode=mode_name,
            seed=seed,
        )
        for path_name in ("source_intervention", "within_state")
        for mode_name in ("mask", "pad", "delete")
        for seed in (1, 2, 3)
    ]
    input_json = _write_input_json(
        tmp_path,
        {
            "schema_version": INPUT_SCHEMA_VERSION,
            "run_id": "dual-path-001",
            "candidates": [
                {
                    "candidate": {"gene_symbol": "STAT3", "intervention_type": "KO"},
                    "candidate_pvalue": 0.01,
                    "observed_direction": "up",
                    "unperturbed_quality_status": "pass",
                    "runs": runs,
                },
                {
                    "candidate": {"gene_symbol": "JUN", "intervention_type": "KO"},
                    "candidate_pvalue": 0.04,
                    "observed_direction": "up",
                    "unperturbed_quality_status": "pass",
                    "runs": runs,
                },
            ],
        },
    )

    output_dir = tmp_path / "reports"
    assert main(["--input-json", str(input_json), "--output-dir", str(output_dir)]) == 0

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidate_count"] == 2
    assert math.isclose(manifest["candidates"][0]["q_value"], 0.02, rel_tol=1e-9)
    assert math.isclose(manifest["candidates"][1]["q_value"], 0.04, rel_tol=1e-9)
    assert Path(manifest["candidates"][0]["artifacts"]["json"]).exists()
    assert Path(manifest["summary_csv"]).exists()


def test_cli_marks_20_null_as_inconclusive_and_99_null_as_formal(tmp_path: Path) -> None:
    output_h5ad = _write_perturbgen_h5ad(tmp_path)
    deg_table_path = tmp_path / "deg.csv"
    _deg_table().to_csv(deg_table_path, index=False)
    null_20_path = tmp_path / "null_20.json"
    null_20_path.write_text(json.dumps([0.1] * 20), encoding="utf-8")
    null_99_path = tmp_path / "null_99.json"
    null_99_path.write_text(json.dumps([0.1] * 99), encoding="utf-8")

    def build_runs(null_path: Path) -> list[dict]:
        return [
            _build_run(
                output_h5ad=output_h5ad,
                deg_table_path=deg_table_path,
                null_distribution_path=null_path,
                path=path_name,
                mode=mode_name,
                seed=seed,
            )
            for path_name in ("source_intervention", "within_state")
            for mode_name in ("mask", "pad", "delete")
            for seed in (1, 2, 3)
        ]

    input_json = _write_input_json(
        tmp_path,
        {
            "schema_version": INPUT_SCHEMA_VERSION,
            "run_id": "dual-path-002",
            "candidates": [
                {
                    "candidate": {"gene_symbol": "SMOKE", "intervention_type": "KO"},
                    "candidate_pvalue": 0.01,
                    "observed_direction": "up",
                    "unperturbed_quality_status": "pass",
                    "runs": build_runs(null_20_path),
                },
                {
                    "candidate": {"gene_symbol": "FORMAL", "intervention_type": "KO"},
                    "candidate_pvalue": 0.01,
                    "observed_direction": "up",
                    "unperturbed_quality_status": "pass",
                    "runs": build_runs(null_99_path),
                },
            ],
        },
    )

    output_dir = tmp_path / "reports"
    main(["--input-json", str(input_json), "--output-dir", str(output_dir)])
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["candidates"][0]["verdict"] == "inconclusive"
    assert "source_intervention_inconclusive" in manifest["candidates"][0]["reasons"]
    assert manifest["candidates"][1]["verdict"] == "pass"


def test_cli_fail_fast_on_missing_fields_and_invalid_path_mode_seed_provenance(tmp_path: Path) -> None:
    output_h5ad = _write_perturbgen_h5ad(tmp_path)
    deg_table_path = tmp_path / "deg.csv"
    _deg_table().to_csv(deg_table_path, index=False)
    null_distribution_path = tmp_path / "null_99.json"
    null_distribution_path.write_text(json.dumps([0.1] * 99), encoding="utf-8")

    bad_run = _build_run(
        output_h5ad=output_h5ad,
        deg_table_path=deg_table_path,
        null_distribution_path=null_distribution_path,
        path="bad_path",
        mode="mask",
        seed=1,
    )
    input_json = _write_input_json(
        tmp_path,
        {
            "schema_version": INPUT_SCHEMA_VERSION,
            "run_id": "dual-path-003",
            "candidates": [
                {
                    "candidate": {"gene_symbol": "STAT3", "intervention_type": "KO"},
                    "candidate_pvalue": 0.01,
                    "observed_direction": "up",
                    "unperturbed_quality_status": "pass",
                    "runs": [bad_run],
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="path must be one of"):
        main(["--input-json", str(input_json), "--output-dir", str(tmp_path / "reports1")])

    bad_run = _build_run(
        output_h5ad=output_h5ad,
        deg_table_path=deg_table_path,
        null_distribution_path=null_distribution_path,
        path="source_intervention",
        mode="mask",
        seed=-1,
    )
    bad_run["h5ad_provenance"] = {}
    input_json = _write_input_json(
        tmp_path,
        {
            "schema_version": INPUT_SCHEMA_VERSION,
            "run_id": "dual-path-004",
            "candidates": [
                {
                    "candidate": {"gene_symbol": "STAT3", "intervention_type": "KO"},
                    "candidate_pvalue": 0.01,
                    "observed_direction": "up",
                    "unperturbed_quality_status": "pass",
                    "runs": [bad_run],
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="seed must be >= 0"):
        main(["--input-json", str(input_json), "--output-dir", str(tmp_path / "reports2")])

    bad_run = _build_run(
        output_h5ad=output_h5ad,
        deg_table_path=deg_table_path,
        null_distribution_path=null_distribution_path,
        path="source_intervention",
        mode="mask",
        seed=1,
    )
    bad_run["h5ad_provenance"] = {}
    input_json = _write_input_json(
        tmp_path,
        {
            "schema_version": INPUT_SCHEMA_VERSION,
            "run_id": "dual-path-005",
            "candidates": [
                {
                    "candidate": {"gene_symbol": "STAT3", "intervention_type": "KO"},
                    "candidate_pvalue": 0.01,
                    "observed_direction": "up",
                    "unperturbed_quality_status": "pass",
                    "runs": [bad_run],
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="h5ad_provenance must be a non-empty mapping"):
        main(["--input-json", str(input_json), "--output-dir", str(tmp_path / "reports3")])

    input_json = _write_input_json(
        tmp_path,
        {
            "schema_version": INPUT_SCHEMA_VERSION,
            "run_id": "dual-path-006",
            "candidates": [
                {
                    "candidate": {"gene_symbol": "STAT3", "intervention_type": "KO"},
                    "observed_direction": "up",
                    "unperturbed_quality_status": "pass",
                    "runs": [
                        _build_run(
                            output_h5ad=output_h5ad,
                            deg_table_path=deg_table_path,
                            null_distribution_path=null_distribution_path,
                            path="source_intervention",
                            mode="mask",
                            seed=1,
                        )
                    ],
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="candidate_pvalue must be provided explicitly"):
        main(["--input-json", str(input_json), "--output-dir", str(tmp_path / "reports4")])


def test_cli_marks_missing_dual_path_as_inconclusive(tmp_path: Path) -> None:
    output_h5ad = _write_perturbgen_h5ad(tmp_path)
    deg_table_path = tmp_path / "deg.csv"
    _deg_table().to_csv(deg_table_path, index=False)
    null_distribution_path = tmp_path / "null_99.json"
    null_distribution_path.write_text(json.dumps([0.1] * 99), encoding="utf-8")

    runs = [
        _build_run(
            output_h5ad=output_h5ad,
            deg_table_path=deg_table_path,
            null_distribution_path=null_distribution_path,
            path="source_intervention",
            mode=mode_name,
            seed=seed,
        )
        for mode_name in ("mask", "pad", "delete")
        for seed in (1, 2, 3)
    ]
    input_json = _write_input_json(
        tmp_path,
        {
            "schema_version": INPUT_SCHEMA_VERSION,
            "run_id": "dual-path-007",
            "candidates": [
                {
                    "candidate": {"gene_symbol": "STAT3", "intervention_type": "KO"},
                    "candidate_pvalue": 0.01,
                    "observed_direction": "up",
                    "unperturbed_quality_status": "pass",
                    "runs": runs,
                }
            ],
        },
    )

    output_dir = tmp_path / "reports"
    main(["--input-json", str(input_json), "--output-dir", str(output_dir)])
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["candidates"][0]["verdict"] == "inconclusive"
    assert "within_state_missing" in manifest["candidates"][0]["reasons"]


def test_cli_resolves_relative_paths_and_rejects_tampered_manifest_and_nonempty_output_dir(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    output_h5ad = _write_perturbgen_h5ad(input_dir)
    deg_table_path = input_dir / "deg.csv"
    _deg_table().to_csv(deg_table_path, index=False)
    null_distribution_path = input_dir / "null_99.json"
    null_distribution_path.write_text(json.dumps([0.1] * 99), encoding="utf-8")
    provenance = _build_provenance(input_dir, output_h5ad, stage="source_intervention")
    within_provenance = _build_provenance(input_dir, output_h5ad, stage="within_state")

    relative_run = _build_run(
        output_h5ad=Path("perturbgen_output.h5ad"),
        deg_table_path=Path("deg.csv"),
        null_distribution_path=Path("null_99.json"),
        path="source_intervention",
        mode="mask",
        seed=1,
        h5ad_provenance={
            "stage_manifest": Path(provenance["stage_manifest"]).name,
            "tokenise_stage_manifest": Path(provenance["tokenise_stage_manifest"]).name,
            "sha256": provenance["sha256"],
        },
    )
    relative_run_within = _build_run(
        output_h5ad=Path("perturbgen_output.h5ad"),
        deg_table_path=Path("deg.csv"),
        null_distribution_path=Path("null_99.json"),
        path="within_state",
        mode="mask",
        seed=1,
        h5ad_provenance={
            "stage_manifest": Path(within_provenance["stage_manifest"]).name,
            "tokenise_stage_manifest": Path(within_provenance["tokenise_stage_manifest"]).name,
            "sha256": within_provenance["sha256"],
        },
    )
    input_json = _write_input_json(
        input_dir,
        {
            "schema_version": INPUT_SCHEMA_VERSION,
            "run_id": "dual-path-008",
            "candidates": [
                {
                    "candidate": {"gene_symbol": "STAT3", "intervention_type": "KO"},
                    "candidate_pvalue": 0.01,
                    "observed_direction": "up",
                    "unperturbed_quality_status": "pass",
                    "runs": [relative_run, relative_run_within],
                }
            ],
        },
    )
    output_dir = tmp_path / "reports"
    main(["--input-json", str(input_json), "--output-dir", str(output_dir)])
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    first_run = manifest["candidates"][0]["input_runs"][0]
    assert Path(first_run["output_h5ad"]).is_absolute()
    assert Path(first_run["h5ad_provenance"]["stage_manifest"]).is_absolute()

    nonempty_output_dir = tmp_path / "occupied"
    nonempty_output_dir.mkdir()
    (nonempty_output_dir / "old.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists and is not empty"):
        main(["--input-json", str(input_json), "--output-dir", str(nonempty_output_dir)])

    tampered_manifest = Path(provenance["stage_manifest"])
    payload = json.loads(tampered_manifest.read_text(encoding="utf-8"))
    payload["outputs"][str(output_h5ad.resolve())]["sha256"] = "0" * 64
    tampered_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        main(["--input-json", str(input_json), "--output-dir", str(tmp_path / "reports_tampered")])


def test_formal_cli_rejects_hand_filled_candidate_pvalue(tmp_path: Path) -> None:
    input_json = _write_input_json(
        tmp_path,
        {
            "schema_version": INPUT_SCHEMA_VERSION,
            "run_id": "formal-reject",
            "evaluation_mode": "formal",
            "candidates": [
                {
                    "candidate": {
                        "gene_symbol": "STAT3",
                        "ensembl_id": "ENSG00000168610",
                        "intervention_type": "KO",
                    },
                    "candidate_pvalue": 0.01,
                    "observed_direction": "up",
                    "unperturbed_quality_status": "pass",
                    "unperturbed_quality": {
                        "source": "extract_unperturbed_quality_from_h5ad",
                        "status": "pass",
                    },
                    "runs": [
                        {
                            "path": "source_intervention",
                            "mode": "mask",
                            "seed": 1,
                            "output_h5ad": str(tmp_path / "missing.h5ad"),
                        }
                    ],
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="formal evaluation rejects"):
        main(["--input-json", str(input_json), "--output-dir", str(tmp_path / "formal_reject")])


def test_formal_cli_aggregates_empirical_p_for_kd_mask(tmp_path: Path) -> None:
    output_h5ad = _write_perturbgen_h5ad(tmp_path)
    deg_table_path = tmp_path / "deg.csv"
    _deg_table().to_csv(deg_table_path, index=False)
    null_distribution_path = tmp_path / "null_99.json"
    null_distribution_path.write_text(json.dumps([0.1] * 99), encoding="utf-8")
    runs = [
        _build_run(
            output_h5ad=output_h5ad,
            deg_table_path=deg_table_path,
            null_distribution_path=null_distribution_path,
            path=path_name,
            mode="mask",
            seed=seed,
        )
        for path_name in ("source_intervention", "within_state")
        for seed in (1, 2, 3)
    ]
    input_json = _write_input_json(
        tmp_path,
        {
            "schema_version": INPUT_SCHEMA_VERSION,
            "run_id": "formal-kd",
            "evaluation_mode": "formal",
            "candidates": [
                {
                    "candidate": {
                        "gene_symbol": "STAT3",
                        "ensembl_id": "ENSG00000168610",
                        "intervention_type": "KD",
                    },
                    "observed_direction": "up",
                    "unperturbed_quality_status": "pass",
                    "unperturbed_quality": {
                        "source": "extract_unperturbed_quality_from_h5ad",
                        "status": "pass",
                    },
                    "runs": runs,
                }
            ],
        },
    )
    output_dir = tmp_path / "formal_reports"
    assert main(["--input-json", str(input_json), "--output-dir", str(output_dir)]) == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["candidates"][0]
    assert entry["verdict"] == "pass"
    assert entry["scientific_acceptance"] is True
    assert entry["evaluation_mode"] == "formal"
    assert entry["evidence_class"] == "empirical_null"
    assert 0.0 < entry["candidate_pvalue"] <= 1.0
