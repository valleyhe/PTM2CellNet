"""M6 frozen-cohort acceptance contract tests (manifest + replay)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.integration.perturbgen.frozen_cohort import (
    FrozenCohortError,
    build_acceptance_plan,
    build_frozen_manifest,
    load_frozen_manifest,
    replay_verdicts,
    verify_eval_input_against_manifest,
)


GENE = "STAT3"
ENSEMBL = "ENSG00000168610"


def _cohort_h5ad(tmp_path: Path, content: bytes | None = None) -> Path:
    path = tmp_path / "cohort.h5ad"
    if content is not None:
        path.write_bytes(content)
        return path
    ad = pytest.importorskip("anndata")
    donors = [f"d{index}" for index in range(1, 6)]
    counts = np.asarray([[10, 5, 2]] * 5 + [[12, 4, 3]] * 5, dtype=np.int64)
    obs = pd.DataFrame(
        {
            "cell_type": ["K562"] * 10,
            "state": ["normal"] * 5 + ["disease"] * 5,
            "donor": donors + donors,
        },
        index=[f"cell_{index}" for index in range(10)],
    )
    var = pd.DataFrame(
        {
            "ensembl_id": [ENSEMBL, "ENSG00000000001", "ENSG00000000002"],
            "gene_symbol": [GENE, "UP_A", "DOWN_A"],
        },
        index=[GENE, "UP_A", "DOWN_A"],
    )
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.layers["counts"] = counts.copy()
    adata.write_h5ad(path)
    return path


def _candidates_csv(tmp_path, rows=("STAT3,ENSG00000168610,KO",)):
    path = tmp_path / "candidates.csv"
    path.write_text(
        "gene_symbol,ensembl_id,intervention_type\n" + "".join(f"{row}\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _manifest_kwargs(tmp_path, **overrides):
    defaults = dict(
        cohort_h5ad=_cohort_h5ad(tmp_path),
        cell_type="K562",
        donor_obs_column="donor",
        train_donors=("d1", "d2"),
        held_out_donors=("d3", "d4", "d5"),
        candidates_csv=_candidates_csv(tmp_path),
        created_at="2026-09-10T00:00:00+00:00",
        modes=("mask", "pad", "delete"),
        seeds=(0, 1, 2),
        matched_nulls=99,
    )
    defaults.update(overrides)
    return defaults


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_result_h5ad(path: Path) -> None:
    ad = pytest.importorskip("anndata")
    baseline = np.asarray(
        [[8.0, 1.0, 100.0], [7.5, 1.2, 110.0], [9.0, 0.8, 120.0]],
        dtype=float,
    )
    perturbed = np.asarray(
        [[2.0, 2.0, 0.0], [2.5, 2.2, 0.0], [1.5, 2.1, 0.0]],
        dtype=float,
    )
    adata = ad.AnnData(
        X=perturbed,
        obs=pd.DataFrame({"donor": ["d3", "d4", "d5"]}, index=["cell_3", "cell_4", "cell_5"]),
        var=pd.DataFrame(
            {
                "ensembl_id": ["ENSG00000000001", "ENSG00000000002", ENSEMBL],
                "gene_symbol": ["UP_A", "DOWN_A", GENE],
            },
            index=["UP_A", "DOWN_A", GENE],
        ),
    )
    adata.layers["true_counts"] = np.ones_like(perturbed)
    adata.layers["pred_counts"] = baseline
    adata.obsm["true_cls"] = np.ones((3, 1))
    adata.obsm["perturbed_cls"] = np.ones((3, 1))
    adata.obsm["mean_cos_similarity"] = np.ones((3, 1))
    adata.varm["gene_cos_similarity"] = np.ones((3, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


def _write_synthetic_run_assets(tmp_path: Path, cohort_h5ad: Path) -> tuple[Path, Path, Path]:
    run_root = tmp_path / "perturbgen" / "KO" / ENSEMBL
    tokenise_dir = run_root / "tokenise"
    artifact_dir = tokenise_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    src_dataset = artifact_dir / "src.dataset"
    src_dataset.write_bytes(b"synthetic-src-dataset")
    tgt_dataset_folder = artifact_dir / "tgt_dataset"
    tgt_dataset_folder.mkdir(exist_ok=True)
    tgt_h5ad_folder = artifact_dir / "tgt_h5ad"
    tgt_h5ad_folder.mkdir(exist_ok=True)
    rowid_to_gene_name = artifact_dir / "rowid_to_gene_name.pkl"
    rowid_to_gene_name.write_bytes(b"synthetic-rowid-map")
    tokenid_to_rowid = artifact_dir / "tokenid_to_rowid.pkl"
    tokenid_to_rowid.write_bytes(b"synthetic-token-map")
    tokenise_artifacts = {
        "src_dataset": str(src_dataset.resolve()),
        "tgt_dataset_folder": str(tgt_dataset_folder.resolve()),
        "src_h5ad": str(cohort_h5ad.resolve()),
        "tgt_h5ad_folder": str(tgt_h5ad_folder.resolve()),
        "rowid_to_gene_name": str(rowid_to_gene_name.resolve()),
        "tokenid_to_rowid": str(tokenid_to_rowid.resolve()),
    }
    tokenise_manifest = tokenise_dir / "stage_manifest.json"
    tokenise_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "tokenise",
                "status": "success",
                "artifacts": tokenise_artifacts,
                "fingerprint_material": {
                    "fingerprint_files": {str(cohort_h5ad.resolve()): _sha256(cohort_h5ad)},
                },
            }
        ),
        encoding="utf-8",
    )

    deg_table = tmp_path / "deg.csv"
    deg_rows = [
        f"{donor},{gene},{effect},0.01"
        for donor in ("d1", "d2")
        for gene, effect in (("UP_A", 2.0), ("DOWN_A", -2.0), (GENE, 3.0))
    ]
    deg_table.write_text("donor,gene_symbol,log2fc,fdr\n" + "\n".join(deg_rows) + "\n", encoding="utf-8")

    distributions = []
    null_ids = [f"ENSG999{index:08d}" for index in range(99)]
    for path_kind in ("source_intervention", "within_state"):
        sequence = "src" if path_kind == "source_intervention" else "tgt"
        for seed in (0, 1, 2):
            for mode in ("mask", "pad", "delete"):
                output_h5ad = (
                    run_root
                    / "perturb"
                    / path_kind
                    / f"{mode}_seed{seed}"
                    / "results"
                    / f"20260913_minference_adata_g{GENE}_s{sequence}_t{mode}.h5ad"
                )
                _write_result_h5ad(output_h5ad)
                stage_manifest = output_h5ad.parents[1] / "stage_manifest.json"
                trainer = {
                    "mapping_dict_path": tokenise_artifacts["rowid_to_gene_name"],
                    "tokenid_to_rowid_path": tokenise_artifacts["tokenid_to_rowid"],
                    "perturbation_mode": mode,
                    "perturbation_sequence": [sequence],
                }
                stage_manifest.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "stage": path_kind,
                            "status": "success",
                            "fingerprint": f"synthetic-{path_kind}-{mode}-{seed}",
                            "fingerprint_material": {
                                "random_seed": seed,
                                "fingerprint_config": {
                                    "stage_config": {
                                        "perturb_config": {
                                            "data": {
                                                "src_dataset_file": tokenise_artifacts["src_dataset"],
                                                "tgt_dataset_folder": tokenise_artifacts["tgt_dataset_folder"],
                                                "src_adata": tokenise_artifacts["src_h5ad"],
                                                "tgt_adata_folder": tokenise_artifacts["tgt_h5ad_folder"],
                                            },
                                            "trainer": trainer,
                                        }
                                    }
                                },
                            },
                            "outputs": {
                                str(output_h5ad.resolve()): {
                                    "kind": "file",
                                    "sha256": _sha256(output_h5ad),
                                }
                            },
                            "artifacts": {"result_h5ad": str(output_h5ad.resolve())},
                        }
                    ),
                    encoding="utf-8",
                )
                distributions.append(
                    {
                        "schema_version": "perturbgen_null_distribution/v1",
                        "candidate_ensembl_id": ENSEMBL,
                        "path": path_kind,
                        "mode": mode,
                        "seed": seed,
                        "required_count": 99,
                        "values": [0.1] * 99,
                        "null_ensembl_ids": null_ids,
                    }
                )
    null_manifest = tmp_path / "null-distributions.json"
    null_manifest.write_text(
        json.dumps(
            {
                "schema_version": "perturbgen_null_distribution/v1",
                "distributions": distributions,
            }
        ),
        encoding="utf-8",
    )
    return run_root, deg_table, null_manifest


def _e2e_report(tmp_path: Path, run_root: Path) -> dict:
    return {
        "schema_version": "davf_perturbgen_e2e/v1",
        "davf_config": "synthetic/davf.yaml",
        "intervention_type": "KO",
        "context_h5ad": str(tmp_path / "cohort.h5ad"),
        "candidates": [
            {
                "intervention_type": "KO",
                "status": "pass",
                "invocation": {
                    "gene_symbol": GENE,
                    "ensembl_id": ENSEMBL,
                    "intervention_type": "KO",
                    "candidate": {
                        "observed_direction": "up",
                        "davf_provenance": "synthetic/davf-evidence.json",
                        "direction_gate_status": "pass",
                    },
                },
            }
        ],
        "perturbgen_runs": [
            {
                "intervention_type": "KO",
                "gene_symbol": GENE,
                "ensembl_id": ENSEMBL,
                "output_root": str(run_root),
                "stages": [],
            }
        ],
    }


def _build_real_eval_input(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict]:
    from scripts.build_dual_path_eval_input import main as assemble_main

    cohort_h5ad = _cohort_h5ad(tmp_path)
    candidates_csv = _candidates_csv(tmp_path)
    run_root, deg_table, null_manifest = _write_synthetic_run_assets(tmp_path, cohort_h5ad)
    e2e_report = tmp_path / "e2e-report.json"
    e2e_report.write_text(json.dumps(_e2e_report(tmp_path, run_root)), encoding="utf-8")
    eval_input = tmp_path / "eval-input.json"
    assert (
        assemble_main(
            [
                "--e2e-report",
                str(e2e_report),
                "--output",
                str(eval_input),
                "--deg-table",
                str(deg_table),
                "--null-distribution-manifest",
                str(null_manifest),
                "--unperturbed-quality-status",
                "pass",
                "--uniform-candidate-pvalue",
                "0.01",
                "--donor-obs-column",
                "donor",
                "--var-gene-column",
                "gene_symbol",
                "--deg-donor-column",
                "donor",
                "--deg-gene-column",
                "gene_symbol",
                "--deg-effect-column",
                "log2fc",
                "--deg-fdr-column",
                "fdr",
            ]
        )
        == 0
    )
    candidate_manifest = tmp_path / "frozen-manifest.json"
    manifest = build_frozen_manifest(
        **_manifest_kwargs(tmp_path, cohort_h5ad=cohort_h5ad, candidates_csv=candidates_csv)
    )
    candidate_manifest.write_text(json.dumps(manifest.to_payload()), encoding="utf-8")
    return cohort_h5ad, candidate_manifest, eval_input, run_root, json.loads(eval_input.read_text(encoding="utf-8"))


def _run_real_evaluator(tmp_path: Path, eval_input: Path) -> tuple[Path, dict]:
    from scripts.evaluate_perturbgen_dual_path import main as evaluator_main

    report_dir = tmp_path / "reports"
    assert evaluator_main(["--input-json", str(eval_input), "--output-dir", str(report_dir)]) == 0
    report_manifest = report_dir / "manifest.json"
    return report_manifest, json.loads(report_manifest.read_text(encoding="utf-8"))


class TestBuildFrozenManifest:
    def test_freeze_records_hash_splits_and_plan(self, tmp_path):
        manifest = build_frozen_manifest(**_manifest_kwargs(tmp_path))
        assert manifest.cohort_sha256 and len(manifest.cohort_sha256) == 64
        assert manifest.train_donors == ("d1", "d2")
        assert manifest.held_out_donors == ("d3", "d4", "d5")
        (candidate,) = manifest.candidates
        assert candidate.modes == ("mask", "pad", "delete")
        assert candidate.seeds == (0, 1, 2)
        assert candidate.matched_nulls == 99

    def test_donor_leakage_is_rejected(self, tmp_path):
        with pytest.raises(FrozenCohortError, match="donor leakage"):
            build_frozen_manifest(
                **_manifest_kwargs(
                    tmp_path,
                    train_donors=("d1", "d5"),
                    held_out_donors=("d3", "d4", "d5"),
                )
            )

    def test_insufficient_held_out_donors_rejected(self, tmp_path):
        with pytest.raises(FrozenCohortError, match="3 held-out donors"):
            build_frozen_manifest(**_manifest_kwargs(tmp_path, held_out_donors=("d3", "d4")))

    def test_candidate_without_mask_rejected(self, tmp_path):
        with pytest.raises(FrozenCohortError, match="'mask' primary mode"):
            build_frozen_manifest(**_manifest_kwargs(tmp_path, modes=("pad",)))

    def test_candidate_without_three_seeds_rejected(self, tmp_path):
        with pytest.raises(FrozenCohortError, match="at least 3 seeds"):
            build_frozen_manifest(**_manifest_kwargs(tmp_path, seeds=(0, 1)))

    def test_candidate_with_few_nulls_rejected(self, tmp_path):
        with pytest.raises(FrozenCohortError, match="99 matched nulls"):
            build_frozen_manifest(**_manifest_kwargs(tmp_path, matched_nulls=20))

    def test_manifest_round_trip_detects_hash_drift(self, tmp_path):
        manifest = build_frozen_manifest(**_manifest_kwargs(tmp_path))
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest.to_payload()), encoding="utf-8")
        loaded = load_frozen_manifest(path)
        assert loaded.cohort_sha256 == manifest.cohort_sha256
        # Mutate the cohort after freezing: the loader must refuse it.
        _cohort_h5ad(tmp_path, content=b"tampered")
        with pytest.raises(FrozenCohortError, match="hash drifted"):
            load_frozen_manifest(path)


class TestAcceptancePlan:
    def test_plan_matrix_counts_every_combination(self, tmp_path):
        manifest = build_frozen_manifest(**_manifest_kwargs(tmp_path))
        plan = build_acceptance_plan(manifest)
        # 1 candidate × 2 paths × 3 seeds × 3 modes = 18 runs
        assert plan["n_runs"] == 18
        assert len(plan["runs"]) == 18
        first = plan["runs"][0]
        assert first["gene_symbol"] == GENE
        assert first["path"] in ("source_intervention", "within_state")
        assert first["mode"] in ("mask", "pad", "delete")


class TestVerifyEvalInput:
    def test_full_coverage_passes(self, tmp_path):
        _, manifest_path, _, _, eval_payload = _build_real_eval_input(tmp_path)
        manifest = load_frozen_manifest(manifest_path)
        verification = verify_eval_input_against_manifest(manifest, eval_payload)
        assert verification["covered"] is True
        assert verification["contract_eligible"] is True
        assert verification["missing_runs"] == []

    def test_missing_run_is_reported(self, tmp_path):
        _, manifest_path, _, _, eval_payload = _build_real_eval_input(tmp_path)
        manifest = load_frozen_manifest(manifest_path)
        candidate = eval_payload["candidates"][0]
        runs = [
            run
            for run in candidate["runs"]
            if not (run["path"] == "within_state" and run["seed"] == 2 and run["mode"] == "delete")
        ]
        missing_payload = dict(eval_payload)
        missing_payload["candidates"] = [dict(candidate, runs=runs)]
        verification = verify_eval_input_against_manifest(manifest, missing_payload)
        assert verification["covered"] is False
        assert verification["missing_runs"] == [f"{ENSEMBL}:within_state:delete:seed2"]

    def test_absent_candidate_is_reported(self, tmp_path):
        _, manifest_path, _, _, eval_payload = _build_real_eval_input(tmp_path)
        manifest = load_frozen_manifest(manifest_path)
        candidate = eval_payload["candidates"][0]
        absent_candidate = dict(
            candidate,
            candidate={
                "gene_symbol": "OTHER",
                "ensembl_id": "ENSG00000000001",
                "intervention_type": "KO",
            },
        )
        verification = verify_eval_input_against_manifest(
            manifest,
            {"schema_version": eval_payload["schema_version"], "candidates": [absent_candidate]},
        )
        assert verification["covered"] is False
        assert verification["missing_runs"] == [
            f"{ENSEMBL}:candidate_absent",
            "ENSG00000000001:extra_candidate",
        ]


class TestReplayVerdicts:
    def test_replay_reproduces_published_verdict(self, tmp_path):
        _, manifest_path, eval_input, _, _ = _build_real_eval_input(tmp_path)
        report_path, report = _run_real_evaluator(tmp_path, eval_input)
        manifest = load_frozen_manifest(manifest_path)
        result = replay_verdicts(report, manifest=manifest)
        assert result["reproduced"] is True
        assert result["mismatches"] == []
        assert result["independent_h5ad_recomputed"] is True
        assert result["candidates"][0]["replayed_verdict"] == report["candidates"][0]["verdict"]

        from scripts.run_frozen_acceptance import main

        verify_output = tmp_path / "verify.json"
        assert (
            main(
                [
                    "--verify",
                    "--manifest",
                    str(manifest_path),
                    "--eval-input",
                    str(eval_input),
                    "--report-manifest",
                    str(report_path),
                    "--output",
                    str(verify_output),
                ]
            )
            == 0
        )
        cli_result = json.loads(verify_output.read_text(encoding="utf-8"))
        assert cli_result["replay"]["independent_h5ad_recomputed"] is True

    def test_tampered_verdict_is_detected(self, tmp_path):
        _, manifest_path, eval_input, _, _ = _build_real_eval_input(tmp_path)
        report_path, report = _run_real_evaluator(tmp_path, eval_input)
        manifest = load_frozen_manifest(manifest_path)
        report["candidates"][0]["verdict"] = "fail"
        result = replay_verdicts(report, manifest=manifest)
        assert result["reproduced"] is False
        assert any("published 'fail'" in mismatch for mismatch in result["mismatches"])

        tampered_report_path = tmp_path / "tampered-report.json"
        tampered_report_path.write_text(json.dumps(report), encoding="utf-8")
        from scripts.run_frozen_acceptance import main

        verify_output = tmp_path / "tampered-verify.json"
        assert (
            main(
                [
                    "--verify",
                    "--manifest",
                    str(manifest_path),
                    "--eval-input",
                    str(eval_input),
                    "--report-manifest",
                    str(tampered_report_path),
                    "--output",
                    str(verify_output),
                ]
            )
            == 1
        )


class TestCli:
    def test_freeze_and_plan_round_trip(self, tmp_path):
        from scripts.run_frozen_acceptance import main

        cohort = _cohort_h5ad(tmp_path)
        candidates = _candidates_csv(tmp_path)
        manifest_path = tmp_path / "manifest.json"
        code = main(
            [
                "--freeze",
                "--cohort-h5ad",
                str(cohort),
                "--cell-type",
                "K562",
                "--train-donors",
                "d1,d2",
                "--held-out-donors",
                "d3,d4,d5",
                "--candidates-csv",
                str(candidates),
                "--output",
                str(manifest_path),
            ]
        )
        assert code == 0
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "ptm2cellnet.frozen-cohort/v1"

        plan_path = tmp_path / "plan.json"
        audit_path = tmp_path / "audit.json"
        code = main(
            [
                "--plan",
                "--manifest",
                str(manifest_path),
                "--plan-output",
                str(plan_path),
                "--output",
                str(audit_path),
            ]
        )
        assert code == 0
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        assert plan["n_runs"] == 18
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        assert audit["donor_leakage"] == "clear"
