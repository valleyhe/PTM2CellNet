"""N-5 eval-assembly tests (E2E report → dual-path evaluation input)."""

from __future__ import annotations

import json

import pytest

from src.integration.perturbgen.eval_assembly import (
    EvalAssemblyError,
    build_eval_input_payload,
    load_candidate_pvalues,
    load_e2e_report,
    resolve_run_artifacts,
)


GENE = "STAT3"
ENSEMBL = "ENSG00000168610"


def _write_tokenise_stage(run_root) -> dict[str, str]:
    tokenise_root = run_root / "tokenise" / "artifacts"
    src_dataset = tokenise_root / "src.dataset"
    src_h5ad = tokenise_root / "src.h5ad"
    tgt_dataset_folder = tokenise_root / "tgt_dataset"
    tgt_h5ad_folder = tokenise_root / "tgt_h5ad"
    src_dataset.parent.mkdir(parents=True, exist_ok=True)
    src_dataset.write_bytes(b"src")
    src_h5ad.write_bytes(b"src-h5ad")
    tgt_dataset_folder.mkdir(exist_ok=True)
    tgt_h5ad_folder.mkdir(exist_ok=True)
    artifacts = {
        "src_dataset": str(src_dataset),
        "src_h5ad": str(src_h5ad),
        "tgt_dataset_folder": str(tgt_dataset_folder),
        "tgt_h5ad_folder": str(tgt_h5ad_folder),
    }
    manifest_path = run_root / "tokenise" / "stage_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "tokenise",
                "status": "success",
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return artifacts


def _write_stage(
    run_root,
    path_kind: str,
    *,
    status: str = "success",
    gene: str = GENE,
    seed: int = 0,
    mode: str = "mask",
    subdir: str | None = None,
) -> None:
    sequence = "src" if path_kind == "source_intervention" else "tgt"
    tokenise_artifacts = _write_tokenise_stage(run_root)
    stage_dir = run_root / "perturb" / path_kind / (subdir or "")
    stage_dir.mkdir(parents=True, exist_ok=True)
    h5ad_name = f"20260910_minference_adata_g{gene}_s{sequence}_t{mode}.h5ad"
    (stage_dir / "results").mkdir(exist_ok=True)
    h5ad = stage_dir / "results" / h5ad_name
    h5ad.write_bytes(b"fake-h5ad")
    manifest = {
        "schema_version": 1,
        "stage": path_kind,
        "status": status,
        "fingerprint": "abc123",
        "fingerprint_material": {
            "random_seed": seed,
            "fingerprint_config": {
                "stage_config": {
                    "perturb_config": {
                        "data": {
                            "src_dataset_file": tokenise_artifacts["src_dataset"],
                            "src_adata": tokenise_artifacts["src_h5ad"],
                            "tgt_dataset_folder": tokenise_artifacts["tgt_dataset_folder"],
                            "tgt_adata_folder": tokenise_artifacts["tgt_h5ad_folder"],
                        }
                    }
                }
            },
        },
        "outputs": {str(h5ad): {"kind": "file", "sha256": "runner-recorded-sha256"}},
        "artifacts": {"result_h5ad": str(h5ad)},
    }
    (stage_dir / "stage_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _e2e_report(tmp_path, run_root) -> dict:
    return {
        "schema_version": "davf_perturbgen_e2e/v1",
        "davf_config": "configs/davf.yaml",
        "intervention_type": "KO",
        "context_h5ad": str(tmp_path / "context.h5ad"),
        "candidates": [
            {
                "intervention_type": "KO",
                "status": "pass",
                "invocation": {
                    "gene_symbol": GENE,
                    "ensembl_id": ENSEMBL,
                    "intervention_type": "KO",
                    "candidate": {
                        "observed_direction": "down",
                        "davf_provenance": "checkpoints/davf/davf_ko_dixit",
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


@pytest.fixture()
def run_root(tmp_path):
    root = tmp_path / "perturbgen" / "KO" / ENSEMBL
    _write_stage(root, "source_intervention", seed=1)
    _write_stage(root, "within_state", seed=2)
    return root


@pytest.fixture()
def deg_and_null(tmp_path):
    deg = tmp_path / "deg.csv"
    deg.write_text("donor,gene_symbol,log2fc,fdr\nd1,G1,1.0,0.01\n", encoding="utf-8")
    null = tmp_path / "null.json"
    null.write_text(json.dumps([0.1, -0.2, 0.0]), encoding="utf-8")
    return deg, null


class TestResolveRunArtifacts:
    def test_binds_both_paths_from_stage_manifests(self, run_root):
        artifacts = resolve_run_artifacts(run_root, GENE)
        assert set(artifacts) == {"source_intervention", "within_state"}
        src = artifacts["source_intervention"]
        assert len(src) == 1
        assert src[0].mode == "mask"
        assert src[0].seed == 1
        assert src[0].fingerprint == "abc123"
        assert src[0].path.endswith(".h5ad")

    def test_multi_seed_and_sensitivity_artifacts_are_all_bound(self, tmp_path):
        root = tmp_path / "perturbgen" / "KO" / ENSEMBL
        for seed in (0, 1, 2):
            _write_stage(root, "source_intervention", seed=seed, mode="mask", subdir=f"mask_seed{seed}")
        for mode in ("pad", "delete"):
            _write_stage(root, "source_intervention", seed=0, mode=mode, subdir=f"{mode}_seed0")
        _write_stage(root, "within_state", seed=0, mode="mask")
        artifacts = resolve_run_artifacts(root, GENE)
        src_modes = {(item.mode, item.seed) for item in artifacts["source_intervention"]}
        assert src_modes == {("mask", 0), ("mask", 1), ("mask", 2), ("pad", 0), ("delete", 0)}

    def test_failed_stage_is_rejected(self, tmp_path):
        root = tmp_path / "run_failed"
        _write_stage(root, "source_intervention", status="failed")
        with pytest.raises(EvalAssemblyError, match="did not succeed"):
            resolve_run_artifacts(root, GENE)

    def test_missing_manifest_is_rejected(self, tmp_path):
        root = tmp_path / "run_empty"
        root.mkdir(parents=True)
        with pytest.raises(EvalAssemblyError, match="no successful perturb artifacts"):
            resolve_run_artifacts(root, GENE)

    def test_gene_mismatch_is_rejected(self, run_root):
        with pytest.raises(EvalAssemblyError, match="targets gene"):
            resolve_run_artifacts(run_root, "BRAF")

    def test_sequence_path_mismatch_is_rejected(self, tmp_path):
        root = tmp_path / "run_mismatch"
        # within_state directory but an src-sequence artifact inside.
        stage_dir = root / "perturb" / "within_state"
        stage_dir.mkdir(parents=True)
        h5ad = stage_dir / "results" / "x_minference_adata_g{}_ssrc_tmask.h5ad".format(GENE)
        h5ad.parent.mkdir(exist_ok=True)
        h5ad.write_bytes(b"fake")
        (stage_dir / "stage_manifest.json").write_text(
            json.dumps(
                {
                    "stage": "within_state",
                    "status": "success",
                    "fingerprint_material": {"random_seed": 0},
                    "artifacts": {"result_h5ad": str(h5ad)},
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(EvalAssemblyError, match="does not match path"):
            resolve_run_artifacts(root, GENE)

    def test_manifest_without_seed_is_rejected(self, tmp_path):
        root = tmp_path / "run_noseed"
        stage_dir = root / "perturb" / "source_intervention"
        stage_dir.mkdir(parents=True)
        h5ad = stage_dir / "x_minference_adata_g{}_ssrc_tmask.h5ad".format(GENE)
        h5ad.write_bytes(b"fake")
        (stage_dir / "stage_manifest.json").write_text(
            json.dumps(
                {
                    "stage": "source_intervention",
                    "status": "success",
                    "artifacts": {"result_h5ad": str(h5ad)},
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(EvalAssemblyError, match="random_seed"):
            resolve_run_artifacts(root, GENE)


class TestBuildEvalInputPayload:
    def _payload(self, tmp_path, run_root, deg_and_null, **kwargs):
        deg, null = deg_and_null
        defaults = dict(
            deg_table_path=deg,
            null_distribution_path=null,
            unperturbed_quality_status="pass",
            uniform_candidate_pvalue=0.04,
            run_id="run-1",
        )
        defaults.update(kwargs)
        return build_eval_input_payload(_e2e_report(tmp_path, run_root), **defaults)

    def test_payload_matches_eval_v1_schema(self, tmp_path, run_root, deg_and_null):
        payload = self._payload(tmp_path, run_root, deg_and_null)
        assert payload["schema_version"] == "perturbgen_dual_path_eval/v1"
        assert payload["run_id"] == "run-1"
        (candidate,) = payload["candidates"]
        assert candidate["candidate"]["gene_symbol"] == GENE
        assert candidate["observed_direction"] == "down"
        assert candidate["unperturbed_quality_status"] == "pass"
        assert candidate["candidate_pvalue"] == 0.04
        runs = {run["path"]: run for run in candidate["runs"]}
        assert set(runs) == {"source_intervention", "within_state"}
        src = runs["source_intervention"]
        for key in (
            "mode",
            "seed",
            "output_h5ad",
            "h5ad_provenance",
            "donor_obs_column",
            "var_gene_column",
            "deg_table_path",
            "null_distribution_path",
            "target_gene",
            "deg_donor_column",
            "deg_gene_column",
            "deg_effect_column",
            "deg_fdr_column",
        ):
            assert key in src
        assert src["target_gene"] == GENE
        assert src["h5ad_provenance"]["stage_manifest"].endswith("stage_manifest.json")
        assert src["h5ad_provenance"]["sha256"] == "runner-recorded-sha256"

    def test_missing_pvalue_for_candidate_is_rejected(self, tmp_path, run_root, deg_and_null):
        deg, null = deg_and_null
        with pytest.raises(EvalAssemblyError, match="no candidate p-value"):
            build_eval_input_payload(
                _e2e_report(tmp_path, run_root),
                deg_table_path=deg,
                null_distribution_path=null,
                unperturbed_quality_status="pass",
                candidate_pvalues={"ENSG00000000099": 0.5},
            )

    def test_pvalue_table_round_trip(self, tmp_path, deg_and_null, run_root):
        deg, null = deg_and_null
        table = tmp_path / "pvalues.csv"
        table.write_text(
            f"ensembl_id,pvalue\n{ENSEMBL},0.033\n",
            encoding="utf-8",
        )
        values = load_candidate_pvalues(table)
        payload = build_eval_input_payload(
            _e2e_report(tmp_path, run_root),
            deg_table_path=deg,
            null_distribution_path=null,
            unperturbed_quality_status="pass",
            candidate_pvalues=values,
        )
        assert payload["candidates"][0]["candidate_pvalue"] == 0.033

    def test_run_without_invocation_is_rejected(self, tmp_path, run_root, deg_and_null):
        report = _e2e_report(tmp_path, run_root)
        report["candidates"] = []
        deg, null = deg_and_null
        with pytest.raises(EvalAssemblyError, match="no passing invocation"):
            build_eval_input_payload(
                report,
                deg_table_path=deg,
                null_distribution_path=null,
                unperturbed_quality_status="pass",
                uniform_candidate_pvalue=0.5,
            )

    def test_failed_gate_invocation_is_rejected(self, tmp_path, run_root, deg_and_null):
        report = _e2e_report(tmp_path, run_root)
        report["candidates"][0]["status"] = "fail"
        deg, null = deg_and_null
        with pytest.raises(EvalAssemblyError, match="no passing invocation"):
            build_eval_input_payload(
                report,
                deg_table_path=deg,
                null_distribution_path=null,
                unperturbed_quality_status="pass",
                uniform_candidate_pvalue=0.5,
            )

    def test_bad_quality_status_is_rejected(self, tmp_path, run_root, deg_and_null):
        with pytest.raises(EvalAssemblyError, match="unperturbed_quality_status"):
            self._payload(tmp_path, run_root, deg_and_null, unperturbed_quality_status="maybe")

    def test_legacy_null_distribution_path_preserves_short_inconclusive_compatibility(
        self, tmp_path, run_root
    ):
        deg = tmp_path / "deg.csv"
        deg.write_text("donor,gene_symbol,log2fc,fdr\nd1,G1,1.0,0.01\n", encoding="utf-8")
        short_null = tmp_path / "short-null.json"
        short_null.write_text(json.dumps([0.1] * 98), encoding="utf-8")
        payload = build_eval_input_payload(
            _e2e_report(tmp_path, run_root),
            deg_table_path=deg,
            null_distribution_path=short_null,
            unperturbed_quality_status="pass",
            uniform_candidate_pvalue=0.5,
        )
        assert payload["candidates"][0]["runs"][0]["null_distribution_path"].endswith("short-null.json")

    def test_manifest_index_binds_values_per_artifact(self, tmp_path, run_root, deg_and_null):
        for path_kind in ("source_intervention", "within_state"):
            for mode in ("mask", "pad"):
                for seed in (3, 4):
                    _write_stage(run_root, path_kind, seed=seed, mode=mode, subdir=f"{mode}_seed{seed}")
        artifacts = resolve_run_artifacts(run_root, GENE)
        distributions = [
            {
                "schema_version": "perturbgen_null_distribution/v1",
                "candidate_ensembl_id": ENSEMBL,
                "path": path_kind,
                "mode": artifact.mode,
                "seed": artifact.seed,
                "required_count": 99,
                "values": [0.1] * 99,
                "null_ensembl_ids": [f"ENSG999{i:08d}" for i in range(99)],
            }
            for path_kind, path_artifacts in artifacts.items()
            for artifact in path_artifacts
        ]
        manifest_path = tmp_path / "null-index.json"
        manifest_path.write_text(
            json.dumps({"schema_version": "perturbgen_null_distribution/v1", "distributions": distributions}),
            encoding="utf-8",
        )
        payload = self._payload(
            tmp_path,
            run_root,
            deg_and_null,
            null_distribution_path=None,
            null_distribution_manifest_path=manifest_path,
        )
        runs = payload["candidates"][0]["runs"]
        assert all(len(run["null_distribution"]) == 99 for run in runs)
        assert all("null_distribution_path" not in run for run in runs)
        assert all(run["null_distribution_manifest_path"] == str(manifest_path.resolve()) for run in runs)
        assert {(run["path"], run["mode"], run["seed"]) for run in runs} == {
            (path_kind, artifact.mode, artifact.seed)
            for path_kind, path_artifacts in artifacts.items()
            for artifact in path_artifacts
        }

        distributions[0]["mode"] = "pad"
        manifest_path.write_text(
            json.dumps({"schema_version": "perturbgen_null_distribution/v1", "distributions": distributions}),
            encoding="utf-8",
        )
        with pytest.raises(EvalAssemblyError, match="does not match"):
            self._payload(
                tmp_path,
                run_root,
                deg_and_null,
                null_distribution_path=None,
                null_distribution_manifest_path=manifest_path,
            )


class TestLoadE2EReport:
    def test_schema_version_is_enforced(self, tmp_path):
        path = tmp_path / "report.json"
        path.write_text(json.dumps({"schema_version": "other/v9"}), encoding="utf-8")
        with pytest.raises(EvalAssemblyError, match="unsupported E2E schema_version"):
            load_e2e_report(path)

    def test_report_without_runs_is_rejected(self, tmp_path):
        path = tmp_path / "report.json"
        path.write_text(
            json.dumps({"schema_version": "davf_perturbgen_e2e/v1", "perturbgen_runs": []}),
            encoding="utf-8",
        )
        with pytest.raises(EvalAssemblyError, match="no perturbgen_runs"):
            load_e2e_report(path)
