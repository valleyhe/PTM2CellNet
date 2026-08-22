import json

from scripts.benchmark_perturbgen import (
    _build_summary,
    _parse_compute_app_memory,
    _rebase_output_root_paths,
    _is_dependency_closed,
    _isolate_tokenise_dataset,
    build_parser,
    run_benchmark,
)


def test_parse_compute_app_memory_only_sums_pipeline_process_tree():
    stdout = "101, 512\n202, 1024\n303, not-a-number\n"

    assert _parse_compute_app_memory(stdout, {101, 303}) == 512.0


def test_summary_reports_peak_gpu_memory_quantiles():
    summary = _build_summary(
        [
            {
                "wall_elapsed_seconds": 1.0,
                "elapsed_seconds": 0.9,
                "rss_mb": 10.0,
                "peak_gpu_memory_mb": 100.0,
                "total_output_bytes": 20,
                "stage_elapsed_seconds": {},
            },
            {
                "wall_elapsed_seconds": 2.0,
                "elapsed_seconds": 1.8,
                "rss_mb": 20.0,
                "peak_gpu_memory_mb": 300.0,
                "total_output_bytes": 40,
                "stage_elapsed_seconds": {},
            },
        ]
    )

    assert summary["peak_gpu_memory_mb"] == {"p50": 200.0, "p95": 290.0}


def test_fresh_stage_selection_requires_dependency_closure():
    assert _is_dependency_closed(["tokenise", "train_mask", "train_decoder", "perturb"])
    assert _is_dependency_closed(["export_gene_embeddings"])
    assert not _is_dependency_closed(["train_decoder"])
    assert not _is_dependency_closed(["tokenise", "perturb"])


def test_isolate_tokenise_dataset_updates_args_and_formula_paths(tmp_path):
    config = {
        "stages": {
            "tokenise": {
                "args": {"dataset": "cohort"},
                "expected_outputs": [{"path": str(tmp_path / "T_perturb/tokenized_data/cohort/dataset_2000_hvg_tgt")}],
            }
        }
    }

    _isolate_tokenise_dataset(config, "run_01")

    assert config["stages"]["tokenise"]["args"]["dataset"] == "cohort_run_01"
    assert "/cohort_run_01/" in config["stages"]["tokenise"]["expected_outputs"][0]["path"]


def test_rebase_output_root_paths_updates_only_paths_below_old_root(tmp_path):
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    external = tmp_path / "asset.ckpt"
    payload = {
        "pipeline": {"output_root": str(old_root)},
        "stages": {
            "train": {"output_dir": str(old_root / "train" / "model")},
            "asset": str(external),
        },
    }

    rebased = _rebase_output_root_paths(payload, old_root=old_root, new_root=new_root)

    assert rebased["pipeline"]["output_root"] == str(new_root.resolve())
    assert rebased["stages"]["train"]["output_dir"] == str((new_root / "train/model").resolve())
    assert rebased["stages"]["asset"] == str(external)


def test_existing_output_benchmark_reports_raw_p50_p95(tmp_path):
    output_root = tmp_path / "run"
    stage_dir = output_root / "perturb"
    artifact = stage_dir / "result.h5ad"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"fixture")
    manifest = {
        "schema_version": 1,
        "stage": "perturb",
        "status": "success",
        "duration_seconds": 2.5,
        "outputs": {str(artifact): {"kind": "file", "sha256": "fixture"}},
        "artifacts": {"result_h5ad": str(artifact)},
        "environment": {"python_version": "3.11.0"},
    }
    (stage_dir / "stage_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--existing-output-root",
            str(output_root),
            "--fixture-type",
            "engineering",
        ]
    )

    payload = run_benchmark(args)

    assert payload["ok"] is True
    assert payload["fixture_type"] == "engineering"
    assert payload["summary"]["sample_count"] == 1
    assert payload["summary"]["elapsed_seconds"] == {"p50": 2.5, "p95": 2.5}
    assert payload["summary"]["peak_gpu_memory_mb"] == {"p50": None, "p95": None}
    assert payload["samples"][0]["rss_mb"] is None
