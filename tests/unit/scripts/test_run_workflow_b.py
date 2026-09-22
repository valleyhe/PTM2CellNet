"""Unit tests for the Workflow B unified lifecycle orchestrator (TD-08)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_workflow_b as wfb  # noqa: E402


def _base_args(tmp_path: Path, **overrides: object) -> SimpleNamespace:
    args = SimpleNamespace(
        scvi_model=str(tmp_path / "scvi"),
        embedding_asset=str(tmp_path / "asset"),
        output_root=str(tmp_path / "run"),
        intervention_type="KO",
        norman_dir=str(tmp_path / "norman"),
        geo_root=None,
        seed=7,
        device="cpu",
        epochs=3,
        batch_size=8,
        learning_rate=1e-4,
        patience=2,
        train_donors=None,
        held_out_donors=None,
        require_donor_split=False,
        stages="verify_assets,build_pairs,train,evaluate",
        benchmark=None,
        old_vocabulary=None,
        new_vocabulary=None,
        davf_paired_results=None,
        dry_run=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestBuildStageArgv:
    def test_build_pairs_prefers_norman_dir(self, tmp_path: Path) -> None:
        args = _base_args(tmp_path)
        argv = wfb.build_stage_argv("build_pairs", args, Path(args.output_root))
        assert argv[:2] == ["--norman-dir", str(tmp_path / "norman")]
        assert "--scvi-model" in argv and "--embedding-asset" in argv
        assert "--seed" in argv and "7" in argv

    def test_build_pairs_uses_geo_root_when_no_norman(self, tmp_path: Path) -> None:
        args = _base_args(tmp_path, norman_dir=None, geo_root=str(tmp_path / "geo"))
        argv = wfb.build_stage_argv("build_pairs", args, Path(args.output_root))
        assert argv[:2] == ["--geo-root", str(tmp_path / "geo")]

    def test_train_passes_donor_split_when_explicit(self, tmp_path: Path) -> None:
        args = _base_args(tmp_path, train_donors="D1,D2", held_out_donors="D3")
        argv = wfb.build_stage_argv("train", args, Path(args.output_root))
        assert "--train-donors" in argv and "D1,D2" in argv
        assert "--held-out-donors" in argv and "D3" in argv
        assert "--require-donor-split" not in argv

    def test_train_appends_require_flag_without_split(self, tmp_path: Path) -> None:
        args = _base_args(tmp_path, require_donor_split=True)
        argv = wfb.build_stage_argv("train", args, Path(args.output_root))
        assert "--require-donor-split" in argv

    def test_train_argv_has_no_empty_strings(self, tmp_path: Path) -> None:
        args = _base_args(tmp_path)
        argv = wfb.build_stage_argv("train", args, Path(args.output_root))
        assert all(part != "" for part in argv)

    def test_gate_e_requires_benchmark(self, tmp_path: Path) -> None:
        args = _base_args(tmp_path)
        with pytest.raises(wfb.WorkflowBError, match="benchmark"):
            wfb.build_stage_argv("gate_e", args, Path(args.output_root))

    def test_gate_e_argv_includes_optional_gate_inputs(self, tmp_path: Path) -> None:
        args = _base_args(
            tmp_path,
            benchmark=str(tmp_path / "bench.csv"),
            old_vocabulary=str(tmp_path / "old.json"),
            new_vocabulary=str(tmp_path / "new.json"),
            davf_paired_results=str(tmp_path / "paired.csv"),
        )
        argv = wfb.build_stage_argv("gate_e", args, Path(args.output_root))
        assert "--benchmark" in argv and "--old-vocabulary" in argv
        assert "--new-vocabulary" in argv and "--davf-paired-results" in argv

    def test_unknown_stage_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(wfb.WorkflowBError, match="unknown stage"):
            wfb.build_stage_argv("nonexistent", _base_args(tmp_path), Path(tmp_path))


class TestStageOutputPaths:
    def test_known_stage_outputs(self, tmp_path: Path) -> None:
        outputs = wfb.stage_output_paths("train", tmp_path)
        assert [path.name for path in outputs] == ["best_model.pt", "training_metrics.json"]

    def test_verify_assets_has_no_outputs(self, tmp_path: Path) -> None:
        assert wfb.stage_output_paths("verify_assets", tmp_path) == []


class TestVerifyAssets:
    def _patch_asset(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from src.models import perturbgen_embedding

        fake = SimpleNamespace(vocab_size=50257, embedding_dim=512, manifest={"schema": "stub"})
        monkeypatch.setattr(perturbgen_embedding, "load_perturbgen_embedding_asset", lambda _: fake)
        asset_dir = tmp_path / "asset"
        asset_dir.mkdir(exist_ok=True)
        (asset_dir / "manifest.json").write_text('{"schema": "stub"}', encoding="utf-8")

    def test_passes_with_explicit_donor_split(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_asset(monkeypatch, tmp_path)
        (tmp_path / "scvi").mkdir()
        args = _base_args(tmp_path, train_donors="D1,D2", held_out_donors="D3")
        checks = wfb.verify_assets(args, ["verify_assets", "train"])
        assert checks["donor_split"]["explicit"] is True
        assert checks["embedding_asset"]["vocab_size"] == 50257

    def test_fails_when_require_donor_split_without_lists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_asset(monkeypatch, tmp_path)
        (tmp_path / "scvi").mkdir()
        args = _base_args(tmp_path, require_donor_split=True)
        with pytest.raises(wfb.WorkflowBError, match="explicit donor split"):
            wfb.verify_assets(args, ["verify_assets", "train"])

    def test_fails_with_one_sided_donor_lists(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_asset(monkeypatch, tmp_path)
        (tmp_path / "scvi").mkdir()
        args = _base_args(tmp_path, train_donors="D1")
        with pytest.raises(wfb.WorkflowBError, match="together"):
            wfb.verify_assets(args, ["verify_assets"])

    def test_fails_when_scvi_dir_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_asset(monkeypatch, tmp_path)
        args = _base_args(tmp_path)
        with pytest.raises(wfb.WorkflowBError, match="scVI model directory not found"):
            wfb.verify_assets(args, ["verify_assets"])

    def test_fails_when_gate_e_selected_without_benchmark(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_asset(monkeypatch, tmp_path)
        (tmp_path / "scvi").mkdir()
        args = _base_args(tmp_path)
        with pytest.raises(wfb.WorkflowBError, match="--benchmark"):
            wfb.verify_assets(args, ["verify_assets", "gate_e"])

    def test_fails_when_benchmark_file_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_asset(monkeypatch, tmp_path)
        (tmp_path / "scvi").mkdir()
        args = _base_args(tmp_path, benchmark=str(tmp_path / "missing.csv"))
        with pytest.raises(wfb.WorkflowBError, match="benchmark file not found"):
            wfb.verify_assets(args, ["verify_assets", "gate_e"])


class TestRunWorkflowB:
    def _patch(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from src.models import perturbgen_embedding

        fake = SimpleNamespace(vocab_size=10, embedding_dim=4, manifest={})
        monkeypatch.setattr(perturbgen_embedding, "load_perturbgen_embedding_asset", lambda _: fake)
        asset_dir = tmp_path / "asset"
        asset_dir.mkdir(exist_ok=True)
        (asset_dir / "manifest.json").write_text("{}", encoding="utf-8")
        scvi = tmp_path / "scvi"
        scvi.mkdir(exist_ok=True)

    def test_rejects_non_empty_run_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch(monkeypatch, tmp_path)
        run = tmp_path / "run"
        run.mkdir()
        (run / "stale.txt").write_text("x", encoding="utf-8")
        args = _base_args(tmp_path)
        with pytest.raises(wfb.WorkflowBError, match="not empty"):
            wfb.run_workflow_b(args)

    def test_rejects_unknown_stage_names(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch(monkeypatch, tmp_path)
        args = _base_args(tmp_path, stages="verify_assets,warp")
        with pytest.raises(wfb.WorkflowBError, match="unknown stages"):
            wfb.run_workflow_b(args)

    def test_dry_run_writes_planned_manifest_without_executing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch, tmp_path)
        args = _base_args(tmp_path, dry_run=True)
        manifest = wfb.run_workflow_b(args)
        run_root = Path(args.output_root)
        assert manifest["schema"] == wfb.RUN_MANIFEST_SCHEMA
        assert manifest["run_status"] == "completed"
        assert manifest["stages"]["verify_assets"]["status"] == "passed"
        assert manifest["stages"]["train"]["status"] == "planned"
        assert "--train-data" in manifest["stages"]["train"]["argv"]
        saved = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
        assert saved["schema"] == wfb.RUN_MANIFEST_SCHEMA
        # dry runs are plans: no seal sidecar
        assert not (run_root / "run_manifest.sha256").exists()

    def test_stage_crash_is_recorded_then_reraised(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch(monkeypatch, tmp_path)
        from scripts import build_davf_latent_pairs

        def boom(argv=None):
            raise RuntimeError("pair construction exploded")

        monkeypatch.setattr(build_davf_latent_pairs, "main", boom)
        args = _base_args(tmp_path)
        with pytest.raises(RuntimeError, match="exploded"):
            wfb.run_workflow_b(args)
        saved = json.loads((Path(args.output_root) / "run_manifest.json").read_text(encoding="utf-8"))
        assert saved["run_status"] == "failed"
        assert saved["stages"]["build_pairs"]["status"] == "failed"
        assert "RuntimeError" in saved["stages"]["build_pairs"]["error"]

    def test_gate_e_verdict_failure_yields_gate_failed_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch, tmp_path)
        benchmark = tmp_path / "bench.csv"
        benchmark.write_text("header\n", encoding="utf-8")
        from scripts import build_davf_latent_pairs, evaluate_gate_e, evaluate_latent_davf, train_latent_davf

        def ok(argv=None):
            return 0

        def gate_fail(argv=None):
            (Path(argv[argv.index("--output") + 1])).write_text('{"gate_e_passed": false}', encoding="utf-8")
            return 1

        monkeypatch.setattr(build_davf_latent_pairs, "main", ok)
        monkeypatch.setattr(train_latent_davf, "main", ok)
        monkeypatch.setattr(evaluate_latent_davf, "main", ok)
        monkeypatch.setattr(evaluate_gate_e, "main", gate_fail)

        # stage_output_paths would demand real artifacts; materialize minimal
        # ones so output hashing has something to bite on
        run_root = tmp_path / "run"
        args = _base_args(tmp_path, stages="all", benchmark=str(benchmark))
        originals = wfb.stage_output_paths

        def fake_stage_output_paths(stage: str, root: Path) -> list[Path]:
            paths = originals(stage, root)
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"stub" if path.suffix != ".json" else b"{}")
            return paths

        monkeypatch.setattr(wfb, "stage_output_paths", fake_stage_output_paths)
        manifest = wfb.run_workflow_b(args)
        assert manifest["run_status"] == "gate_failed"
        assert manifest["stages"]["gate_e"]["status"] == "gate_failed"
        assert manifest["stages"]["gate_e"]["outputs"][0]["sha256"]
        sidecar = wfb.seal_run_manifest(run_root)
        assert sidecar.is_file()
        assert len(sidecar.read_text(encoding="utf-8").split()) == 2

    def test_seal_requires_manifest(self, tmp_path: Path) -> None:
        with pytest.raises(wfb.WorkflowBError, match="run manifest missing"):
            wfb.seal_run_manifest(tmp_path)


class TestMainExitCodes:
    def test_main_returns_zero_on_dry_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
        from src.models import perturbgen_embedding

        fake = SimpleNamespace(vocab_size=10, embedding_dim=4, manifest={})
        monkeypatch.setattr(perturbgen_embedding, "load_perturbgen_embedding_asset", lambda _: fake)
        asset_dir = tmp_path / "asset"
        asset_dir.mkdir()
        (asset_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (tmp_path / "scvi").mkdir()
        argv = [
            "--scvi-model",
            str(tmp_path / "scvi"),
            "--embedding-asset",
            str(tmp_path / "asset"),
            "--output-root",
            str(tmp_path / "run"),
            "--intervention-type",
            "KO",
            "--norman-dir",
            str(tmp_path / "norman"),
            "--dry-run",
        ]
        assert wfb.main(argv) == 0
        summary = json.loads(capsys.readouterr().out)
        assert summary["run_status"] == "completed"
