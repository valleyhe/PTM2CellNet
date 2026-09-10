import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.integration.perturbgen.config_builder import GeneratedFileSpec, OutputCheck, StagePlan
from src.integration.perturbgen.dimensions import PerturbGenDimensions
from src.integration.perturbgen.env_guard import ExternalEnvironmentReport, PROJECT_ROOT, validate_repo_roots
from src.integration.perturbgen.runner import PerturbGenResumeError, PerturbGenRunner, PerturbGenStageError


def _write(path: Path, text: str = "ok") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _env_report() -> ExternalEnvironmentReport:
    roots = validate_repo_roots(project_root=PROJECT_ROOT, perturbgen_repo_root=PROJECT_ROOT / "ref/Perturbgen-src")
    return ExternalEnvironmentReport(
        python_path=Path(sys.executable),
        python_version="3.11.0",
        roots=roots,
        dependency_hashes={},
        asset_hashes={},
        issues=(),
        warnings=(),
    )


def _plan(
    tmp_path: Path,
    *,
    name: str = "tokenise",
    argv: tuple[str, ...],
    expected_output: Path,
    fingerprint_file: Path,
    uses_gpu: bool = False,
    driver: str = "script",
) -> StagePlan:
    output_dir = tmp_path / name
    return StagePlan(
        name=name,
        driver=driver,
        argv=argv,
        cwd=PROJECT_ROOT,
        output_dir=output_dir,
        timeout_seconds=5,
        uses_gpu=uses_gpu,
        output_root=tmp_path,
        external_python=Path(sys.executable),
        expected_outputs=(OutputCheck(expected_output, "json" if expected_output.suffix == ".json" else "file"),),
        generated_files=(),
        fingerprint_paths=(fingerprint_file,),
        dependency_files=(),
        asset_paths=(),
        resource_estimate={},
        fingerprint_config={"stage_version": 1},
        roots=None,
    )


def test_runner_uses_argv_array_and_shell_false(tmp_path, monkeypatch):
    fingerprint_file = _write(tmp_path / "input.txt", "fp")
    output_file = tmp_path / "tokenise" / "done.txt"
    plan = _plan(
        tmp_path,
        argv=(sys.executable, "-c", "print('ok')"),
        expected_output=output_file,
        fingerprint_file=fingerprint_file,
    )
    runner = PerturbGenRunner(gpu_lock_file=tmp_path / "gpu.lock")
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        _write(output_file, "done")
        return subprocess.CompletedProcess(argv, 0, stdout="stdout", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.run_stage(plan, env_report=_env_report(), resume=False)

    assert result.status == "success"
    assert isinstance(captured["argv"], list)
    assert captured["kwargs"]["shell"] is False


def test_runner_rejects_resume_when_fingerprint_changes(tmp_path):
    output_file = tmp_path / "tokenise" / "done.txt"
    fingerprint_file = _write(tmp_path / "input.txt", "one")
    script = _write(
        tmp_path / "write_done.py",
        "from pathlib import Path\nimport sys\nPath(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)\nPath(sys.argv[1]).write_text('done', encoding='utf-8')\n",
    )
    plan = _plan(
        tmp_path,
        argv=(sys.executable, str(script), str(output_file)),
        expected_output=output_file,
        fingerprint_file=fingerprint_file,
    )
    runner = PerturbGenRunner(gpu_lock_file=tmp_path / "gpu.lock")
    runner.run_stage(plan, env_report=_env_report(), resume=False)

    fingerprint_file.write_text("two", encoding="utf-8")
    with pytest.raises(PerturbGenResumeError, match="fingerprint changed"):
        runner.run_stage(plan, env_report=_env_report(), resume=True)


def test_runner_rejects_resume_when_output_hash_changes(tmp_path):
    output_file = tmp_path / "tokenise" / "done.txt"
    fingerprint_file = _write(tmp_path / "input.txt", "one")
    script = _write(
        tmp_path / "write_done.py",
        "from pathlib import Path\nimport sys\nPath(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)\nPath(sys.argv[1]).write_text('done', encoding='utf-8')\n",
    )
    plan = _plan(
        tmp_path,
        argv=(sys.executable, str(script), str(output_file)),
        expected_output=output_file,
        fingerprint_file=fingerprint_file,
    )
    runner = PerturbGenRunner(gpu_lock_file=tmp_path / "gpu.lock")
    runner.run_stage(plan, env_report=_env_report(), resume=False)
    output_file.write_text("tampered", encoding="utf-8")
    with pytest.raises(PerturbGenResumeError, match="outputs changed"):
        runner.run_stage(plan, env_report=_env_report(), resume=True)


def test_runner_rejects_dirty_json_output(tmp_path):
    output_file = tmp_path / "report" / "bad.json"
    fingerprint_file = _write(tmp_path / "input.txt", "one")
    script = _write(
        tmp_path / "write_bad_json.py",
        "from pathlib import Path\nimport sys\nPath(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)\nPath(sys.argv[1]).write_text('not-json', encoding='utf-8')\n",
    )
    plan = _plan(
        tmp_path,
        name="report",
        argv=(sys.executable, str(script), str(output_file)),
        expected_output=output_file,
        fingerprint_file=fingerprint_file,
    )
    runner = PerturbGenRunner(gpu_lock_file=tmp_path / "gpu.lock")

    with pytest.raises(PerturbGenStageError, match="invalid JSON output"):
        runner.run_stage(plan, env_report=_env_report(), resume=False)


def test_runner_discovers_one_dynamic_output_and_records_artifact(tmp_path):
    fingerprint_file = _write(tmp_path / "input.txt", "one")
    discovery_root = tmp_path / "train_mask" / "model" / "checkpoints"
    dynamic_output = discovery_root / "20260822_train_masking-epoch=00.ckpt"
    script = _write(
        tmp_path / "write_checkpoint.py",
        "from pathlib import Path\nimport sys\np=Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('weights', encoding='utf-8')\n",
    )
    plan = _plan(
        tmp_path,
        name="train_mask",
        argv=(sys.executable, str(script), str(dynamic_output)),
        expected_output=discovery_root,
        fingerprint_file=fingerprint_file,
    )
    plan = StagePlan(
        **{
            **plan.__dict__,
            "expected_outputs": (
                OutputCheck(
                    discovery_root,
                    "file",
                    name="checkpoint",
                    discover_glob="*.ckpt",
                ),
            ),
        }
    )
    runner = PerturbGenRunner(gpu_lock_file=tmp_path / "gpu.lock")

    result = runner.run_stage(plan, env_report=_env_report())
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.artifacts == {"checkpoint": str(dynamic_output.resolve())}
    assert manifest["artifacts"] == result.artifacts
    assert manifest["outputs"][str(dynamic_output.resolve())]["sha256"]


@pytest.mark.parametrize("count", [0, 2])
def test_runner_rejects_zero_or_multiple_dynamic_outputs(tmp_path, count):
    discovery_root = tmp_path / "train_mask" / "model" / "checkpoints"
    discovery_root.mkdir(parents=True)
    for index in range(count):
        _write(discovery_root / f"model-{index}.ckpt")
    runner = PerturbGenRunner(gpu_lock_file=tmp_path / "gpu.lock")

    with pytest.raises(PerturbGenStageError, match="expected exactly one match"):
        runner._resolve_discovered_outputs(
            (
                OutputCheck(
                    discovery_root,
                    "file",
                    name="checkpoint",
                    discover_glob="*.ckpt",
                ),
            )
        )


def test_runner_resolves_upstream_artifact_in_argv_generated_yaml_and_fingerprint(tmp_path):
    checkpoint = _write(tmp_path / "upstream" / "mask.ckpt", "weights")
    output_file = tmp_path / "train_decoder" / "done.txt"
    generated_path = tmp_path / "train_decoder" / "generated" / "config.yaml"
    script = _write(
        tmp_path / "consume.py",
        (
            "from pathlib import Path\n"
            "import sys, yaml\n"
            "ckpt=Path(sys.argv[1]); cfg=yaml.safe_load(Path(sys.argv[2]).read_text(encoding='utf-8'))\n"
            "assert cfg['model']['checkpoint'] == str(ckpt)\n"
            "out=Path(sys.argv[3]); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(ckpt.read_text(), encoding='utf-8')\n"
        ),
    )
    plan = StagePlan(
        name="train_decoder",
        driver="script",
        argv=(
            sys.executable,
            str(script),
            "@artifact:train_mask:checkpoint",
            str(generated_path),
            str(output_file),
        ),
        cwd=PROJECT_ROOT,
        output_dir=tmp_path / "train_decoder",
        timeout_seconds=5,
        uses_gpu=False,
        output_root=tmp_path,
        external_python=Path(sys.executable),
        expected_outputs=(OutputCheck(output_file, "file"),),
        generated_files=(
            GeneratedFileSpec(
                generated_path,
                "yaml",
                {"model": {"checkpoint": "@artifact:train_mask:checkpoint"}},
            ),
        ),
        fingerprint_paths=("@artifact:train_mask:checkpoint",),
        fingerprint_config={"checkpoint": "@artifact:train_mask:checkpoint"},
    )
    runner = PerturbGenRunner(gpu_lock_file=tmp_path / "gpu.lock")

    result = runner.run_stage(
        plan,
        env_report=_env_report(),
        artifact_registry={("train_mask", "checkpoint"): checkpoint},
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest["command"][2] == str(checkpoint.resolve())
    assert manifest["fingerprint_material"]["fingerprint_files"][str(checkpoint.resolve())]


def test_runner_resolves_auto_perturb_dimensions_from_tokenized_artifacts(tmp_path, monkeypatch):
    generated_path = tmp_path / "perturb" / "generated" / "config.yaml"
    payload = {
        "data": {
            "src_dataset_file": str(tmp_path / "src.dataset"),
            "tgt_dataset_folder": str(tmp_path / "tgt"),
        },
        "trainer": {
            "tgt_vocab_size": "auto",
            "max_seq_length": "auto",
        },
        "datamodule": {"max_len": "auto"},
    }
    plan = StagePlan(
        name="perturb",
        driver="perturb_script",
        argv=(),
        cwd=PROJECT_ROOT,
        output_dir=tmp_path / "perturb",
        timeout_seconds=5,
        uses_gpu=False,
        output_root=tmp_path,
        external_python=Path(sys.executable),
        expected_outputs=(),
        generated_files=(GeneratedFileSpec(generated_path, "yaml", payload),),
        fingerprint_paths=(),
        dependency_files=(),
        asset_paths=(),
        resource_estimate={},
        fingerprint_config={},
        roots=None,
    )
    monkeypatch.setattr(
        "src.integration.perturbgen.runner.derive_perturbgen_dimensions",
        lambda *args, **kwargs: PerturbGenDimensions(
            tgt_vocab_size=2005,
            max_seq_length=248,
        ),
    )

    resolved = PerturbGenRunner(gpu_lock_file=tmp_path / "gpu.lock")._resolve_auto_perturb_dimensions(plan)

    generated_payload = resolved.generated_files[0].payload
    assert generated_payload["trainer"]["tgt_vocab_size"] == 2005
    assert generated_payload["trainer"]["max_seq_length"] == 248
    assert generated_payload["datamodule"]["max_len"] == 248
    assert payload["trainer"]["tgt_vocab_size"] == "auto"


def test_runner_rejects_unresolved_upstream_artifact(tmp_path):
    plan = _plan(
        tmp_path,
        argv=(sys.executable, "@artifact:train_mask:checkpoint"),
        expected_output=tmp_path / "tokenise" / "done.txt",
        fingerprint_file=_write(tmp_path / "input.txt"),
    )
    runner = PerturbGenRunner(gpu_lock_file=tmp_path / "gpu.lock")

    with pytest.raises(PerturbGenStageError, match="unresolved upstream artifact"):
        runner.run_stage(plan, env_report=_env_report())


def test_runner_rejects_stale_external_output_before_stage_start(tmp_path):
    external_output = _write(tmp_path / "shared_tokenized" / "normal.dataset" / "data.bin")
    plan = _plan(
        tmp_path,
        argv=(sys.executable, "-c", "raise SystemExit('must not run')"),
        expected_output=external_output.parent,
        fingerprint_file=_write(tmp_path / "input.txt"),
    )
    plan = StagePlan(
        **{
            **plan.__dict__,
            "expected_outputs": (OutputCheck(external_output.parent, "directory"),),
        }
    )
    runner = PerturbGenRunner(gpu_lock_file=tmp_path / "gpu.lock")

    with pytest.raises(PerturbGenStageError, match="already exists before stage start"):
        runner.run_stage(plan, env_report=_env_report())


def test_runner_rejects_incomplete_embedding_asset(tmp_path):
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "manifest.json").write_text("{}", encoding="utf-8")
    runner = PerturbGenRunner(gpu_lock_file=tmp_path / "gpu.lock")
    with pytest.raises(PerturbGenStageError, match="invalid PerturbGen embedding asset"):
        runner._validate_outputs(
            (OutputCheck(asset, "perturbgen_embedding_asset"),)
        )


def test_runner_timeout_is_propagated(tmp_path):
    output_file = tmp_path / "tokenise" / "done.txt"
    fingerprint_file = _write(tmp_path / "input.txt", "one")
    script = _write(
        tmp_path / "sleep.py",
        "import time\ntime.sleep(2)\n",
    )
    plan = StagePlan(
        **{
            **_plan(
                tmp_path,
                argv=(sys.executable, str(script)),
                expected_output=output_file,
                fingerprint_file=fingerprint_file,
            ).__dict__,
            "timeout_seconds": 1,
        }
    )
    runner = PerturbGenRunner(gpu_lock_file=tmp_path / "gpu.lock")

    with pytest.raises(subprocess.TimeoutExpired):
        runner.run_stage(plan, env_report=_env_report(), resume=False)


def test_gpu_lock_serializes_parallel_gpu_stages(tmp_path):
    shared_log = tmp_path / "shared.log"
    fingerprint_a = _write(tmp_path / "fp_a.txt", "a")
    fingerprint_b = _write(tmp_path / "fp_b.txt", "b")
    script = _write(
        tmp_path / "gpu_stage.py",
        (
            "from pathlib import Path\n"
            "import sys, time\n"
            "out = Path(sys.argv[1])\n"
            "log = Path(sys.argv[2])\n"
            "out.parent.mkdir(parents=True, exist_ok=True)\n"
            "log.parent.mkdir(parents=True, exist_ok=True)\n"
            "with log.open('a', encoding='utf-8') as handle:\n"
            "    handle.write('start\\n')\n"
            "time.sleep(0.25)\n"
            "out.write_text('done', encoding='utf-8')\n"
            "with log.open('a', encoding='utf-8') as handle:\n"
            "    handle.write('end\\n')\n"
        ),
    )
    runner = PerturbGenRunner(gpu_lock_file=tmp_path / "gpu.lock")
    plan_a = _plan(
        tmp_path,
        name="gpu_a",
        argv=(sys.executable, str(script), str(tmp_path / "gpu_a" / "done.txt"), str(shared_log)),
        expected_output=tmp_path / "gpu_a" / "done.txt",
        fingerprint_file=fingerprint_a,
        uses_gpu=True,
    )
    plan_b = _plan(
        tmp_path,
        name="gpu_b",
        argv=(sys.executable, str(script), str(tmp_path / "gpu_b" / "done.txt"), str(shared_log)),
        expected_output=tmp_path / "gpu_b" / "done.txt",
        fingerprint_file=fingerprint_b,
        uses_gpu=True,
    )

    errors = []

    def _run(plan):
        try:
            runner.run_stage(plan, env_report=_env_report(), resume=False)
        except Exception as exc:  # pragma: no cover - surfaced by assert below
            errors.append(exc)

    start = time.monotonic()
    first = threading.Thread(target=_run, args=(plan_a,))
    second = threading.Thread(target=_run, args=(plan_b,))
    first.start()
    second.start()
    first.join()
    second.join()
    elapsed = time.monotonic() - start

    assert not errors
    assert elapsed >= 0.45
    assert shared_log.read_text(encoding="utf-8").count("start") == 2


def test_disk_budget_requires_double_estimate_plus_ten_gib(tmp_path, monkeypatch):
    fingerprint = _write(tmp_path / "input.txt")
    plan = _plan(
        tmp_path,
        argv=(sys.executable, "-c", "pass"),
        expected_output=tmp_path / "tokenise" / "done.txt",
        fingerprint_file=fingerprint,
    )
    plan = StagePlan(
        **{
            **plan.__dict__,
            "fingerprint_config": {"estimated_total_output_bytes": 1024},
        }
    )
    monkeypatch.setattr(
        "src.integration.perturbgen.runner.shutil.disk_usage",
        lambda path: SimpleNamespace(free=10 * 1024**3),
    )
    runner = PerturbGenRunner(gpu_lock_file=tmp_path / "gpu.lock")
    with pytest.raises(PerturbGenStageError, match="insufficient disk space"):
        runner._ensure_disk_budget(plan)
