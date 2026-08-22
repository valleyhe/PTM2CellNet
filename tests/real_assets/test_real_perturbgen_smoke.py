"""Real-asset smoke for the PerturbGen bridge (M5 / T3).

This suite is explicitly opt-in and never runs in default CI.

Supported modes:

1. Run a real pipeline benchmark from a fully resolved config:
   * PTM2CELLNET_PERTURBGEN_REAL_CONFIG
2. Validate an already completed real run:
   * PTM2CELLNET_PERTURBGEN_REAL_OUTPUT_ROOT

Optional env vars:

* PTM2CELLNET_PERTURBGEN_REAL_STAGES - whitespace-separated stage list.
* PTM2CELLNET_PERTURBGEN_REAL_PATH - source_intervention / within_state / both.
* PTM2CELLNET_PERTURBGEN_REAL_TIMEOUT_S - outer timeout for benchmark mode.

When neither a runnable config nor an existing real output root is provided,
the test is skipped with a clear reason.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pytest

from src.integration.perturbgen.config_builder import STAGE_ORDER, load_pipeline_config
from src.integration.perturbgen.results import validate_perturbgen_h5ad_schema
from tests.real_assets import real_assets_enabled, record_evidence


pytestmark = [
    pytest.mark.real_assets,
    pytest.mark.skipif(
        not real_assets_enabled(),
        reason=(
            "PerturbGen real-asset smoke is opt-in. Set "
            "PTM2CELLNET_RUN_REAL_ASSET_TESTS=1 and provide either "
            "PTM2CELLNET_PERTURBGEN_REAL_CONFIG or "
            "PTM2CELLNET_PERTURBGEN_REAL_OUTPUT_ROOT."
        ),
    ),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _benchmark_script() -> Path:
    path = _repo_root() / "scripts" / "benchmark_perturbgen.py"
    assert path.is_file(), f"{path} not found"
    return path


def _parse_stages() -> list[str]:
    raw = os.environ.get("PTM2CELLNET_PERTURBGEN_REAL_STAGES", "").strip()
    if not raw:
        return list(STAGE_ORDER)
    stages = raw.split()
    invalid = sorted(set(stages) - set(STAGE_ORDER))
    if invalid:
        pytest.skip(f"invalid PTM2CELLNET_PERTURBGEN_REAL_STAGES values: {invalid}")
    return stages


def _discover_perturb_h5ad(output_root: Path) -> tuple[Path, ...]:
    stage_names = {"perturb", "source_intervention", "within_state"}
    manifests = sorted(output_root.rglob("stage_manifest.json"))
    matches: list[Path] = []
    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "success" or payload.get("stage") not in stage_names:
            continue
        artifact_path = _extract_h5ad_from_manifest(payload)
        if artifact_path is not None:
            matches.append(artifact_path)
    unique_matches = sorted({path.resolve() for path in matches})
    if not unique_matches:
        pytest.fail(
            "real PerturbGen smoke requires .h5ad outputs resolved from successful "
            f"perturb stage manifests; found none under {output_root}"
        )
    return tuple(unique_matches)


def _extract_h5ad_from_manifest(payload: dict[str, Any]) -> Path | None:
    artifacts = payload.get("artifacts", {})
    if isinstance(artifacts, dict):
        result_h5ad = artifacts.get("result_h5ad")
        if isinstance(result_h5ad, str) and result_h5ad.endswith(".h5ad"):
            return Path(result_h5ad)

    outputs = payload.get("outputs", {})
    if not isinstance(outputs, dict):
        return None
    h5ad_candidates = [Path(path) for path in outputs.keys() if str(path).endswith(".h5ad")]
    if len(h5ad_candidates) == 1:
        return h5ad_candidates[0]
    if len(h5ad_candidates) > 1:
        pytest.fail(
            "successful perturb stage manifest contains multiple .h5ad outputs, "
            f"cannot choose uniquely: {[str(path) for path in h5ad_candidates]}"
        )
    return None


def test_real_perturbgen_smoke():
    config_env = os.environ.get("PTM2CELLNET_PERTURBGEN_REAL_CONFIG")
    existing_output_env = os.environ.get("PTM2CELLNET_PERTURBGEN_REAL_OUTPUT_ROOT")
    path_mode = os.environ.get("PTM2CELLNET_PERTURBGEN_REAL_PATH", "").strip() or None
    stages = _parse_stages()
    timeout_s = int(os.environ.get("PTM2CELLNET_PERTURBGEN_REAL_TIMEOUT_S", "7200"))

    if not config_env and not existing_output_env:
        pytest.skip(
            "Set PTM2CELLNET_PERTURBGEN_REAL_CONFIG to run a real pipeline smoke, "
            "or PTM2CELLNET_PERTURBGEN_REAL_OUTPUT_ROOT to validate existing real artifacts."
        )

    benchmark_json = (_repo_root() / "outputs" / "real_assets" / "perturbgen_real_benchmark.json").resolve()
    benchmark_json.parent.mkdir(parents=True, exist_ok=True)
    benchmark_cmd = [
        sys.executable,
        str(_benchmark_script()),
        "--fixture-type",
        "real",
        "--output",
        str(benchmark_json),
    ]
    mode: str
    asset_label: str
    output_root: Path

    if existing_output_env:
        mode = "validate_existing_output"
        output_root = Path(existing_output_env).expanduser()
        if not output_root.exists():
            pytest.skip(f"PTM2CELLNET_PERTURBGEN_REAL_OUTPUT_ROOT not found: {output_root}")
        benchmark_cmd.extend(["--existing-output-root", str(output_root.resolve())])
        asset_label = f"existing_output_root={output_root}"
    else:
        mode = "run_pipeline"
        config_path = Path(config_env).expanduser()
        if not config_path.is_file():
            pytest.skip(f"PTM2CELLNET_PERTURBGEN_REAL_CONFIG not found: {config_path}")
        try:
            loaded = load_pipeline_config(config_path)
        except Exception as exc:
            pytest.skip(f"real PerturbGen config is not runnable in this environment: {exc}")
        output_root = Path(str(loaded["pipeline"]["output_root"])).expanduser()
        benchmark_cmd.extend(
            [
                "--config",
                str(config_path.resolve()),
                "--stages",
                *stages,
                "--iterations",
                "1",
                "--timeout-seconds",
                str(timeout_s),
            ]
        )
        if path_mode is not None:
            benchmark_cmd.extend(["--path", path_mode])
        asset_label = f"config={config_path}"

    start = time.time()
    outcome = "pass"
    err: str | None = None
    returncode: int | None = None
    benchmark_summary: dict | None = None
    validated_h5ad: list[str] = []
    stderr_tail = ""
    stdout_tail = ""
    try:
        proc = subprocess.run(
            benchmark_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(_repo_root()),
            check=False,
        )
        returncode = proc.returncode
        stdout_tail = proc.stdout[-2000:]
        stderr_tail = proc.stderr[-2000:]
        if proc.returncode != 0:
            outcome = "fail"
            err = f"benchmark subprocess exited {proc.returncode}"
            pytest.fail(f"benchmark_perturbgen.py exited {proc.returncode}.\nstderr tail:\n{stderr_tail}")

        assert benchmark_json.is_file(), f"missing benchmark JSON: {benchmark_json}"
        payload = json.loads(benchmark_json.read_text(encoding="utf-8"))
        assert payload.get("ok") is True, payload
        assert payload.get("fixture_type") == "real", payload
        assert payload.get("samples"), payload
        benchmark_summary = payload.get("summary", {})

        sample = payload["samples"][0]
        resolved_root = sample.get("output_root")
        if resolved_root:
            output_root = Path(resolved_root)
        assert output_root.exists(), f"benchmark reported missing output_root: {output_root}"

        for h5ad_path in _discover_perturb_h5ad(output_root):
            summary = validate_perturbgen_h5ad_schema(h5ad_path)
            validated_h5ad.append(str(h5ad_path))
            assert summary.n_obs > 0 and summary.n_vars > 0
            assert "true_counts" in summary.layers and "pred_counts" in summary.layers
    except subprocess.TimeoutExpired as exc:
        outcome = "fail"
        err = f"timeout after {exc.timeout}s"
        raise
    except Exception as exc:  # pragma: no cover
        outcome = "fail"
        err = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record_evidence(
            "test_real_perturbgen_smoke",
            asset=asset_label,
            outcome=outcome,
            duration_s=time.time() - start,
            extra={
                "mode": mode,
                "stages": stages,
                "path_mode": path_mode,
                "returncode": returncode,
                "output_root": str(output_root) if "output_root" in locals() else None,
                "validated_h5ad": validated_h5ad,
                "benchmark_json": str(benchmark_json),
                "benchmark_summary": benchmark_summary,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "error": err,
            },
        )
