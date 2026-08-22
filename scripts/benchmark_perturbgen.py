#!/usr/bin/env python3
"""Benchmark or summarize a PerturbGen pipeline run and emit JSON.

This script intentionally reports raw measurements only. It does not turn the
numbers into PASS/FAIL performance claims because fixture shape, hardware, and
selected stages may differ between runs.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.perturbgen.config_builder import STAGE_ORDER, load_pipeline_config  # noqa: E402


RUN_PIPELINE_SCRIPT = PROJECT_ROOT / "scripts" / "run_perturbgen_pipeline.py"
STAGE_MANIFEST = "stage_manifest.json"
_RSS_MARKER = "__PTM2CELLNET_MAXRSS_KB__="
_RSS_PATTERN = re.compile(rf"{_RSS_MARKER}(?P<rss_kb>\d+)")
_SAFE_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--config",
        type=Path,
        help="Run the pipeline from this config and benchmark the resulting run(s).",
    )
    target.add_argument(
        "--existing-output-root",
        type=Path,
        help="Summarize an already completed real or engineering PerturbGen run.",
    )
    parser.add_argument(
        "--fixture-type",
        choices=("engineering", "real"),
        required=True,
        help="Tag the measurement source. This label is operator-supplied, not inferred.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_ORDER,
        default=list(STAGE_ORDER),
        help="Stages to execute when --config is used.",
    )
    parser.add_argument(
        "--path",
        choices=("source_intervention", "within_state", "both"),
        default=None,
        help="Optional perturbation path selection when executing the pipeline.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="How many benchmark runs to execute for --config mode.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing output root in --config mode. Only valid with --iterations 1.",
    )
    parser.add_argument(
        "--run-id-prefix",
        default="benchmark",
        help="Prefix for generated output-root subdirectories in --config mode.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="Optional outer timeout for each benchmark invocation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON payload.",
    )
    return parser


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _dir_size_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def _rebase_output_root_paths(value: Any, *, old_root: Path, new_root: Path) -> Any:
    if isinstance(value, str):
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            return value
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(old_root)
        except ValueError:
            return value
        return str((new_root / relative).resolve())
    if isinstance(value, list):
        return [_rebase_output_root_paths(item, old_root=old_root, new_root=new_root) for item in value]
    if isinstance(value, dict):
        return {
            key: _rebase_output_root_paths(item, old_root=old_root, new_root=new_root) for key, item in value.items()
        }
    return value


def _is_dependency_closed(stages: list[str]) -> bool:
    selected = set(stages)
    prerequisites = {
        "train_mask": {"tokenise"},
        "train_decoder": {"tokenise", "train_mask"},
        "perturb": {"tokenise", "train_mask", "train_decoder"},
    }
    return all(prerequisites.get(stage, set()) <= selected for stage in selected)


def _isolate_tokenise_dataset(config: dict[str, Any], run_name: str) -> None:
    tokenise = config["stages"]["tokenise"]
    args = tokenise["args"]
    old_dataset = str(args["dataset"])
    new_dataset = f"{old_dataset}_{run_name}"
    args["dataset"] = new_dataset
    for output in tokenise.get("expected_outputs", []):
        raw_path = output.get("path")
        if not isinstance(raw_path, str):
            continue
        parts = list(Path(raw_path).parts)
        positions = [index for index, part in enumerate(parts) if part == old_dataset]
        if len(positions) != 1:
            raise ValueError(f"tokenise expected output path must contain the dataset segment exactly once: {raw_path}")
        parts[positions[0]] = new_dataset
        output["path"] = str(Path(*parts))


def _load_stage_manifests(output_root: Path) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(output_root.rglob(STAGE_MANIFEST)):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        stage = payload.get("stage")
        if isinstance(stage, str) and stage:
            manifests[stage] = payload
    if not manifests:
        raise FileNotFoundError(f"no {STAGE_MANIFEST} files found under {output_root}")
    return manifests


def _stage_output_sizes(manifests: dict[str, dict[str, Any]]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for stage, payload in manifests.items():
        stage_total = 0
        for output_path in payload.get("outputs", {}).keys():
            path = Path(output_path)
            if path.is_file():
                stage_total += path.stat().st_size
            elif path.is_dir():
                stage_total += _dir_size_bytes(path)
        sizes[stage] = stage_total
    return sizes


def _summarize_completed_run(
    output_root: Path,
    *,
    fixture_type: str,
    iteration: int,
    rss_mb: float | None,
    invoked_argv: list[str] | None,
) -> dict[str, Any]:
    manifests = _load_stage_manifests(output_root)
    stage_elapsed = {
        stage: float(payload["duration_seconds"])
        for stage, payload in manifests.items()
        if "duration_seconds" in payload
    }
    stage_status = {stage: str(payload.get("status", "unknown")) for stage, payload in manifests.items()}
    stage_output_bytes = _stage_output_sizes(manifests)
    report_path = output_root / "report" / "pipeline_report.json"
    perturb_outputs = sorted(
        output_path
        for payload in manifests.values()
        for output_path in payload.get("outputs", {}).keys()
        if str(output_path).endswith(".h5ad")
    )
    first_stage = manifests[sorted(manifests)[0]]
    environment = first_stage.get("environment", {})

    sample: dict[str, Any] = {
        "iteration": iteration,
        "fixture_type": fixture_type,
        "output_root": str(output_root),
        "report_path": str(report_path) if report_path.is_file() else None,
        "elapsed_seconds": float(sum(stage_elapsed.values())),
        "rss_mb": rss_mb,
        "total_output_bytes": int(sum(stage_output_bytes.values())),
        "run_output_root_bytes": _dir_size_bytes(output_root),
        "stage_elapsed_seconds": stage_elapsed,
        "stage_status": stage_status,
        "stage_output_bytes": stage_output_bytes,
        "perturb_output_h5ad": perturb_outputs,
        "environment": environment,
    }
    if invoked_argv is not None:
        sample["invoked_argv"] = invoked_argv
    return sample


def _run_nvidia_smi() -> dict[str, Any] | None:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        proc = subprocess.run(
            [
                binary,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    devices = []
    for line in proc.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) == 3:
            devices.append(
                {
                    "name": parts[0],
                    "memory_total": parts[1],
                    "driver_version": parts[2],
                }
            )
    return {"binary": binary, "devices": devices}


def _runtime_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "generated_at": _now_iso(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "benchmark_python": sys.version,
        "benchmark_python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "time_binary": shutil.which("time"),
        "project_root": str(PROJECT_ROOT),
    }
    try:
        import torch

        metadata["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "device_names": (
                [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
                if torch.cuda.is_available()
                else []
            ),
        }
    except Exception as exc:  # pragma: no cover - torch import depends on env
        metadata["torch"] = {"import_error": f"{type(exc).__name__}: {exc}"}
    nvidia_smi = _run_nvidia_smi()
    if nvidia_smi is not None:
        metadata["nvidia_smi"] = nvidia_smi
    return metadata


def _parse_timed_stderr(stderr: str) -> tuple[str, float | None]:
    matches = list(_RSS_PATTERN.finditer(stderr))
    rss_mb = None
    if matches:
        rss_kb = int(matches[-1].group("rss_kb"))
        rss_mb = float(rss_kb / 1024.0)
        cleaned = _RSS_PATTERN.sub("", stderr).strip()
        return cleaned, rss_mb
    return stderr, None


def _descendant_pids(root_pid: int) -> set[int]:
    """Return a Linux process tree rooted at ``root_pid`` without psutil."""

    pending = [int(root_pid)]
    found: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.add(pid)
        children_file = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            raw = children_file.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        pending.extend(int(item) for item in raw.split() if item.isdigit())
    return found


def _parse_compute_app_memory(stdout: str, pids: set[int]) -> float:
    total = 0.0
    for line in stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            used_mb = float(parts[1])
        except ValueError:
            continue
        if pid in pids:
            total += used_mb
    return total


def _query_process_tree_gpu_memory_mb(root_pid: int) -> float | None:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    proc = subprocess.run(
        [
            binary,
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    if proc.returncode != 0:
        return None
    return _parse_compute_app_memory(proc.stdout, _descendant_pids(root_pid))


def _monitor_gpu_memory(
    root_pid: int,
    stop: threading.Event,
    samples: list[float],
) -> None:
    while not stop.is_set():
        try:
            used_mb = _query_process_tree_gpu_memory_mb(root_pid)
        except (OSError, subprocess.SubprocessError):
            used_mb = None
        if used_mb is not None:
            samples.append(float(used_mb))
        stop.wait(0.25)


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _run_with_resource_metrics(
    argv: list[str],
    *,
    cwd: Path,
    timeout_seconds: int | None,
) -> tuple[subprocess.CompletedProcess[str], float, float | None, float | None, int]:
    started = time.perf_counter()
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd),
        start_new_session=True,
    )
    stop = threading.Event()
    gpu_samples: list[float] = []
    monitor = threading.Thread(
        target=_monitor_gpu_memory,
        args=(process.pid, stop, gpu_samples),
        daemon=True,
    )
    monitor.start()
    try:
        stdout, raw_stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        raise
    finally:
        stop.set()
        monitor.join(timeout=2)
    wall_elapsed = time.perf_counter() - started
    stderr, rss_mb = _parse_timed_stderr(raw_stderr)
    completed = subprocess.CompletedProcess(
        argv,
        process.returncode,
        stdout=stdout,
        stderr=stderr,
    )
    peak_gpu_memory_mb = max(gpu_samples) if gpu_samples else None
    return completed, wall_elapsed, rss_mb, peak_gpu_memory_mb, len(gpu_samples)


def _build_iteration_output_root(
    base_output_root: Path,
    *,
    run_id_prefix: str,
    fixture_type: str,
    iteration: int,
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return base_output_root / "benchmark_runs" / f"{run_id_prefix}_{fixture_type}_{stamp}_{iteration:02d}"


def _execute_pipeline_iteration(
    config_path: Path,
    args: argparse.Namespace,
    *,
    iteration: int,
) -> dict[str, Any]:
    config = load_pipeline_config(config_path)
    base_output_root = Path(str(config["pipeline"]["output_root"])).expanduser().resolve()
    if args.resume:
        if args.iterations != 1:
            raise ValueError("--resume only supports --iterations 1")
        iteration_output_root = base_output_root
    else:
        iteration_output_root = _build_iteration_output_root(
            base_output_root,
            run_id_prefix=str(args.run_id_prefix),
            fixture_type=str(args.fixture_type),
            iteration=iteration,
        )

    config_copy = deepcopy(config)
    config_copy["pipeline"]["output_root"] = str(iteration_output_root)
    config_copy = _rebase_output_root_paths(
        config_copy,
        old_root=base_output_root,
        new_root=iteration_output_root,
    )
    config_copy["pipeline"]["output_root"] = str(iteration_output_root)
    if "tokenise" in args.stages and not args.resume:
        _isolate_tokenise_dataset(config_copy, iteration_output_root.name)

    with tempfile.TemporaryDirectory(prefix="perturbgen-benchmark-") as temp_dir:
        temp_config = Path(temp_dir) / f"benchmark_config_{iteration:02d}.yaml"
        temp_config.write_text(
            yaml.safe_dump(config_copy, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        cmd = [
            sys.executable,
            str(RUN_PIPELINE_SCRIPT),
            "--config",
            str(temp_config),
            "--stages",
            *args.stages,
        ]
        if args.path is not None:
            cmd.extend(["--path", args.path])
        if args.resume:
            cmd.append("--resume")

        wrapped_cmd = cmd
        time_binary = shutil.which("time")
        if time_binary is not None:
            wrapped_cmd = [time_binary, "-f", f"{_RSS_MARKER}%M", *cmd]

        proc, wall_elapsed, rss_mb, peak_gpu_memory_mb, gpu_sample_count = _run_with_resource_metrics(
            wrapped_cmd,
            cwd=PROJECT_ROOT,
            timeout_seconds=args.timeout_seconds,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"PerturbGen benchmark run failed: returncode={proc.returncode}, stderr_tail={proc.stderr[-2000:]}"
            )

    sample = _summarize_completed_run(
        iteration_output_root,
        fixture_type=str(args.fixture_type),
        iteration=iteration,
        rss_mb=rss_mb,
        invoked_argv=cmd,
    )
    sample["wall_elapsed_seconds"] = float(wall_elapsed)
    sample["peak_gpu_memory_mb"] = peak_gpu_memory_mb
    sample["gpu_memory_sample_count"] = gpu_sample_count
    if peak_gpu_memory_mb is None:
        sample["gpu_memory_unavailable_reason"] = "nvidia-smi compute-app sampling returned no process-tree samples"
    sample["stdout_tail"] = proc.stdout[-2000:]
    sample["stderr_tail"] = proc.stderr[-2000:]
    return sample


def _build_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    wall_elapsed = [float(item["wall_elapsed_seconds"]) for item in samples if "wall_elapsed_seconds" in item]
    elapsed = [float(item["elapsed_seconds"]) for item in samples if "elapsed_seconds" in item]
    rss = [float(item["rss_mb"]) for item in samples if item.get("rss_mb") is not None]
    output_bytes = [float(item["total_output_bytes"]) for item in samples if "total_output_bytes" in item]
    peak_gpu_memory = [
        float(item["peak_gpu_memory_mb"]) for item in samples if item.get("peak_gpu_memory_mb") is not None
    ]

    stage_names = sorted(
        {stage_name for item in samples for stage_name in item.get("stage_elapsed_seconds", {}).keys()}
    )
    stage_elapsed_summary = {}
    for stage_name in stage_names:
        values = [
            float(item["stage_elapsed_seconds"][stage_name])
            for item in samples
            if stage_name in item.get("stage_elapsed_seconds", {})
        ]
        stage_elapsed_summary[stage_name] = {
            "p50": _quantile(values, 0.50),
            "p95": _quantile(values, 0.95),
        }

    return {
        "sample_count": len(samples),
        "wall_elapsed_seconds": {
            "p50": _quantile(wall_elapsed, 0.50),
            "p95": _quantile(wall_elapsed, 0.95),
        },
        "elapsed_seconds": {
            "p50": _quantile(elapsed, 0.50),
            "p95": _quantile(elapsed, 0.95),
        },
        "rss_mb": {
            "p50": _quantile(rss, 0.50),
            "p95": _quantile(rss, 0.95),
        },
        "peak_gpu_memory_mb": {
            "p50": _quantile(peak_gpu_memory, 0.50),
            "p95": _quantile(peak_gpu_memory, 0.95),
        },
        "total_output_bytes": {
            "p50": _quantile(output_bytes, 0.50),
            "p95": _quantile(output_bytes, 0.95),
        },
        "stage_elapsed_seconds": stage_elapsed_summary,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")
    if _SAFE_RUN_ID_PATTERN.fullmatch(str(args.run_id_prefix)) is None:
        raise ValueError("--run-id-prefix must contain only letters, digits, dot, underscore, or hyphen")
    if args.config is not None and not args.resume and not _is_dependency_closed(args.stages):
        raise ValueError(
            "fresh benchmark stage selection is missing upstream dependencies; "
            "include tokenise/train_mask/train_decoder or use --resume"
        )

    samples: list[dict[str, Any]] = []
    if args.config is not None:
        config_path = args.config.expanduser().resolve(strict=True)
        for iteration in range(1, args.iterations + 1):
            samples.append(_execute_pipeline_iteration(config_path, args, iteration=iteration))
    else:
        output_root = args.existing_output_root.expanduser().resolve(strict=True)
        sample = _summarize_completed_run(
            output_root,
            fixture_type=str(args.fixture_type),
            iteration=1,
            rss_mb=None,
            invoked_argv=None,
        )
        sample["rss_unavailable_reason"] = (
            "existing-output-root mode summarizes stage manifests only; those manifests do not record peak RSS"
        )
        sample["peak_gpu_memory_mb"] = None
        sample["gpu_memory_unavailable_reason"] = "existing-output-root mode cannot reconstruct peak GPU memory"
        samples.append(sample)

    return {
        "ok": True,
        "benchmark_name": "perturbgen_pipeline",
        "fixture_type": args.fixture_type,
        "measurement_notice": (
            "Raw measurements only. Do not treat these numbers as a release "
            "threshold or biological claim without separately freezing the "
            "fixture, hardware, and stage selection."
        ),
        "selected_stages": list(args.stages),
        "selected_path": args.path,
        "runtime_metadata": _runtime_metadata(),
        "samples": samples,
        "summary": _build_summary(samples),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run_benchmark(args)
    except (FileNotFoundError, ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        error_payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(error_payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
