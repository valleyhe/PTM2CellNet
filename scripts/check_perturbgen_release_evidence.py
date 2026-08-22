#!/usr/bin/env python3
"""Hard release gate for PerturbGen M5 real-assets evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


FORMAL_REQUIRED_PATHS = ("source_intervention", "within_state")
DEFAULT_MAX_AGE_HOURS = 48
_FUTURE_SKEW_SECONDS = 300


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        required=True,
        help="JSON evidence emitted by tests/real_assets/test_real_perturbgen_smoke.py.",
    )
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        default=None,
        help="Optional explicit benchmark JSON path. Defaults to evidence.extra.benchmark_json.",
    )
    parser.add_argument(
        "--mode",
        choices=("formal", "smoke"),
        default="formal",
        help="formal requires dual-path evidence; smoke only requires one validated real .h5ad.",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=DEFAULT_MAX_AGE_HOURS,
        help="Maximum allowed evidence age in hours.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for the rendered gate report.",
    )
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _resolve_path(value: str | Path | None, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (base_dir / raw).resolve()


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _record_failure(failures: list[dict[str, str]], code: str, detail: str) -> None:
    failures.append({"code": code, "detail": detail})


def _require_positive_quantiles(
    summary: dict[str, Any],
    metric: str,
    failures: list[dict[str, str]],
) -> None:
    payload = summary.get(metric)
    if not isinstance(payload, dict):
        _record_failure(
            failures,
            "benchmark_metric_missing",
            f"benchmark summary missing {metric} quantiles",
        )
        return
    for quantile in ("p50", "p95"):
        value = payload.get(quantile)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            _record_failure(
                failures,
                "benchmark_metric_invalid",
                f"benchmark summary {metric}.{quantile} must be a positive number",
            )


def check_release_evidence(args: argparse.Namespace) -> dict[str, Any]:
    checked_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    failures: list[dict[str, str]] = []
    evidence_path = args.evidence.expanduser().resolve()
    benchmark_path = None
    evidence_payload: dict[str, Any] = {}
    benchmark_payload: dict[str, Any] = {}

    if not math.isfinite(float(args.max_age_hours)) or args.max_age_hours <= 0:
        _record_failure(
            failures,
            "max_age_hours_invalid",
            "max_age_hours must be a positive finite number",
        )

    try:
        evidence_payload = _load_json(evidence_path)
    except ValueError as exc:
        _record_failure(failures, "evidence_missing_or_invalid", str(exc))
    else:
        if evidence_payload.get("test") != "test_real_perturbgen_smoke":
            _record_failure(
                failures,
                "evidence_test_mismatch",
                "evidence.test must equal test_real_perturbgen_smoke",
            )
        outcome = evidence_payload.get("outcome")
        if outcome != "pass":
            _record_failure(
                failures,
                "evidence_outcome_not_pass",
                f"expected outcome=pass, got {outcome!r}",
            )

        timestamp_unix = evidence_payload.get("timestamp_unix")
        now_ts = datetime.now(timezone.utc).timestamp()
        age_hours = None
        if not isinstance(timestamp_unix, int):
            _record_failure(
                failures,
                "evidence_timestamp_missing",
                "timestamp_unix must be an integer unix timestamp",
            )
        else:
            age_seconds = now_ts - float(timestamp_unix)
            age_hours = round(age_seconds / 3600.0, 3)
            if age_seconds < -_FUTURE_SKEW_SECONDS:
                _record_failure(
                    failures,
                    "evidence_timestamp_future",
                    f"timestamp_unix={timestamp_unix} is unexpectedly in the future",
                )
            elif age_seconds > float(args.max_age_hours) * 3600.0:
                _record_failure(
                    failures,
                    "evidence_expired",
                    f"evidence age {age_hours}h exceeds limit {args.max_age_hours}h",
                )
        evidence_payload["_age_hours"] = age_hours

        extra = evidence_payload.get("extra")
        if not isinstance(extra, dict):
            _record_failure(
                failures,
                "evidence_extra_missing",
                "evidence.extra must be a JSON object",
            )
            extra = {}

        benchmark_path = args.benchmark_json
        if benchmark_path is None:
            benchmark_path = _resolve_path(
                extra.get("benchmark_json"),
                base_dir=evidence_path.parent,
            )
        else:
            benchmark_path = benchmark_path.expanduser().resolve()
        if benchmark_path is None:
            _record_failure(
                failures,
                "benchmark_path_missing",
                "benchmark JSON path not provided and evidence.extra.benchmark_json missing",
            )

        validated_h5ad = [
            _resolve_path(item, base_dir=evidence_path.parent) for item in _as_str_list(extra.get("validated_h5ad"))
        ]
        validated_h5ad = [path for path in validated_h5ad if path is not None]
        if not validated_h5ad:
            _record_failure(
                failures,
                "validated_h5ad_missing",
                "evidence.extra.validated_h5ad must contain at least one real .h5ad path",
            )
        for path in validated_h5ad:
            if path.suffix != ".h5ad":
                _record_failure(
                    failures,
                    "validated_h5ad_invalid_suffix",
                    f"validated artifact is not .h5ad: {path}",
                )
            elif not path.is_file():
                _record_failure(
                    failures,
                    "validated_h5ad_missing_file",
                    f"validated .h5ad not found: {path}",
                )

        if args.mode == "formal" and len(validated_h5ad) < 2:
            _record_failure(
                failures,
                "dual_path_h5ad_required",
                "formal mode requires at least two validated real .h5ad outputs",
            )

    if benchmark_path is not None:
        try:
            benchmark_payload = _load_json(benchmark_path)
        except ValueError as exc:
            _record_failure(failures, "benchmark_missing_or_invalid", str(exc))
        else:
            if benchmark_payload.get("ok") is not True:
                _record_failure(
                    failures,
                    "benchmark_not_ok",
                    f"expected benchmark ok=true, got {benchmark_payload.get('ok')!r}",
                )
            if benchmark_payload.get("fixture_type") != "real":
                _record_failure(
                    failures,
                    "benchmark_not_real_fixture",
                    "benchmark fixture_type must be 'real'",
                )

            samples = benchmark_payload.get("samples")
            if not isinstance(samples, list) or not samples:
                _record_failure(
                    failures,
                    "benchmark_samples_missing",
                    "benchmark JSON must contain at least one sample",
                )
                samples = []

            summary = benchmark_payload.get("summary")
            if not isinstance(summary, dict):
                _record_failure(
                    failures,
                    "benchmark_summary_missing",
                    "benchmark summary must be a JSON object",
                )
                summary = {}
            if args.mode == "formal":
                for metric in (
                    "wall_elapsed_seconds",
                    "rss_mb",
                    "peak_gpu_memory_mb",
                    "total_output_bytes",
                ):
                    _require_positive_quantiles(summary, metric, failures)

            successful_stage_names: set[str] = set()
            benchmark_h5ad_paths: set[Path] = set()
            sample_audit: list[dict[str, Any]] = []
            for index, sample in enumerate(samples):
                if not isinstance(sample, dict):
                    _record_failure(
                        failures,
                        "benchmark_sample_invalid",
                        f"sample[{index}] must be a JSON object",
                    )
                    continue
                stage_status = sample.get("stage_status")
                if not isinstance(stage_status, dict) or not stage_status:
                    _record_failure(
                        failures,
                        "benchmark_stage_status_missing",
                        f"sample[{index}] missing non-empty stage_status",
                    )
                    continue
                non_success = {
                    stage: status
                    for stage, status in stage_status.items()
                    if not isinstance(stage, str) or status != "success"
                }
                if non_success:
                    _record_failure(
                        failures,
                        "benchmark_stage_not_success",
                        f"sample[{index}] has non-success stage status: {non_success}",
                    )
                successful_stage_names.update(
                    stage for stage, status in stage_status.items() if isinstance(stage, str) and status == "success"
                )
                if args.mode == "formal":
                    sample_success = {
                        stage
                        for stage, status in stage_status.items()
                        if isinstance(stage, str) and status == "success"
                    }
                    missing_sample_paths = sorted(set(FORMAL_REQUIRED_PATHS) - sample_success)
                    if missing_sample_paths:
                        _record_failure(
                            failures,
                            "dual_path_stage_missing_in_sample",
                            f"sample[{index}] missing successful stages {missing_sample_paths}",
                        )

                sample_h5ad = [
                    _resolve_path(item, base_dir=benchmark_path.parent)
                    for item in _as_str_list(sample.get("perturb_output_h5ad"))
                ]
                sample_h5ad = [path for path in sample_h5ad if path is not None]
                if not sample_h5ad:
                    _record_failure(
                        failures,
                        "benchmark_h5ad_missing",
                        f"sample[{index}] missing perturb_output_h5ad entries",
                    )
                elif args.mode == "formal" and len(set(sample_h5ad)) < 2:
                    _record_failure(
                        failures,
                        "dual_path_h5ad_missing_in_sample",
                        f"sample[{index}] requires at least two distinct perturb h5ad outputs",
                    )
                for path in sample_h5ad:
                    benchmark_h5ad_paths.add(path)

                if args.mode == "formal":
                    gpu_sample_count = sample.get("gpu_memory_sample_count")
                    if (
                        not isinstance(gpu_sample_count, int)
                        or isinstance(gpu_sample_count, bool)
                        or gpu_sample_count <= 0
                    ):
                        _record_failure(
                            failures,
                            "gpu_memory_samples_missing",
                            f"sample[{index}] must contain positive gpu_memory_sample_count",
                        )

                sample_audit.append(
                    {
                        "index": index,
                        "output_root": sample.get("output_root"),
                        "stage_status": stage_status,
                        "perturb_output_h5ad": [str(path) for path in sample_h5ad],
                    }
                )

            if args.mode == "formal":
                missing_paths = sorted(set(FORMAL_REQUIRED_PATHS) - successful_stage_names)
                if missing_paths:
                    _record_failure(
                        failures,
                        "dual_path_stage_missing",
                        f"formal mode requires successful stages {missing_paths}",
                    )

            validated_paths = set()
            extra = evidence_payload.get("extra", {})
            if isinstance(extra, dict):
                for item in _as_str_list(extra.get("validated_h5ad")):
                    path = _resolve_path(item, base_dir=evidence_path.parent)
                    if path is not None:
                        validated_paths.add(path)
            if validated_paths and benchmark_h5ad_paths:
                missing_from_benchmark = sorted(
                    str(path) for path in validated_paths if path not in benchmark_h5ad_paths
                )
                if missing_from_benchmark:
                    _record_failure(
                        failures,
                        "validated_h5ad_not_in_benchmark",
                        "validated .h5ad missing from benchmark sample outputs: " + ", ".join(missing_from_benchmark),
                    )
            benchmark_payload["_sample_audit"] = sample_audit

    report = {
        "ok": not failures,
        "gate": "perturbgen_m5_release_evidence",
        "mode": args.mode,
        "checked_at": checked_at,
        "max_age_hours": args.max_age_hours,
        "inputs": {
            "evidence": str(evidence_path),
            "benchmark_json": str(benchmark_path) if benchmark_path is not None else None,
        },
        "failures": failures,
        "evidence": {
            "test": evidence_payload.get("test"),
            "asset": evidence_payload.get("asset"),
            "outcome": evidence_payload.get("outcome"),
            "duration_s": evidence_payload.get("duration_s"),
            "timestamp_unix": evidence_payload.get("timestamp_unix"),
            "age_hours": evidence_payload.get("_age_hours"),
            "validated_h5ad": (
                evidence_payload.get("extra", {}).get("validated_h5ad")
                if isinstance(evidence_payload.get("extra"), dict)
                else None
            ),
        },
        "benchmark": {
            "ok": benchmark_payload.get("ok"),
            "fixture_type": benchmark_payload.get("fixture_type"),
            "sample_count": (
                len(benchmark_payload.get("samples", [])) if isinstance(benchmark_payload.get("samples"), list) else 0
            ),
            "summary": benchmark_payload.get("summary"),
            "samples": benchmark_payload.get("_sample_audit"),
        },
    }
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = check_release_evidence(args)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
