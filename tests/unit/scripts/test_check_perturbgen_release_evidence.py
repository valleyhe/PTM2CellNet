import json
from pathlib import Path
import time

from scripts.check_perturbgen_release_evidence import build_parser, check_release_evidence


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _benchmark_payload(*, fixture_type: str = "real", stages: dict[str, str], h5ad: list[Path]) -> dict:
    return {
        "ok": True,
        "fixture_type": fixture_type,
        "samples": [
            {
                "output_root": str(h5ad[0].parent),
                "stage_status": stages,
                "perturb_output_h5ad": [str(path) for path in h5ad],
                "gpu_memory_sample_count": 4,
            }
        ],
        "summary": {
            "sample_count": 1,
            "wall_elapsed_seconds": {"p50": 10.0, "p95": 10.0},
            "rss_mb": {"p50": 512.0, "p95": 512.0},
            "peak_gpu_memory_mb": {"p50": 2048.0, "p95": 2048.0},
            "total_output_bytes": {"p50": 4096.0, "p95": 4096.0},
        },
    }


def _evidence_payload(*, benchmark_json: Path, validated_h5ad: list[Path], age_seconds: int = 0) -> dict:
    return {
        "test": "test_real_perturbgen_smoke",
        "asset": "config=/tmp/real.yaml",
        "outcome": "pass",
        "duration_s": 12.3,
        "timestamp_unix": int(time.time()) - age_seconds,
        "extra": {
            "benchmark_json": str(benchmark_json),
            "validated_h5ad": [str(path) for path in validated_h5ad],
        },
    }


def test_release_gate_formal_passes_with_dual_path_outputs(tmp_path):
    src_h5ad = tmp_path / "source_intervention.h5ad"
    tgt_h5ad = tmp_path / "within_state.h5ad"
    src_h5ad.write_bytes(b"src")
    tgt_h5ad.write_bytes(b"tgt")
    benchmark = _write_json(
        tmp_path / "benchmark.json",
        _benchmark_payload(
            stages={"source_intervention": "success", "within_state": "success"},
            h5ad=[src_h5ad, tgt_h5ad],
        ),
    )
    evidence = _write_json(
        tmp_path / "evidence.json",
        _evidence_payload(benchmark_json=benchmark, validated_h5ad=[src_h5ad, tgt_h5ad]),
    )

    args = build_parser().parse_args(["--evidence", str(evidence)])
    payload = check_release_evidence(args)

    assert payload["ok"] is True
    assert payload["benchmark"]["fixture_type"] == "real"
    assert payload["benchmark"]["sample_count"] == 1
    assert payload["failures"] == []


def test_release_gate_fails_when_evidence_is_expired(tmp_path):
    src_h5ad = tmp_path / "source_intervention.h5ad"
    tgt_h5ad = tmp_path / "within_state.h5ad"
    src_h5ad.write_bytes(b"src")
    tgt_h5ad.write_bytes(b"tgt")
    benchmark = _write_json(
        tmp_path / "benchmark.json",
        _benchmark_payload(
            stages={"source_intervention": "success", "within_state": "success"},
            h5ad=[src_h5ad, tgt_h5ad],
        ),
    )
    evidence = _write_json(
        tmp_path / "evidence.json",
        _evidence_payload(
            benchmark_json=benchmark,
            validated_h5ad=[src_h5ad, tgt_h5ad],
            age_seconds=60 * 60 * 49,
        ),
    )

    args = build_parser().parse_args(["--evidence", str(evidence), "--max-age-hours", "48"])
    payload = check_release_evidence(args)

    assert payload["ok"] is False
    assert any(item["code"] == "evidence_expired" for item in payload["failures"])


def test_release_gate_fails_when_benchmark_fixture_is_not_real(tmp_path):
    src_h5ad = tmp_path / "source_intervention.h5ad"
    tgt_h5ad = tmp_path / "within_state.h5ad"
    src_h5ad.write_bytes(b"src")
    tgt_h5ad.write_bytes(b"tgt")
    benchmark = _write_json(
        tmp_path / "benchmark.json",
        _benchmark_payload(
            fixture_type="engineering",
            stages={"source_intervention": "success", "within_state": "success"},
            h5ad=[src_h5ad, tgt_h5ad],
        ),
    )
    evidence = _write_json(
        tmp_path / "evidence.json",
        _evidence_payload(benchmark_json=benchmark, validated_h5ad=[src_h5ad, tgt_h5ad]),
    )

    args = build_parser().parse_args(["--evidence", str(evidence)])
    payload = check_release_evidence(args)

    assert payload["ok"] is False
    assert any(item["code"] == "benchmark_not_real_fixture" for item in payload["failures"])


def test_release_gate_smoke_mode_allows_single_validated_h5ad(tmp_path):
    src_h5ad = tmp_path / "source_intervention.h5ad"
    src_h5ad.write_bytes(b"src")
    benchmark = _write_json(
        tmp_path / "benchmark.json",
        _benchmark_payload(
            stages={"source_intervention": "success"},
            h5ad=[src_h5ad],
        ),
    )
    evidence = _write_json(
        tmp_path / "evidence.json",
        _evidence_payload(benchmark_json=benchmark, validated_h5ad=[src_h5ad]),
    )

    args = build_parser().parse_args(["--evidence", str(evidence), "--mode", "smoke"])
    payload = check_release_evidence(args)

    assert payload["ok"] is True
    assert payload["mode"] == "smoke"


def test_release_gate_formal_rejects_missing_resource_metrics(tmp_path):
    src_h5ad = tmp_path / "source_intervention.h5ad"
    tgt_h5ad = tmp_path / "within_state.h5ad"
    src_h5ad.write_bytes(b"src")
    tgt_h5ad.write_bytes(b"tgt")
    benchmark_payload = _benchmark_payload(
        stages={"source_intervention": "success", "within_state": "success"},
        h5ad=[src_h5ad, tgt_h5ad],
    )
    benchmark_payload["summary"]["peak_gpu_memory_mb"] = {"p50": None, "p95": None}
    benchmark_payload["samples"][0]["gpu_memory_sample_count"] = 0
    benchmark = _write_json(tmp_path / "benchmark.json", benchmark_payload)
    evidence = _write_json(
        tmp_path / "evidence.json",
        _evidence_payload(
            benchmark_json=benchmark,
            validated_h5ad=[src_h5ad, tgt_h5ad],
        ),
    )

    payload = check_release_evidence(build_parser().parse_args(["--evidence", str(evidence)]))

    assert payload["ok"] is False
    assert any(
        item["code"] in {"benchmark_metric_invalid", "gpu_memory_samples_missing"} for item in payload["failures"]
    )


def test_release_gate_formal_rejects_non_finite_metric(tmp_path):
    src_h5ad = tmp_path / "source_intervention.h5ad"
    tgt_h5ad = tmp_path / "within_state.h5ad"
    src_h5ad.write_bytes(b"src")
    tgt_h5ad.write_bytes(b"tgt")
    benchmark_payload = _benchmark_payload(
        stages={"source_intervention": "success", "within_state": "success"},
        h5ad=[src_h5ad, tgt_h5ad],
    )
    benchmark_payload["summary"]["rss_mb"]["p95"] = float("nan")
    benchmark = _write_json(tmp_path / "benchmark.json", benchmark_payload)
    evidence = _write_json(
        tmp_path / "evidence.json",
        _evidence_payload(benchmark_json=benchmark, validated_h5ad=[src_h5ad, tgt_h5ad]),
    )

    payload = check_release_evidence(build_parser().parse_args(["--evidence", str(evidence)]))

    assert payload["ok"] is False
    assert any(item["code"] == "benchmark_metric_invalid" for item in payload["failures"])
