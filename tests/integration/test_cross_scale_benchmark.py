"""Lightweight performance-script smoke test."""

import argparse

from scripts.benchmark_cross_scale import run_benchmark


def test_cross_scale_benchmark_reports_latency_and_shapes():
    report = run_benchmark(
        argparse.Namespace(
            batch_size=2,
            signal_nodes=8,
            cell_genes=12,
            iterations=2,
            warmup=1,
            seed=42,
            threads=1,
        )
    )
    assert report["benchmark_kind"] == "engineering_fixture_only"
    assert report["latency_ms_mean"] >= 0.0
    assert report["output_shapes"]["sensitivity_matrix"] == [8, 8]
