#!/usr/bin/env python3
"""Measure a small CPU cross-scale forward pass and emit JSON metrics.

This benchmark is an engineering smoke measurement for the explicit graph and
decoder contracts. It uses deterministic in-memory fixture embeddings and must
not be interpreted as a biological or production-capacity benchmark.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cross_scale import CrossScaleConfig, CrossScalePTM2CellNet  # noqa: E402


def _chain(num_nodes: int) -> torch.Tensor:
    source = torch.arange(num_nodes - 1, dtype=torch.long)
    destination = source + 1
    return torch.stack([torch.cat([source, destination]), torch.cat([destination, source])])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--signal-nodes", type=int, default=32)
    parser.add_argument("--cell-genes", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _rss_mb() -> float:
    # Linux reports KiB; macOS reports bytes. The execution environment is
    # Linux, but the branch keeps the output interpretable elsewhere.
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(usage / 1024.0 if sys.platform != "darwin" else usage / (1024.0 * 1024.0))


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if min(args.batch_size, args.signal_nodes, args.cell_genes, args.iterations, args.warmup) < 0:
        raise ValueError("benchmark dimensions and iteration counts cannot be negative")
    if args.batch_size == 0 or args.signal_nodes < 2 or args.cell_genes < 2 or args.iterations == 0:
        raise ValueError("batch-size, graph sizes and iterations must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    config = CrossScaleConfig(
        protein_dim=16,
        signal_input_dim=16,
        signal_hidden_dim=24,
        signal_output_dim=16,
        cell_gene_feature_dim=8,
        cell_hidden_dim=24,
        num_cell_genes=args.cell_genes,
        num_cell_states=4,
        num_ptm_types=4,
        max_position=max(args.signal_nodes + 2, 64),
        dropout=0.0,
        signal_layers=2,
    )
    model = CrossScalePTM2CellNet(
        config=config,
        plm_backbone_dims={"ankh39": 11, "esm2": 13, "prott5": 17},
        data_manifest="data/manifests/datasets.yaml",
        signal_graph_version="benchmark-fixture-v1",
    ).eval()
    batch = {
        "ankh39_embeddings": torch.randn(args.batch_size, args.signal_nodes, 11),
        "esm2_embeddings": torch.randn(args.batch_size, args.signal_nodes, 13),
        "prott5_embeddings": torch.randn(args.batch_size, args.signal_nodes, 17),
        "signal_edge_index": _chain(args.signal_nodes),
        "signal_gene_map": torch.ones(args.signal_nodes, args.cell_genes),
        "cell_edge_index": _chain(args.cell_genes),
        "cell_gene_features": torch.randn(args.batch_size, args.cell_genes, 8),
        "gene_mask": torch.ones(args.batch_size, args.cell_genes),
    }

    with torch.no_grad():
        for _ in range(args.warmup):
            model(batch)
        latencies: list[float] = []
        before_rss = _rss_mb()
        for _ in range(args.iterations):
            start = time.perf_counter()
            output = model(batch)
            latencies.append((time.perf_counter() - start) * 1000.0)
        after_rss = _rss_mb()

    latency_tensor = torch.tensor(latencies, dtype=torch.float64)
    return {
        "benchmark_kind": "engineering_fixture_only",
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "seed": args.seed,
        "threads": args.threads,
        "batch_size": args.batch_size,
        "signal_nodes": args.signal_nodes,
        "cell_genes": args.cell_genes,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "latency_ms_mean": float(latency_tensor.mean()),
        "latency_ms_p50": float(torch.quantile(latency_tensor, 0.50)),
        "latency_ms_p95": float(torch.quantile(latency_tensor, 0.95)),
        "throughput_samples_per_second": float(args.batch_size / (latency_tensor.mean().item() / 1000.0)),
        "rss_mb_before": before_rss,
        "rss_mb_after": after_rss,
        "rss_mb_delta": max(0.0, after_rss - before_rss),
        "output_shapes": {
            "delta_expression": list(output["delta_expression"].shape),
            "cell_state_logits": list(output["cell_state_logits"].shape),
            "sensitivity_matrix": list(output["sensitivity_matrix"].shape),
        },
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_benchmark(args)
    except (ValueError, RuntimeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
