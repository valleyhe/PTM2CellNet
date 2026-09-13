#!/usr/bin/env python3
"""Evaluate Gate-E: gene-vocabulary migration and DAVF non-inferiority.

Implements the scientific quality gate of the integration proposal §5.4
that had no code before (analysis 2026-09-10 §4.1 N-3).  The benchmark is a
frozen CSV of at least 200 traceable PTM→gene samples; vocabulary migration
checks gene coverage, token collisions and PTM action-code invariance; DAVF
non-inferiority bootstraps the paired direction-accuracy difference between
the frozen old baseline and the new vocabulary runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.perturbgen.gate_e import (  # noqa: E402
    DEFAULT_NONINFERIORITY_MARGIN,
    MIN_BENCHMARK_SAMPLES,
    build_gate_e_report,
    evaluate_davf_noninferiority,
    evaluate_vocabulary_migration,
    load_benchmark,
    load_paired_results,
    load_vocabulary,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--old-vocabulary", type=Path)
    parser.add_argument("--new-vocabulary", type=Path)
    parser.add_argument("--davf-paired-results", type=Path)
    parser.add_argument("--intervention-type", choices=("KO", "KD", "OE"), default="KO")
    parser.add_argument("--min-samples", type=int, default=MIN_BENCHMARK_SAMPLES)
    parser.add_argument("--noninferiority-margin", type=float, default=DEFAULT_NONINFERIORITY_MARGIN)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.old_vocabulary and not args.new_vocabulary:
        raise SystemExit("--old-vocabulary requires --new-vocabulary")
    if not args.old_vocabulary and not args.davf_paired_results:
        raise SystemExit("nothing to evaluate: provide vocabularies and/or --davf-paired-results")

    benchmark = load_benchmark(args.benchmark)
    benchmark.validate(min_samples=args.min_samples)

    vocabulary_metrics = None
    if args.old_vocabulary and args.new_vocabulary:
        vocabulary_metrics = evaluate_vocabulary_migration(
            benchmark,
            load_vocabulary(args.old_vocabulary),
            load_vocabulary(args.new_vocabulary),
            intervention_type=args.intervention_type,
        )

    davf_metrics = None
    if args.davf_paired_results:
        davf_metrics = evaluate_davf_noninferiority(
            load_paired_results(args.davf_paired_results),
            margin=args.noninferiority_margin,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        )

    report = build_gate_e_report(
        benchmark=benchmark,
        vocabulary_metrics=vocabulary_metrics,
        davf_metrics=davf_metrics,
        thresholds={"min_benchmark_samples": float(args.min_samples)},
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "gate_e_passed": report["gate_e_passed"]}))
    return 0 if report["gate_e_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
