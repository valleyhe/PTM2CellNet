#!/usr/bin/env python3
"""Run the reproducible PMADS + Ridge baseline.

The command consumes a user-provided CSV/TSV/Parquet file.  It never generates
synthetic PMADS rows implicitly; use a checked-in test fixture only for CI
smoke tests and mark that artifact with ``--demo-data``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.pmads_ridge import run_pmads_ridge, save_baseline_artifact  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="PMADS-compatible CSV/TSV/Parquet input")
    parser.add_argument("--target", default="label", help="target column (default: label)")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None, help="optional data manifest to link in provenance")
    parser.add_argument("--group-col", default=None, help="optional protein/accession group column")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--strict-quality", action="store_true")
    parser.add_argument("--demo-data", action="store_true", help="mark output as engineering-only fixture evidence")
    return parser


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"baseline input does not exist: {path}")
    suffixes = {suffix.lower() for suffix in path.suffixes}
    if ".parquet" in suffixes:
        return pd.read_parquet(path)
    if ".tsv" in suffixes or ".txt" in suffixes:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        frame = _read_frame(args.input)
        result = run_pmads_ridge(
            frame,
            target_col=args.target,
            task=args.task,
            alpha=args.alpha,
            test_size=args.test_size,
            validation_size=args.validation_size,
            seed=args.seed,
            group_col=args.group_col,
            strict_quality=args.strict_quality,
        )
        paths = save_baseline_artifact(
            result,
            args.output_dir,
            input_path=args.input,
            manifest_path=args.manifest,
            parameters={
                "target": args.target,
                "task": args.task,
                "alpha": args.alpha,
                "seed": args.seed,
                "test_size": args.test_size,
                "validation_size": args.validation_size,
                "group_col": args.group_col,
                "strict_quality": args.strict_quality,
            },
            is_demo_data=args.demo_data,
        )
    except (FileNotFoundError, ValueError, TypeError, OSError, ImportError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "ok": True,
                "metrics": result["metrics"],
                "artifacts": paths,
                "split_sizes": {name: int(len(result["split"][name])) for name in ("train", "validation", "test")},
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
