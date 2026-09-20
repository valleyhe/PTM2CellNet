#!/usr/bin/env python3
"""Write a deterministic PTM-activity smoke bundle and optionally run stages 1/3/4/5.

This is an engineering fixture: it does not run KSTAR, does not use the frozen
OmniPath TF-only network, and must not be reported as biology PASS.
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

from src.analysis.ptm_smoke import write_and_run  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="run stages 1/3/4/5 on the smoke bundle (skip stage 2 / KSTAR)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = write_and_run(args.output_dir, run_pipeline=args.run_pipeline)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"[generate_ptm_smoke] 失败：{exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
