#!/usr/bin/env python3
"""从版本化 JSON 重放 PerturbGen 双路径评估并产出报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.perturbgen.replay_evaluation import (  # noqa: E402,F401
    INPUT_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    _extract_run,
    replay_dual_path_evaluation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    input_path = args.input_json.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output_dir already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = json.loads(input_path.read_text(encoding="utf-8"))
    replay_dual_path_evaluation(spec, input_path=input_path, output_dir=output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
