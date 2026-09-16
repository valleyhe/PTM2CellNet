#!/usr/bin/env python3
"""Assemble the Gate-E vocabulary-migration benchmark CSV from local PTM databases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.gate_e_benchmark import (  # noqa: E402
    GateEBenchmarkBuildError,
    build_gate_e_benchmark,
    write_gate_e_benchmark,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cplm-file",
        type=Path,
        default=PROJECT_ROOT / "data/raw/cplm/Homo sapiens.txt",
    )
    parser.add_argument(
        "--dbptm-phosphorylation",
        type=Path,
        default=PROJECT_ROOT / "data/raw/dbptm/Phosphorylation",
        help="dbptm phosphorylation table; UniProt accessions are resolved via the CPLM symbol pairs",
    )
    parser.add_argument(
        "--ensembl-mapping",
        type=Path,
        default=PROJECT_ROOT / "ref/Perturbgen-src/perturbgen/pp/ensembl_mapping_dict_gc95M.pkl",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        frame, manifest = build_gate_e_benchmark(
            cplm_path=args.cplm_file,
            dbptm_phosphorylation_path=args.dbptm_phosphorylation,
            ensembl_mapping_path=args.ensembl_mapping,
        )
        write_gate_e_benchmark(frame, manifest, args.output_csv, args.manifest_output)
    except (GateEBenchmarkBuildError, FileNotFoundError) as exc:
        print(f"[build_gate_e_benchmark] 组装失败：{exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output_csv),
                "rows": manifest["benchmark_rows"],
                "ptm_types": manifest["ptm_type_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
