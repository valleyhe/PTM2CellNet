#!/usr/bin/env python3
"""Unified formal Workflow A acceptance verification (Gate-4/5).

Consumes one E2E gate report plus the formal evidence artifacts (frozen
cohort manifest, matched-null distribution manifests, unperturbed quality,
dual-path eval input and report manifest) and renders a single reviewable
verdict.  Missing artifacts surface as ``blocked`` with an explicit
``missing_inputs`` list — they are never silently skipped, and a blocked or
failed verdict is never rendered as a pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.integration.perturbgen.formal_verification import verify_formal_workflow_a

_EXIT_PASS = 0
_EXIT_FAIL = 1
_EXIT_BLOCKED = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2e-report", type=Path, required=True, help="E2E gate report JSON (davf_perturbgen_e2e/v1).")
    parser.add_argument(
        "--frozen-manifest",
        type=Path,
        default=None,
        help="Frozen cohort manifest produced by run_frozen_acceptance.py --freeze.",
    )
    parser.add_argument(
        "--null-distribution-manifest",
        type=Path,
        action="append",
        default=None,
        help="Matched-null distribution manifest (perturbgen_null_distribution/v1); repeatable.",
    )
    parser.add_argument(
        "--quality-json",
        type=Path,
        default=None,
        help="Unperturbed quality payload from extract_unperturbed_quality_from_h5ad; "
        "omit to read quality from --eval-input candidates instead.",
    )
    parser.add_argument("--eval-input", type=Path, default=None, help="Dual-path eval input JSON.")
    parser.add_argument("--report-manifest", type=Path, default=None, help="Dual-path report manifest JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Verification report output path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = verify_formal_workflow_a(
        e2e_report=args.e2e_report,
        frozen_manifest=args.frozen_manifest,
        null_distribution_manifests=args.null_distribution_manifest or (),
        quality_payload=args.quality_json,
        eval_input=args.eval_input,
        report_manifest=args.report_manifest,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "verdict": payload["verdict"]}, ensure_ascii=False))
    if payload["verdict"] == "fail":
        return _EXIT_FAIL
    if payload["verdict"] == "blocked":
        return _EXIT_BLOCKED
    return _EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
