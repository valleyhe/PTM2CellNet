#!/usr/bin/env python3
"""Assemble a dual-path evaluation input JSON from a DAVF→PerturbGen E2E report.

Closes gap N-5: the E2E report (``davf_perturbgen_e2e/v1``) and the replay
evaluator (``perturbgen_dual_path_eval/v1``) used to be bridged by hand.
The perturb h5ad paths are read exclusively from the successful perturb
stage manifests; the DEG table, null distribution, candidate p-values and
unperturbed quality status are explicit inputs because the E2E report does
not carry them.
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

from src.integration.perturbgen.eval_assembly import (  # noqa: E402
    build_eval_input_payload,
    load_candidate_pvalues,
    load_e2e_report,
    write_eval_input,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2e-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deg-table", type=Path, required=True)
    null_group = parser.add_mutually_exclusive_group(required=True)
    null_group.add_argument("--null-distribution", type=Path)
    null_group.add_argument("--null-distribution-manifest", type=Path)
    parser.add_argument(
        "--unperturbed-quality-status",
        choices=("pass", "fail", "inconclusive"),
        default=None,
        help="hand-filled quality status; engineering only. Formal mode requires --unperturbed-quality",
    )
    parser.add_argument(
        "--unperturbed-quality",
        type=Path,
        help="JSON produced by extract_unperturbed_quality_from_h5ad (required for --evaluation-mode formal)",
    )
    parser.add_argument("--candidate-pvalues", type=Path, help="CSV of ensembl_id,pvalue")
    parser.add_argument("--uniform-candidate-pvalue", type=float)
    parser.add_argument(
        "--evaluation-mode",
        choices=("engineering", "formal"),
        default="engineering",
        help="formal rejects uniform/external p-values and hand-filled quality",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--donor-obs-column", default="donor")
    parser.add_argument("--var-gene-column", default="__index__")
    parser.add_argument("--deg-donor-column", default="donor")
    parser.add_argument("--deg-gene-column", default="gene_symbol")
    parser.add_argument("--deg-effect-column", default="log2fc")
    parser.add_argument("--deg-fdr-column", default="fdr")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.candidate_pvalues and args.uniform_candidate_pvalue is not None:
        raise SystemExit("provide either --candidate-pvalues or --uniform-candidate-pvalue, not both")
    quality_payload = None
    if args.unperturbed_quality is not None:
        loaded = json.loads(args.unperturbed_quality.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise SystemExit("--unperturbed-quality must be a JSON object")
        quality_payload = loaded
    if args.evaluation_mode == "formal":
        if args.uniform_candidate_pvalue is not None or args.candidate_pvalues is not None:
            raise SystemExit(
                "formal evaluation-mode rejects --uniform-candidate-pvalue and --candidate-pvalues; "
                "q_value is aggregated from empirical null runs after extraction"
            )
        if quality_payload is None:
            raise SystemExit("formal evaluation-mode requires --unperturbed-quality JSON from h5ad extraction")
    elif args.unperturbed_quality_status is None and quality_payload is None:
        raise SystemExit("engineering evaluation-mode requires --unperturbed-quality-status or --unperturbed-quality")
    report = load_e2e_report(args.e2e_report)
    payload = build_eval_input_payload(
        report,
        deg_table_path=args.deg_table,
        null_distribution_path=args.null_distribution,
        null_distribution_manifest_path=args.null_distribution_manifest,
        unperturbed_quality_status=args.unperturbed_quality_status or "inconclusive",
        candidate_pvalues=(load_candidate_pvalues(args.candidate_pvalues) if args.candidate_pvalues else None),
        uniform_candidate_pvalue=args.uniform_candidate_pvalue,
        evaluation_mode=args.evaluation_mode,
        unperturbed_quality=quality_payload,
        run_id=args.run_id or args.e2e_report.stem,
        donor_obs_column=args.donor_obs_column,
        var_gene_column=args.var_gene_column,
        deg_donor_column=args.deg_donor_column,
        deg_gene_column=args.deg_gene_column,
        deg_effect_column=args.deg_effect_column,
        deg_fdr_column=args.deg_fdr_column,
    )
    output = write_eval_input(payload, args.output)
    print(
        json.dumps(
            {"output": str(output), "candidates": len(payload["candidates"])},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
