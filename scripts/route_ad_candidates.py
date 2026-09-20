#!/usr/bin/env python3
"""Route the frozen AD candidates onto the §6 exploratory / engineering split.

Consumes the existing DAVF axis-coverage audit, optional observed DEG table,
public perturbation inventory, GRN support table, external-evidence payloads
and the APOE anchor backtest. Writes candidates, semantic contexts, a routing
report and exploratory sidecars. Never emits a PerturbGen invocation and never
overwrites the registered GEARS/Geneformer fail assets.
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

from src.analysis.ad_candidate_routing import (  # noqa: E402
    ADCandidateRoutingError,
    load_axis_audit,
    load_grn_support,
    load_public_perturbation_inventory,
    parse_anchor_verdict,
    route_frozen_ad_candidates,
    summarize_observed_gate,
    write_routing_outputs,
)
from src.analysis.ad_research_decision import (  # noqa: E402
    FROZEN_AD_RESEARCH_DECISION,
    PUBLIC_PERTURBATION_OUT_OF_SCOPE,
    load_ad_research_decision,
)
from src.analysis.ptm_gene_score import PTMGeneScoreError, load_deg_table  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--axis-audit-tsv",
        type=Path,
        required=True,
        help="davf_axis_coverage_audit.tsv from scripts/audit_davf_axis_coverage.py",
    )
    parser.add_argument(
        "--deg-table",
        type=Path,
        default=None,
        help="donor-level AD DEG aggregate TSV; omitted observed evidence is inconclusive",
    )
    parser.add_argument(
        "--public-inventory-tsv",
        type=Path,
        default=None,
        help="optional public perturbation inventory TSV (target-specific KO/KD rows). "
        "The frozen 2026-09-18 decision refuses this flag.",
    )
    parser.add_argument(
        "--grn-support-tsv",
        type=Path,
        default=None,
        help="optional GRN support TSV with has_tf_support/has_direct_path",
    )
    parser.add_argument(
        "--external-evidence-json",
        type=Path,
        action="append",
        default=[],
        help="optional external-perturbation-evidence/v1 payload; repeatable",
    )
    parser.add_argument(
        "--anchor-backtest",
        type=Path,
        default=None,
        help="optional apoe_anchor_backtest.json; fail keeps assets supplementary-only",
    )
    parser.add_argument(
        "--contract-to-apoe-ko",
        action="store_true",
        help="B6 contraction: keep only APOE KO engineering; do not claim five-candidate coverage",
    )
    parser.add_argument(
        "--research-decision-json",
        type=Path,
        default=None,
        help="optional ptm2cellnet.ad-research-decision/v1 JSON; default is the frozen 2026-09-18 decision",
    )
    parser.add_argument(
        "--research-objective", default="association", choices=("association", "replication", "reversal")
    )
    parser.add_argument("--observed-cohort", default="GSE174367")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict:
    payload = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ADCandidateRoutingError(f"JSON root must be an object: {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        axis_coverage = load_axis_audit(args.axis_audit_tsv)
        decision = (
            load_ad_research_decision(args.research_decision_json)
            if args.research_decision_json is not None
            else FROZEN_AD_RESEARCH_DECISION
        )
        if (
            args.public_inventory_tsv is not None
            and decision.public_perturbation_policy == PUBLIC_PERTURBATION_OUT_OF_SCOPE
        ):
            raise ADCandidateRoutingError(
                "public Perturb-seq inventory is out of scope under the frozen 2026-09-18 decision"
            )
        deg_frame = load_deg_table(args.deg_table) if args.deg_table is not None else None
        observed = summarize_observed_gate(deg_frame, max_fdr=0.05, admission_rule=decision.observed_admission_rule)
        inventory = load_public_perturbation_inventory(args.public_inventory_tsv)
        grn_support = load_grn_support(args.grn_support_tsv)
        assets = [_load_json(path) for path in args.external_evidence_json]
        anchor = _load_json(args.anchor_backtest) if args.anchor_backtest is not None else None
        report = route_frozen_ad_candidates(
            axis_coverage=axis_coverage,
            observed=observed,
            inventory=inventory,
            grn_support=grn_support,
            external_assets=assets,
            anchor_verdict=parse_anchor_verdict(anchor),
            contract_to_apoe_ko=args.contract_to_apoe_ko,
            research_objective=args.research_objective,
            observed_cohort=args.observed_cohort,
            decision=decision,
            sources={
                "axis_audit": str(args.axis_audit_tsv.expanduser().resolve(strict=True)),
                "deg_table": "not_provided"
                if args.deg_table is None
                else str(args.deg_table.expanduser().resolve(strict=True)),
                "public_inventory": "not_provided"
                if args.public_inventory_tsv is None
                else str(args.public_inventory_tsv.expanduser().resolve(strict=True)),
                "anchor_backtest": "not_provided"
                if args.anchor_backtest is None
                else str(args.anchor_backtest.expanduser().resolve(strict=True)),
                "research_decision": "frozen_2026_09_18"
                if args.research_decision_json is None
                else str(args.research_decision_json.expanduser().resolve(strict=True)),
            },
        )
        outputs = write_routing_outputs(report, args.output_dir)
    except (ADCandidateRoutingError, PTMGeneScoreError, FileNotFoundError, ValueError) as exc:
        print(f"[route_ad_candidates] 分流失败：{exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "n_formal_invocations": 0,
                "may_enter_lineage": False,
                "biology_pass": False,
                "coverage_claim": report.coverage_claim,
                "observed_admission_rule": decision.observed_admission_rule,
                "kd_policy": decision.kd_policy,
                "outputs": {key: str(path) for key, path in outputs.items()},
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
