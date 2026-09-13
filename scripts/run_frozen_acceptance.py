#!/usr/bin/env python3
"""M6 frozen-cohort scientific acceptance orchestration (proposal §7.2 M6).

Three explicit modes:

* ``--freeze`` builds the frozen cohort manifest (cohort h5ad hash, explicit
  train/held-out donor split, candidate CSV, uniform seed/mode/null plan);
* ``--plan`` audits donor leakage and emits the acceptance run matrix;
* ``--verify`` checks a dual-path evaluation input against the frozen plan
  and independently replays the published verdicts.

The statistical kernel stays in ``results.py``/``dual_path.py``; this CLI
only freezes contracts and replays decisions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.perturbgen.frozen_cohort import (  # noqa: E402
    build_acceptance_plan,
    build_frozen_manifest,
    load_frozen_manifest,
    replay_verdicts,
    verify_eval_input_against_manifest,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_mutually_exclusive_group(required=True)
    sub.add_argument("--freeze", action="store_true")
    sub.add_argument("--plan", action="store_true")
    sub.add_argument("--verify", action="store_true")

    parser.add_argument("--cohort-h5ad", type=Path)
    parser.add_argument("--cell-type")
    parser.add_argument("--donor-obs-column", default="donor")
    parser.add_argument("--train-donors", help="comma-separated donor labels")
    parser.add_argument("--held-out-donors", help="comma-separated donor labels")
    parser.add_argument("--candidates-csv", type=Path)
    parser.add_argument("--modes", default="mask,pad,delete")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--matched-nulls", type=int, default=99)
    parser.add_argument("--source-config")

    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--eval-input", type=Path)
    parser.add_argument("--report-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _split_labels(raw: str | None) -> list[str]:
    if not raw:
        return []
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise SystemExit("donor lists must contain at least one label")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output is None:
        raise SystemExit("--output is required")

    if args.freeze:
        missing = [
            name
            for name, value in (
                ("--cohort-h5ad", args.cohort_h5ad),
                ("--cell-type", args.cell_type),
                ("--train-donors", args.train_donors),
                ("--held-out-donors", args.held_out_donors),
                ("--candidates-csv", args.candidates_csv),
            )
            if not value
        ]
        if missing:
            raise SystemExit(f"--freeze requires {', '.join(missing)}")
        manifest = build_frozen_manifest(
            cohort_h5ad=args.cohort_h5ad,
            cell_type=args.cell_type,
            donor_obs_column=args.donor_obs_column,
            train_donors=_split_labels(args.train_donors),
            held_out_donors=_split_labels(args.held_out_donors),
            candidates_csv=args.candidates_csv,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            modes=[item.strip() for item in args.modes.split(",") if item.strip()],
            seeds=[int(item) for item in args.seeds.split(",") if item.strip()],
            matched_nulls=args.matched_nulls,
            source_config=args.source_config,
        )
        payload = manifest.to_payload()
    elif args.plan:
        if args.manifest is None or args.plan_output is None:
            raise SystemExit("--plan requires --manifest and --plan-output")
        manifest = load_frozen_manifest(args.manifest)
        plan = build_acceptance_plan(manifest)
        plan_path = args.plan_output.expanduser().resolve()
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        payload = {
            "schema_version": "ptm2cellnet.frozen-acceptance-audit/v1",
            "manifest": str(args.manifest.expanduser().resolve()),
            "plan": str(plan_path),
            "n_runs": plan["n_runs"],
            "donor_leakage": "clear",
        }
    else:
        if args.manifest is None or args.eval_input is None or args.report_manifest is None:
            raise SystemExit("--verify requires --manifest, --eval-input and --report-manifest")
        manifest = load_frozen_manifest(args.manifest)
        eval_payload = json.loads(args.eval_input.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
        verification = verify_eval_input_against_manifest(manifest, eval_payload)
        report_payload = json.loads(args.report_manifest.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
        result: dict = {
            "verification": verification,
            "replay": replay_verdicts(report_payload, manifest=manifest),
        }
        payload = result

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output)}, ensure_ascii=False))
    if args.verify:
        verification = payload["verification"]
        replay = payload["replay"]
        if (
            not verification["covered"]
            or not verification["formal_plan_complete"]
            or not replay["reproduced"]
            or not replay["independent_h5ad_recomputed"]
        ):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
