#!/usr/bin/env python3
"""Plan or collect matched-null PerturbGen perturb stages.

This CLI does not guess DEG columns.  GPU execution must supply an explicit
rescue extractor via the Python API; the command itself supports:

* ``--dry-run``: write the planned null stage identities
* ``--records-json``: collect already extracted rescue records into
  ``perturbgen_null_distribution/v1``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.perturbgen.null_generation import (  # noqa: E402
    NullGenerationError,
    collect_null_stage_records,
    run_matched_null_stages,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--e2e-report", type=Path, required=True)
    parser.add_argument("--path", required=True, choices=("source_intervention", "within_state"))
    parser.add_argument("--mode", required=True, choices=("mask", "pad", "delete", "overexpress"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--records-json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ensembl-to-symbol", type=Path, default=None)
    args = parser.parse_args(argv)

    mapping: dict[str, str] | None = None
    if args.ensembl_to_symbol is not None:
        payload = json.loads(args.ensembl_to_symbol.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SystemExit("ensembl-to-symbol JSON must be an object")
        mapping = {str(key): str(value) for key, value in payload.items()}

    if args.records_json is not None:
        records = json.loads(args.records_json.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise SystemExit("records-json must be a list of null stage records")
        selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        report = json.loads(args.e2e_report.read_text(encoding="utf-8"))
        from src.integration.perturbgen.null_generation import candidate_identity_from_e2e_report

        candidate = candidate_identity_from_e2e_report(report)
        distribution = collect_null_stage_records(
            records,
            candidate_ensembl_id=candidate["ensembl_id"],
            path=args.path,
            mode=args.mode,
            seed=args.seed,
            required_count=int(selection["required_count"]),
            selection_manifest_path=args.selection_manifest,
            output_path=args.output,
        )
        print(json.dumps({key: distribution[key] for key in ("schema_version", "required_count", "candidate_ensembl_id")}))
        return 0

    try:
        payload = run_matched_null_stages(
            args.selection_manifest,
            args.base_config,
            args.e2e_report,
            args.path,
            args.mode,
            args.seed,
            args.output_root,
            execute=not args.dry_run,
            dry_run=args.dry_run,
            ensembl_to_symbol=mapping,
            selection_manifest_path=args.selection_manifest,
            output_path=None if args.dry_run else args.output,
        )
    except NullGenerationError as exc:
        raise SystemExit(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
