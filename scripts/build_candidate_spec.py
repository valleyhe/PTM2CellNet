#!/usr/bin/env python3
"""Build the DAVF candidate spec JSON from upstream PTM and GSE artifacts.

Closes the manual hand-off gaps N-1/N-2 of the mainline: the PTM-site
prediction CSV is converted into provenance-bearing proposals and the GSE
donor-level direction evidence is joined automatically, so the candidate
spec consumed by ``run_davf_perturbgen_e2e.py`` no longer has to be written
by hand.

Example::

    python scripts/build_candidate_spec.py \
        --ptm-site-predictions outputs/ptm_sites/predictions.csv \
        --gene-map outputs/ptm_sites/gene_map.csv \
        --direction-evidence outputs/gse/direction_evidence.csv \
        --context-h5ad data/processed/context.h5ad \
        --provenance "ptm-site-model/run-1" \
        --proposed-direction down \
        --output outputs/e2e/candidate_spec.json

The PTM-side direction claim is an explicit scientific input: provide a
uniform ``--proposed-direction`` and optionally a site-level
``--proposal-direction-map`` CSV (protein_id, position, ptm_type,
proposed_direction).  The direction is never derived from the observed
expression evidence, because that would turn the three-way direction gate
into circular validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.perturbgen.candidate_spec import (  # noqa: E402
    build_candidate_spec_payload,
    build_spec_candidates,
    load_direction_evidence,
    load_direction_map,
    load_gene_map,
    load_ptm_site_predictions,
    write_candidate_spec,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptm-site-predictions", type=Path, required=True)
    parser.add_argument("--gene-map", type=Path, required=True)
    parser.add_argument("--direction-evidence", type=Path, required=True)
    parser.add_argument("--context-h5ad", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument(
        "--proposed-direction",
        choices=("up", "down"),
        required=True,
        help="PTM-side direction claim applied to every emitted site",
    )
    parser.add_argument(
        "--proposal-direction-map",
        type=Path,
        help="optional CSV overriding the direction for specific (protein_id, position, ptm_type)",
    )
    parser.add_argument("--site-probability-threshold", type=float, default=0.5)
    parser.add_argument("--context-cell-index", type=int, default=0)
    parser.add_argument(
        "--cell-type",
        action="append",
        dest="cell_types",
        help="restrict the evidence join to this cell type; repeatable",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.site_probability_threshold < 0 or args.site_probability_threshold > 1:
        raise SystemExit("--site-probability-threshold must be within [0, 1]")
    if args.context_cell_index < 0:
        raise SystemExit("--context-cell-index must be >= 0")

    predictions = load_ptm_site_predictions(args.ptm_site_predictions)
    gene_map = load_gene_map(args.gene_map)
    evidence = load_direction_evidence(args.direction_evidence)
    overrides = load_direction_map(args.proposal_direction_map) if args.proposal_direction_map else None

    candidates, summary = build_spec_candidates(
        predictions,
        gene_map,
        evidence,
        provenance=args.provenance,
        proposed_direction=args.proposed_direction,
        direction_overrides=overrides,
        site_probability_threshold=args.site_probability_threshold,
        context_cell_index=args.context_cell_index,
        cell_types=args.cell_types,
    )
    payload: dict[str, Any] = build_candidate_spec_payload(
        context_h5ad=args.context_h5ad,
        candidates=candidates,
        summary=summary,
        sources={
            "ptm_site_predictions": args.ptm_site_predictions,
            "gene_map": args.gene_map,
            "direction_evidence": args.direction_evidence,
            "proposal_direction_map": args.proposal_direction_map,
        },
    )
    output = write_candidate_spec(payload, args.output)
    print(json.dumps({"output": str(output), "summary": payload["build_summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
