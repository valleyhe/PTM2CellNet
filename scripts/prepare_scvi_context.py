#!/usr/bin/env python3
"""Prepare a scVI-gene-axis-aligned DAVF context from a cohort AnnData."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import anndata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.scvi_context import (  # noqa: E402
    MISSING_GENE_POLICIES,
    ScviContextError,
    VALID_DAVF_ROUTES,
    prepare_scvi_context,
    write_scvi_context,
)
from src.models.scvi_adapter import ScVIAdapter  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-h5ad", type=Path, required=True)
    parser.add_argument("--scvi-model", type=Path, required=True, help="route scVI checkpoint directory")
    parser.add_argument("--gene-aliases", type=Path, required=True, help="route prepared.gene_aliases.tsv")
    parser.add_argument("--davf-route", required=True, choices=VALID_DAVF_ROUTES)
    parser.add_argument(
        "--davf-batch",
        required=True,
        help="batch value from the route's scVI batch registry to bind every context cell to",
    )
    parser.add_argument(
        "--missing-gene-policy",
        required=True,
        choices=MISSING_GENE_POLICIES,
        help="zero_fill: axis genes absent from the cohort become all-zero columns; fail: abort with the gene list",
    )
    parser.add_argument(
        "--batch-binding-rationale",
        required=True,
        help="verbatim research rationale for binding the cohort to this training batch; recorded in the manifest",
    )
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cohort_path = args.cohort_h5ad.expanduser().resolve(strict=True)
        scvi_path = args.scvi_model.expanduser().resolve(strict=True)
        aliases_path = args.gene_aliases.expanduser().resolve(strict=True)
        cohort = anndata.read_h5ad(cohort_path)
        adapter = ScVIAdapter.from_trained_model(scvi_path)
        context, manifest = prepare_scvi_context(
            cohort,
            adapter=adapter,
            davf_route=args.davf_route,
            davf_batch=args.davf_batch,
            missing_gene_policy=args.missing_gene_policy,
            batch_binding_rationale=args.batch_binding_rationale,
            gene_aliases_path=aliases_path,
            counts_layer=args.counts_layer,
        )
        write_scvi_context(context, manifest, args.output_h5ad, args.manifest_output)
    except (ScviContextError, FileNotFoundError) as exc:
        print(f"[prepare_scvi_context] 输入契约失败：{exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output_h5ad),
                "manifest": str(args.manifest_output),
                "davf_route": args.davf_route,
                "davf_batch": args.davf_batch,
                "n_missing_axis_genes": manifest["n_missing_axis_genes"],
                "n_dropped_cohort_genes": manifest["n_dropped_cohort_genes"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
