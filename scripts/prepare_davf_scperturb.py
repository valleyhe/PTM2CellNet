#!/usr/bin/env python3
"""Prepare real scPerturb H5AD files for a formal KO/KD DAVF run.

The output is an exact 4018-gene count matrix. All observed target genes that
are present in the verified PerturbGen vocabulary are retained first; the
remaining columns are selected deterministically by the source ``ncells``
feature score. The script never invents a target mapping or a gene column.

Examples::

    python scripts/prepare_davf_scperturb.py \
        --modality KO \
        --input data/raw/scperturb/DixitRegev2016_K562_TFs_7_days.h5ad \
        --input data/raw/scperturb/DixitRegev2016_K562_TFs_13_days.h5ad \
        --embedding-asset outputs/perturbgen/embedding_asset_20260822 \
        --output data/processed/davf_scperturb/ko/prepared.h5ad
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.davf_scperturb import prepare_scperturb_anndata
from src.models.perturbgen_embedding import load_perturbgen_embedding_asset


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare real scPerturb data for formal DAVF")
    parser.add_argument("--modality", required=True, choices=("KO", "KD"))
    parser.add_argument("--input", action="append", required=True, help="input H5AD; repeat for multiple sources")
    parser.add_argument("--embedding-asset", required=True)
    parser.add_argument("--output", required=True, help="prepared H5AD output path")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    asset = load_perturbgen_embedding_asset(args.embedding_asset)
    report = prepare_scperturb_anndata(
        args.input,
        modality=args.modality,
        asset_gene_to_token=asset.gene_to_token,
        output_path=args.output,
        embedding_asset_path=args.embedding_asset,
        embedding_manifest=asset.manifest,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "modality": report["modality"],
                "output": report["output"],
                "shape": report["shape"],
                "target_genes": len(report["target_genes"]),
                "sources": len(report["sources"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
