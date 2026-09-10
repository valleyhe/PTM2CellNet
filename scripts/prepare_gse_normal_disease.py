#!/usr/bin/env python3
"""Prepare a public GSE 10x normal/disease cohort for state evidence.

The output is deliberately marked observational-only.  It is suitable for a
GSE-specific scVI model and donor-level direction statistics, not for creating
causal KO/KD DAVF supervision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.gse_normal_disease import (
    build_gse_sample_specs,
    prepare_gse_normal_disease,
)
from src.models.gene_vocabulary import normalize_ensembl_id
from src.models.perturbgen_embedding import load_perturbgen_embedding_asset


def _sample_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _asset_gene_ids(asset: object) -> tuple[str, ...]:
    """Keep only biological Ensembl entries; PerturbGen also stores special tokens."""

    gene_to_token = getattr(asset, "gene_to_token", None)
    if not isinstance(gene_to_token, dict):
        raise ValueError("PerturbGen embedding asset does not expose a gene_to_token mapping")
    gene_ids: list[str] = []
    for value in gene_to_token:
        try:
            gene_ids.append(normalize_ensembl_id(value))
        except ValueError:
            # <cls>/<eos>/<mask>/<pad> are valid model vocabulary entries,
            # but they are not genes and must not enter the scVI axis.
            continue
    if not gene_ids:
        raise ValueError("PerturbGen embedding asset contains no Ensembl gene entries")
    return tuple(dict.fromkeys(gene_ids))


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GSE normal/disease 10x counts for DAVF direction evidence")
    parser.add_argument("--raw-dir", required=True, help="directory containing per-sample GSE 10x files")
    parser.add_argument("--annotation", required=True, help="GSE cell annotation CSV/CSV.GZ")
    parser.add_argument("--normal-samples", required=True, help="comma-separated normal sample labels")
    parser.add_argument("--disease-samples", required=True, help="comma-separated disease sample labels")
    parser.add_argument("--embedding-asset", required=True, help="verified PerturbGen embedding asset directory")
    parser.add_argument("--output", required=True, help="prepared H5AD output path")
    parser.add_argument("--dataset-accession", default="GSE214695")
    parser.add_argument("--min-donors", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.min_donors <= 0:
        raise ValueError("--min-donors must be positive")
    asset = load_perturbgen_embedding_asset(args.embedding_asset)
    samples = build_gse_sample_specs(
        args.raw_dir,
        normal_samples=_sample_list(args.normal_samples),
        disease_samples=_sample_list(args.disease_samples),
    )
    result = prepare_gse_normal_disease(
        samples,
        annotation_path=args.annotation,
        candidate_gene_ids=_asset_gene_ids(asset),
        n_genes=4018,
        dataset_accession=args.dataset_accession,
        embedding_asset_path=args.embedding_asset,
        embedding_manifest=asset.manifest,
        output_path=args.output,
        min_donors=args.min_donors,
    )
    print(json.dumps(result.report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
