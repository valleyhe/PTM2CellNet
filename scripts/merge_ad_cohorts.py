#!/usr/bin/env python3
"""Merge standardized AD cohort h5ads into one combined cohort for DEG.

The merge is a plain donor-pooling concat: cells keep their cohort of origin
(``obs['dataset']``), the gene axis becomes the intersection of canonical
ENSG ids, and no batch correction is applied — cohort composition effects
enter the donor-level test variance instead. That estimand choice is recorded
verbatim in the provenance JSON and must not be silently changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import anndata
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", type=Path, action="append", required=True, help="standardized cohort h5ad (repeat once per cohort)"
    )
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cohorts = [anndata.read_h5ad(path) for path in args.cohort]
    common = set(cohorts[0].var["ensembl_id"].astype(str))
    for cohort in cohorts[1:]:
        common &= set(cohort.var["ensembl_id"].astype(str))
    if not common:
        raise ValueError("cohorts share no canonical ENSG genes")

    parts = []
    symbol_by_ensembl: dict[str, str] = {}
    for cohort in cohorts:
        keep = cohort.var["ensembl_id"].astype(str).isin(common).to_numpy()
        part = cohort[:, keep].copy()
        part.X = sp.csr_matrix(part.layers[args.counts_layer])
        part.var.index = part.var["ensembl_id"].astype(str)
        for ensembl_id, symbol in zip(part.var["ensembl_id"], part.var["gene_symbol"], strict=True):
            symbol_by_ensembl.setdefault(str(ensembl_id), str(symbol))
        parts.append(part)
    merged = anndata.concat(parts, axis=0, join="outer", index_unique=None)
    merged.X = sp.csr_matrix(merged.X)
    merged.var = pd.DataFrame(
        {
            "ensembl_id": merged.var_names.astype(str),
            "gene_symbol": [symbol_by_ensembl.get(str(value), "") for value in merged.var_names],
        },
        index=pd.Index(merged.var_names.astype(str), name=None),
    )
    merged.layers[args.counts_layer] = merged.X.copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.write_h5ad(args.output, compression="gzip")

    provenance = {
        "schema_version": "ptm2cellnet.ad-cohort-merge/v1",
        "cohorts": [str(path) for path in args.cohort],
        "estimand_note": (
            "plain donor pooling without batch correction; cohort composition effects "
            "enter the donor-level test variance"
        ),
        "n_genes_per_cohort": [int(cohort.n_vars) for cohort in cohorts],
        "n_genes_common": len(common),
        "n_cells_merged": int(merged.n_obs),
        "donors_by_dataset": {
            str(dataset): int(group["donor"].nunique()) for dataset, group in merged.obs.groupby("dataset")
        },
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(args.output), "genes": len(common), "cells": int(merged.n_obs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
