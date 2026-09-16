#!/usr/bin/env python3
"""Repair a scPerturb h5ad whose var['ensembl_id'] is partly symbolic.

Some scPerturb exports (e.g. FrangiehIzar2021_RNA) carry gene symbols or
aliases inside ``var['ensembl_id']`` for thousands of rows. This script
rewrites the column via the PerturbGen ensembl mapping (symbol -> ENSG,
keys uppercased, ENSG-keyed entries skipped) and drops rows that stay
unmapped; ``var_names`` keep their original symbols and the counts matrix is
untouched. The output feeds ``prepare_davf_scperturb.py`` directly.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys
from typing import Sequence

import anndata
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gene_vocabulary import normalize_ensembl_id  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ensembl-mapping", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    with Path(args.ensembl_mapping).open("rb") as handle:
        payload = pickle.load(file=handle)
    symbol_to_ensembl: dict[str, str] = {}
    for key, value in payload.items():
        text_key = str(key).strip().upper()
        text_value = str(value).strip()
        if text_key.startswith("ENSG") or not text_key:
            continue
        symbol_to_ensembl.setdefault(text_key, text_value)

    adata = anndata.read_h5ad(args.input)
    raw = adata.var["ensembl_id"].astype(str)
    fixed, n_already_valid, n_remapped, n_dropped = [], 0, 0, 0
    for position, value in enumerate(raw):
        text = value.strip()
        try:
            normalize_ensembl_id(text)
            fixed.append(normalize_ensembl_id(text))
            n_already_valid += 1
            continue
        except ValueError:
            pass
        mapped = symbol_to_ensembl.get(text.upper())
        if mapped is None:
            mapped = symbol_to_ensembl.get(str(adata.var_names[position]).strip().upper())
        if mapped is None:
            fixed.append(None)
            n_dropped += 1
        else:
            fixed.append(normalize_ensembl_id(mapped))
            n_remapped += 1

    keep = [value is not None for value in fixed]
    output = adata[:, keep].copy()
    ensembl_values = [value for value in fixed if value is not None]
    output.var["ensembl_id"] = ensembl_values
    if "gene_symbol" not in output.var.columns:
        output.var["gene_symbol"] = [str(name) for name in adata.var_names[keep]]

    # Remapped aliases can collide with already-valid ENSG rows; collapse
    # duplicates by summing their count columns (same rule as the cohort
    # standardizer).
    counts_frame = output.var[["ensembl_id"]]
    if counts_frame["ensembl_id"].duplicated().any():
        import numpy as np
        import scipy.sparse as sp

        order = sorted(set(ensembl_values))
        new_index = {gene_id: position for position, gene_id in enumerate(order)}
        coo = sp.csr_matrix(output.X).tocoo()
        new_cols = np.asarray([new_index[gene_id] for gene_id in np.asarray(ensembl_values)[coo.col]])
        merged = sp.coo_matrix((coo.data, (coo.row, new_cols)), shape=(output.n_obs, len(order))).tocsr()
        symbol_map: dict[str, str] = {}
        for gene_id, symbol in zip(ensembl_values, output.var["gene_symbol"], strict=True):
            symbol_map.setdefault(gene_id, str(symbol))
        output = anndata.AnnData(
            X=merged,
            obs=output.obs.copy(),
            var=pd.DataFrame(
                {
                    "ensembl_id": order,
                    "gene_symbol": [symbol_map[gene_id] for gene_id in order],
                },
                index=pd.Index([symbol_map[gene_id] for gene_id in order], name=None),
            ),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.write_h5ad(args.output, compression="gzip")
    stats = {
        "schema_version": "ptm2cellnet.scperturb-ensembl-remap/v1",
        "input": str(args.input),
        "output": str(args.output),
        "ensembl_mapping": str(args.ensembl_mapping),
        "n_genes_input": int(adata.n_vars),
        "n_already_valid_ensembl": n_already_valid,
        "n_remapped_from_symbol": n_remapped,
        "n_dropped_unmapped": n_dropped,
        "n_genes_output": int(output.n_vars),
    }
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
