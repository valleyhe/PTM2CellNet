#!/usr/bin/env python3
"""Audit AD candidate genes against DAVF route axes and local scPerturb sources.

Answers the coverage question behind 方案 §7.3-U3: for each candidate gene,
whether it is present in each route's frozen 4018-gene alias axis, whether it
is measurable (var) or perturbed (obs target) in any local scPerturb dataset,
and whether it resolves in the PerturbGen embedding vocabulary. The output is
decision support for "retrain a wider axis (Workflow B)" vs "restrict formal
candidates to covered genes" — it never mutates any asset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gene_vocabulary import normalize_ensembl_id  # noqa: E402

DEFAULT_ROUTES = ("ko", "kd")
ROUTE_ALIAS_PATHS = {
    "ko": PROJECT_ROOT / "data/processed/davf_scperturb/ko/prepared.gene_aliases.tsv",
    "kd": PROJECT_ROOT / "data/processed/davf_scperturb/kd/prepared.gene_aliases.tsv",
}
DEFAULT_CANDIDATES = (
    ("APOE", "ENSG00000130203"),
    ("APP", "ENSG00000142192"),
    ("PSEN1", "ENSG00000080815"),
    ("BACE1", "ENSG00000186318"),
    ("MAPT", "ENSG00000186868"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates-tsv",
        type=Path,
        default=None,
        help="TSV with columns gene_symbol,ensembl_id; defaults to the frozen AD candidate set",
    )
    parser.add_argument("--routes", nargs="*", default=list(DEFAULT_ROUTES), choices=list(DEFAULT_ROUTES))
    parser.add_argument(
        "--scperturb-dir",
        type=Path,
        default=PROJECT_ROOT / "data/raw/scperturb",
        help="directory of raw scPerturb h5ad files to scan for measured/perturbed genes",
    )
    parser.add_argument("--embedding-vocab", type=Path, default=None, help="PerturbGen token dict pickle")
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser.parse_args(argv)


def _load_candidates(path: Path | None) -> list[tuple[str, str]]:
    if path is None:
        return list(DEFAULT_CANDIDATES)
    frame = pd.read_csv(path, sep="\t")
    missing = [column for column in ("gene_symbol", "ensembl_id") if column not in frame.columns]
    if missing:
        raise ValueError(f"candidates TSV is missing columns: {missing}")
    candidates = []
    for _, row in frame.iterrows():
        candidates.append((str(row["gene_symbol"]).strip(), normalize_ensembl_id(str(row["ensembl_id"]))))
    if not candidates:
        raise ValueError("candidates TSV is empty")
    return candidates


def _route_axis_symbols(path: Path) -> tuple[set[str], set[str]]:
    table = pd.read_csv(path, sep="\t")
    return set(table["ensembl_id"].astype(str)), set(table["gene_symbol"].astype(str))


def _scan_scperturb(directory: Path, symbols: set[str]) -> tuple[dict[str, dict[str, list[str]]], list[dict[str, str]]]:
    import anndata

    result: dict[str, dict[str, list[str]]] = {symbol: {"measured_in": [], "perturbed_in": []} for symbol in symbols}
    unreadable: list[dict[str, str]] = []
    for path in sorted(directory.glob("*.h5ad")):
        try:
            adata = anndata.read_h5ad(path, backed="r")
        except (OSError, ValueError) as exc:
            unreadable.append({"file": path.name, "error": str(exc)[:200]})
            continue
        try:
            var_symbols = set(str(value) for value in adata.var_names)
            target_columns = [
                column for column in ("target", "gene", "gene_symbol", "perturbation") if column in adata.obs.columns
            ]
            perturbed: set[str] = set()
            for column in target_columns:
                values = adata.obs[column].dropna().astype(str)
                perturbed.update(
                    value.strip() for value in values.unique() if value.strip().lower() not in {"control", "nan"}
                )
            for symbol in symbols:
                if symbol in var_symbols:
                    result[symbol]["measured_in"].append(path.stem)
                if symbol in perturbed:
                    result[symbol]["perturbed_in"].append(path.stem)
        finally:
            adata.file.close()
    return result, unreadable


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        candidates = _load_candidates(args.candidates_tsv)
        route_axes = {route: _route_axis_symbols(ROUTE_ALIAS_PATHS[route]) for route in args.routes}

        vocabulary: set[str] | None = None
        if args.embedding_vocab is not None:
            import pickle

            with args.embedding_vocab.open("rb") as handle:
                token_dict = pickle.load(handle)
            if not isinstance(token_dict, dict):
                raise ValueError(f"embedding vocabulary must be a dict pickle: {args.embedding_vocab}")
            vocabulary = set(str(key) for key in token_dict)

        symbols = {symbol for symbol, _ in candidates}
        scan, unreadable = _scan_scperturb(args.scperturb_dir, symbols) if args.scperturb_dir.is_dir() else ({}, [])

        rows = []
        for symbol, ensembl_id in candidates:
            row: dict[str, object] = {
                "gene_symbol": symbol,
                "ensembl_id": ensembl_id,
                "embedding_vocab": None if vocabulary is None else ensembl_id in vocabulary,
            }
            for route, (ensembl_axis, symbol_axis) in route_axes.items():
                row[f"{route}_in_axis"] = ensembl_id in ensembl_axis or symbol in symbol_axis
            if scan:
                entry = scan.get(symbol, {"measured_in": [], "perturbed_in": []})
                row["scperturb_measured_in"] = ";".join(entry["measured_in"])
                row["scperturb_perturbed_in"] = ";".join(entry["perturbed_in"])
            rows.append(row)

        frame = pd.DataFrame(rows)
        args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output_tsv, sep="\t", index=False)

        records = frame.to_dict("records")
        per_route_covered = {
            route: sorted(str(row["gene_symbol"]) for row in records if bool(row.get(f"{route}_in_axis")))
            for route in args.routes
        }
        manifest = {
            "schema_version": "ptm2cellnet.davf-axis-coverage-audit/v1",
            "candidates": [{"gene_symbol": symbol, "ensembl_id": ensembl_id} for symbol, ensembl_id in candidates],
            "routes": {route: {"alias_path": str(ROUTE_ALIAS_PATHS[route])} for route in args.routes},
            "embedding_vocab": None if args.embedding_vocab is None else str(args.embedding_vocab),
            "scperturb_dir": str(args.scperturb_dir) if args.scperturb_dir.is_dir() else None,
            "scperturb_unreadable_files": unreadable,
            "axis_covered_per_route": per_route_covered,
            "formal_candidate_eligible": (
                sorted(set.intersection(*(set(genes) for genes in per_route_covered.values())))
                if per_route_covered
                else []
            ),
            "formal_candidate_eligible_note": (
                "axis membership is not formal eligibility; KO and KD are independent routes; "
                "formal invocation still requires observed FDR<=0.05, a real target-specific "
                "z0/z1 pair, and the three-way gate. Use scripts/route_ad_candidates.py."
            ),
            "route_specific_axis_covered": per_route_covered,
        }
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (ValueError, FileNotFoundError) as exc:
        print(f"[audit_davf_axis_coverage] 审计失败：{exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"ok": True, "output": str(args.output_tsv), "manifest": str(args.manifest_output)}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
