#!/usr/bin/env python3
"""Create donor-level normal/disease direction evidence from prepared GSE data."""

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
    GSE_COUNTS_LAYER,
    summarize_normal_disease_directions,
)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize donor-level GSE normal/disease directions")
    parser.add_argument("--input", required=True, help="prepared GSE H5AD")
    parser.add_argument("--output", required=True, help="direction evidence CSV")
    parser.add_argument("--cell-type", action="append", dest="cell_types", help="optional cell type filter; repeatable")
    parser.add_argument("--min-donors", type=int, default=3)
    parser.add_argument("--counts-layer", default=GSE_COUNTS_LAYER)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.min_donors <= 0:
        raise ValueError("--min-donors must be positive")
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise ImportError("anndata is required to summarize GSE directions") from exc

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"prepared GSE H5AD not found: {input_path}")
    adata = ad.read_h5ad(input_path)
    evidence = summarize_normal_disease_directions(
        adata,
        cell_types=args.cell_types,
        counts_layer=args.counts_layer,
        min_donors=args.min_donors,
    )
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(output_path, index=False)
    report = {
        "schema_version": "ptm2cellnet.gse-direction-evidence.v1",
        "input": str(input_path),
        "output": str(output_path),
        "rows": int(len(evidence)),
        "cell_types": sorted(evidence["cell_type"].unique().tolist()),
        "direction_counts": {
            str(key): int(value) for key, value in evidence["observed_direction"].value_counts().to_dict().items()
        },
        "skipped_cell_types": evidence.attrs.get("skipped_cell_types", {}),
        "counts_layer": args.counts_layer,
        "effect_scale": "donor_mean_log2_normalized_counts",
        "observational_only": True,
        "formal_perturbgen_ready": False,
    }
    report_path = output_path.with_suffix(".manifest.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
