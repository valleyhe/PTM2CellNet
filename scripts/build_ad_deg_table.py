#!/usr/bin/env python3
"""Build aggregate and donor-level AD DEG tables from a cohort AnnData file."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Sequence

import anndata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.ad_deg_table import (  # noqa: E402
    ADDEGError,
    build_ad_deg_tables,
    manifest_payload,
    write_ad_deg_tables,
    write_manifest,
)
from src.analysis.ptm_research_config import PTMResearchConfigError, load_ptm_research_config  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="frozen ptm_research_config.yaml")
    parser.add_argument("--cohort-h5ad", type=Path, default=None, help="cohort AnnData input")
    parser.add_argument("--cell-type-obs-column", default="cell_type")
    parser.add_argument("--state-obs-column", default="state")
    parser.add_argument("--donor-obs-column", default="donor")
    parser.add_argument("--output-tsv", type=Path, required=True, help="aggregate AD DEG table output")
    parser.add_argument("--donor-level-output", type=Path, required=True, help="donor-level AD DEG table output")
    parser.add_argument("--manifest-output", type=Path, required=True, help="AD DEG manifest output")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config_path = args.config.expanduser().resolve(strict=True)
        config = load_ptm_research_config(config_path)
        cohort_h5ad = (
            args.cohort_h5ad.expanduser().resolve(strict=True)
            if args.cohort_h5ad is not None
            else (config_path.parent / config.cohort_h5ad).expanduser().resolve(strict=True)
        )
        aggregate_output = args.output_tsv.expanduser().resolve()
        donor_output = args.donor_level_output.expanduser().resolve()
        manifest_output = args.manifest_output.expanduser().resolve()
        adata = anndata.read_h5ad(cohort_h5ad)
        aggregate, donor, audit = build_ad_deg_tables(
            adata,
            cell_types=config.cell_types,
            cohort_pairing=config.cohort_pairing,
            min_donors_per_state=config.min_donors_per_state,
            cell_type_column=args.cell_type_obs_column,
            state_column=args.state_obs_column,
            donor_column=args.donor_obs_column,
            donor_aggregation=config.deg_donor_aggregation,
        )
        write_ad_deg_tables(aggregate, donor, aggregate_output, donor_output)
        manifest = manifest_payload(
            audit,
            aggregate_output=str(aggregate_output),
            donor_output=str(donor_output),
        )
        manifest.update(
            {
                "schema_version": "ptm2cellnet.ad-deg-manifest/v1",
                "input_h5ad": str(cohort_h5ad),
                "config": asdict(config),
                "columns": {
                    "cell_type": args.cell_type_obs_column,
                    "state": args.state_obs_column,
                    "donor": args.donor_obs_column,
                },
                "cell_types": list(config.cell_types),
                "cohort_pairing": config.cohort_pairing,
                "output_contracts": {
                    "aggregate": {
                        "path": str(aggregate_output),
                        "format": "tsv" if aggregate_output.suffix.lower() in {".tsv", ".tab"} else "csv",
                        "columns": list(aggregate.columns),
                    },
                    "donor_level": {
                        "path": str(donor_output),
                        "format": "tsv" if donor_output.suffix.lower() in {".tsv", ".tab"} else "csv",
                        "columns": list(donor.columns),
                    },
                },
            }
        )
        write_manifest(manifest, str(manifest_output))
    except (ADDEGError, PTMResearchConfigError, FileNotFoundError) as exc:
        print(f"[build_ad_deg_table] 输入契约失败：{exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(aggregate_output),
                "donor_level_output": str(donor_output),
                "manifest": str(manifest_output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
