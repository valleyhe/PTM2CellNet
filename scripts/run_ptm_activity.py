#!/usr/bin/env python3
"""Stage 1 of the PTM-activity mainline (方案 §5.1/§7 阶段 1).

Standardize a ``ptm_site_quantification.tsv`` (validated against the §4.1
contract) using the frozen ``ptm_research_config.yaml``: explicit replicate
policy, site/total-protein normalization on the single declared value
scale, protein → canonical Ensembl mapping via an explicit gene map, and a
``ptm_input_manifest.json`` with full provenance.

Activity inference itself (KSTAR/PhosR, 方案 §5.2) runs in its own
environment; its ``ptm_activity.tsv`` output is validated downstream by
``build_ptm_global_gene_scores.py`` — this CLI does not compute activities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.ptm_activity import (  # noqa: E402
    PTMActivityContractError,
    load_gene_map,
    load_ptm_site_quantification,
    standardize_ptm_input,
    write_ptm_input_manifest,
)
from src.analysis.ptm_research_config import PTMResearchConfigError, load_ptm_research_config  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="frozen ptm_research_config.yaml")
    parser.add_argument("--input-tsv", type=Path, required=True, help="ptm_site_quantification.tsv (§4.1)")
    parser.add_argument("--gene-map", type=Path, required=True, help="protein_id/gene_symbol/ensembl_id TSV")
    parser.add_argument("--output-tsv", type=Path, required=True, help="standardized PTM table output")
    parser.add_argument("--manifest-output", type=Path, required=True, help="ptm_input_manifest.json output")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_ptm_research_config(args.config)
        frame = load_ptm_site_quantification(args.input_tsv)
        gene_map = load_gene_map(args.gene_map)
        standardized, audit = standardize_ptm_input(frame, config=config, gene_map=gene_map)
    except (PTMResearchConfigError, PTMActivityContractError, FileNotFoundError) as exc:
        print(f"[run_ptm_activity] 输入契约失败：{exc}", file=sys.stderr)
        return 1
    input_path = args.input_tsv.expanduser().resolve(strict=True)
    gene_map_path = args.gene_map.expanduser().resolve(strict=True)
    output_path = args.output_tsv.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(output_path, sep="\t", index=False)
    manifest_path = write_ptm_input_manifest(
        args.manifest_output,
        input_path=input_path,
        audit=audit,
        config=config,
        gene_map_path=gene_map_path,
        output_path=output_path,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output_path),
                "manifest": str(manifest_path),
                "n_rows_input": audit.n_rows_input,
                "n_rows_after_replicates": audit.n_rows_after_replicates,
                "n_unmapped_proteins": audit.n_unmapped_proteins,
                "n_rows_without_donor": audit.n_rows_without_donor,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
