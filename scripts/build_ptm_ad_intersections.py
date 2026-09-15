#!/usr/bin/env python3
"""Stage 4 of the PTM-activity mainline (方案 §5.4/§7 阶段 4).

Join the §4.4 global gene-score table with the donor-level AD DEG table per
frozen cell type on canonical Ensembl: direction-match flag, explicit
PTM_only / AD_only / concordant / discordant membership, and a
formal/exploratory evidence tier from the frozen AD FDR and donor-support
thresholds. Emits per-cell-type ``ptm_ad_intersection_<cell_type>.tsv``,
one ``intersection_summary.json`` (with non-independent
``n_cell_types_supported`` bookkeeping) and one ``target_set_manifest.json``
for stage 5.
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

from src.analysis.ptm_gene_score import (  # noqa: E402
    PTMGeneScoreError,
    intersect_gene_scores_with_deg,
    load_deg_table,
    load_gene_score_table,
    write_intersection_outputs,
)
from src.analysis.ptm_research_config import PTMResearchConfigError, load_ptm_research_config  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="frozen ptm_research_config.yaml")
    parser.add_argument("--gene-scores-tsv", type=Path, required=True, help="ptm_global_gene_scores.tsv (§4.4)")
    parser.add_argument("--deg-table", type=Path, required=True, help="donor-level AD DEG table (§4.5)")
    parser.add_argument("--output-dir", type=Path, required=True, help="per-cell-type intersection output directory")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_ptm_research_config(args.config)
        score_frame = load_gene_score_table(args.gene_scores_tsv)
        deg_frame = load_deg_table(args.deg_table)
        intersections = intersect_gene_scores_with_deg(score_frame, deg_frame, config=config)
    except (PTMResearchConfigError, PTMGeneScoreError, FileNotFoundError) as exc:
        print(f"[build_ptm_ad_intersections] 输入契约失败：{exc}", file=sys.stderr)
        return 1
    output_dir = args.output_dir.expanduser().resolve()
    outputs = write_intersection_outputs(
        intersections,
        output_dir=output_dir,
        config=config,
        score_table_path=args.gene_scores_tsv.expanduser().resolve(strict=True),
        deg_table_path=args.deg_table.expanduser().resolve(strict=True),
    )
    summary = {
        "cell_types": {
            cell_type: {
                "n_formal_concordant": summary.n_formal_concordant,
                "n_concordant": summary.n_concordant,
                "n_discordant": summary.n_discordant,
                "n_ptm_only": summary.n_ptm_only,
                "n_ad_only": summary.n_ad_only,
            }
            for cell_type, (_frame, summary) in sorted(intersections.items())
        }
    }
    print(
        json.dumps(
            {
                "ok": True,
                "output_dir": str(output_dir),
                "summary": str(outputs["summary"]),
                "target_set_manifest": str(outputs["target_set_manifest"]),
                **summary,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
