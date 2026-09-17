#!/usr/bin/env python3
"""Assemble the external perturbation-model evidence payload for a candidate list.

Consumes a frozen external prediction asset
(``ptm2cellnet.external-perturbation-prediction/v1`` manifest + h5ad produced
by an external-model script) and writes the field-separated evidence JSON
(``ptm2cellnet.external-perturbation-evidence/v1``) that downstream report
lineage can record verbatim. Candidates absent from the asset fail loudly —
the formal consumer never silently drops coverage gaps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.perturbgen.external_perturbation_evidence import (  # noqa: E402
    ExternalPerturbationEvidenceError,
    evidence_missing_candidates,
    external_evidence_to_payload,
    load_external_prediction_asset,
)
from src.models.gene_vocabulary import normalize_ensembl_id  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-manifest", type=Path, required=True, help="external-perturbation-prediction/v1 manifest path"
    )
    parser.add_argument(
        "--candidates-tsv",
        type=Path,
        required=True,
        help="TSV with columns ensembl_id and gene_symbol requested for this evidence payload",
    )
    parser.add_argument("--output", type=Path, required=True, help="evidence JSON output path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import pandas as pd

    try:
        candidates_path = args.candidates_tsv.expanduser().resolve(strict=True)
        frame = pd.read_csv(candidates_path, sep="\t")
        missing_columns = [column for column in ("ensembl_id", "gene_symbol") if column not in frame.columns]
        if missing_columns:
            raise ExternalPerturbationEvidenceError(f"candidates TSV is missing columns: {missing_columns}")
        requested = {
            normalize_ensembl_id(str(value).strip()): str(symbol).strip()
            for value, symbol in zip(frame["ensembl_id"], frame["gene_symbol"], strict=True)
        }
        evidence = load_external_prediction_asset(args.prediction_manifest)
        missing = evidence_missing_candidates(evidence, requested)
        if missing:
            raise ExternalPerturbationEvidenceError(
                f"external prediction asset does not cover requested candidates: {list(missing)}; "
                "extend the asset in the external environment before assembly"
            )
        payload = external_evidence_to_payload(evidence)
        payload["requested_candidates"] = {ensembl_id: requested[ensembl_id] for ensembl_id in sorted(requested)}
        payload["missing_candidates"] = list(missing)
        payload["candidates_tsv"] = str(candidates_path)

        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (ExternalPerturbationEvidenceError, FileNotFoundError, ValueError) as exc:
        print(f"[assemble_external_evidence] 输入契约失败：{exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(output), "n_candidates": len(payload["candidates"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
