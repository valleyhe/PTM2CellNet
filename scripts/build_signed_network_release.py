#!/usr/bin/env python3
"""Assemble a frozen signed-network release from an OmniPath interactions export.

Consumes the OmniPath REST ``/interactions`` export (``datasets=dorothea,tf_target``,
``dorothea_levels=A..E``, ``genesymbols=1``, ``fields=sources,evidences``) and emits
the nine-column edge contract consumed by ``src/analysis/signed_network.py``:

    source_id, target_id, edge_type, effect_sign, site, species, evidence, confidence, release

Rules (方案 §4.3, all recorded in the release manifest):

* sign: ``consensus_stimulation``/``consensus_inhibition`` win; otherwise
  ``is_stimulation``/``is_inhibition``; both set or neither set drops the edge
  (unsigned edges never enter formal propagation).
* ids: canonical ENSG via the PerturbGen ensembl mapping (gene symbol keyed);
  unmapped endpoints drop the edge and are counted.
* ``edge_type`` is ``tf_regulation`` for both source datasets (TF→gene).
* ``confidence`` = min(1.0, 0.5 + 0.1 × (n_unique_resources − 1)); the formula
  is a frozen calibration choice, recorded verbatim — single-resource edges
  weigh 0.5, six or more resources weigh 1.0.
* ``site`` stays empty (TF regulation has no residue anchor); ``species`` is 9606;
  ``evidence`` is the unique resource list.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gene_vocabulary import normalize_ensembl_id  # noqa: E402

SIGNED_NETWORK_COLUMNS = (
    "source_id",
    "target_id",
    "edge_type",
    "effect_sign",
    "site",
    "species",
    "evidence",
    "confidence",
    "release",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omnipath-tsv", type=Path, required=True)
    parser.add_argument("--ensembl-mapping", type=Path, required=True)
    parser.add_argument("--release", required=True, help="frozen release identifier, e.g. omnipath-2026-09-16")
    parser.add_argument("--edge-type", default="tf_regulation")
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser.parse_args(argv)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _edge_sign(row: pd.Series) -> int | None:
    consensus_up = _truthy(row.get("consensus_stimulation", False))
    consensus_down = _truthy(row.get("consensus_inhibition", False))
    if consensus_up and consensus_down:
        return None
    if consensus_up or consensus_down:
        return 1 if consensus_up else -1
    is_up = _truthy(row.get("is_stimulation", False))
    is_down = _truthy(row.get("is_inhibition", False))
    if is_up and is_down:
        return None
    if is_up or is_down:
        return 1 if is_up else -1
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    frame = pd.read_csv(args.omnipath_tsv, sep="\t", dtype=str, keep_default_na=False)

    with Path(args.ensembl_mapping).open("rb") as handle:
        mapping_payload = pickle.load(handle)
    symbol_to_ensembl: dict[str, str] = {}
    for symbol, ensembl_id in mapping_payload.items():
        text_symbol = str(symbol).strip().upper()
        text_id = str(ensembl_id).strip()
        if text_symbol.startswith("ENSG") or not text_symbol:
            continue
        symbol_to_ensembl[text_symbol] = text_id

    rows: list[dict[str, object]] = []
    n_input = len(frame)
    n_unsigned_or_conflicting = 0
    n_unmapped = 0
    n_self_loops = 0
    for record in frame.to_dict("records"):
        sign = _edge_sign(record)
        if sign is None:
            n_unsigned_or_conflicting += 1
            continue
        source_symbol = str(record.get("source_genesymbol", "")).strip().upper()
        target_symbol = str(record.get("target_genesymbol", "")).strip().upper()
        source_id = symbol_to_ensembl.get(source_symbol)
        target_id = symbol_to_ensembl.get(target_symbol)
        if source_id is None or target_id is None:
            n_unmapped += 1
            continue
        try:
            source_id = normalize_ensembl_id(source_id)
            target_id = normalize_ensembl_id(target_id)
        except ValueError:
            n_unmapped += 1
            continue
        if source_id == target_id:
            n_self_loops += 1
            continue
        resources = sorted({item.strip() for item in str(record.get("sources", "")).split(";") if item.strip()})
        confidence = min(1.0, 0.5 + 0.1 * (len(resources) - 1))
        rows.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "edge_type": args.edge_type,
                "effect_sign": sign,
                "site": "",
                "species": "9606",
                "evidence": ";".join(resources),
                "confidence": f"{confidence:.3f}",
                "release": args.release,
            }
        )

    output = pd.DataFrame(rows, columns=list(SIGNED_NETWORK_COLUMNS))
    # Parallel edges with the same (source, target, edge_type, site): keep the
    # signedness decision of the strongest-confidence row; signed_network.py
    # enforces the final same-sign-collapse / sign-conflict-drop rules on load.
    output = output.sort_values(
        ["source_id", "target_id", "edge_type", "confidence"], ascending=[True, True, True, False]
    ).drop_duplicates(subset=["source_id", "target_id", "edge_type", "effect_sign"], keep="first")

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_tsv, sep="\t", index=False)

    manifest = {
        "schema_version": "ptm2cellnet.signed-network-release/v1",
        "release": args.release,
        "source": {
            "omnipath_export": str(args.omnipath_tsv),
            "query": (
                "https://omnipathdb.org/interactions?datasets=dorothea,tf_target"
                "&genesymbols=1&fields=sources,evidences&dorothea_levels=A,B,C,D,E"
            ),
            "ensembl_mapping": str(args.ensembl_mapping),
        },
        "edge_type": args.edge_type,
        "sign_rule": "consensus_(stimulation|inhibition) first, then is_(stimulation|inhibition); both/neither -> dropped",
        "confidence_formula": "min(1.0, 0.5 + 0.1 * (n_unique_resources - 1))",
        "counts": {
            "input_rows": n_input,
            "unsigned_or_conflicting_dropped": n_unsigned_or_conflicting,
            "unmapped_dropped": n_unmapped,
            "self_loops_dropped": n_self_loops,
            "released_edges": len(output),
            "activating": int((output["effect_sign"] == 1).sum()),
            "inhibiting": int((output["effect_sign"] == -1).sum()),
        },
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(args.output_tsv), "edges": len(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
