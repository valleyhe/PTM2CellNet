#!/usr/bin/env python3
"""Export frozen GEARS prediction assets for unseen mainline candidates.

Runs in the external GEARS environment (``pip install cell-gears``; see
docs/guides/perturbgen_bridge.md §6b). GEARS is the only evaluated source whose
GO-graph channel can query genes never perturbed in training — the 2026-09-17
vocabulary audit showed the five AD candidates are absent from every trained
CRISPR corpus (STATE/Nadig 2,024; LPM essentialome 2,285; GWPS covers only
APOE/MAPT/PSEN1), while GEARS' gene2go graph contains all five with dense GO
neighbourhoods.

The script trains (or loads) a GEARS model on the Norman 2019 corpus and
predicts each candidate as an unseen single-gene perturbation. Output is the
same ``external-perturbation-prediction/v1`` asset consumed by
``src/integration/perturbgen/external_perturbation_evidence.py`` with
``evidence_kind="go_extrapolation"`` and
``perturbation_semantics="unseen_perturbation_extrapolation"`` — the evidence is
a GO-graph extrapolation in a K562 training context, never a disease-context
intervention or causal validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-tsv", type=Path, required=True, help="ensembl_id/gene_symbol columns")
    parser.add_argument(
        "--gears-data-dir",
        type=Path,
        required=True,
        help="directory holding the Norman corpus (perturb_processed.h5ad + go.csv) and gene2go.pkl",
    )
    parser.add_argument("--model-ckpt", type=Path, default=None, help="trained GEARS checkpoint (train if omitted)")
    parser.add_argument(
        "--gene-set-pkl",
        type=Path,
        default=None,
        help="pickle of gene symbols defining the perturbation-graph node set; include candidates that are "
        "absent from the default graph so the GO channel can embed them (requires training with this graph)",
    )
    parser.add_argument("--train-epochs", type=int, default=10, help="training epochs when no checkpoint is given")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser.parse_args(argv)


def _load_candidates(path: Path) -> list[tuple[str, str]]:
    import pandas as pd

    frame = pd.read_csv(path, sep="\t")
    missing = [column for column in ("ensembl_id", "gene_symbol") if column not in frame.columns]
    if missing:
        raise SystemExit(f"candidates TSV is missing columns: {missing}")
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for ensembl_id, gene_symbol in zip(frame["ensembl_id"], frame["gene_symbol"], strict=True):
        text = str(ensembl_id).strip()
        symbol = str(gene_symbol).strip()
        if not text or text in seen:
            raise SystemExit(f"candidates TSV has an empty or duplicated ensembl_id: {ensembl_id!r}")
        seen.add(text)
        candidates.append((text, symbol))
    if not candidates:
        raise SystemExit("candidates TSV contains no rows")
    return candidates


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from gears import GEARS, PertData
    except ImportError:
        print(
            "[run_gears_predictions] cell-gears is not installed; run this script inside the external "
            "GEARS environment (docs/guides/perturbgen_bridge.md §6b)",
            file=sys.stderr,
        )
        return 2

    import numpy as np
    import pandas as pd

    EXTERNAL_PREDICTION_SCHEMA_VERSION = "ptm2cellnet.external-perturbation-prediction/v1"

    candidates = _load_candidates(args.candidates_tsv.expanduser().resolve(strict=True))
    data_dir = args.gears_data_dir.expanduser().resolve(strict=True)

    pert_data = PertData(
        str(data_dir.parent),
        gene_set_path=str(args.gene_set_pkl.expanduser().resolve(strict=True)) if args.gene_set_pkl else None,
    )
    pert_data.load(data_name="norman")
    pert_data.prepare_split(split="simulation", seed=args.seed)
    pert_data.get_dataloader(batch_size=128)

    model = GEARS(pert_data, weight_bias_track=False)
    if args.model_ckpt is not None:
        model.load_pretrained(str(args.model_ckpt))
    else:
        model.model_initialize(hidden_size=64)
        model.train(epochs=args.train_epochs)
        save_dir = data_dir.parent / "gears_ckpt"
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(str(save_dir))

    # Unseen single-gene queries (list of lists is the GEARS predict contract);
    # the perturbed-minus-control baseline is the model's ctrl_expression.
    perturbation_queries = [[symbol] for _, symbol in candidates]
    results = model.predict(perturbation_queries)
    perturbed_predictions = np.asarray([results["_".join(query)] for query in perturbation_queries])
    control_expression = np.asarray(model.ctrl_expression.detach().cpu()).reshape(1, -1)
    delta_matrix = perturbed_predictions - control_expression

    # Map readouts to canonical Ensembl. Norman var_names already are Ensembl
    # ids; only map when a symbol readout space is used (other corpora), and
    # drop unmapped columns explicitly (never guess an id).
    symbol_to_ensembl: dict[str, str] = {}
    var_symbols = (
        pert_data.adata.var_names.tolist() if hasattr(pert_data, "adata") else [str(name) for name in model.node_list]
    )
    unmapped: list[str] = []
    for symbol in var_symbols:
        text = str(symbol)
        if text.startswith("ENSG"):
            symbol_to_ensembl[text] = text
            continue
        mapped = _resolve_symbol_to_ensembl(text)
        if mapped is None:
            unmapped.append(text)
        else:
            symbol_to_ensembl[text] = mapped
    if unmapped:
        print(
            f"[run_gears_predictions] dropping {len(unmapped)} readout symbols without canonical Ensembl: "
            f"{unmapped[:10]}{'...' if len(unmapped) > 10 else ''}",
            file=sys.stderr,
        )
    keep_positions = [position for position, symbol in enumerate(var_symbols) if str(symbol) in symbol_to_ensembl]
    readout_ids = [symbol_to_ensembl[str(var_symbols[position])] for position in keep_positions]
    delta_matrix = delta_matrix[:, keep_positions]
    if not readout_ids:
        raise SystemExit("no readout symbol resolved to canonical Ensembl; check the symbol mapping source")

    import anndata as ad

    output_h5ad = args.output_h5ad.expanduser().resolve()
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata = ad.AnnData(
        X=np.asarray(delta_matrix, dtype=float),
        obs=pd.DataFrame(
            {
                "ensembl_id": [ensembl_id for ensembl_id, _ in candidates],
                "gene_symbol": [symbol for _, symbol in candidates],
                "context": ["HumanCellLine_K562_10xChromium3-scRNA-seq_Norman2019"] * len(candidates),
                "perturbation_semantics": ["unseen_perturbation_extrapolation"] * len(candidates),
            },
            index=[symbol for _, symbol in candidates],
        ),
    )
    adata.var_names = readout_ids
    adata.write_h5ad(output_h5ad)

    manifest_output = args.manifest_output.expanduser().resolve()
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": EXTERNAL_PREDICTION_SCHEMA_VERSION,
        "evidence_kind": "go_extrapolation",
        "model": {
            "source": "stanford_gears/cell-gears",
            "cell_gears_commit": _cell_gears_commit(),
            "training_config": "norman_simulation_split",
            "license": "MIT (GEARS); predictions inherit Norman 2019 dataset terms",
        },
        "context": "HumanCellLine_K562_10xChromium3-scRNA-seq_Norman2019",
        "perturbation_semantics": "unseen_perturbation_extrapolation",
        "seeds": [args.seed],
        "aggregation": "single-seed unseen query prediction minus ctrl prediction (GEARS GO channel)",
        "predictions_h5ad": str(output_h5ad),
        "predictions_sha256": hashlib.sha256(output_h5ad.read_bytes()).hexdigest(),
        "readout_vocab": "canonical Ensembl (Norman symbols mapped; unmapped dropped and logged)",
        "unmapped_readout_symbols": unmapped,
    }
    manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_h5ad), "manifest": str(manifest_output)}, ensure_ascii=False))
    return 0


def _resolve_symbol_to_ensembl(symbol: str) -> str | None:
    """Resolve a HGNC symbol to canonical Ensembl via the project mapping table."""

    mapping_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "gene_symbol_ensembl_map.tsv"
    if not _SYMBOL_CACHE:
        if not mapping_path.exists():
            raise SystemExit(f"symbol mapping table not found: {mapping_path}; regenerate it before running")
        import pandas as pd

        for raw_symbol, raw_ensembl in zip(
            pd.read_csv(mapping_path, sep="\t")["gene_symbol"],
            pd.read_csv(mapping_path, sep="\t")["ensembl_id"],
            strict=True,
        ):
            _SYMBOL_CACHE[str(raw_symbol).upper()] = str(raw_ensembl)
    return _SYMBOL_CACHE.get(symbol.upper())


_SYMBOL_CACHE: dict[str, str] = {}


def _cell_gears_commit() -> str:
    try:
        from importlib.metadata import version

        return f"pypi:cell-gears=={version('cell-gears')}"
    except Exception:  # noqa: BLE001 - provenance is best-effort metadata
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
