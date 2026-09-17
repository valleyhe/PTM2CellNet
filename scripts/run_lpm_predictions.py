#!/usr/bin/env python3
"""Export frozen LPM prediction assets for mainline direction evidence.

Runs in the *external* perturblib environment (see docs/guides/perturbgen_bridge.md
§7): the main requirements-core environment never installs perturblib. The
script loads LPM checkpoints trained with the paper configuration
(``replogle_k562_paper_lpm``, 5 seeds), predicts the readout response for every
requested candidate perturbation plus the unperturbed control, averages the
``perturbed - control`` delta over seeds and writes the frozen asset consumed by
``src/integration/perturbgen/external_perturbation_evidence.py``:

- ``<output-h5ad>``: obs = candidate perturbations (ensembl_id, gene_symbol,
  context, perturbation_semantics), var = canonical Ensembl readout ids,
  X = per-seed-mean delta matrix.
- ``<manifest-output>``: ``ptm2cellnet.lpm-prediction/v1`` manifest binding the
  h5ad sha256, perturblib commit, training config, context, seeds, aggregation
  and license.

Nothing is imputed: candidates absent from the model's perturbation vocabulary
fail loudly with the full missing list.
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
    parser.add_argument(
        "--candidates-tsv",
        type=Path,
        required=True,
        help="TSV with columns ensembl_id and gene_symbol (one candidate per row)",
    )
    parser.add_argument(
        "--trained-model-dir",
        type=Path,
        required=True,
        help="directory holding the 5-seed LPM checkpoints from perturb-gym training",
    )
    parser.add_argument(
        "--context",
        default="HumanCellLine_K562_10xChromium3-scRNA-seq_Replogle22",
        help="perturblib context id the model was trained on (default: Replogle K562)",
    )
    parser.add_argument(
        "--perturbation-semantics",
        default="crispri_kd",
        choices=("crispri_kd", "shrna_kd", "crispra_oe"),
        help="intervention semantics declared on every exported row (default: crispri_kd)",
    )
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument(
        "--perturblib-commit",
        default="",
        help="git commit of the perturblib checkout used for training/inference",
    )
    parser.add_argument(
        "--training-config",
        default="replogle_k562_paper_lpm",
        help="perturb-gym training config id used to train the checkpoints",
    )
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
        if not text or text in seen:
            raise SystemExit(f"candidates TSV has an empty or duplicated ensembl_id: {ensembl_id!r}")
        seen.add(text)
        candidates.append((text, str(gene_symbol).strip()))
    if not candidates:
        raise SystemExit("candidates TSV contains no rows")
    return candidates


def _predict_deltas(args: argparse.Namespace, candidates: list[tuple[str, str]]) -> tuple["object", list[int]]:
    """Run every seed checkpoint and average perturbed-minus-control deltas."""

    import perturb_lib as plib

    context = args.context
    pdata = plib.load_plibdata(context)
    vocabulary = set(pdata.perturbation_symbols) if hasattr(pdata, "perturbation_symbols") else None
    if vocabulary is not None:
        missing = [symbol for _, symbol in candidates if symbol not in vocabulary]
        if missing:
            raise SystemExit(
                "candidates absent from the LPM perturbation vocabulary (train or correct the context): "
                f"{sorted(set(missing))}"
            )

    seed_dirs = sorted(path for path in args.trained_model_dir.iterdir() if path.is_dir())
    if not seed_dirs:
        raise SystemExit(f"--trained-model-dir contains no per-seed checkpoint directories: {args.trained_model_dir}")
    seeds: list[int] = []
    per_seed_perturbed = []
    per_seed_control = None
    for seed_dir in seed_dirs:
        seed = _seed_from_dirname(seed_dir.name)
        seeds.append(seed)
        model = plib.load_trained_model(seed_dir)
        perturbed_x = _prediction_frame(model, pdata, [symbol for _, symbol in candidates])
        control_x = _prediction_frame(model, pdata, [plib.ControlSymbol])
        per_seed_perturbed.append(perturbed_x)
        if per_seed_control is None:
            per_seed_control = control_x
        else:
            per_seed_control = per_seed_control + control_x
    mean_perturbed = sum(per_seed_perturbed) / len(per_seed_perturbed)
    mean_control = per_seed_control / len(seeds)
    delta_matrix = mean_perturbed - mean_control
    return delta_matrix, seeds


def _seed_from_dirname(name: str) -> int:
    digits = "".join(character for character in name if character.isdigit())
    return int(digits) if digits else -1


def _prediction_frame(model, pdata, symbols: list[str]):
    """Predict readout responses for ``symbols``; returns a candidates × readouts matrix."""

    import numpy as np

    predicted = np.asarray(model.predict(_restricted_pdata(pdata, symbols)))
    if predicted.ndim == 1:
        predicted = predicted.reshape(len(symbols), -1)
    return predicted


def _restricted_pdata(pdata, symbols: list[str]):
    """Build the prediction input restricted to ``symbols`` in the training context."""

    import pandas as pd

    frame = pdata.data if hasattr(pdata, "data") else None
    if frame is None:
        raise SystemExit(
            "unexpected PlibData layout: pass a PlibData whose .data frame carries "
            "perturbation/readout/context symbol columns (see perturblib tutorials)"
        )
    columns = {str(column) for column in frame.columns}
    perturbation_column = next(name for name in ("perturbation", "perturbation_symbol") if name in columns)
    context_column = next(name for name in ("context", "context_id") if name in columns)
    readout_column = next(name for name in ("readout", "readout_symbol") if name in columns)
    readout_symbols = sorted(set(frame[readout_column]))
    rows = [
        {perturbation_column: symbol, readout_column: readout, context_column: pdata.context_ids[0]}
        for symbol in symbols
        for readout in readout_symbols
    ]
    return pdata.__class__(pd.DataFrame(rows), context_ids=pdata.context_ids, covariates=pdata.covariates)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import perturb_lib  # noqa: F401
    except ImportError:
        print(
            "[run_lpm_predictions] perturblib is not installed; run this script inside the external "
            "LPM environment (docs/guides/perturbgen_bridge.md §7)",
            file=sys.stderr,
        )
        return 2

    candidates = _load_candidates(args.candidates_tsv.expanduser().resolve(strict=True))
    delta_matrix, seeds = _predict_deltas(args, candidates)

    import anndata as ad
    import numpy as np
    import pandas as pd

    from src.integration.perturbgen.external_perturbation_evidence import EXTERNAL_PREDICTION_SCHEMA_VERSION

    output_h5ad = args.output_h5ad.expanduser().resolve()
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata = ad.AnnData(
        X=np.asarray(delta_matrix, dtype=float),
        obs=pd.DataFrame(
            {
                "ensembl_id": [ensembl_id for ensembl_id, _ in candidates],
                "gene_symbol": [symbol for _, symbol in candidates],
                "context": [args.context] * len(candidates),
                "perturbation_semantics": [args.perturbation_semantics] * len(candidates),
            },
            index=[symbol for _, symbol in candidates],
        ),
    )
    adata.write_h5ad(output_h5ad)

    manifest_output = args.manifest_output.expanduser().resolve()
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": EXTERNAL_PREDICTION_SCHEMA_VERSION,
        "evidence_kind": "trained_response",
        "model": {
            "source": "gsk_lpm/perturblib",
            "perturblib_commit": args.perturblib_commit,
            "training_config": args.training_config,
            "license": "Apache-2.0 (perturblib); predictions inherit Replogle dataset terms",
        },
        "context": args.context,
        "perturbation_semantics": args.perturbation_semantics,
        "seeds": seeds,
        "aggregation": "mean over seed checkpoints of (perturbed - control) predicted readout",
        "predictions_h5ad": str(output_h5ad),
        "predictions_sha256": hashlib.sha256(output_h5ad.read_bytes()).hexdigest(),
        "readout_vocab": "canonical Ensembl (see h5ad var_names)",
    }
    manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"output": str(output_h5ad), "manifest": str(manifest_output), "seeds": seeds}, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
