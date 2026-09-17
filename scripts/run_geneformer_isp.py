#!/usr/bin/env python3
"""Export frozen Geneformer in-silico perturbation assets for mainline candidates.

Runs in the external Geneformer environment (``pip install geneformer``; see
docs/guides/perturbgen_bridge.md §6b). Geneformer is the only evaluated source
that can run counterfactuals **inside our own disease cells**: the tokenizer
consumes the standardized AD cohort directly (Ensembl var, raw counts), and the
pretrained masked-gene model reports per-gene shifts after each candidate is
deleted / down-regulated / over-expressed in the chosen disease cell subset.

Output is the same ``external-perturbation-prediction/v1`` asset consumed by
``src/integration/perturbgen/external_perturbation_evidence.py`` with
``evidence_kind="network_counterfactual"`` and
``perturbation_semantics="in_silico_perturbation"`` — the evidence is a network
counterfactual, not a trained perturbation response and never causal validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-tsv", type=Path, required=True, help="ensembl_id/gene_symbol columns")
    parser.add_argument(
        "--input-h5ad", type=Path, required=True, help="disease-cell subset (Ensembl var, counts layer)"
    )
    parser.add_argument(
        "--cell-type-obs-value",
        required=True,
        help="value of the cell_type obs column to keep (e.g. EX); disease cells only, decided upstream",
    )
    parser.add_argument("--model-dir", type=Path, required=True, help="directory with the Geneformer-V2 checkpoint")
    parser.add_argument("--median-file", type=Path, required=True, help="gene_median_dictionary_gc104M.pkl")
    parser.add_argument("--token-dict", type=Path, required=True, help="token_dictionary_gc104M.pkl")
    parser.add_argument(
        "--perturb-mode",
        choices=("delete", "inhibit", "overexpress", "activate"),
        default="delete",
        help="Geneformer perturb_type (official vocabulary); inhibit/activate shift the token rank",
    )
    parser.add_argument("--n-cells", type=int, default=250, help="number of tokenized cells used per ISP run")
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import geneformer  # noqa: F401
    except ImportError:
        print(
            "[run_geneformer_isp] geneformer is not installed; run this script inside the external "
            "Geneformer environment (docs/guides/perturbgen_bridge.md §6b)",
            file=sys.stderr,
        )
        return 2

    import anndata as ad
    import numpy as np
    import pandas as pd

    EXTERNAL_PREDICTION_SCHEMA_VERSION = "ptm2cellnet.external-perturbation-prediction/v1"

    candidates = pd.read_csv(args.candidates_tsv.expanduser().resolve(strict=True), sep="\t")
    for column in ("ensembl_id", "gene_symbol"):
        if column not in candidates.columns:
            raise SystemExit(f"candidates TSV is missing column: {column}")

    work_dir = args.output_h5ad.expanduser().resolve().parent / "_isp_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    # 1) Restrict to the requested disease cells and tokenize (raw counts, Ensembl var).
    adata = ad.read_h5ad(args.input_h5ad.expanduser().resolve(strict=True))
    adata = adata[adata.obs["cell_type"].astype(str) == args.cell_type_obs_value].copy()
    adata = adata[adata.obs["state"].astype(str) == "disease"].copy()
    if adata.n_obs == 0:
        raise SystemExit(f"no disease cells for cell_type {args.cell_type_obs_value!r} in the input h5ad")
    adata = adata[: max(args.n_cells, adata.n_obs)] if adata.n_obs <= args.n_cells else adata[: args.n_cells]
    adata.X = adata.layers["counts"].copy()
    # Geneformer tokenizer contract: explicit ensembl_id var column + n_counts obs column.
    adata.var["ensembl_id"] = adata.var_names.astype(str)
    adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
    subset_path = work_dir / "input_subset.h5ad"
    adata.write_h5ad(subset_path)

    from geneformer import InSilicoPerturber, TranscriptomeTokenizer

    tokenizer = TranscriptomeTokenizer({"cell_type": "cell_type"})
    tokenizer.tokenize_data(str(work_dir), str(work_dir), "tokenized", file_format="h5ad")
    tokenized = work_dir / "tokenized.dataset"

    # 2) One ISP run per candidate; emb_mode="cell_and_gene" reports per-gene shifts.
    perturb_kwargs = {
        "delete": {"perturb_type": "delete"},
        "inhibit": {"perturb_type": "inhibit", "perturb_rank_shift": 1},
        "overexpress": {"perturb_type": "overexpress"},
        "activate": {"perturb_type": "activate", "perturb_rank_shift": 1},
    }[args.perturb_mode]
    model_directory = str(args.model_dir.expanduser().resolve(strict=True))
    per_candidate_cos: list[tuple[str, str, dict[int, float]]] = []
    for ensembl_id, gene_symbol in zip(candidates["ensembl_id"], candidates["gene_symbol"], strict=True):
        isp = InSilicoPerturber(
            emb_mode="cls_and_gene",
            genes_to_perturb=[str(ensembl_id)],
            token_dictionary_file=str(args.token_dict.expanduser().resolve(strict=True)),
            forward_batch_size=8,
            **perturb_kwargs,
        )
        isp_output_prefix = f"isp_{gene_symbol}"
        already_computed = sorted(work_dir.glob(f"in_silico_*_{isp_output_prefix}_gene_embs_dict_*.pickle"))
        if not already_computed:
            isp.perturb_data(
                model_directory,
                str(tokenized),
                str(work_dir),
                isp_output_prefix,
            )
        token_cos = _load_gene_shifts(work_dir, isp_output_prefix)
        per_candidate_cos.append((str(ensembl_id), str(gene_symbol), token_cos))

    # 3) Assemble the candidates x readouts delta matrix in canonical Ensembl space.
    import pickle

    with open(args.token_dict.expanduser().resolve(strict=True), "rb") as handle:
        token_dictionary = pickle.load(handle)
    token_to_ensembl = {}
    for token, token_id in token_dictionary.items():
        if isinstance(token, str) and token.startswith("ENSG"):
            token_to_ensembl[token_id] = token

    rows = []
    row_index = []
    unmapped_tokens: set[int] = set()
    for ensembl_id, gene_symbol, token_cos in per_candidate_cos:
        mapped = {token_to_ensembl[token]: cos for token, cos in token_cos.items() if token in token_to_ensembl}
        unmapped_tokens.update(token for token in token_cos if token not in token_to_ensembl)
        rows.append(mapped)
        row_index.append((ensembl_id, gene_symbol))
    # Each perturbation run reports only the genes observed in the perturbed
    # cells; export the intersection so every matrix cell is a real observed
    # value — zero-filling or NaN padding is not allowed.
    readout_intersection = set.intersection(*(set(row) for row in rows)) if rows else set()
    dropped_union = (set.union(*(set(row) for row in rows)) - readout_intersection) if rows else set()
    readout_ids = sorted(readout_intersection)
    delta_matrix = np.array([[row[readout] for readout in readout_ids] for row in rows], dtype=float)

    output_h5ad = args.output_h5ad.expanduser().resolve()
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata_out = ad.AnnData(
        X=delta_matrix,
        obs=pd.DataFrame(
            {
                "ensembl_id": [row[0] for row in row_index],
                "gene_symbol": [row[1] for row in row_index],
                "context": f"GSE174367_disease_{args.cell_type_obs_value}_ISP_{args.perturb_mode}",
                "perturbation_semantics": ["in_silico_perturbation"] * len(row_index),
            },
            index=[row[1] for row in row_index],
        ),
    )
    adata_out.var_names = readout_ids
    adata_out.write_h5ad(output_h5ad)

    manifest_output = args.manifest_output.expanduser().resolve()
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": EXTERNAL_PREDICTION_SCHEMA_VERSION,
        "evidence_kind": "network_counterfactual",
        "model": {
            "source": "geneformer/V2-104M",
            "geneformer_commit": _geneformer_commit(),
            "training_config": "Genecorpus-104M pretrained (masked gene modeling)",
            "license": "MIT (Geneformer); Genecorpus terms apply to the pretrained weights",
        },
        "context": f"GSE174367_disease_{args.cell_type_obs_value}_ISP_{args.perturb_mode}",
        "perturbation_semantics": "in_silico_perturbation",
        "seeds": [0],
        "aggregation": f"gene-embedding cosine similarities from {args.perturb_mode} counterfactuals on {adata.n_obs} disease cells (influence spectrum; no up/down expression direction)",
        "predictions_h5ad": str(output_h5ad),
        "predictions_sha256": hashlib.sha256(output_h5ad.read_bytes()).hexdigest(),
        "readout_vocab": "canonical Ensembl (Geneformer token space; unmapped tokens dropped)",
        "unmapped_token_count": len(unmapped_tokens),
        "input_h5ad": str(args.input_h5ad.expanduser().resolve()),
        "cell_type_obs_value": args.cell_type_obs_value,
        "n_cells": int(adata.n_obs),
    }
    manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_h5ad), "manifest": str(manifest_output)}, ensure_ascii=False))
    return 0


def _load_gene_shifts(work_dir: Path, prefix: str) -> dict[int, float]:
    """Read the Geneformer gene-embedding cosine-similarity pickle for one run.

    The official v2 ``InSilicoPerturber`` reports *embedding* cosine similarities
    between original and perturbed states, not expression deltas: the value for
    (perturbed_gene, affected_gene) measures how much the perturbation moves the
    affected gene's embedding (cos -> 1 means unchanged). The evidence is an
    influence spectrum and carries no up/down expression direction.
    """

    import pickle

    matches = sorted(
        path for path in work_dir.glob(f"in_silico_*_{prefix}_gene_embs_dict_*.pickle") if "filtered" not in path.name
    )
    if not matches:
        raise SystemExit(
            f"Geneformer wrote no gene_embs pickle for {prefix!r} under {work_dir}; "
            "align the ISP output layout with the installed geneformer version"
        )
    with matches[-1].open("rb") as handle:
        payload = pickle.load(handle)
    token_cos: dict[int, float] = {}
    for (_perturbed_token, affected_token), value in payload.items():
        cos = value[0] if isinstance(value, list) else value
        token_cos[int(affected_token)] = float(cos)
    return token_cos


def _geneformer_commit() -> str:
    try:
        from importlib.metadata import version

        return f"pypi:geneformer=={version('geneformer')}"
    except Exception:  # noqa: BLE001 - provenance is best-effort metadata
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
