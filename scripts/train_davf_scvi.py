#!/usr/bin/env python3
"""Train the exact 64-dimensional scVI model used by one DAVF modality.

The input H5AD must be produced by ``prepare_davf_scperturb.py``. A new scVI
model is trained for each modality so KO and KD never share incompatible
latent coordinates. The saved model uses ``davf_batch`` and has exactly 4018
decoder genes in the H5AD order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.davf_scperturb import FORMAL_DAVF_LATENT_DIM, FORMAL_DAVF_NUM_GENES
from src.models.scvi_adapter import ScVIAdapter, ScVIAdapterConfig


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a modality-specific scVI model for DAVF")
    parser.add_argument("--input", required=True, help="prepared 4018-gene H5AD")
    parser.add_argument("--modality", required=True, choices=("KO", "KD"))
    parser.add_argument("--output", required=True, help="scVI model directory")
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.max_epochs <= 0 or args.batch_size <= 0 or args.n_layers <= 0:
        raise ValueError("max-epochs, batch-size and n-layers must be positive")
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("--accelerator gpu was requested but CUDA is unavailable")

    import anndata as ad
    import scvi
    from scvi.model import SCVI

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"prepared H5AD not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"scVI output already exists; pass --overwrite explicitly: {output_path}")

    adata = ad.read_h5ad(input_path)
    preparation = adata.uns.get("davf_preparation")
    if not isinstance(preparation, dict):
        raise ValueError("input H5AD is missing uns['davf_preparation']")
    if preparation.get("modality") != args.modality:
        raise ValueError(
            f"input H5AD modality={preparation.get('modality')!r} does not match {args.modality!r}"
        )
    if adata.n_vars != FORMAL_DAVF_NUM_GENES:
        raise ValueError(f"input H5AD must have {FORMAL_DAVF_NUM_GENES} genes, got {adata.n_vars}")
    if tuple(map(str, adata.var_names)) != tuple(preparation.get("gene_names", ())):
        raise ValueError("input H5AD gene order does not match its preparation manifest")
    if "davf_batch" not in adata.obs:
        raise ValueError("input H5AD must contain the davf_batch covariate")
    if adata.n_obs < 10:
        raise ValueError("input H5AD must contain at least ten cells")
    if adata.obs["davf_batch"].isna().any():
        raise ValueError("davf_batch contains missing values")

    adata.obs["davf_batch"] = adata.obs["davf_batch"].astype("category")
    scvi.settings.seed = int(args.seed)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    SCVI.setup_anndata(adata, batch_key="davf_batch")
    model = SCVI(
        adata,
        n_latent=FORMAL_DAVF_LATENT_DIM,
        n_layers=args.n_layers,
        gene_likelihood="nb",
    )
    model.train(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        accelerator=args.accelerator,
        devices=1,
        early_stopping=args.early_stopping,
        check_val_every_n_epoch=1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path), overwrite=args.overwrite, save_anndata=False)

    # Reload through the project's adapter. This catches a saved registry or
    # device contract problem before pair construction starts.
    adapter = ScVIAdapter.from_trained_model(
        output_path,
        config=ScVIAdapterConfig(
            model_path=str(output_path),
            n_latent=FORMAL_DAVF_LATENT_DIM,
            batch_key="davf_batch",
            device="cpu",
        ),
        adata=adata,
    )
    adapter.validate_compatibility(
        expected_latent_dim=FORMAL_DAVF_LATENT_DIM,
        expected_num_genes=FORMAL_DAVF_NUM_GENES,
        expected_gene_names=tuple(map(str, adata.var_names)),
    )
    report = {
        "schema_version": "ptm2cellnet.davf-scvi-training.v1",
        "modality": args.modality,
        "direction_code": 0 if args.modality == "KO" else 1,
        "input": str(input_path),
        "model_path": str(output_path),
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "latent_dim": adapter.n_latent,
        "num_genes": adapter.n_genes,
        "gene_names": list(adapter.gene_names),
        "batch_key": "davf_batch",
        "max_epochs": int(args.max_epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "accelerator": args.accelerator,
    }
    (output_path / "davf_scvi_training.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    report = train(args)
    print(
        json.dumps(
            {
                "ok": True,
                "modality": report["modality"],
                "model_path": report["model_path"],
                "shape": report["shape"],
                "latent_dim": report["latent_dim"],
                "num_genes": report["num_genes"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
