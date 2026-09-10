#!/usr/bin/env python3
"""Train and validate the 64-dimensional scVI model for a prepared GSE cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.gse_normal_disease import (
    FORMAL_DAVF_NUM_GENES,
    GSE_COUNTS_LAYER,
    GSE_NORMAL_DISEASE_SCHEMA_VERSION,
    _validate_direction_input,
)
from src.models.scvi_adapter import ScVIAdapter, ScVIAdapterConfig


FORMAL_LATENT_DIM = 64


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GSE-specific 64-dimensional scVI model")
    parser.add_argument("--input", required=True, help="prepared GSE H5AD")
    parser.add_argument("--output", required=True, help="scVI model directory")
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_gse_scvi_input(adata: object) -> dict[str, object]:
    """Validate the exact input contract before scVI mutates AnnData setup."""

    preparation = getattr(adata, "uns", {}).get("gse_normal_disease")
    if not isinstance(preparation, dict) or preparation.get("schema_version") != GSE_NORMAL_DISEASE_SCHEMA_VERSION:
        raise ValueError("input H5AD is missing the current gse_normal_disease preparation manifest")
    if getattr(adata, "n_vars", 0) != FORMAL_DAVF_NUM_GENES:
        raise ValueError(f"input H5AD must have {FORMAL_DAVF_NUM_GENES} genes")
    counts, obs, gene_names = _validate_direction_input(adata, counts_layer=GSE_COUNTS_LAYER)
    if tuple(map(str, getattr(adata, "var_names", ()))) != gene_names:
        raise ValueError("input H5AD var_names must equal var['ensembl_id'] in the same order")
    if "davf_batch" not in obs.columns:
        raise ValueError("input H5AD must contain davf_batch")
    if obs["davf_batch"].isna().any():
        raise ValueError("davf_batch contains missing values")
    normal_donors = set(obs.loc[obs["state"].astype(str) == "normal", "donor"].astype(str))
    disease_donors = set(obs.loc[obs["state"].astype(str) == "disease", "donor"].astype(str))
    if len(normal_donors) < 3 or len(disease_donors) < 3:
        raise ValueError("GSE scVI input requires at least three donors in each state")
    if getattr(adata, "n_obs", 0) < 10:
        raise ValueError("input H5AD must contain at least ten cells")
    return {
        "schema_version": GSE_NORMAL_DISEASE_SCHEMA_VERSION,
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "gene_names": list(gene_names),
        "normal_donors": sorted(normal_donors),
        "disease_donors": sorted(disease_donors),
        "counts_layer": GSE_COUNTS_LAYER,
        "counts_shape": list(counts.shape),
        "observational_only": True,
    }


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.max_epochs <= 0 or args.batch_size <= 0 or args.n_layers <= 0:
        raise ValueError("max-epochs, batch-size and n-layers must be positive")
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("--accelerator gpu was requested but CUDA is unavailable")
    try:
        import anndata as ad
        import scvi
        from scvi.model import SCVI
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise ImportError("anndata and scvi-tools are required for GSE scVI training") from exc

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"prepared GSE H5AD not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"scVI output already exists; pass --overwrite explicitly: {output_path}")
    adata = ad.read_h5ad(input_path)
    validation = validate_gse_scvi_input(adata)
    adata.obs["davf_batch"] = adata.obs["davf_batch"].astype("category")
    scvi.settings.seed = int(args.seed)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    SCVI.setup_anndata(adata, layer=GSE_COUNTS_LAYER, batch_key="davf_batch")
    model = SCVI(
        adata,
        n_latent=FORMAL_LATENT_DIM,
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
    adapter = ScVIAdapter.from_trained_model(
        output_path,
        config=ScVIAdapterConfig(
            model_path=str(output_path),
            n_latent=FORMAL_LATENT_DIM,
            gene_layer=GSE_COUNTS_LAYER,
            batch_key="davf_batch",
            device="cpu",
        ),
        adata=adata,
    )
    adapter.validate_compatibility(
        expected_latent_dim=FORMAL_LATENT_DIM,
        expected_num_genes=FORMAL_DAVF_NUM_GENES,
        expected_gene_names=tuple(map(str, adata.var_names)),
    )
    latent = np.asarray(adapter.encode(adata, batch_size=args.batch_size))
    if latent.shape != (adata.n_obs, FORMAL_LATENT_DIM) or not np.isfinite(latent).all():
        raise RuntimeError(f"GSE scVI encoder returned invalid shape {latent.shape}")
    report = {
        "schema_version": "ptm2cellnet.gse-scvi-training.v1",
        "input": str(input_path),
        "model_path": str(output_path),
        "latent_dim": adapter.n_latent,
        "num_genes": adapter.n_genes,
        "gene_names": list(adapter.gene_names),
        "batch_key": "davf_batch",
        "gene_layer": GSE_COUNTS_LAYER,
        "max_epochs": int(args.max_epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "accelerator": args.accelerator,
        "validation": validation,
        "observational_only": True,
        "formal_davf_checkpoint": False,
    }
    (output_path / "gse_scvi_training.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    report = train(args)
    print(
        json.dumps(
            {
                "ok": True,
                "model_path": report["model_path"],
                "latent_dim": report["latent_dim"],
                "num_genes": report["num_genes"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
