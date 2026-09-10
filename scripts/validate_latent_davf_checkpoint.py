#!/usr/bin/env python3
"""Validate a formal current LatentDAVF checkpoint.

The command intentionally rejects historical checkpoints, including artifacts
whose filenames or embedded labels say ``latent_dim=64``/``num_genes=4018``
but whose state dict belongs to the old ``delta_mlp`` model.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.davf_checkpoint_contract import load_current_davf_checkpoint
from src.models.perturbgen_embedding import load_perturbgen_embedding_asset
from src.models.scvi_adapter import ScVIAdapter, ScVIAdapterConfig


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a current formal LatentDAVF checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scvi-model", required=True)
    parser.add_argument("--embedding-asset", required=True)
    parser.add_argument(
        "--intervention-type",
        required=True,
        choices=("KO", "KD", "OE"),
        help="the direction this checkpoint is expected to serve",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    asset = load_perturbgen_embedding_asset(args.embedding_asset)
    adapter = ScVIAdapter.from_trained_model(
        args.scvi_model,
        config=ScVIAdapterConfig(
            model_path=str(args.scvi_model),
            n_latent=64,
            device=args.device,
        ),
    )
    payload, config = load_current_davf_checkpoint(
        args.checkpoint,
        asset=asset,
        scvi_adapter=adapter,
        map_location=args.device,
        expected_intervention_type=args.intervention_type,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
                "schema_version": payload["schema_version"],
                "model_type": payload["model_type"],
                "latent_dim": config.latent_dim,
                "num_genes": config.num_genes,
                "intervention_type": payload.get("training", {}).get("intervention_type"),
                "direction_code": payload.get("training", {}).get("direction_code"),
                "perturbgen_embedding_shape": list(asset.embeddings.shape),
                "scvi_shape": [adapter.n_genes, adapter.n_latent],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
