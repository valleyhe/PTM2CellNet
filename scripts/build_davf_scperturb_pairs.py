#!/usr/bin/env python3
"""Build leakage-free latent-pair NPZ files for a KO or KD DAVF model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.davf_scperturb import build_scperturb_latent_pairs
from src.models.perturbgen_embedding import load_perturbgen_embedding_asset


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict latent DAVF pairs from prepared scPerturb data")
    parser.add_argument("--input", required=True, help="prepared H5AD")
    parser.add_argument("--scvi-model", required=True, help="matching 64-dimensional scVI model directory")
    parser.add_argument("--embedding-asset", required=True)
    parser.add_argument("--modality", required=True, choices=("KO", "KD"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cells-per-target", type=int, default=256)
    parser.add_argument("--encoder-batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu", help="scVI encoding device, e.g. cpu or cuda")
    parser.add_argument(
        "--split-strategy",
        default="target",
        choices=("target", "cell"),
        help="target: unseen-target extrapolation; cell: held-out cells for known targets",
    )
    parser.add_argument(
        "--control-baseline",
        default="mean",
        choices=("mean", "cell"),
        help="mean: historical control mean; cell: same-batch control cell for E2E context compatibility",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    asset = load_perturbgen_embedding_asset(args.embedding_asset)
    report = build_scperturb_latent_pairs(
        args.input,
        scvi_model_path=args.scvi_model,
        embedding_asset=asset,
        embedding_asset_path=args.embedding_asset,
        output_dir=args.output_dir,
        modality=args.modality,
        seed=args.seed,
        max_cells_per_target=args.max_cells_per_target,
        encoder_batch_size=args.encoder_batch_size,
        device=args.device,
        split_strategy=args.split_strategy,
        control_baseline=args.control_baseline,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    print(json.dumps({"ok": True, "modality": args.modality, "splits": report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
