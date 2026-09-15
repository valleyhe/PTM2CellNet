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
    parser.add_argument(
        "--donor-obs-column",
        default=None,
        help="obs column carrying donor identity; required with --train-donors/--held-out-donors",
    )
    parser.add_argument(
        "--train-donors",
        default=None,
        help="comma-separated training donor IDs; row-level donor binding (F-03)",
    )
    parser.add_argument(
        "--held-out-donors",
        default=None,
        help="comma-separated held-out donor IDs; their cells only enter the test NPZ",
    )
    parser.add_argument(
        "--donor-split-json",
        type=Path,
        default=None,
        help="write the canonical donor split payload next to pair_manifest.json",
    )
    parser.add_argument(
        "--state-obs-column",
        default=None,
        help="obs column carrying the disease state; required with --require-state-coverage",
    )
    parser.add_argument(
        "--require-state-coverage",
        action="store_true",
        help=(
            "fail when the held-out donor pool covers fewer than two states of "
            "--state-obs-column; the observed coverage is recorded in pair_manifest.json"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.require_state_coverage and args.state_obs_column is None:
        raise SystemExit("--require-state-coverage requires --state-obs-column")
    asset = load_perturbgen_embedding_asset(args.embedding_asset)
    donor_split_payload = None
    if args.train_donors is not None or args.held_out_donors is not None:
        if args.donor_obs_column is None:
            raise SystemExit("--train-donors/--held-out-donors require --donor-obs-column")
        from src.integration.perturbgen.donor_split import build_donor_split

        try:
            donor_split = build_donor_split(args.train_donors or "", args.held_out_donors or "")
        except Exception as exc:  # noqa: BLE001 - CLI boundary reports the cause verbatim
            raise SystemExit(str(exc)) from exc
        donor_split_payload = donor_split.to_payload()
        if args.donor_split_json is not None:
            donor_split_payload_path = Path(args.donor_split_json).expanduser().resolve()
            donor_split_payload_path.parent.mkdir(parents=True, exist_ok=True)
            donor_split_payload_path.write_text(
                json.dumps(donor_split_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
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
        donor_obs_column=args.donor_obs_column,
        donor_split=donor_split_payload,
        state_obs_column=args.state_obs_column,
        require_state_coverage=args.require_state_coverage,
    )
    print(json.dumps({"ok": True, "modality": args.modality, "splits": report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
