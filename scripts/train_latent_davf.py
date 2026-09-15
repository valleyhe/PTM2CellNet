#!/usr/bin/env python3
"""Train the current formal latent DAVF checkpoint.

This entry point is deliberately separate from ``finetune_davf_e2e.py``.
That script trains a downstream PTM2CellNet classifier head; this script
trains the current ODE-based ``LatentDAVF`` itself with flow matching.

The input NPZ must follow :mod:`src.data.latent_davf_dataset` and contain
observed scVI latent pairs plus PerturbGen token IDs.  It is invalid to use a
gene-space delta table here because it does not identify the scVI latent pair
or the perturbation direction.

Example::

    python scripts/train_latent_davf.py \
        --train-data data/processed/davf_latent/train.npz \
        --val-data data/processed/davf_latent/val.npz \
        --scvi-model checkpoints/scvi/ibd_norman_model \
        --embedding-asset outputs/perturbgen/embedding_asset_20260822 \
        --output checkpoints/davf/latent_davf_perturbgen_4018
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.latent_davf_dataset import load_latent_davf_pairs
from src.models.davf_checkpoint_contract import (
    FORMAL_DAVF_LATENT_DIM,
    FORMAL_DAVF_NUM_GENES,
    build_current_davf_checkpoint,
    load_current_davf_checkpoint,
)
from src.models.latent_davf import LatentDAVF, LatentDAVFConfig
from src.models.perturbgen_embedding import load_perturbgen_embedding_asset
from src.models.scvi_adapter import ScVIAdapter, ScVIAdapterConfig
from src.integration.perturbgen.donor_split import (
    DonorSplitError,
    bind_frozen_donor_split,
    load_donor_split,
    optional_donor_split_from_args,
)

logger = logging.getLogger("train_latent_davf")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the formal current LatentDAVF checkpoint")
    parser.add_argument("--train-data", required=True, help="validated latent-pair training NPZ")
    parser.add_argument("--val-data", required=True, help="validated latent-pair validation NPZ")
    parser.add_argument("--scvi-model", required=True, help="trained scVI model directory")
    parser.add_argument(
        "--intervention-type",
        required=True,
        choices=("KO", "KD", "OE"),
        help="the single intervention direction represented by both pair files",
    )
    parser.add_argument(
        "--embedding-asset",
        required=True,
        help="verified PerturbGen embedding asset directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="output directory; writes best_model.pt and training_metrics.json",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--endpoint-loss-weight",
        type=float,
        default=1.0,
        help="weight of the t=0 boundary velocity loss; default 1.0",
    )
    parser.add_argument(
        "--direction-loss-weight",
        type=float,
        default=1.0,
        help="weight of the t=0 latent direction loss; default 1.0",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:N; default auto")
    parser.add_argument(
        "--train-donors",
        default=None,
        help="comma-separated training donor IDs; required with --held-out-donors",
    )
    parser.add_argument(
        "--held-out-donors",
        default=None,
        help="comma-separated held-out donor IDs; required with --train-donors",
    )
    parser.add_argument(
        "--frozen-cohort-manifest",
        default=None,
        help="optional frozen M6 manifest whose train/held-out lists must match this split",
    )
    parser.add_argument(
        "--require-donor-split",
        action="store_true",
        help="fail if train/held-out donor lists are omitted",
    )
    return parser.parse_args(argv)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(value: Optional[str]) -> torch.device:
    if value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {device} but CUDA is unavailable")
    return device


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _flow_matching_loss(model: LatentDAVF, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    outputs = model.forward_flow_matching(
        batch["z_0"],
        batch["z_1"],
        gene_ids=batch["gene_ids"],
        directions=batch["directions"],
        magnitudes=batch.get("magnitudes"),
        attention_mask=batch["attention_mask"],
    )
    return outputs["loss"]


def _endpoint_losses(
    model: LatentDAVF,
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anchor the velocity at the actual integration start ``t=0``.

    The production predictor starts Euler integration at ``t=0``.  Sampling
    ``t`` from a continuous uniform distribution never evaluates that boundary
    exactly, so a model can obtain a low interior flow loss while producing an
    unusable initial velocity.  This loss makes the training objective include
    the endpoint that inference actually uses.
    """

    batch_size = int(batch["z_0"].shape[0])
    t_zero = torch.zeros(
        batch_size,
        device=batch["z_0"].device,
        dtype=batch["z_0"].dtype,
    )
    outputs = model.forward_flow_matching(
        batch["z_0"],
        batch["z_1"],
        gene_ids=batch["gene_ids"],
        directions=batch["directions"],
        magnitudes=batch.get("magnitudes"),
        t=t_zero,
        attention_mask=batch["attention_mask"],
    )
    direction_loss = (
        1.0
        - F.cosine_similarity(
            outputs["v_t"],
            outputs["u_t"],
            dim=1,
            eps=1e-8,
        ).mean()
    )
    return outputs["loss"], direction_loss


def _endpoint_velocity_loss(model: LatentDAVF, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Return the endpoint magnitude loss for compatibility with callers."""

    endpoint_loss, _ = _endpoint_losses(model, batch)
    return endpoint_loss


def _run_epoch(
    model: LatentDAVF,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_grad_norm: float = 0.0,
    endpoint_loss_weight: float = 1.0,
    direction_loss_weight: float = 1.0,
) -> dict[str, float]:
    if endpoint_loss_weight < 0 or direction_loss_weight < 0:
        raise ValueError("endpoint-loss-weight and direction-loss-weight must be non-negative")
    training = optimizer is not None
    model.train(training)
    total_flow_loss = 0.0
    total_endpoint_loss = 0.0
    total_direction_loss = 0.0
    total_objective = 0.0
    total_samples = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
            flow_loss = _flow_matching_loss(model, batch)
            endpoint_loss, direction_loss = _endpoint_losses(model, batch)
            loss = flow_loss + endpoint_loss_weight * endpoint_loss + direction_loss_weight * direction_loss
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        else:
            # A fixed interpolation time makes validation comparable between
            # epochs; training still samples t uniformly in [0, 1].
            with torch.no_grad():
                flow_outputs = model.forward_flow_matching(
                    batch["z_0"],
                    batch["z_1"],
                    gene_ids=batch["gene_ids"],
                    directions=batch["directions"],
                    magnitudes=batch.get("magnitudes"),
                    t=torch.full(
                        (batch["z_0"].shape[0],),
                        0.5,
                        device=device,
                        dtype=batch["z_0"].dtype,
                    ),
                    attention_mask=batch["attention_mask"],
                )
                flow_loss = flow_outputs["loss"]
                endpoint_loss, direction_loss = _endpoint_losses(model, batch)
                loss = flow_loss + endpoint_loss_weight * endpoint_loss + direction_loss_weight * direction_loss
        count = int(batch["z_0"].shape[0])
        total_flow_loss += float(flow_loss.detach().cpu()) * count
        total_endpoint_loss += float(endpoint_loss.detach().cpu()) * count
        total_direction_loss += float(direction_loss.detach().cpu()) * count
        total_objective += float(loss.detach().cpu()) * count
        total_samples += count
    if total_samples == 0:
        raise RuntimeError("latent DAVF loader produced no samples")
    return {
        "loss": total_objective / total_samples,
        "flow_loss": total_flow_loss / total_samples,
        "endpoint_loss": total_endpoint_loss / total_samples,
        "direction_loss": total_direction_loss / total_samples,
    }


def _build_model(asset_embeddings: torch.Tensor) -> LatentDAVF:
    config = LatentDAVFConfig(
        latent_dim=FORMAL_DAVF_LATENT_DIM,
        num_genes=FORMAL_DAVF_NUM_GENES,
    )
    return LatentDAVF(config, pretrained_gene_embeddings=asset_embeddings)


def _validate_intervention_direction(dataset: Any, *, split: str, intervention_type: str) -> None:
    expected_code = {"KO": 0, "KD": 1, "OE": 2}[intervention_type]
    active = dataset.directions[dataset.attention_mask.to(dtype=torch.bool)]
    if active.numel() == 0 or set(int(value) for value in active.detach().cpu().tolist()) != {expected_code}:
        observed = sorted(set(int(value) for value in active.detach().cpu().tolist()))
        raise ValueError(
            f"{split} contains active directions {observed}; expected only "
            f"{intervention_type} (direction code {expected_code})"
        )


def _validate_donor_split_metadata(dataset: Any, split_name: str, donor_split: Any) -> None:
    if donor_split is None:
        return

    metadata = getattr(dataset, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{split_name} NPZ metadata must contain donor_split and dataset.donor_rows")

    metadata_split = metadata.get("donor_split")
    if not isinstance(metadata_split, Mapping):
        raise ValueError(f"{split_name} NPZ metadata must contain a donor_split mapping")
    recorded = load_donor_split(metadata_split)
    if recorded.sha256 != donor_split.sha256:
        raise ValueError(
            f"{split_name} NPZ donor_split sha256 {recorded.sha256} does not match CLI split {donor_split.sha256}"
        )

    dataset_metadata = metadata.get("dataset")
    donor_rows = dataset_metadata.get("donor_rows") if isinstance(dataset_metadata, Mapping) else None
    if not isinstance(donor_rows, list) or not donor_rows:
        raise ValueError(f"{split_name} NPZ metadata.dataset.donor_rows must be a non-empty per-row donor list")

    train_donors = set(donor_split.train_donors)
    for row_index, row in enumerate(donor_rows):
        row_donors = [row] if isinstance(row, str) else row
        if (
            not isinstance(row_donors, list)
            or not row_donors
            or not all(isinstance(donor, str) and donor for donor in row_donors)
        ):
            raise ValueError(f"{split_name} NPZ metadata.dataset.donor_rows[{row_index}] is invalid")
        invalid = sorted(set(row_donors) - train_donors)
        if invalid:
            raise ValueError(
                f"{split_name} NPZ metadata.dataset.donor_rows[{row_index}] contains donor(s) "
                f"outside CLI train_donors: {invalid}"
            )


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.max_grad_norm < 0:
        raise ValueError("learning-rate must be positive; weight-decay/max-grad-norm must be non-negative")
    if args.endpoint_loss_weight < 0:
        raise ValueError("endpoint-loss-weight must be non-negative")
    if args.direction_loss_weight < 0:
        raise ValueError("direction-loss-weight must be non-negative")
    if args.patience < 0 or args.num_workers < 0:
        raise ValueError("patience and num-workers must be non-negative")

    _set_seed(args.seed)
    device = _resolve_device(args.device)
    try:
        donor_split = optional_donor_split_from_args(
            train_donors=args.train_donors,
            held_out_donors=args.held_out_donors,
            require=bool(args.require_donor_split or args.frozen_cohort_manifest),
        )
    except DonorSplitError as exc:
        raise ValueError(str(exc)) from exc
    if donor_split is not None and args.frozen_cohort_manifest:
        from src.integration.perturbgen.frozen_cohort import load_frozen_manifest

        frozen = load_frozen_manifest(args.frozen_cohort_manifest)
        bind_frozen_donor_split(donor_split, frozen)
    asset = load_perturbgen_embedding_asset(args.embedding_asset)
    adapter = ScVIAdapter.from_trained_model(
        args.scvi_model,
        config=ScVIAdapterConfig(
            model_path=str(args.scvi_model),
            n_latent=FORMAL_DAVF_LATENT_DIM,
            device=str(device),
        ),
    )
    adapter.validate_compatibility(
        expected_latent_dim=FORMAL_DAVF_LATENT_DIM,
        expected_num_genes=FORMAL_DAVF_NUM_GENES,
    )

    train_dataset = load_latent_davf_pairs(
        args.train_data,
        latent_dim=FORMAL_DAVF_LATENT_DIM,
        perturbgen_vocab_size=asset.vocab_size,
        expected_scvi_model_path=args.scvi_model,
        expected_scvi_gene_names=adapter.gene_names,
        expected_embedding_manifest=asset.manifest,
        expected_embedding_path=args.embedding_asset,
        expected_embedding_dim=asset.embedding_dim,
        expected_intervention_type=args.intervention_type,
    )
    val_dataset = load_latent_davf_pairs(
        args.val_data,
        latent_dim=FORMAL_DAVF_LATENT_DIM,
        perturbgen_vocab_size=asset.vocab_size,
        expected_scvi_model_path=args.scvi_model,
        expected_scvi_gene_names=adapter.gene_names,
        expected_embedding_manifest=asset.manifest,
        expected_embedding_path=args.embedding_asset,
        expected_embedding_dim=asset.embedding_dim,
        expected_intervention_type=args.intervention_type,
    )
    _validate_intervention_direction(
        train_dataset,
        split="train",
        intervention_type=args.intervention_type,
    )
    _validate_intervention_direction(
        val_dataset,
        split="val",
        intervention_type=args.intervention_type,
    )
    for split_name, dataset in (("train", train_dataset), ("val", val_dataset)):
        _validate_donor_split_metadata(dataset, split_name, donor_split)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = _build_model(asset.embeddings).to(device)
    if model.gene_embed_table is None:
        raise RuntimeError("current LatentDAVF must contain the PerturbGen gene embedding table")
    model.gene_embed_table.weight.requires_grad_(False)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    metrics: list[dict[str, float | int]] = []
    best_val_endpoint_loss = float("inf")
    best_val_flow_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    logger.info(
        "training current LatentDAVF: device=%s train=%d val=%d asset=%dx%d scVI=%dx%d",
        device,
        len(train_dataset),
        len(val_dataset),
        asset.vocab_size,
        asset.embedding_dim,
        adapter.n_genes,
        adapter.n_latent,
    )
    for epoch in range(1, args.epochs + 1):
        train_stats = _run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            max_grad_norm=args.max_grad_norm,
            endpoint_loss_weight=args.endpoint_loss_weight,
            direction_loss_weight=args.direction_loss_weight,
        )
        val_stats = _run_epoch(
            model,
            val_loader,
            device,
            endpoint_loss_weight=args.endpoint_loss_weight,
            direction_loss_weight=args.direction_loss_weight,
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_flow_loss": train_stats["flow_loss"],
            "train_endpoint_loss": train_stats["endpoint_loss"],
            "train_direction_loss": train_stats["direction_loss"],
            "val_loss": val_stats["loss"],
            "val_flow_loss": val_stats["flow_loss"],
            "val_endpoint_loss": val_stats["endpoint_loss"],
            "val_direction_loss": val_stats["direction_loss"],
        }
        metrics.append(epoch_metrics)
        logger.info(
            "epoch=%d train_loss=%.6f train_endpoint=%.6f train_direction=%.6f "
            "val_loss=%.6f val_flow=%.6f val_endpoint=%.6f val_direction=%.6f",
            epoch,
            train_stats["loss"],
            train_stats["endpoint_loss"],
            train_stats["direction_loss"],
            val_stats["loss"],
            val_stats["flow_loss"],
            val_stats["endpoint_loss"],
            val_stats["direction_loss"],
        )

        if val_stats["endpoint_loss"] < best_val_endpoint_loss:
            best_val_endpoint_loss = val_stats["endpoint_loss"]
            best_val_flow_loss = val_stats["flow_loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            payload = build_current_davf_checkpoint(
                model,
                asset=asset,
                embedding_asset_path=args.embedding_asset,
                scvi_model_path=args.scvi_model,
                scvi_adapter=adapter,
                training={
                    "epoch": epoch,
                    "best_val_loss": best_val_endpoint_loss,
                    "best_val_endpoint_loss": best_val_endpoint_loss,
                    "best_val_flow_loss": best_val_flow_loss,
                    "endpoint_loss_weight": args.endpoint_loss_weight,
                    "direction_loss_weight": args.direction_loss_weight,
                    "seed": args.seed,
                    "train_samples": len(train_dataset),
                    "val_samples": len(val_dataset),
                    "data_contract": "ptm2cellnet.latent-davf-pairs.v1",
                    "gene_embeddings_frozen": True,
                    "intervention_type": args.intervention_type,
                    "direction_code": {"KO": 0, "KD": 1, "OE": 2}[args.intervention_type],
                    "donor_split": None if donor_split is None else donor_split.to_payload(),
                    "donor_split_status": "bound" if donor_split is not None else "unspecified",
                },
            )
            torch.save(payload, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience > 0:
                logger.info("early stopping at epoch %d", epoch)
                break

    if not checkpoint_path.is_file():
        raise RuntimeError("training completed without exporting a best current DAVF checkpoint")

    # Re-open the serialized file through the safe current-checkpoint path. It
    # verifies that the artifact, not merely the in-memory model, is usable.
    load_current_davf_checkpoint(
        checkpoint_path,
        asset=asset,
        scvi_adapter=adapter,
        map_location="cpu",
        expected_intervention_type=args.intervention_type,
    )
    report = {
        "ok": True,
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_endpoint_loss,
        "best_val_endpoint_loss": best_val_endpoint_loss,
        "best_val_flow_loss": best_val_flow_loss,
        "endpoint_loss_weight": args.endpoint_loss_weight,
        "direction_loss_weight": args.direction_loss_weight,
        "device": str(device),
        "scvi_model": str(Path(args.scvi_model).expanduser().resolve()),
        "embedding_asset": str(Path(args.embedding_asset).expanduser().resolve()),
        "asset_shape": [asset.vocab_size, asset.embedding_dim],
        "scvi_shape": [adapter.n_genes, adapter.n_latent],
        "donor_split": None if donor_split is None else donor_split.to_payload(),
        "donor_split_status": "bound" if donor_split is not None else "unspecified",
        "metrics": metrics,
    }
    (output_dir / "training_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    report = train(args)
    print(json.dumps({key: value for key, value in report.items() if key != "metrics"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
