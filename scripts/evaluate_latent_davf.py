#!/usr/bin/env python3
"""Evaluate a current latent DAVF checkpoint on a held-out latent-pair split."""

from __future__ import annotations

import argparse
import json
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
    load_current_davf_checkpoint,
)
from src.models.latent_davf import LatentDAVF
from src.models.perturbgen_embedding import load_perturbgen_embedding_asset
from src.models.scvi_adapter import ScVIAdapter, ScVIAdapterConfig


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a current latent DAVF checkpoint")
    parser.add_argument("--test-data", required=True, help="held-out latent-pair NPZ")
    parser.add_argument("--checkpoint", required=True, help="current schema-v2 DAVF checkpoint")
    parser.add_argument("--scvi-model", required=True, help="matching scVI model directory")
    parser.add_argument("--embedding-asset", required=True, help="verified PerturbGen embedding asset")
    parser.add_argument(
        "--intervention-type",
        required=True,
        choices=("KO", "KD", "OE"),
    )
    parser.add_argument(
        "--context-h5ad",
        default=None,
        help="real AnnData context for scVI decoding; defaults to pair metadata provenance",
    )
    parser.add_argument("--output", required=True, help="JSON evaluation report")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument(
        "--gene-direction-epsilon",
        type=float,
        default=0.0,
        help="absolute decoded expression delta treated as inconclusive; default 0",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:N; default auto")
    return parser.parse_args(argv)


def _resolve_device(value: Optional[str]) -> torch.device:
    if value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {device} but CUDA is unavailable")
    return device


def compute_endpoint_metrics(
    z_0: torch.Tensor,
    z_1: torch.Tensor,
    prediction: torch.Tensor,
) -> dict[str, float]:
    """Compute endpoint and identity-baseline errors for one or more batches."""

    if z_0.ndim != 2 or z_1.shape != z_0.shape or prediction.shape != z_0.shape:
        raise ValueError("z_0, z_1, and prediction must have the same 2D shape")
    if not torch.isfinite(z_0).all() or not torch.isfinite(z_1).all() or not torch.isfinite(prediction).all():
        raise ValueError("latent evaluation inputs must be finite")

    endpoint_error = prediction - z_1
    identity_error = z_0 - z_1
    target_delta = z_1 - z_0
    predicted_delta = prediction - z_0
    identity_mse = float(identity_error.square().mean().detach().cpu())
    if identity_mse <= 0.0:
        raise ValueError("held-out latent pairs have zero identity baseline error")
    endpoint_mse = float(endpoint_error.square().mean().detach().cpu())
    target_delta_mean_square = target_delta.square().mean()
    if float(target_delta_mean_square.detach().cpu()) <= 0.0:
        raise ValueError("held-out latent pairs have zero target delta")
    direction_mask = target_delta.abs() > 1e-8
    direction_count = int(direction_mask.sum().detach().cpu())
    if direction_count == 0:
        raise ValueError("held-out latent pairs have no non-zero target direction")
    return {
        "endpoint_mse": endpoint_mse,
        "endpoint_mae": float(endpoint_error.abs().mean().detach().cpu()),
        "identity_mse": identity_mse,
        "relative_to_identity_mse": endpoint_mse / identity_mse,
        "target_delta_rmse": float(target_delta.square().mean().sqrt().detach().cpu()),
        "predicted_delta_rmse": float(predicted_delta.square().mean().sqrt().detach().cpu()),
        "latent_delta_cosine_mean": float(
            F.cosine_similarity(predicted_delta, target_delta, dim=1, eps=1e-8).mean().detach().cpu()
        ),
        "latent_delta_sign_accuracy": float(
            (torch.sign(predicted_delta[direction_mask]) == torch.sign(target_delta[direction_mask]))
            .to(dtype=torch.float32)
            .mean()
            .detach()
            .cpu()
        ),
        "predicted_delta_norm_ratio": float(
            (predicted_delta.square().mean() / target_delta_mean_square).sqrt().detach().cpu()
        ),
    }


def compute_gene_direction_metrics(
    baseline_expression: np.ndarray | torch.Tensor,
    observed_expression: np.ndarray | torch.Tensor,
    prediction_expression: np.ndarray | torch.Tensor,
    *,
    epsilon: float = 0.0,
) -> dict[str, float]:
    """Compare decoded target-gene endpoints and direction signs.

    ``observed_expression`` comes from the held-out pair's real target latent,
    while ``prediction_expression`` comes from DAVF's predicted latent. A
    zero observed delta is reported as inconclusive and is not included in the
    sign-accuracy denominator.
    """

    baseline = torch.as_tensor(baseline_expression, dtype=torch.float32)
    observed = torch.as_tensor(observed_expression, dtype=torch.float32)
    prediction = torch.as_tensor(prediction_expression, dtype=torch.float32)
    if baseline.ndim != 1 or observed.shape != baseline.shape or prediction.shape != baseline.shape:
        raise ValueError("decoded target-gene arrays must be 1-D and have the same shape")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("gene-direction epsilon must be finite and >= 0")
    if not torch.isfinite(baseline).all() or not torch.isfinite(observed).all() or not torch.isfinite(prediction).all():
        raise ValueError("decoded target-gene arrays must be finite")

    observed_delta = observed - baseline
    predicted_delta = prediction - baseline
    observed_nonzero = observed_delta.abs() > epsilon
    nonzero_count = int(observed_nonzero.sum().item())
    if nonzero_count == 0:
        raise ValueError("held-out target-gene deltas are all inconclusive")
    observed_delta_mse = observed_delta.square().mean()
    predicted_nonzero = predicted_delta.abs() > epsilon
    return {
        "gene_endpoint_mse": float((prediction - observed).square().mean().item()),
        "gene_endpoint_mae": float((prediction - observed).abs().mean().item()),
        "gene_observed_delta_rmse": float(observed_delta_mse.sqrt().item()),
        "gene_predicted_delta_rmse": float(predicted_delta.square().mean().sqrt().item()),
        "gene_direction_sign_accuracy": float(
            (torch.sign(predicted_delta[observed_nonzero]) == torch.sign(observed_delta[observed_nonzero]))
            .to(dtype=torch.float32)
            .mean()
            .item()
        ),
        "gene_direction_sign_total": float(nonzero_count),
        "gene_direction_inconclusive_fraction": float((~observed_nonzero).to(dtype=torch.float32).mean().item()),
        "gene_predicted_nonzero_fraction": float(predicted_nonzero.to(dtype=torch.float32).mean().item()),
        "gene_predicted_delta_norm_ratio": float((predicted_delta.square().mean() / observed_delta_mse).sqrt().item()),
    }


def _load_evaluation_context(
    dataset: Any,
    test_data: str | Path,
    explicit_context_path: str | Path | None,
) -> tuple[Any, np.ndarray, Path]:
    """Load the real AnnData and target-cell IDs needed for gene decoding."""

    with np.load(Path(test_data).expanduser().resolve(), allow_pickle=False) as npz:
        if "target_cell_ids" not in npz.files:
            raise ValueError("gene-level DAVF evaluation requires target_cell_ids in the latent-pair NPZ")
        target_cell_ids = np.asarray(npz["target_cell_ids"])
    if target_cell_ids.ndim != 1 or target_cell_ids.dtype.kind not in {"U", "S"}:
        raise ValueError("target_cell_ids must be a 1-D string array")
    if target_cell_ids.shape[0] != len(dataset):
        raise ValueError("target_cell_ids length does not match latent-pair samples")

    context_value: str | Path | None = explicit_context_path
    if context_value is None:
        metadata = getattr(dataset, "metadata", {})
        dataset_metadata = metadata.get("dataset") if isinstance(metadata, Mapping) else None
        if isinstance(dataset_metadata, Mapping):
            context_value = dataset_metadata.get("prepared_anndata")
    if not isinstance(context_value, (str, Path)) or not str(context_value):
        raise ValueError("gene-level DAVF evaluation requires --context-h5ad or dataset.prepared_anndata provenance")

    context_path = Path(context_value).expanduser().resolve()
    if not context_path.is_file():
        raise FileNotFoundError(f"DAVF evaluation context AnnData not found: {context_path}")
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise ImportError("anndata is required for gene-level DAVF evaluation") from exc
    context = ad.read_h5ad(context_path)
    missing = context.obs_names.get_indexer(target_cell_ids)
    if (missing < 0).any():
        missing_ids = target_cell_ids[missing < 0][:3].tolist()
        raise ValueError(f"evaluation context is missing target cell IDs: {missing_ids}")
    return context, target_cell_ids, context_path


def _resolve_target_decoder_indices(
    gene_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    token_to_gene: Mapping[int, str],
    scvi_adapter: Any,
) -> np.ndarray:
    """Resolve one active PerturbGen token per pair to live scVI columns."""

    active = attention_mask.to(dtype=torch.bool)
    active_counts = active.sum(dim=1)
    if not torch.all(active_counts == 1):
        raise ValueError("gene-level DAVF evaluation requires exactly one active target token per sample")
    active_tokens = gene_ids[active].detach().cpu().tolist()
    try:
        target_gene_names = [token_to_gene[int(token)] for token in active_tokens]
    except KeyError as exc:
        raise ValueError(f"PerturbGen token {exc.args[0]!r} has no Ensembl ID in the verified asset") from exc
    indices = scvi_adapter.resolve_target_gene_indices(target_gene_names)
    return np.asarray(indices, dtype=np.int64)


def _evaluate_model(
    model: LatentDAVF,
    dataset: Any,
    *,
    embedding_asset: Any,
    scvi_adapter: Any,
    context_adata: Any,
    target_cell_ids: np.ndarray,
    gene_direction_epsilon: float,
    batch_size: int,
    num_workers: int,
    num_steps: int,
    device: torch.device,
) -> dict[str, float | int]:
    if batch_size <= 0 or num_workers < 0 or num_steps <= 0:
        raise ValueError("batch-size and num-steps must be positive; num-workers must be non-negative")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    totals = {
        "endpoint_squared_error": 0.0,
        "endpoint_absolute_error": 0.0,
        "identity_squared_error": 0.0,
        "target_delta_squared": 0.0,
        "predicted_delta_squared": 0.0,
        "direction_cosine": 0.0,
        "direction_sign_correct": 0,
        "direction_sign_total": 0,
        "gene_endpoint_squared_error": 0.0,
        "gene_endpoint_absolute_error": 0.0,
        "gene_observed_delta_squared": 0.0,
        "gene_predicted_delta_squared": 0.0,
        "gene_direction_sign_correct": 0,
        "gene_direction_sign_total": 0,
        "gene_direction_inconclusive": 0,
        "gene_predicted_nonzero": 0,
        "gene_samples": 0,
        "elements": 0,
        "samples": 0,
    }
    token_to_gene = {int(token): str(gene) for gene, token in embedding_asset.gene_to_token.items()}
    context_offset = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            z_0 = batch["z_0"].to(device=device, dtype=torch.float32)
            z_1 = batch["z_1"].to(device=device, dtype=torch.float32)
            prediction = model.predict(
                z_0,
                gene_ids=batch["gene_ids"].to(device),
                directions=batch["directions"].to(device),
                magnitudes=None if "magnitudes" not in batch else batch["magnitudes"].to(device),
                num_steps=num_steps,
                attention_mask=batch["attention_mask"].to(device),
            )
            if prediction.shape != z_1.shape:
                raise ValueError(f"DAVF endpoint prediction shape {tuple(prediction.shape)} does not match z_1")
            endpoint_error = prediction - z_1
            identity_error = z_0 - z_1
            target_delta = z_1 - z_0
            predicted_delta = prediction - z_0
            count = int(z_0.shape[0])
            totals["endpoint_squared_error"] += float(endpoint_error.square().sum().cpu())
            totals["endpoint_absolute_error"] += float(endpoint_error.abs().sum().cpu())
            totals["identity_squared_error"] += float(identity_error.square().sum().cpu())
            totals["target_delta_squared"] += float(target_delta.square().sum().cpu())
            totals["predicted_delta_squared"] += float(predicted_delta.square().sum().cpu())
            totals["direction_cosine"] += float(
                F.cosine_similarity(predicted_delta, target_delta, dim=1, eps=1e-8).sum().cpu()
            )
            direction_mask = target_delta.abs() > 1e-8
            totals["direction_sign_correct"] += int(
                (torch.sign(predicted_delta[direction_mask]) == torch.sign(target_delta[direction_mask])).sum().cpu()
            )
            totals["direction_sign_total"] += int(direction_mask.sum().cpu())
            totals["elements"] += int(z_0.numel())
            totals["samples"] += count

            batch_cell_ids = target_cell_ids[context_offset : context_offset + count]
            context_offset += count
            context_batch = context_adata[list(batch_cell_ids)].copy()
            decoder_indices = _resolve_target_decoder_indices(
                batch["gene_ids"],
                batch["attention_mask"],
                token_to_gene=token_to_gene,
                scvi_adapter=scvi_adapter,
            )
            baseline_expression = np.asarray(
                scvi_adapter.decode(z_0.detach().cpu().numpy(), adata=context_batch),
                dtype=np.float64,
            )
            observed_expression = np.asarray(
                scvi_adapter.decode(z_1.detach().cpu().numpy(), adata=context_batch),
                dtype=np.float64,
            )
            predicted_expression = np.asarray(
                scvi_adapter.decode(prediction.detach().cpu().numpy(), adata=context_batch),
                dtype=np.float64,
            )
            expected_expression_shape = (count, scvi_adapter.n_genes)
            if (
                baseline_expression.shape != expected_expression_shape
                or observed_expression.shape != expected_expression_shape
                or predicted_expression.shape != expected_expression_shape
            ):
                raise ValueError("scVI decoder returned an unexpected expression shape during evaluation")
            if (
                not np.isfinite(baseline_expression).all()
                or not np.isfinite(observed_expression).all()
                or not np.isfinite(predicted_expression).all()
            ):
                raise ValueError("scVI decoder returned non-finite expression values during evaluation")
            row_indices = np.arange(count)
            baseline_target = baseline_expression[row_indices, decoder_indices]
            observed_target = observed_expression[row_indices, decoder_indices]
            predicted_target = predicted_expression[row_indices, decoder_indices]
            # Compute once per batch to fail on an all-inconclusive batch only
            # at the aggregate level below.
            observed_delta = observed_target - baseline_target
            predicted_delta = predicted_target - baseline_target
            observed_nonzero = np.abs(observed_delta) > gene_direction_epsilon
            totals["gene_endpoint_squared_error"] += float(np.square(predicted_target - observed_target).sum())
            totals["gene_endpoint_absolute_error"] += float(np.abs(predicted_target - observed_target).sum())
            totals["gene_observed_delta_squared"] += float(np.square(observed_delta).sum())
            totals["gene_predicted_delta_squared"] += float(np.square(predicted_delta).sum())
            totals["gene_direction_sign_correct"] += int(
                (np.sign(predicted_delta[observed_nonzero]) == np.sign(observed_delta[observed_nonzero])).sum()
            )
            totals["gene_direction_sign_total"] += int(observed_nonzero.sum())
            totals["gene_direction_inconclusive"] += int((~observed_nonzero).sum())
            totals["gene_predicted_nonzero"] += int((np.abs(predicted_delta) > gene_direction_epsilon).sum())
            totals["gene_samples"] += count

    if totals["samples"] == 0 or totals["elements"] == 0:
        raise RuntimeError("held-out latent DAVF loader produced no samples")
    identity_mse = totals["identity_squared_error"] / totals["elements"]
    if identity_mse <= 0.0:
        raise ValueError("held-out latent pairs have zero identity baseline error")
    if totals["direction_sign_total"] == 0:
        raise ValueError("held-out latent pairs have no non-zero target direction")
    if totals["gene_samples"] != totals["samples"]:
        raise RuntimeError("gene-level evaluation did not decode every latent-pair sample")
    if totals["gene_direction_sign_total"] == 0:
        raise ValueError("held-out target-gene deltas are all inconclusive")
    target_delta_mse = totals["target_delta_squared"] / totals["elements"]
    if target_delta_mse <= 0.0:
        raise ValueError("held-out latent pairs have zero target delta")
    endpoint_mse = totals["endpoint_squared_error"] / totals["elements"]
    return {
        "samples": int(totals["samples"]),
        "endpoint_mse": endpoint_mse,
        "endpoint_mae": totals["endpoint_absolute_error"] / totals["elements"],
        "identity_mse": identity_mse,
        "relative_to_identity_mse": endpoint_mse / identity_mse,
        "target_delta_rmse": (totals["target_delta_squared"] / totals["elements"]) ** 0.5,
        "predicted_delta_rmse": (totals["predicted_delta_squared"] / totals["elements"]) ** 0.5,
        "latent_delta_cosine_mean": totals["direction_cosine"] / totals["samples"],
        "latent_delta_sign_accuracy": (totals["direction_sign_correct"] / totals["direction_sign_total"]),
        "predicted_delta_norm_ratio": (totals["predicted_delta_squared"] / totals["target_delta_squared"]) ** 0.5,
        "gene_endpoint_mse": totals["gene_endpoint_squared_error"] / totals["gene_samples"],
        "gene_endpoint_mae": totals["gene_endpoint_absolute_error"] / totals["gene_samples"],
        "gene_observed_delta_rmse": (totals["gene_observed_delta_squared"] / totals["gene_samples"]) ** 0.5,
        "gene_predicted_delta_rmse": (totals["gene_predicted_delta_squared"] / totals["gene_samples"]) ** 0.5,
        "gene_direction_sign_accuracy": (totals["gene_direction_sign_correct"] / totals["gene_direction_sign_total"]),
        "gene_direction_sign_total": int(totals["gene_direction_sign_total"]),
        "gene_direction_inconclusive_fraction": (totals["gene_direction_inconclusive"] / totals["gene_samples"]),
        "gene_predicted_nonzero_fraction": (totals["gene_predicted_nonzero"] / totals["gene_samples"]),
        "gene_predicted_delta_norm_ratio": (
            totals["gene_predicted_delta_squared"] / totals["gene_observed_delta_squared"]
        )
        ** 0.5,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = _resolve_device(args.device)
    if args.batch_size <= 0 or args.num_workers < 0 or args.num_steps <= 0:
        raise ValueError("batch-size and num-steps must be positive; num-workers must be non-negative")
    if args.gene_direction_epsilon < 0 or not np.isfinite(args.gene_direction_epsilon):
        raise ValueError("gene-direction-epsilon must be finite and >= 0")

    asset = load_perturbgen_embedding_asset(args.embedding_asset)
    dataset = load_latent_davf_pairs(
        args.test_data,
        latent_dim=FORMAL_DAVF_LATENT_DIM,
        perturbgen_vocab_size=asset.vocab_size,
        expected_embedding_manifest=asset.manifest,
        expected_embedding_path=args.embedding_asset,
        expected_embedding_dim=asset.embedding_dim,
        expected_intervention_type=args.intervention_type,
    )
    context_adata, target_cell_ids, context_path = _load_evaluation_context(
        dataset,
        args.test_data,
        args.context_h5ad,
    )
    first_context = context_adata[list(target_cell_ids[: min(args.batch_size, len(dataset))])].copy()
    adapter = ScVIAdapter.from_trained_model(
        args.scvi_model,
        config=ScVIAdapterConfig(
            model_path=str(args.scvi_model),
            n_latent=FORMAL_DAVF_LATENT_DIM,
            batch_key="davf_batch",
            device=str(device),
        ),
        adata=first_context,
    )
    adapter.validate_compatibility(
        expected_latent_dim=FORMAL_DAVF_LATENT_DIM,
        expected_num_genes=FORMAL_DAVF_NUM_GENES,
    )
    # Re-run the provenance check after loading the adapter with real decoder
    # context. This validates the pair gene order against live scVI names.
    dataset = load_latent_davf_pairs(
        args.test_data,
        latent_dim=FORMAL_DAVF_LATENT_DIM,
        perturbgen_vocab_size=asset.vocab_size,
        expected_scvi_model_path=args.scvi_model,
        expected_scvi_gene_names=adapter.gene_names,
        expected_embedding_manifest=asset.manifest,
        expected_embedding_path=args.embedding_asset,
        expected_embedding_dim=asset.embedding_dim,
        expected_intervention_type=args.intervention_type,
    )
    payload, config = load_current_davf_checkpoint(
        args.checkpoint,
        asset=asset,
        scvi_adapter=adapter,
        map_location=device,
        expected_intervention_type=args.intervention_type,
    )
    model = LatentDAVF(config, pretrained_gene_embeddings=asset.embeddings).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    metrics = _evaluate_model(
        model,
        dataset,
        embedding_asset=asset,
        scvi_adapter=adapter,
        context_adata=context_adata,
        target_cell_ids=target_cell_ids,
        gene_direction_epsilon=args.gene_direction_epsilon,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_steps=args.num_steps,
        device=device,
    )
    report = {
        "schema_version": "ptm2cellnet.latent-davf-evaluation.v2",
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "test_data": str(Path(args.test_data).expanduser().resolve()),
        "context_h5ad": str(context_path),
        "scvi_model": str(Path(args.scvi_model).expanduser().resolve()),
        "embedding_asset": str(Path(args.embedding_asset).expanduser().resolve()),
        "intervention_type": args.intervention_type,
        "direction_code": {"KO": 0, "KD": 1, "OE": 2}[args.intervention_type],
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_steps": int(args.num_steps),
        "gene_direction_epsilon": float(args.gene_direction_epsilon),
        "latent_dim": FORMAL_DAVF_LATENT_DIM,
        "num_genes": FORMAL_DAVF_NUM_GENES,
        "metrics": metrics,
    }
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    report = evaluate(args)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
