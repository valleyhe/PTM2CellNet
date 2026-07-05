#!/usr/bin/env python3
"""
DEPRECATED — 此文件已被 scripts/finetune_davf_e2e.py 取代。

生产版本使用正式的 PTMDataset / PTMDirectionMapper / DAVFInferenceModule 管道。
本实验版本保留用于历史参考。如需实验性脚本，请使用生产版本中的 --stage1-epochs 0
仅运行分类头训练。

Status: deprecated, replaced by scripts/finetune_davf_e2e.py (2026-07-05)

---
Original docstring below:
---

Fine-tune a pretrained DAVF model for a downstream task.

EXPERIMENTAL (F-01, 25% complete).
This script demonstrates the DAVF fine-tuning workflow but uses a placeholder
downstream head. For production use, replace DownstreamHead with your
domain-specific task head.

Usage (experimental only):
    PTM2CELLNET_ALLOW_EXPERIMENTAL=1 python scripts/experimental/finetune_davf_e2e.py \\
        --davf-checkpoint path/to/davf.pt \\
        --data-path path/to/training_data.csv \\
        --output-dir outputs/experimental/davf_e2e
"""

import argparse
import contextlib
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Experimental guard — must be before project imports that may fail
_EXPERIMENTAL_GUARD = os.environ.get("PTM2CELLNET_ALLOW_EXPERIMENTAL", "").lower() in ("1", "true", "yes")
if not _EXPERIMENTAL_GUARD:
    raise RuntimeError(
        "finetune_davf_e2e.py is an EXPERIMENTAL script (F-01 status: 25% complete). "
        "The DownstreamHead is a generic placeholder — it does not produce scientifically "
        "valid output. To use this script anyway, set PTM2CELLNET_ALLOW_EXPERIMENTAL=1 "
        "and replace DownstreamHead with your domain-specific task head. "
        "See docs/项目代码现状系统性复核报告_2026-07-05_v11.md for details."
    )

import torch
import torch.nn as nn
import torch.optim as optim

from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="DAVF End-to-End Fine-Tuning")
    parser.add_argument("--checkpoint_path", required=True, help="Path to pretrained DAVF checkpoint")
    parser.add_argument("--data_path", required=True, help="Path to training data CSV")
    parser.add_argument("--output_dir", default="outputs/davf_finetuned", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for DeltaProjection")
    parser.add_argument("--backbone_lr", type=float, default=1e-5, help="Learning rate for backbone (when unfrozen)")
    parser.add_argument("--unfreeze_after", type=int, default=None,
                        help="Epoch after which to unfreeze DAVF backbone (None = never unfreeze)")
    parser.add_argument("--state_space", default="scvi_latent", choices=["scvi_latent", "gene"],
                        help="DAVF state space mode")
    parser.add_argument("--feature_dim", type=int, default=128, help="DAVF feature dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="DAVF hidden dimension")
    parser.add_argument("--latent_dim", type=int, default=10, help="DAVF latent dimension")
    parser.add_argument("--num_genes", type=int, default=5000, help="Number of genes")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Load pretrained DAVF module (frozen by default)
    logger.info("Loading pretrained DAVFInferenceModule from %s", args.checkpoint_path)
    config = DAVFInferenceConfig(
        state_space=args.state_space,
        checkpoint_path=args.checkpoint_path,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_genes=args.num_genes,
        freeze=True,  # Start frozen
        device=device,
    )
    model = DAVFInferenceModule(config)
    logger.info("DAVF module loaded. Frozen: %s", config.freeze)

    # Step 2: Create a simple downstream task head
    class DownstreamHead(nn.Module):
        """EXPERIMENTAL placeholder downstream head (F-01).

        This is a generic binary classification head for demonstration purposes only.
        In production, replace with your domain-specific task head that matches your
        data schema and scientific requirements.

        Classes: PTM effect prediction (binary: significant vs not-significant)
        """

        def __init__(self, feature_dim, num_classes=2):
            super().__init__()
            self.head = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim // 2, num_classes),
            )

        def forward(self, x):
            return self.head(x)

    downstream_head = DownstreamHead(args.feature_dim).to(device)

    # Step 3: Configure optimizer with differential learning rates
    # DeltaProjection + downstream head get higher LR
    trainable_params = list(model.delta_projection.parameters()) + list(downstream_head.parameters())
    optimizer = optim.Adam(trainable_params, lr=args.lr)

    # Step 4: Load training data
    logger.info("Loading training data from %s", args.data_path)
    df = pd.read_csv(args.data_path)

    # Resolve sequence column — try common alternatives
    seq_col = None
    for candidate in ["sequence_window", "sequence", "peptide", "window"]:
        if candidate in df.columns:
            seq_col = candidate
            break
    if seq_col is None:
        raise ValueError(
            f"No sequence column found in {args.data_path}. "
            f"Expected one of: sequence_window, sequence, peptide, window. "
            f"Got columns: {list(df.columns)}"
        )

    # Resolve label column
    if "label" not in df.columns:
        raise ValueError(
            f"No 'label' column found in {args.data_path}. "
            f"Got columns: {list(df.columns)}"
        )

    df = df[[seq_col, "label"]].dropna()
    logger.info(
        "Loaded %d samples (seq_col='%s', label_col='label')",
        len(df), seq_col,
    )

    # Step 5: Create Dataset and DataLoader
    class SequenceDataset(torch.utils.data.Dataset):
        """Simple sequence→tensor dataset for fine-tuning."""

        def __init__(self, dataframe: pd.DataFrame, seq_col: str):
            self.sequences = dataframe[seq_col].tolist()
            self.labels = dataframe["label"].tolist()

        def __len__(self) -> int:
            return len(self.sequences)

        def __getitem__(self, idx: int):
            seq = str(self.sequences[idx])
            # Map each character to an integer index: ord(c) - ord('A'), capped at 25
            indices = [min(max(ord(c) - ord("A"), 0), 25) for c in seq.upper()]
            return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

    dataset = SequenceDataset(df, seq_col)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Step 6: Loss function and training loop
    criterion = nn.CrossEntropyLoss()

    logger.info("Starting training for %d epochs", args.epochs)
    for epoch in range(args.epochs):
        model.train()
        downstream_head.train()
        epoch_loss = 0.0
        num_batches = 0

        # Optionally unfreeze backbone after specified epoch
        if args.unfreeze_after is not None and epoch == args.unfreeze_after:
            logger.info("Unfreezing DAVF backbone at epoch %d", epoch)
            model.unfreeze_davf()
            # Add backbone parameters with lower learning rate
            backbone_params = []
            if hasattr(model, "latent_davf"):
                backbone_params.extend(model.latent_davf.parameters())
            elif hasattr(model, "gene_encoder"):
                backbone_params.extend(model.gene_encoder.parameters())
            optimizer = optim.Adam([
                {"params": trainable_params, "lr": args.lr},
                {"params": backbone_params, "lr": args.backbone_lr},
            ])

        for batch_sequences, batch_labels in train_loader:
            batch_sequences = batch_sequences.to(device)
            batch_labels = batch_labels.to(device)

            # Forward through DAVF
            with torch.no_grad() if not model.training else contextlib.nullcontext():
                davf_output = model(batch_sequences)
            features = davf_output.davf_features  # [B, feature_dim]

            # Forward through downstream head
            logits = downstream_head(features)    # [B, num_classes]
            loss = criterion(logits, batch_labels)

            # Backward + step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info("Epoch %d/%d — avg_loss=%.4f", epoch + 1, args.epochs, avg_loss)

    # Step 7: Save fine-tuned model
    output_path = os.path.join(args.output_dir, "davf_finetuned.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "downstream_head_state_dict": downstream_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epochs_completed": args.epochs,
        "training_completed": True,
    }, output_path)
    logger.info("Saved fine-tuned checkpoint to %s", output_path)

    logger.info("Fine-tuning complete.")


if __name__ == "__main__":
    main()
