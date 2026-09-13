#!/usr/bin/env python3
"""
DAVF E2E fine-tuning script (production, V2).

Fine-tune a pretrained DAVF model for downstream PTM cell-state classification.
Uses the formal project data pipeline (PTMDataset, PTMDirectionMapper,
DAVFInferenceModule) with a configurable task head.

Two-stage schedule:
    Stage 1 (frozen DAVF, warmup): DAVF backbone frozen, train head only.
    Stage 2 (unfrozen, fine-tune): All parameters unfrozen, lower LR.

Usage:
    python scripts/finetune_davf_e2e.py \\
        --data data/processed/ptm_labeled.csv \\
        --checkpoint checkpoints/davf/best_model.pt \\
        --output outputs/davf_e2e_finetune \\
        --stage1-epochs 5 --stage2-epochs 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocess import DataPreprocessor
from src.data.datasets import PTMDataset, DatasetConfig
from src.data.features import FeatureExtractor
from src.data.schemas import PTMSite
from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig
from src.models.ptm_direction_mapper import PTMDirectionMapper
from src.evaluation.metrics import (
    calculate_accuracy,
    calculate_precision,
    calculate_recall,
    calculate_f1_score,
)
from src.training.losses import FocalLoss
from src.utils.logging import setup_logger
from src.utils.io import save_json

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Task head
# ---------------------------------------------------------------------------

class TaskHead(nn.Module):
    """Configurable classification head for DAVF features.

    Takes DAVF features [B, feature_dim] and produces class logits [B, num_classes].
    """

    def __init__(self, feature_dim: int = 128, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = feature_dim // 2
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAVF E2E Fine-Tuning (production)")
    parser.add_argument("--data", type=str, required=True, help="训练数据 CSV 路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="预训练 DAVF checkpoint 路径")
    parser.add_argument("--output", type=str, default="outputs/davf_e2e_finetune", help="输出目录")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径 (可选)")
    # Two-stage schedule
    parser.add_argument("--stage1-epochs", type=int, default=5, help="阶段1（DAVF冻结）训练轮数")
    parser.add_argument("--stage2-epochs", type=int, default=15, help="阶段2（DAVF解冻）训练轮数")
    parser.add_argument("--stage1-lr", type=float, default=1e-3, help="阶段1学习率")
    parser.add_argument("--stage2-lr", type=float, default=1e-5, help="阶段2学习率")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--feature-dim", type=int, default=128, help="DAVF 特征维度")
    parser.add_argument(
        "--embedding-asset",
        type=str,
        required=True,
        help="PerturbGen embedding asset 目录 (manifest.json + vocabulary.json + "
        "gene_embeddings.safetensors, schema v2)。必须提供；禁止回退到 Geneformer mapper。",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="分类头 dropout")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_path: str, config: Optional[Dict[str, Any]] = None):
    """Load and preprocess labeled PTM data into train/val/test splits."""
    logger.info("加载数据: %s", data_path)
    df = pd.read_csv(data_path)
    logger.info("原始数据: %d 条记录", len(df))

    preprocessor = DataPreprocessor(config or {})
    train_df, val_df, test_df = preprocessor.preprocess_pipeline(
        df, sequence_col="sequence", ptm_col="ptm_sites", label_col="cell_state"
    )
    logger.info("数据划分: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


def _davf_collate_fn(batch):
    """Custom collate for DAVF finetuning with variable-length sequences.

    PTMDataset returns variable-length tensors (sequence, ptm_mask, ptm_types)
    and variable-length Python lists (davf_sites, davf_gene_names, davf_type_names,
    davf_attention_mask). The default collate_fn cannot handle these, so we:
    - Pad 1D tensors to the max length in the batch.
    - Keep variable-length list fields as raw lists (outer list of per-sample lists).
    - Stack scalar tensors normally.
    """
    import torch.utils.data._utils.collate as torch_collate

    elem = batch[0]
    out: Dict[str, Any] = {}
    # Keys whose values are Python lists (variable length — keep as outer list)
    _LIST_KEYS = {"davf_sites", "davf_gene_names", "davf_type_names", "davf_attention_mask"}
    # Keys whose values are scalar tensors — stack normally
    _SCALAR_KEYS = {"label", "sequence_length"}

    for key in elem:
        values = [d[key] for d in batch]
        if key in _LIST_KEYS:
            # Keep as outer list: [sample1_list, sample2_list, ...]
            out[key] = values
        elif key in _SCALAR_KEYS:
            out[key] = torch.stack(values)
        else:
            # Tensor fields — pad 1D tensors to max length in batch
            try:
                out[key] = torch_collate.default_collate(values)
            except (RuntimeError, TypeError):
                # Variable-length 1D tensors: pad to max_len
                max_len = max(v.shape[0] for v in values)
                dtype = values[0].dtype
                device = values[0].device
                padded = torch.zeros(len(values), max_len, dtype=dtype, device=device)
                for i, v in enumerate(values):
                    padded[i, : v.shape[0]] = v
                out[key] = padded
    return out


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int,
    config: Optional[Dict[str, Any]] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders with DAVF-aware PTMDataset."""
    feature_extractor = FeatureExtractor(config or {})
    dataset_config = DatasetConfig(
        use_davf=True,
        use_feature_extractor=False,
    )

    train_ds = PTMDataset(train_df, feature_extractor, dataset_config, training=True)
    val_ds = PTMDataset(val_df, feature_extractor, dataset_config, training=False)
    test_ds = PTMDataset(test_df, feature_extractor, dataset_config, training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=_davf_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_davf_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_davf_collate_fn)

    logger.info("DataLoader: train=%d, val=%d, test=%d batches", len(train_loader), len(val_loader), len(test_loader))
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model(
    checkpoint_path: str,
    feature_dim: int,
    device: torch.device,
    embedding_asset_path: str,
) -> tuple[DAVFInferenceModule, PTMDirectionMapper]:
    """Build DAVF inference module and a PerturbGen-locked direction mapper."""
    if not str(embedding_asset_path).strip():
        raise ValueError(
            "finetune_davf_e2e.build_model requires embedding_asset_path; "
            "the default Geneformer PTMDirectionMapper is forbidden because token IDs "
            "must not index a PerturbGen gene_embed_table"
        )
    davf_config = DAVFInferenceConfig(
        checkpoint_path=checkpoint_path,
        freeze=True,  # Start frozen for stage 1
        feature_dim=feature_dim,
        device=str(device),
        embedding_asset_path=embedding_asset_path,
    )
    davf_module = DAVFInferenceModule(davf_config).to(device)
    logger.info("DAVFInferenceModule 已加载 (model_source=%s)", davf_module.model_source)
    mapper = davf_module.build_perturbgen_direction_mapper()
    logger.info("PTMDirectionMapper locked to PerturbGen embedding asset vocabulary")
    return davf_module, mapper


# ---------------------------------------------------------------------------
# Batch conversion helpers
# ---------------------------------------------------------------------------

def _build_ptm_sites_from_batch(
    davf_sites: List[List[int]],
    davf_type_names: List[List[str]],
    davf_gene_names: List[List[str]],
) -> List[List[PTMSite]]:
    """Convert per-sample DAVF site lists into PTMSite dataclass lists.

    PTMDataset returns davf_sites (positions), davf_type_names, and
    davf_gene_names as parallel lists.  PTMDirectionMapper.map_batch()
    expects List[List[PTMSite]], so we construct the dataclass objects here.
    """
    result: List[List[PTMSite]] = []
    for sites, types, _genes in zip(davf_sites, davf_type_names, davf_gene_names, strict=False):
        ptm_sites = [
            PTMSite(position=pos, type=ptm_type)
            for pos, ptm_type in zip(sites, types, strict=False)
        ]
        result.append(ptm_sites)
    return result


def _move_to_device(obj: Any, device: torch.device) -> Any:
    """Move tensors to device; leave lists and other types unchanged."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    davf_module: DAVFInferenceModule,
    mapper: PTMDirectionMapper,
    head: TaskHead,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on a dataloader."""
    criterion = FocalLoss()
    davf_module.eval()
    head.eval()

    total_loss = 0.0
    n_batches = 0
    all_labels: List[int] = []
    all_preds: List[int] = []

    for batch in loader:
        batch = _move_to_device(batch, device)

        # Extract DAVF inputs from PTMDataset batch.
        # PTMDataset returns these as Python lists (not tensors).
        davf_sites: Optional[List[List[int]]] = batch.get("davf_sites")
        davf_gene_names: Optional[List[List[str]]] = batch.get("davf_gene_names")
        davf_type_names: Optional[List[List[str]]] = batch.get("davf_type_names")

        if davf_sites is None or davf_gene_names is None or davf_type_names is None:
            logger.warning("批次缺少 DAVF 字段 (davf_sites/davf_gene_names/davf_type_names)，跳过")
            continue

        # Convert to PTMSite objects for the mapper
        ptm_sites = _build_ptm_sites_from_batch(davf_sites, davf_type_names, davf_gene_names)

        # Map PTM sites to DAVF inputs (gene_ids, directions, attention_mask)
        mapper_output = mapper.map_batch(ptm_sites, davf_gene_names)
        # Move mapper output tensors to device
        mapper_output.gene_ids = mapper_output.gene_ids.to(device)
        mapper_output.directions = mapper_output.directions.to(device)
        mapper_output.attention_mask = mapper_output.attention_mask.to(device)

        # Forward through DAVF
        davf_output = davf_module(mapper_output)
        features = davf_output.davf_features  # [B, feature_dim]

        # Forward through task head
        logits = head(features)
        loss = criterion(logits, batch["label"])

        total_loss += float(loss.item())
        n_batches += 1

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(batch["label"].detach().cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    metrics: Dict[str, float] = {"loss": total_loss / max(n_batches, 1)}
    if len(y_true) > 0:
        metrics["accuracy"] = float(calculate_accuracy(y_true, y_pred))
        metrics["precision"] = float(calculate_precision(y_true, y_pred, average="weighted"))
        metrics["recall"] = float(calculate_recall(y_true, y_pred, average="weighted"))
        metrics["f1"] = float(calculate_f1_score(y_true, y_pred, average="weighted"))
    return metrics


def train_epoch(
    davf_module: DAVFInferenceModule,
    mapper: PTMDirectionMapper,
    head: TaskHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch. Returns average loss."""
    criterion = FocalLoss()
    davf_module.train()
    head.train()

    running_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = _move_to_device(batch, device)

        davf_sites: Optional[List[List[int]]] = batch.get("davf_sites")
        davf_gene_names: Optional[List[List[str]]] = batch.get("davf_gene_names")
        davf_type_names: Optional[List[List[str]]] = batch.get("davf_type_names")

        if davf_sites is None or davf_gene_names is None or davf_type_names is None:
            continue

        # Convert to PTMSite objects for the mapper
        ptm_sites = _build_ptm_sites_from_batch(davf_sites, davf_type_names, davf_gene_names)

        mapper_output = mapper.map_batch(ptm_sites, davf_gene_names)
        mapper_output.gene_ids = mapper_output.gene_ids.to(device)
        mapper_output.directions = mapper_output.directions.to(device)
        mapper_output.attention_mask = mapper_output.attention_mask.to(device)

        davf_output = davf_module(mapper_output)
        features = davf_output.davf_features

        logits = head(features)
        loss = criterion(logits, batch["label"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        n_batches += 1

    return running_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    config: Optional[Dict[str, Any]] = None
    if args.config:
        from src.utils.config import Config
        config = Config.from_yaml(args.config).to_dict()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("使用设备: %s", device)

    # 1. Load data
    train_df, val_df, test_df = load_data(args.data, config)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, args.batch_size, config,
    )
    num_classes = len(train_df["cell_state"].unique())
    logger.info("类别数: %d", num_classes)

    # 2. Build model
    davf_module, mapper = build_model(
        args.checkpoint, args.feature_dim, device, args.embedding_asset
    )
    head = TaskHead(feature_dim=args.feature_dim, num_classes=num_classes, dropout=args.dropout).to(device)

    results: Dict[str, Any] = {"args": vars(args), "num_classes": num_classes}

    # 3. Stage 1 — frozen DAVF, train head only
    if args.stage1_epochs > 0:
        logger.info("=" * 60)
        logger.info("Stage 1: DAVF frozen, 训练分类头 (epochs=%d, lr=%g)", args.stage1_epochs, args.stage1_lr)
        davf_module.freeze_davf()
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.stage1_lr, weight_decay=args.weight_decay)

        stage1_history: Dict[str, list] = {"train_loss": [], "val_metrics": []}
        best_val_f1 = 0.0
        for epoch in range(1, args.stage1_epochs + 1):
            t0 = time.time()
            train_loss = train_epoch(davf_module, mapper, head, train_loader, optimizer, device)
            val_metrics = evaluate(davf_module, mapper, head, val_loader, device)
            stage1_history["train_loss"].append(train_loss)
            stage1_history["val_metrics"].append(val_metrics)

            logger.info(
                "[Stage1] epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  val_f1=%.4f  (%.1fs)",
                epoch, args.stage1_epochs, train_loss,
                val_metrics.get("loss", float("inf")), val_metrics.get("accuracy", 0.0),
                val_metrics.get("f1", 0.0), time.time() - t0,
            )

            if val_metrics.get("f1", 0.0) > best_val_f1:
                best_val_f1 = val_metrics.get("f1", 0.0)
                torch.save({
                    "head_state_dict": head.state_dict(),
                    "davf_state_dict": davf_module.state_dict(),
                    "config": vars(args),
                }, os.path.join(args.output, "best_model_stage1.pt"))

        results["stage1"] = {"history": stage1_history, "best_val_f1": best_val_f1}

    # 4. Stage 2 — unfreeze DAVF, fine-tune all
    if args.stage2_epochs > 0:
        logger.info("=" * 60)
        logger.info("Stage 2: DAVF 解冻, 端到端微调 (epochs=%d, lr=%g)", args.stage2_epochs, args.stage2_lr)
        davf_module.unfreeze_davf()
        optimizer = torch.optim.AdamW(
            list(davf_module.parameters()) + list(head.parameters()),
            lr=args.stage2_lr,
            weight_decay=args.weight_decay,
        )

        stage2_history: Dict[str, list] = {"train_loss": [], "val_metrics": []}
        best_val_f1 = 0.0
        for epoch in range(1, args.stage2_epochs + 1):
            t0 = time.time()
            train_loss = train_epoch(davf_module, mapper, head, train_loader, optimizer, device)
            val_metrics = evaluate(davf_module, mapper, head, val_loader, device)
            stage2_history["train_loss"].append(train_loss)
            stage2_history["val_metrics"].append(val_metrics)

            logger.info(
                "[Stage2] epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  val_f1=%.4f  (%.1fs)",
                epoch, args.stage2_epochs, train_loss,
                val_metrics.get("loss", float("inf")), val_metrics.get("accuracy", 0.0),
                val_metrics.get("f1", 0.0), time.time() - t0,
            )

            if val_metrics.get("f1", 0.0) > best_val_f1:
                best_val_f1 = val_metrics.get("f1", 0.0)
                torch.save({
                    "head_state_dict": head.state_dict(),
                    "davf_state_dict": davf_module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "config": vars(args),
                }, os.path.join(args.output, "best_model_stage2.pt"))

        results["stage2"] = {"history": stage2_history, "best_val_f1": best_val_f1}

    # 5. Final test evaluation
    test_metrics = evaluate(davf_module, mapper, head, test_loader, device)
    results["test_metrics"] = test_metrics
    logger.info("测试集指标: %s", json.dumps(test_metrics, indent=2))

    # 6. Save final artifacts
    torch.save({
        "head_state_dict": head.state_dict(),
        "davf_state_dict": davf_module.state_dict(),
        "config": vars(args),
    }, os.path.join(args.output, "final_model.pt"))
    save_json(results, os.path.join(args.output, "finetune_results.json"))
    logger.info("结果已保存至 %s", args.output)


if __name__ == "__main__":
    main()
