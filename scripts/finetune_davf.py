#!/usr/bin/env python3
"""
DAVF 端到端微调脚本 (V22-01).

Implements the DAVF fine-tuning training loop that was previously missing
(REQUIREMENTS V22-01). It exercises the freeze/unfreeze API of
``DAVFInferenceModule`` (decisions D-15 / D-16 / D-17) and trains a
``PTM2CellNet`` backbone that consumes DAVF features end-to-end.

Two-stage schedule (decisions D-15..D-17):
    Stage 1 (frozen DAVF, warmup):
        - DAVF backbone frozen (``freeze_davf``), only the DeltaProjection
          head + PTM2CellNet predictor are trainable.
        - Uses a higher learning rate to adapt the new heads quickly.
    Stage 2 (unfrozen, fine-tune):
        - All parameters unfrozen (``unfreeze_davf``).
        - Lower learning rate to avoid catastrophic forgetting of the
          pretrained DAVF ODE flow.

The script reuses the existing ``Trainer`` / data pipeline so it stays
consistent with ``scripts/train.py`` and ``scripts/train_binary.py``.

Usage:
    python scripts/finetune_davf.py \\
        --config configs/default.yaml \\
        --data data/processed/ptm_labeled.csv \\
        --checkpoint checkpoints/latent_davf_ibd_norman/best_model.pt \\
        --stage1-epochs 5 --stage2-epochs 15 \\
        --stage1-lr 1e-3 --stage2-lr 1e-5 \\
        --output outputs/davf_finetune
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import Config
from src.utils.logging import setup_logger
from src.utils.io import save_json
from src.utils.dependency_check import check_extras, format_dependency_table
from src.data.preprocess import DataPreprocessor
from src.data.datasets import PTMDataset
from src.data.features import FeatureExtractor
from src.models.architectures import PTM2CellNet
from src.models.davf_inference import DAVFInferenceModule
from src.training.losses import FocalLoss
from src.evaluation.metrics import (
    calculate_accuracy,
    calculate_precision,
    calculate_recall,
    calculate_f1_score,
    calculate_auc_roc,
)

logger = setup_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="DAVF 端到端微调脚本 (V22-01)")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="配置文件路径（首次运行推荐使用 configs/smoke/ 下的配置）")
    parser.add_argument("--data", type=str, required=True,
                        help="训练数据 CSV 路径")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/latent_davf_ibd_norman/best_model.pt",
                        help="预训练 LatentDAVF checkpoint 路径")
    parser.add_argument("--output", type=str, default="outputs/davf_finetune",
                        help="输出目录")
    # Two-stage schedule
    parser.add_argument("--stage1-epochs", type=int, default=5,
                        help="阶段1（DAVF冻结）训练轮数")
    parser.add_argument("--stage2-epochs", type=int, default=15,
                        help="阶段2（DAVF解冻）训练轮数")
    parser.add_argument("--stage1-lr", type=float, default=1e-3,
                        help="阶段1学习率（仅头部可训练）")
    parser.add_argument("--stage2-lr", type=float, default=1e-5,
                        help="阶段2学习率（全模型可训练）")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批次大小")
    parser.add_argument("--freeze-davf-stage1", action="store_true", default=True,
                        help="阶段1冻结DAVF（默认True，遵循D-15）")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="权重衰减")
    return parser.parse_args()


def load_data(data_path: str, config: Config):
    """Load and preprocess labeled PTM data into train/val/test splits."""
    logger.info("加载数据: %s", data_path)
    df = pd.read_csv(data_path)
    logger.info("原始数据: %d 条记录", len(df))

    preprocessor = DataPreprocessor(config.to_dict())
    train_df, val_df, test_df = preprocessor.preprocess_pipeline(
        df, sequence_col="sequence", ptm_col="ptm_sites", label_col="cell_state"
    )
    logger.info(
        "数据划分: train=%d, val=%d, test=%d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


def create_dataloaders(train_df, val_df, test_df, config: Config, batch_size: int):
    """Create train/val/test dataloaders."""
    feature_extractor = FeatureExtractor(config.to_dict())
    cfg = config.to_dict()
    train_ds = PTMDataset(train_df, feature_extractor, cfg)
    val_ds = PTMDataset(val_df, feature_extractor, cfg)
    test_ds = PTMDataset(test_df, feature_extractor, cfg)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    logger.info(
        "DataLoader: train=%d, val=%d, test=%d batches",
        len(train_loader), len(val_loader), len(test_loader),
    )
    return train_loader, val_loader, test_loader


def build_model(config: Config, checkpoint_path: str, num_classes: int) -> PTM2CellNet:
    """Build a PTM2CellNet backbone wired with a DAVF feature extractor.

    Uses :meth:`PTM2CellNet.from_config` which properly translates config
    dict keys (e.g. ``max_sequence_length`` → ``max_seq_len``, ``hidden_dim``
    → ``embed_dim``) into :class:`PTM2CellNetBase.__init__` keyword arguments
    and wires up the DAVF inference module when ``model.use_davf`` is set.
    """
    config_dict = config.to_dict()
    model_cfg = config_dict.setdefault("model", {})
    model_cfg["use_davf"] = True
    # Move inline ``davf`` sub-config to the key ``from_config`` expects
    # and inject the checkpoint path / training freeze.
    davf_cfg = model_cfg.setdefault("davf_config", model_cfg.pop("davf", {}))
    davf_cfg["checkpoint_path"] = checkpoint_path
    davf_cfg["freeze"] = True  # stage 1 default
    if "num_classes" in model_cfg:
        model_cfg["num_classes"] = num_classes

    model = PTM2CellNet.from_config(config_dict)
    return model


def _count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def configure_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    """Configure AdamW optimizer for the current trainable parameters.

    Raises:
        RuntimeError: If no parameters are trainable (e.g. everything frozen).
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(
            "No trainable parameters found; cannot configure optimizer. "
            "Ensure at least one module is unfrozen before training."
        )
    return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)


def _find_davf_module(model: torch.nn.Module) -> Optional[DAVFInferenceModule]:
    """Locate the DAVFInferenceModule inside a PTM2CellNet (if any)."""
    for module in model.modules():
        if isinstance(module, DAVFInferenceModule):
            return module
    return None


def train_stage(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    stage_name: str,
    output_dir: str,
    monitor: str = "val_loss",
) -> Dict[str, Any]:
    """Run a single training stage and return history + best metrics."""
    logger.info("=" * 60)
    logger.info("%s 开始 (epochs=%d, lr=%g)", stage_name, epochs, lr)
    logger.info("可训练参数: %d", _count_trainable(model))
    logger.info("=" * 60)

    optimizer = configure_optimizer(model, lr, weight_decay)
    criterion = FocalLoss()
    best_val_loss = float("inf")
    best_metrics: Dict[str, Any] = {}
    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for batch in train_loader:
            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs["logits"], batch["label"])
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        val_metrics = evaluate(model, val_loader, device=device)
        val_loss = val_metrics.get("loss", float("inf"))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics.get("accuracy", 0.0))

        logger.info(
            "[%s] epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  (%.1fs)",
            stage_name, epoch, epochs, train_loss, val_loss,
            val_metrics.get("accuracy", 0.0), time.time() - t0,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = dict(val_metrics)
            os.makedirs(output_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "best_model_davf_finetuned.pt"),
            )

    return {"history": history, "best_metrics": best_metrics, "best_val_loss": best_val_loss}


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate model on a dataloader; returns loss + classification metrics."""
    criterion = FocalLoss()
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_labels: list = []
    all_probs: list = []
    all_preds: list = []

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        outputs = model(batch)
        loss = criterion(outputs["logits"], batch["label"])
        total_loss += float(loss.item())
        n_batches += 1

        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1) if logits.ndim == 2 else logits
        preds = torch.argmax(probs, dim=-1)
        all_probs.append(probs.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(batch["label"].detach().cpu().numpy())

    import numpy as np

    y_true = np.concatenate(all_labels) if all_labels else np.array([])
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    y_prob = np.concatenate(all_probs) if all_probs else np.array([])

    metrics: Dict[str, float] = {"loss": total_loss / max(1, n_batches)}
    if len(y_true):
        metrics["accuracy"] = float(calculate_accuracy(y_true, y_pred))
        metrics["precision"] = float(calculate_precision(y_true, y_pred, average="weighted"))
        metrics["recall"] = float(calculate_recall(y_true, y_pred, average="weighted"))
        metrics["f1"] = float(calculate_f1_score(y_true, y_pred, average="weighted"))
        try:
            if y_prob.shape[1] == 2:
                metrics["auc_roc"] = float(calculate_auc_roc(y_true, y_prob[:, 1]))
        except Exception:
            pass
    return metrics


def _report_optional_dependencies() -> None:
    """Log which optional deps are available for the DAVF gene-space workflow.

    The latent-space fine-tune itself does not require scvi-tools, but the
    end-to-end ``gene -> latent -> gene`` round-trip (used by
    :class:`src.models.scvi_adapter.ScVIAdapter`) does. Reporting availability
    up front lets users install ``.[analysis]`` before hitting an ImportError.
    """
    statuses = check_extras(["scvi", "anndata"])
    logger.info(format_dependency_table(statuses))


def main():
    """Main entry point."""
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Preflight: report optional-dependency availability so users learn *now*
    # (with an actionable pip hint) whether the full gene-space DAVF pipeline
    # is usable, rather than discovering it mid-training via a stack trace.
    # Latent-only fine-tuning does not require scvi, so we only warn here.
    _report_optional_dependencies()

    config = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    # 1. Data
    train_df, val_df, test_df = load_data(args.data, config)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, config, args.batch_size
    )
    num_classes = int(config.get("model.num_classes", len(set(train_df["cell_state"]))))
    logger.info("类别数: %d", num_classes)

    # 2. Build model with DAVF feature extractor (frozen by default → stage 1)
    model = build_model(config, args.checkpoint, num_classes).to(device)
    davf_module = _find_davf_module(model)
    if davf_module is None:
        logger.warning(
            "未在模型中找到 DAVFInferenceModule；将仅对 PTM2CellNet 主干进行训练。"
        )
    else:
        logger.info("已挂载 DAVFInferenceModule (checkpoint=%s)", args.checkpoint)

    results: Dict[str, Any] = {"args": vars(args)}

    # 3. Stage 1 — frozen DAVF, train heads only (D-15)
    if args.stage1_epochs > 0 and davf_module is not None and args.freeze_davf_stage1:
        davf_module.freeze_davf()
        stage1 = train_stage(
            model, train_loader, val_loader,
            epochs=args.stage1_epochs, lr=args.stage1_lr,
            weight_decay=args.weight_decay, device=device,
            stage_name="Stage1 (DAVF frozen)",
            output_dir=args.output,
        )
        results["stage1"] = stage1

    # 4. Stage 2 — unfreeze, fine-tune all (D-16)
    if args.stage2_epochs > 0 and davf_module is not None:
        davf_module.unfreeze_davf()
        logger.info("已解冻 DAVF 参数以进行端到端微调 (D-16)")
        stage2 = train_stage(
            model, train_loader, val_loader,
            epochs=args.stage2_epochs, lr=args.stage2_lr,
            weight_decay=args.weight_decay, device=device,
            stage_name="Stage2 (DAVF unfrozen)",
            output_dir=args.output,
        )
        results["stage2"] = stage2

    # 5. Final test evaluation
    test_metrics = evaluate(model, test_loader, device=device)
    results["test_metrics"] = test_metrics
    logger.info("测试集指标: %s", json.dumps(test_metrics, indent=2))

    # 6. Save artifacts
    torch.save(model.state_dict(), os.path.join(args.output, "final_model.pt"))
    save_json(results, os.path.join(args.output, "finetune_results.json"))
    logger.info("结果已保存至 %s", args.output)


if __name__ == "__main__":
    main()
