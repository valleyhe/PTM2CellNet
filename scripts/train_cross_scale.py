#!/usr/bin/env python3
"""Train the opt-in cross-scale model from versioned NPZ split archives."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cross_scale_dataset import CrossScaleDataModule  # noqa: E402
from src.models.cross_scale import CrossScalePTM2CellNet  # noqa: E402
from src.training.cross_scale_trainer import CrossScaleTrainer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cross_scale/smoke.yaml"))
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("配置顶层必须是 mapping")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _load_config(args.config)
        training = config.get("training", {})
        if not isinstance(training, dict):
            raise ValueError("training 配置必须是 mapping")
        seed = int(args.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

        model = CrossScalePTM2CellNet.from_config(config)
        data_module = CrossScaleDataModule(
            args.train_data,
            args.val_data,
            args.test_data,
            batch_size=int(args.batch_size or training.get("batch_size", 8)),
            num_workers=int(training.get("num_workers", 0)),
            seed=seed,
            backbone_names=model.protein_encoder.backbone_names,
        )
        learning_rate = float(training.get("learning_rate", 1e-3))
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=float(training.get("weight_decay", 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(training.get("lr_factor", 0.5)),
            patience=int(training.get("lr_patience", 2)),
        )
        device = args.device or str(training.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        labels_config = config.get("labels", {})
        labels = labels_config.get("cell_states") if isinstance(labels_config, dict) else None
        expected_states = model.cross_scale_config.num_cell_states
        if (
            not isinstance(labels, list)
            or len(labels) != expected_states
            or not all(isinstance(label, str) and label for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError(f"labels.cell_states 必须包含 {expected_states} 个唯一非空字符串")
        context = {
            "config": config,
            "data_contracts": data_module.contracts(),
            "seed": seed,
            "label_vocabulary": labels,
        }
        trainer = CrossScaleTrainer(
            model,
            optimizer,
            scheduler=scheduler,
            device=device,
            gradient_clip_norm=float(training.get("gradient_clip_norm", 1.0)),
            accumulation_steps=int(training.get("accumulation_steps", 1)),
            mixed_precision=bool(training.get("mixed_precision", False)),
            artifact_context=context,
        )
        if args.resume is not None:
            trainer.load_checkpoint(args.resume)
        max_epochs = int(args.epochs or training.get("max_epochs", 10))
        result = trainer.fit(
            data_module.train_dataloader(),
            data_module.val_dataloader(),
            max_epochs=max_epochs,
            output_dir=args.output,
            patience=int(training.get("early_stopping_patience", 5)),
        )
        trainer.load_checkpoint(args.output / "best.pt")
        test_metrics = trainer.evaluate(data_module.test_dataloader())
        trainer.write_artifact_manifest(args.output, test_metrics=test_metrics)
        args.output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.config, args.output / "config.yaml")
        metrics = {**result, "test_metrics": test_metrics}
        (args.output / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"ok": True, "output": str(args.output), **metrics}, ensure_ascii=False, indent=2))
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
