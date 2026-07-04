#!/usr/bin/env python3
"""
PTM2CellNet 二分类微调脚本
使用 ptm_integrated_human_labeled.csv (112,012条, Quiescent vs Activated)
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import Config
from src.utils.logging import setup_logger
from src.utils.io import save_model, save_json
from src.data.preprocess import DataPreprocessor
from src.data.datasets import PTMDataset
from src.data.features import FeatureExtractor
from src.models.architectures import PTM2CellNet
from src.training.trainers import Trainer
from src.training.callbacks import ModelCheckpoint, EarlyStopping
from src.training.losses import FocalLoss
from src.evaluation.metrics import (
    calculate_accuracy, calculate_precision, calculate_recall,
    calculate_f1_score, calculate_auc_roc,
)

logger = setup_logger(__name__)

DEFAULT_LABEL_MAP = {"Quiescent": 0, "Activated": 1}


def load_binary_data(data_path: str, config: Config, label_map: Optional[dict] = None):
    """加载二分类数据"""
    logger.info("加载数据: %s", data_path)
    df = pd.read_csv(data_path)
    logger.info("原始数据: %d 条记录", len(df))

    # 标签映射
    label_map = label_map if label_map is not None else DEFAULT_LABEL_MAP
    df["label"] = df["cell_state"].map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    
    logger.info("标签分布:")
    for label, count in df["cell_state"].value_counts().items():
        logger.info("  %s: %d (%.1f%%)", label, count, count/len(df)*100)
    
    # 预处理
    preprocessor = DataPreprocessor(config.to_dict())
    train_df, val_df, test_df = preprocessor.preprocess_pipeline(
        df, sequence_col="sequence", ptm_col="ptm_sites", label_col="cell_state"
    )
    
    logger.info("数据划分: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


def create_dataloaders(train_df, val_df, test_df, config: Config, batch_size: int = 64):
    """创建数据加载器"""
    feature_extractor = FeatureExtractor(config.to_dict())
    
    train_dataset = PTMDataset(train_df, feature_extractor, config.to_dict())
    val_dataset = PTMDataset(val_df, feature_extractor, config.to_dict())
    test_dataset = PTMDataset(test_df, feature_extractor, config.to_dict())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    logger.info("DataLoader: train=%d, val=%d, test=%d batches",
                len(train_loader), len(val_loader), len(test_loader))
    
    return train_loader, val_loader, test_loader


def train(config_path: str, output_dir: str, label_map: Optional[dict] = None):
    """主训练流程"""
    logger.info("=" * 60)
    logger.info("PTM2CellNet 二分类微调开始")
    logger.info("=" * 60)

    # 加载配置
    config = Config.from_yaml(config_path)
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    data_path = config.get("data.data_path", "data/processed/ptm_integrated_human_labeled.csv")
    train_df, val_df, test_df = load_binary_data(data_path, config, label_map=label_map)
    
    # 创建DataLoader
    batch_size = config.get("training.batch_size", 64)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, config, batch_size
    )
    
    # 创建模型
    num_classes = config.get("model.num_classes", 2)
    model = PTM2CellNet.from_config(config.to_dict())
    
    device = config.get("training.device", "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("模型参数量: %d", sum(p.numel() for p in model.parameters()))
    logger.info("设备: %s", device)
    
    # 创建训练器
    trainer = Trainer(model, config.to_dict(), device=device)
    
    # 损失函数
    loss_type = config.get("training.loss_type", "cross_entropy")
    if loss_type == "focal":
        alpha = config.get("training.focal_alpha", 0.75)
        gamma = config.get("training.focal_gamma", 2.0)
        loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
        logger.info("使用 Focal Loss (alpha=%.2f, gamma=%.2f)", alpha, gamma)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        logger.info("使用 CrossEntropy Loss")
    
    # 回调
    checkpoint_path = os.path.join(output_dir, "best_model.pt")
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
        EarlyStopping(patience=config.get("training.early_stopping_patience", 8), monitor="val_loss"),
    ]
    
    # 编译训练器
    trainer.compile(loss_fn=loss_fn, callbacks=callbacks)
    
    # 训练
    max_epochs = config.get("training.max_epochs", 50)
    t0 = time.time()
    trainer.fit(train_loader, val_loader, max_epochs=max_epochs)
    train_time = time.time() - t0
    
    logger.info("训练完成! 耗时: %.1f 分钟", train_time / 60)
    
    # 最终评估
    logger.info("=" * 60)
    logger.info("最终评估")
    logger.info("=" * 60)

    # 在验证集上评估
    val_logs = trainer.validate(model, val_loader)
    logger.info("验证集结果: %s", val_logs)

    # 在测试集上评估：遍历全部 batch，正确收集预测概率、预测类别与真实标签
    import numpy as np
    all_probs: list = []
    all_preds: list = []
    all_labels: list = []
    model.eval()
    label_key = "label"
    with torch.no_grad():
        for batch in test_loader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            outputs = model(batch)
            if isinstance(outputs, dict):
                logits = outputs.get("logits")
                probs = outputs.get("probabilities")
            else:
                logits = outputs
                probs = None
            if probs is None:
                probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            labels = batch.get(label_key)
            if labels is None:
                labels = batch.get("labels")
            if labels is None:
                raise KeyError(
                    f"测试 batch 缺少标签键 '{label_key}'/'labels'，无法计算测试指标"
                )
            all_labels.append(labels.cpu() if isinstance(labels, torch.Tensor) else torch.as_tensor(labels))

    probs_tensor = torch.cat(all_probs, dim=0)
    preds_tensor = torch.cat(all_preds, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    # 计算测试指标
    labels_np = labels_tensor.numpy()
    preds_np = preds_tensor.numpy()
    probs_np = probs_tensor.numpy()
    test_metrics = {
        "accuracy": calculate_accuracy(labels_np, preds_np),
        "precision_macro": calculate_precision(labels_np, preds_np, average="macro"),
        "recall_macro": calculate_recall(labels_np, preds_np, average="macro"),
        "f1_macro": calculate_f1_score(labels_np, preds_np, average="macro"),
    }
    try:
        test_metrics["auc_roc"] = calculate_auc_roc(labels_np, probs_np)
    except Exception as exc:  # 单类别等边界情况下 AUC 可能无法计算
        logger.warning("无法计算 AUC-ROC: %s", exc)

    logger.info("测试集指标: %s", test_metrics)
    test_metrics_path = os.path.join(output_dir, "test_metrics.json")
    save_json(test_metrics, test_metrics_path)
    logger.info("测试指标已保存: %s", test_metrics_path)

    # 保存训练曲线
    training_logs = {
        "train_losses": trainer.train_losses,
        "val_losses": trainer.val_losses,
        "best_val_loss": min(trainer.val_losses) if trainer.val_losses else None,
        "train_time_minutes": train_time / 60,
        "val_metrics": val_logs,
        "test_metrics": test_metrics,
    }
    
    logs_path = os.path.join(output_dir, "training_logs.json")
    save_json(training_logs, logs_path)
    logger.info("训练日志已保存: %s", logs_path)
    
    return training_logs


def parse_args():
    parser = argparse.ArgumentParser(description="PTM2CellNet 二分类微调")
    parser.add_argument("--config", type=str, default="configs/training/finetune_binary.yaml")
    parser.add_argument("--output", type=str, default="outputs/models/binary_finetune")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--label-map", type=str, default=None,
                        help='JSON label mapping, e.g. \'{"Quiescent": 0, "Activated": 1}\' '
                             '(default: {"Quiescent": 0, "Activated": 1})')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 解析 label_map：未提供时使用默认值，保持向后兼容
    label_map = DEFAULT_LABEL_MAP
    if args.label_map:
        try:
            label_map = json.loads(args.label_map)
        except json.JSONDecodeError as exc:
            logger.error("无效的 --label-map JSON: %s", exc)
            sys.exit(1)
        if not all(isinstance(v, int) for v in label_map.values()):
            logger.error("无效的 --label-map: 所有值必须是整数")
            sys.exit(1)
    logger.info("使用标签映射: %s", label_map)

    # 覆盖配置
    if args.epochs or args.batch_size or args.lr or args.device:
        config = Config.from_yaml(args.config)
        if args.epochs:
            config.set("training.max_epochs", args.epochs)
        if args.batch_size:
            config.set("training.batch_size", args.batch_size)
        if args.lr:
            config.set("training.learning_rate", args.lr)
        if args.device:
            config.set("training.device", args.device)
        config.save(args.config)

    train(args.config, args.output, label_map=label_map)
