#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多任务PTM位点预测训练脚本
功能: 联合训练多种PTM类型，提升K修饰区分能力
"""

import sys
import argparse
import logging
from pathlib import Path
from collections import defaultdict
import json

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.utils.io import safe_torch_load
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

from src.models.multitask_ptm import MultiTaskPTMPredictor, MultiTaskLoss
from src.data.multitask_dataset import (
    MultiTaskPTMDataModule,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, criterion, device, ptm_types):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = defaultdict(int)
    total = defaultdict(int)

    for batch in tqdm(dataloader, desc="Training"):
        sequence_indices = batch['sequence_indices'].to(device)
        ptm_types_batch = batch['ptm_types']

        # 按PTM类型分组预测
        outputs = {}
        targets = {}

        for ptm_type in ptm_types:
            # 筛选该PTM类型的样本
            indices = [i for i, t in enumerate(ptm_types_batch) if t == ptm_type]
            if not indices:
                continue

            ptm_inputs = sequence_indices[indices]
            output = model(ptm_inputs, ptm_type)

            outputs[ptm_type] = output
            targets[ptm_type] = batch['labels'][ptm_type].to(device)

        # 计算损失
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['total_loss']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # 统计准确率
        for ptm_type in outputs:
            preds = outputs[ptm_type]['logits'].argmax(dim=-1)
            correct[ptm_type] += (preds == targets[ptm_type]).sum().item()
            total[ptm_type] += targets[ptm_type].size(0)

    metrics = {
        'loss': total_loss / len(dataloader),
    }
    for ptm_type in ptm_types:
        if total[ptm_type] > 0:
            metrics[f'{ptm_type}_acc'] = correct[ptm_type] / total[ptm_type]

    return metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, ptm_types):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = defaultdict(int)
    total = defaultdict(int)
    all_probs = defaultdict(list)
    all_labels = defaultdict(list)

    for batch in tqdm(dataloader, desc="Evaluating"):
        sequence_indices = batch['sequence_indices'].to(device)
        ptm_types_batch = batch['ptm_types']

        outputs = {}
        targets = {}

        for ptm_type in ptm_types:
            indices = [i for i, t in enumerate(ptm_types_batch) if t == ptm_type]
            if not indices:
                continue

            ptm_inputs = sequence_indices[indices]
            output = model(ptm_inputs, ptm_type)

            outputs[ptm_type] = output
            targets[ptm_type] = batch['labels'][ptm_type].to(device)

            # 收集预测概率
            probs = output['probs'][:, 1].cpu().numpy()
            all_probs[ptm_type].extend(probs)
            all_labels[ptm_type].extend(targets[ptm_type].cpu().numpy())

        loss_dict = criterion(outputs, targets)
        total_loss += loss_dict['total_loss'].item()

        for ptm_type in outputs:
            preds = outputs[ptm_type]['logits'].argmax(dim=-1)
            correct[ptm_type] += (preds == targets[ptm_type]).sum().item()
            total[ptm_type] += targets[ptm_type].size(0)

    # 计算AUROC
    from sklearn.metrics import roc_auc_score, f1_score

    metrics = {
        'loss': total_loss / len(dataloader),
    }

    for ptm_type in ptm_types:
        if total[ptm_type] > 0:
            metrics[f'{ptm_type}_acc'] = correct[ptm_type] / total[ptm_type]

            if len(all_probs[ptm_type]) > 0 and len(set(all_labels[ptm_type])) > 1:
                metrics[f'{ptm_type}_auroc'] = roc_auc_score(all_labels[ptm_type], all_probs[ptm_type])

                preds = (np.array(all_probs[ptm_type]) > 0.5).astype(int)
                metrics[f'{ptm_type}_f1'] = f1_score(all_labels[ptm_type], preds)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='多任务PTM预测训练')
    parser.add_argument('--data-dir', '-d', type=str, default='data/processed',
                        help='数据目录')
    parser.add_argument('--output-dir', '-o', type=str, default='outputs/multitask_ptm',
                        help='输出目录')
    parser.add_argument('--ptm-types', '-p', type=str, nargs='+',
                        default=['Phosphorylation', 'Acetylation', 'Ubiquitination',
                                 'Methylation', 'Sumoylation', 'Succinylation'],
                        help='PTM类型')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='每种PTM类型的最大样本数')
    parser.add_argument('--batch-size', '-b', type=int, default=256,
                        help='批大小')
    parser.add_argument('--epochs', '-e', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--embed-dim', type=int, default=64,
                        help='嵌入维度')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='隐藏层维度')
    parser.add_argument('--share-encoder', action='store_true', default=True,
                        help='共享编码器')
    parser.add_argument('--gpus', type=int, default=1,
                        help='GPU数量')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpus > 0 else 'cpu')
    logger.info(f"使用设备: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    # 准备数据
    logger.info("准备数据...")
    data_module = MultiTaskPTMDataModule(
        data_dir=args.data_dir,
        ptm_types=args.ptm_types,
        batch_size=args.batch_size,
        max_samples_per_type=args.max_samples,
    )
    data_module.setup()

    # 创建模型
    logger.info("创建模型...")
    model = MultiTaskPTMPredictor(
        vocab_size=21,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        encoder_type='cnn',
        ptm_types=args.ptm_types,
        share_encoder=args.share_encoder,
    )
    model.to(device)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    # 损失函数
    criterion = MultiTaskLoss(args.ptm_types)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练循环
    best_auroc = 0
    history = []

    logger.info("=" * 60)
    logger.info("开始训练")
    logger.info("=" * 60)

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        # 训练
        train_metrics = train_epoch(
            model, data_module.train_dataloader(),
            optimizer, criterion, device, args.ptm_types
        )

        # 验证
        val_metrics = evaluate(
            model, data_module.val_dataloader(),
            criterion, device, args.ptm_types
        )

        # 更新学习率
        scheduler.step()

        # 记录
        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
        })

        # 打印指标
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")

        for ptm_type in args.ptm_types:
            auroc_key = f'{ptm_type}_auroc'
            if auroc_key in val_metrics:
                logger.info(f"  {ptm_type}: AUROC={val_metrics[auroc_key]:.4f}, "
                           f"Acc={val_metrics.get(f'{ptm_type}_acc', 0):.4f}")

        # 保存最佳模型
        avg_auroc = np.mean([val_metrics.get(f'{ptm}_auroc', 0) for ptm in args.ptm_types
                            if f'{ptm}_auroc' in val_metrics])

        if avg_auroc > best_auroc:
            best_auroc = avg_auroc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auroc': best_auroc,
            }, checkpoint_dir / 'best_model.ckpt')
            logger.info(f"保存最佳模型: AUROC={best_auroc:.4f}")

    # 测试
    logger.info("\n" + "=" * 60)
    logger.info("测试集评估")
    logger.info("=" * 60)

    # 加载最佳模型
    # 训练保存的 checkpoint 包含 optimizer state（含 numpy 标量），
    # 需显式使用 weights_only=False 加载（文件由本项目自身写入，可信）。
    logger.warning(
        "⚠️ SECURITY RISK: Loading checkpoint with weights_only=False at %s. "
        "This allows arbitrary code execution via pickle deserialization. "
        "Safe because this file was written by the current training run. "
        "For external checkpoints, prefer weights_only=True with allowed_classes=.",
        checkpoint_dir / 'best_model.ckpt',
    )
    checkpoint = safe_torch_load(checkpoint_dir / 'best_model.ckpt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(
        model, data_module.test_dataloader(),
        criterion, device, args.ptm_types
    )

    for ptm_type in args.ptm_types:
        auroc_key = f'{ptm_type}_auroc'
        if auroc_key in test_metrics:
            logger.info(f"{ptm_type}: AUROC={test_metrics[auroc_key]:.4f}, "
                       f"Acc={test_metrics.get(f'{ptm_type}_acc', 0):.4f}, "
                       f"F1={test_metrics.get(f'{ptm_type}_f1', 0):.4f}")

    # 保存结果
    results = {
        'args': vars(args),
        'history': history,
        'test_metrics': test_metrics,
        'best_auroc': best_auroc,
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n结果保存至: {output_dir}")


if __name__ == '__main__':
    main()