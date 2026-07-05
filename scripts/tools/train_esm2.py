#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ESM-2预训练模型微调训练脚本
功能: 使用ESM-2蛋白质语言模型进行PTM位点预测
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import json
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ESM2PTMDataset(Dataset):
    """ESM-2 PTM数据集"""

    def __init__(self, samples, ptm_type):
        self.samples = samples
        self.ptm_type = ptm_type

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'sequence': sample['sequence_window'],
            'label': sample['label'],
            'gene': sample.get('uniprot_id', 'unknown'),
            'position': sample.get('position', 0),
        }


def collate_esm2_batch(batch):
    """ESM-2批处理函数"""
    sequences = [item['sequence'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])

    return {
        'sequences': sequences,
        'labels': labels,
    }


def load_ptm_data(data_dir, ptm_type, max_samples=None):
    """加载PTM数据"""
    file_path = Path(data_dir) / f"ptm_train_{ptm_type.lower()}.csv"
    if not file_path.exists():
        return []

    df = pd.read_csv(file_path)

    if max_samples and len(df) > max_samples:
        pos_df = df[df['label'] == 1].sample(max_samples // 2, random_state=42)
        neg_df = df[df['label'] == 0].sample(max_samples // 2, random_state=42)
        df = pd.concat([pos_df, neg_df])

    samples = []
    for _, row in df.iterrows():
        samples.append({
            'sequence_window': row['sequence_window'],
            'label': int(row['label']),
            'uniprot_id': row.get('uniprot_id', 'unknown'),
            'position': row.get('position', 0),
        })

    return samples


class ESM2FineTunedModel(nn.Module):
    """ESM-2微调模型"""

    def __init__(
        self,
        esm_model='esm2_t12_35M_UR50D',
        hidden_dim=256,
        num_classes=2,
        freeze_esm=True,
        dropout=0.1,
    ):
        super().__init__()

        self.freeze_esm = freeze_esm
        self.esm_model_name = esm_model

        # 延迟加载ESM
        self.esm = None
        self.alphabet = None
        self.batch_converter = None
        self.esm_dim = 480  # 默认

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.esm_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _load_esm(self):
        """延迟加载ESM模型"""
        if self.esm is not None:
            return

        try:
            import esm
            logger.info(f"加载ESM-2模型: {self.esm_model_name}")

            self.esm, self.alphabet = esm.pretrained.load_model_and_alphabet(
                self.esm_model_name
            )
            self.batch_converter = self.alphabet.get_batch_converter()

            # 获取embedding维度
            self.esm_dim = self.esm.args.embed_dim

            # 更新分类器输入维度
            self.classifier[0] = nn.Linear(self.esm_dim, self.classifier[0].out_features)

            # 冻结参数
            if self.freeze_esm:
                for param in self.esm.parameters():
                    param.requires_grad = False
                self.esm.eval()
                logger.info("ESM-2参数已冻结")

        except ImportError:
            raise ImportError("请安装fair-esm: pip install fair-esm")

    def encode_sequences(self, sequences):
        """编码序列"""
        self._load_esm()

        # 准备数据
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)

        # 移动到设备
        device = next(self.classifier.parameters()).device
        batch_tokens = batch_tokens.to(device)

        # 编码
        with torch.no_grad() if self.freeze_esm else torch.enable_grad():
            results = self.esm(
                batch_tokens,
                repr_layers=[self.esm.num_layers],
                return_contacts=False,
            )

        # 获取CLS token表示
        token_repr = results['representations'][self.esm.num_layers]
        cls_repr = token_repr[:, 0]  # CLS token

        return cls_repr

    def forward(self, sequences):
        """前向传播"""
        embeddings = self.encode_sequences(sequences)
        logits = self.classifier(embeddings)
        probs = torch.softmax(logits, dim=-1)
        return {'logits': logits, 'probs': probs, 'embeddings': embeddings}


def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
        sequences = batch['sequences']
        labels = batch['labels'].to(device)

        # 前向传播
        output = model(sequences)
        loss = criterion(output['logits'], labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # 统计
        preds = output['logits'].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_probs.extend(output['probs'][:, 1].detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 计算AUROC
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total,
        'auroc': auroc,
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        sequences = batch['sequences']
        labels = batch['labels'].to(device)

        output = model(sequences)
        loss = criterion(output['logits'], labels)

        total_loss += loss.item()

        preds = output['logits'].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_probs.extend(output['probs'][:, 1].cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 计算AUROC
    from sklearn.metrics import roc_auc_score, f1_score
    auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    pred_labels = (np.array(all_probs) > 0.5).astype(int)
    f1 = f1_score(all_labels, pred_labels)

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total,
        'auroc': auroc,
        'f1': f1,
    }


def main():
    parser = argparse.ArgumentParser(description='ESM-2 PTM预测训练')
    parser.add_argument('--data-dir', default='data/processed')
    parser.add_argument('--output-dir', default='outputs/esm2_ptm')
    parser.add_argument('--ptm-type', default='Phosphorylation')
    parser.add_argument('--max-samples', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=16)  # ESM-2需要较小batch
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--esm-model', default='esm2_t12_35M_UR50D')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--freeze-esm', action='store_true', default=True)
    parser.add_argument('--unfreeze-esm', action='store_false', dest='freeze_esm')
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpus > 0 else 'cpu')
    logger.info(f"使用设备: {device}")

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)

    # 加载数据
    logger.info(f"加载{args.ptm_type}数据...")
    samples = load_ptm_data(args.data_dir, args.ptm_type, args.max_samples)

    if not samples:
        logger.error(f"未找到{args.ptm_type}数据")
        return

    random.shuffle(samples)
    n_train = int(len(samples) * 0.8)
    n_val = int(len(samples) * 0.1)

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    logger.info(f"数据划分: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    # 创建数据集
    train_dataset = ESM2PTMDataset(train_samples, args.ptm_type)
    val_dataset = ESM2PTMDataset(val_samples, args.ptm_type)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_esm2_batch,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_esm2_batch,
        num_workers=0,
    )

    # 创建模型
    logger.info("创建ESM-2模型...")
    model = ESM2FineTunedModel(
        esm_model=args.esm_model,
        hidden_dim=args.hidden_dim,
        freeze_esm=args.freeze_esm,
    )
    model.to(device)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # 训练
    best_auroc = 0
    logger.info("=" * 60)
    logger.info("开始ESM-2训练")
    logger.info("=" * 60)

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)

        # 验证
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        # 打印
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, AUROC: {train_metrics['auroc']:.4f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, AUROC: {val_metrics['auroc']:.4f}, F1: {val_metrics['f1']:.4f}")

        # 保存最佳模型
        if val_metrics['auroc'] > best_auroc:
            best_auroc = val_metrics['auroc']
            torch.save(model.state_dict(), output_dir / 'checkpoints' / 'best_model.pt')
            logger.info(f"保存最佳模型: AUROC={best_auroc:.4f}")

    logger.info(f"\n训练完成，最佳AUROC: {best_auroc:.4f}")

    # 保存结果
    results = {
        'ptm_type': args.ptm_type,
        'esm_model': args.esm_model,
        'best_auroc': best_auroc,
        'args': vars(args),
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()