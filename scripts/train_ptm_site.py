#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PTM位点预测训练脚本
功能: 训练PTM位点二分类预测模型
"""

import sys
from pathlib import Path

# 确保项目根目录在sys.path中
def _ensure_project_root():
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

_ensure_project_root()

import argparse
import lightning as L
from lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from src.data.ptm_site_dataset import PTMSiteDataModule
from src.models.ptm_site_predictor import create_model
from src.training.ptm_site_lightning import PTMSiteLightning


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="PTM位点预测训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 数据参数
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/ptm_train_phosphorylation.csv",
        help="训练数据CSV路径",
    )
    parser.add_argument(
        "--ptm-type",
        type=str,
        default=None,
        help="筛选特定PTM类型",
    )
    # 模型参数
    parser.add_argument(
        "--encoder",
        type=str,
        default="cnn",
        choices=["cnn", "transformer", "lstm"],
        help="编码器类型",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=64,
        help="嵌入维度",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="隐藏层维度",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="编码器层数",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout率",
    )
    # 训练参数
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="最大训练轮数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="批次大小",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="学习率",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="权重衰减",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="GPU数量",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="训练精度",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="数据加载进程数",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="验证集比例",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="测试集比例",
    )
    # 输出参数
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ptm_pretrain",
        help="输出目录",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复训练的检查点路径",
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("PTM位点预测训练")
    print("=" * 60)
    print(f"数据文件: {args.data}")
    print(f"编码器类型: {args.encoder}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    # 创建数据模块
    print("\n加载数据...")
    data_module = PTMSiteDataModule(
        train_path=args.data,
        ptm_type=args.ptm_type,
        window_size=31,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
    )
    data_module.setup()

    # 创建模型
    print("\n创建模型...")
    model_config = {
        'vocab_size': 21,
        'embed_dim': args.embed_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'num_heads': 4,
        'dropout': args.dropout,
        'encoder_type': args.encoder,
        'window_size': 31,
        'num_classes': 2,
    }
    model = create_model(model_config)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")

    # Lightning模块
    training_config = {
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'max_epochs': args.max_epochs,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
    }
    lightning_model = PTMSiteLightning(model, {**model_config, **training_config})

    # 回调
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"ptm-{args.encoder}-{{epoch:02d}}-{{val_auroc:.4f}}",
            monitor='val_auroc',
            mode='max',
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor='val_auroc',
            patience=10,
            mode='max',
            verbose=True,
        ),
    ]

    # Logger
    logger = L.loggers.TensorBoardLogger(
        save_dir=output_dir / "logs",
        name=f"ptm_{args.encoder}",
    )

    # Trainer
    trainer = L.Trainer(
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        deterministic=False,
        gradient_clip_val=1.0,
    )

    # 开始训练
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    trainer.fit(
        lightning_model,
        datamodule=data_module,
        ckpt_path=args.resume,
    )

    # 测试
    print("\n" + "=" * 60)
    print("测试模型...")
    print("=" * 60)

    test_results = trainer.test(datamodule=data_module)

    print("\n测试结果:")
    for key, value in test_results[0].items():
        print(f"  {key}: {value:.4f}")

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"模型保存在: {checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
