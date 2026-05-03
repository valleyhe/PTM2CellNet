#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lightning训练脚本
使用PyTorch Lightning进行模型训练
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

from src.utils.config import Config
from src.data.loaders import DataLoader
from src.data.preprocess import DataPreprocessor
from src.data.lightning_datamodule import PTMDataModule
from src.models.architectures import PTM2CellNet
from src.training.lightning_module import PTM2CellNetLightning
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="PTM2CellNet Lightning训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lightning.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/sample_data.csv",
        help="训练数据路径",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复训练的检查点路径",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="最大训练轮数（覆盖配置文件）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批次大小（覆盖配置文件）",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="学习率（覆盖配置文件）",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="GPU数量",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["32", "16-mixed", "bf16-mixed"],
        help="训练精度",
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 加载配置
    logger.info(f"加载配置文件: {args.config}")
    config = Config.from_yaml(args.config)

    # 命令行参数覆盖配置
    if args.max_epochs is not None:
        config.set("training.max_epochs", args.max_epochs)
    if args.batch_size is not None:
        config.set("training.batch_size", args.batch_size)
    if args.learning_rate is not None:
        config.set("training.learning_rate", args.learning_rate)
    if args.gpus is not None:
        config.set("training.devices", args.gpus)
    if args.precision is not None:
        config.set("training.precision", args.precision)

    # 加载和预处理数据
    logger.info(f"加载数据: {args.data}")
    loader = DataLoader()
    df = loader.load_from_csv(args.data)

    preprocessor = DataPreprocessor(config.to_dict())
    train_df, val_df, test_df = preprocessor.preprocess_pipeline(df)

    # 创建数据模块
    logger.info("创建数据模块...")
    data_module = PTMDataModule(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=config.to_dict(),
    )

    # 创建模型
    logger.info("创建模型...")
    model = PTM2CellNet.from_config(config.to_dict())
    lightning_model = PTM2CellNetLightning(model, config.to_dict())

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数量: {total_params / 1e6:.2f}M")
    logger.info(f"可训练参数量: {trainable_params / 1e6:.2f}M")

    # 配置回调
    callbacks = []

    # 学习率监控
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # 模型检查点
    checkpoint_config = config.get("training.checkpoint", {})
    if checkpoint_config.get("enabled", True):
        checkpoint_dir = config.get("paths.outputs_models", "outputs/models")
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=checkpoint_config.get(
                "filename", "ptm2cellnet-{epoch:02d}-{val_loss:.4f}"
            ),
            monitor=checkpoint_config.get("monitor", "val_loss"),
            mode=checkpoint_config.get("mode", "min"),
            save_top_k=checkpoint_config.get("save_top_k", 3),
            save_last=True,
        )
        callbacks.append(checkpoint_callback)
        logger.info(f"检查点保存目录: {checkpoint_dir}")

    # 早停
    early_stopping_config = config.get("training.early_stopping", {})
    if early_stopping_config.get("enabled", True):
        early_stop_callback = EarlyStopping(
            monitor=early_stopping_config.get("monitor", "val_loss"),
            patience=early_stopping_config.get("patience", 10),
            mode=early_stopping_config.get("mode", "min"),
            verbose=True,
        )
        callbacks.append(early_stop_callback)
        logger.info(f"早停配置: patience={early_stopping_config.get('patience', 10)}")

    # 配置日志
    logger_config = config.get("training.logger", {})
    if logger_config.get("type", "tensorboard") == "tensorboard":
        log_dir = logger_config.get("save_dir", "outputs/logs")
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        tb_logger = L.loggers.TensorBoardLogger(
            save_dir=log_dir,
            name=logger_config.get("name", "lightning_logs"),
        )
        loggers = [tb_logger]
        logger.info(f"TensorBoard日志目录: {log_dir}")
    elif logger_config.get("type") == "wandb":
        wandb_logger = L.loggers.WandbLogger(
            project="ptm2cellnet",
            name=logger_config.get("name", "lightning_run"),
            save_dir=logger_config.get("save_dir", "outputs/logs"),
        )
        loggers = [wandb_logger]
    else:
        loggers = None

    # 创建Trainer
    trainer = L.Trainer(
        accelerator=config.get("training.accelerator", "gpu"),
        devices=config.get("training.devices", 1),
        strategy=config.get("training.strategy", "auto"),
        precision=config.get("training.precision", "32"),
        max_epochs=config.get("training.max_epochs", 100),
        accumulate_grad_batches=config.get("training.accumulate_grad_batches", 1),
        gradient_clip_val=config.get("training.gradient_clip_val", 0.0),
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        deterministic=False,
    )

    # 开始训练
    logger.info("=" * 60)
    logger.info("开始训练...")
    logger.info("=" * 60)

    trainer.fit(
        lightning_model,
        datamodule=data_module,
        ckpt_path=args.resume,
    )

    # 测试
    logger.info("=" * 60)
    logger.info("测试模型...")
    logger.info("=" * 60)

    test_results = trainer.test(datamodule=data_module)

    # 打印测试结果
    logger.info("测试结果:")
    for key, value in test_results[0].items():
        logger.info(f"  {key}: {value:.4f}")

    logger.info("=" * 60)
    logger.info("训练完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
