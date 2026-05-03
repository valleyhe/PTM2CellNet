#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预训练模型训练脚本
使用预训练模型（ESM-2, ProtBERT等）进行训练
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
from lightning.pytorch.callbacks import (
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


# 支持的预训练模型
PRETRAINED_MODELS = {
    "esm2_8M": "configs/pretrained/esm2_8m.yaml",
    "esm2_35M": "configs/pretrained/esm2_35m.yaml",
    "esm2_150M": "configs/pretrained/esm2_150m.yaml",
    "esm2_650M": "configs/pretrained/esm2_650m.yaml",
    "protbert": "configs/pretrained/protbert.yaml",
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="PTM2CellNet 预训练模型训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="esm2_150M",
        choices=list(PRETRAINED_MODELS.keys()),
        help="预训练模型类型",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（覆盖默认配置）",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/sample_data.csv",
        help="训练数据路径",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="冻结预训练模型权重",
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
        help="最大训练轮数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批次大小",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="学习率",
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
        default="bf16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="训练精度",
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 确定配置文件
    if args.config is not None:
        config_path = args.config
    else:
        config_path = PRETRAINED_MODELS.get(args.model)
        if config_path is None:
            logger.error(f"未知的预训练模型: {args.model}")
            sys.exit(1)

    # 加载配置
    logger.info(f"使用预训练模型: {args.model}")
    logger.info(f"加载配置文件: {config_path}")
    config = Config.from_yaml(config_path)

    # 确保encoder_type与模型名称匹配
    config.set("model.encoder_type", args.model.lower().replace("m", "M"))

    # 命令行参数覆盖配置
    if args.freeze:
        config.set("model.freeze_encoder", True)
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

    # 创建模型
    logger.info("创建模型...")
    model = PTM2CellNet.from_config(config.to_dict())
    lightning_model = PTM2CellNetLightning(model, config.to_dict())

    # 如果使用ESM编码器，从模型中取出tokenizer
    tokenizer = None
    if hasattr(model.encoder, "tokenizer"):
        tokenizer = model.encoder.tokenizer
        logger.info(f"使用ESM tokenizer: {type(tokenizer).__name__}")

    # 创建数据模块
    logger.info("创建数据模块...")
    data_module = PTMDataModule(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=config.to_dict(),
        tokenizer=tokenizer,
    )

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数量: {total_params / 1e6:.2f}M")
    logger.info(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    logger.info(f"编码器类型: {config.get('model.encoder_type')}")
    logger.info(f"冻结编码器: {config.get('model.freeze_encoder', False)}")

    # 配置回调
    callbacks = []
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # 模型检查点
    checkpoint_config = config.get("training.checkpoint", {})
    if checkpoint_config.get("enabled", True):
        checkpoint_dir = config.get("paths.outputs_models", f"outputs/models/{args.model}")
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{args.model}-{{epoch:02d}}-{{val_loss:.4f}}",
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
            patience=early_stopping_config.get("patience", 5),
            mode=early_stopping_config.get("mode", "min"),
            verbose=True,
        )
        callbacks.append(early_stop_callback)

    # 配置日志
    logger_config = config.get("training.logger", {})
    log_dir = logger_config.get("save_dir", "outputs/logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    tb_logger = L.loggers.TensorBoardLogger(
        save_dir=log_dir,
        name=logger_config.get("name", args.model),
    )

    # 创建Trainer
    trainer = L.Trainer(
        accelerator=config.get("training.accelerator", "gpu"),
        devices=config.get("training.devices", 1),
        strategy=config.get("training.strategy", "auto"),
        precision=config.get("training.precision", "bf16-mixed"),
        max_epochs=config.get("training.max_epochs", 50),
        accumulate_grad_batches=config.get("training.accumulate_grad_batches", 4),
        gradient_clip_val=config.get("training.gradient_clip_val", 1.0),
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=10,
    )

    # 开始训练
    logger.info("=" * 60)
    logger.info(f"开始训练 {args.model}...")
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
    logger.info(f"{args.model} 训练完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
