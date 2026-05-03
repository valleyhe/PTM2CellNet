"""
训练脚本
功能概述: 训练PTM2CellNet模型的主脚本
设计思路: 整合数据加载、模型创建、训练流程
"""

import argparse
import json
import os
import sys


def _ensure_project_root() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PTM2CellNet训练脚本")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--data", type=str, default=None, help="训练数据路径")
    parser.add_argument("--output", type=str, default="outputs", help="输出目录")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    return parser.parse_args()


def main():
    """主函数"""
    _ensure_project_root()
    from src.utils.config import Config
    from src.utils.logging import setup_logger, get_timestamped_log_filename
    from src.data.loaders import DataLoader
    from src.data.preprocess import DataPreprocessor
    from src.data.datasets import PTMDataModule
    from src.models.architectures import PTM2CellNet
    from src.training.losses import FocalLoss
    from src.training.optimizers import configure_optimizer
    from src.training.callbacks import ModelCheckpoint, EarlyStopping
    from src.training.trainers import Trainer
    from src.evaluation.evaluators import Evaluator
    from src.evaluation.visualization import plot_training_curves

    logger = setup_logger(__name__, get_timestamped_log_filename("train"))
    logger.info("=" * 60)
    logger.info("PTM2CellNet 训练开始")
    logger.info("=" * 60)

    args = parse_args()

    config = Config.from_yaml(args.config)

    if args.epochs is not None:
        config.set("training.max_epochs", args.epochs)
    if args.batch_size is not None:
        config.set("training.batch_size", args.batch_size)
    if args.lr is not None:
        config.set("training.learning_rate", args.lr)

    logger.info("步骤 1: 加载数据")
    loader = DataLoader(config.to_dict())

    if args.data:
        df = loader.load_from_csv(args.data)
    else:
        df = loader.load_sample_data(num_samples=500)

    logger.info("步骤 2: 数据预处理")
    preprocessor = DataPreprocessor(config.to_dict())
    train_df, val_df, test_df = preprocessor.preprocess_pipeline(df)

    logger.info("步骤 3: 创建数据模块")
    batch_size = config.get("training.batch_size", 32)
    datamodule = PTMDataModule(
        train_df, val_df, test_df,
        config=config.to_dict(),
        batch_size=batch_size,
    )
    cell_states = datamodule.get_labels()

    # Create WeightedRandomSampler for balanced batches
    import numpy as np
    import torch
    from torch.utils.data import WeightedRandomSampler

    label_col = config.get("data.label_column", "cell_state")
    train_labels = train_df[label_col].map({cls: idx for idx, cls in enumerate(cell_states)}).values
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(train_labels),
        replacement=True
    )
    logger.info("使用WeightedRandomSampler平衡批次: 类别分布 %s", class_counts.tolist())

    # Create train dataloader with sampler
    from torch.utils.data import DataLoader
    train_dataset = datamodule.train_dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )

    logger.info("步骤 4: 创建模型")
    config.set("model.num_classes", len(cell_states))
    model = PTM2CellNet.from_config(config.to_dict())
    logger.info("模型创建完成: %s", model.encoder_type)

    logger.info("步骤 5: 配置训练器")
    trainer = Trainer(model, config=config.to_dict())

    loss_fn = FocalLoss(gamma=2.0)
    optimizer, scheduler = configure_optimizer(model, config.to_dict())

    trainer.compile(
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.output, "models", "best_model.pt"),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=config.get("training.early_stopping_patience", 10),
    )

    trainer.add_callback(checkpoint_callback)
    trainer.add_callback(early_stop_callback)

    logger.info("步骤 6: 开始训练")
    trainer.fit(
        train_loader,
        datamodule.val_dataloader(),
    )

    logger.info("步骤 7: 绘制训练曲线")
    logs = {
        "train_loss": trainer.train_losses,
        "val_loss": trainer.val_losses,
    }
    plot_training_curves(
        logs,
        os.path.join(args.output, "results", "training_curves.png"),
    )

    logger.info("步骤 8: 评估模型")
    evaluator = Evaluator(model, config=config.to_dict(), task_type="classification")
    test_metrics = evaluator.evaluate(datamodule.test_dataloader())

    metrics_path = os.path.join(args.output, "results", "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info("测试指标已保存: %s", metrics_path)

    logger.info("=" * 60)
    logger.info("训练完成")
    if trainer.val_losses:
        logger.info("最佳验证损失: %.4f", min(trainer.val_losses))
    else:
        logger.info("未进行验证，无法计算最佳验证损失")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
