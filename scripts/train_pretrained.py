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
import os
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger

from src.utils.config import Config
from src.data.loaders import DataLoader
from src.data.preprocess import DataPreprocessor
from src.data.lightning_datamodule import PTMLightningDataModule
from src.models.architectures import PTM2CellNet
from src.training.artifacts import export_inference_artifact, write_artifact_manifest
from src.training.lightning_module import PTM2CellNetLightning
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


# 支持的预训练模型
PRETRAINED_MODELS = {
    "esm2_8M": "configs/pretrained/esm2_8m.yaml",
    "esm2_35M": "configs/pretrained/esm2_35m.yaml",
    "esm2_150M": "configs/pretrained/esm2_150m.yaml",
    "esm2_650M": "configs/pretrained/esm2_650m.yaml",
    "esm3_sm_open": "configs/pretrained/esm3_sm_open.yaml",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子以确保可复现性
    import random
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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
    # PTM2CellNet 内部会对 encoder_type 做 .lower()，并使用
    # ``encoder_type.startswith("esm2")`` 判定；model_size 通过
    # ``split("_")[1]`` 取得（如 "esm2_8m" -> "8m"），ESM2Encoder 内部
    # 再 normalize 为大写 "8M"。因此这里只需小写化，避免历史上的
    # ``.replace("m", "M")`` 把 "esm2_8M" 错误变成 "esM2_8M"。
    config.set("model.encoder_type", args.model.lower())

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
    # P1-3: detect the synthetic sample fixture so the resulting artifact is
    # correctly marked model_kind=demo and downstream tooling can warn against
    # biological interpretation.
    _data_basename = os.path.basename(args.data).lower()
    is_demo_data = "sample" in _data_basename or "synthetic" in _data_basename
    data_source = args.data

    preprocessor = DataPreprocessor(config.to_dict())
    try:
        train_df, val_df, test_df = preprocessor.preprocess_pipeline(df)
    except Exception as exc:
        # P1-3: 将预处理期空数据集等错误转为明确的退出码与可操作信息
        raise SystemExit(f"数据预处理失败: {exc}") from exc

    # P0-2: 派生标签映射并同步到 config（与 train.py / train_lightning.py 一致），
    # 否则预训练模型产物无法被 predict.py / API 可靠解码标签。
    from src.data.labels import derive_label_mapping

    cell_states, label_to_idx = derive_label_mapping(train_df)
    config.set("data.cell_states", cell_states)
    config.set("data.label_to_idx", label_to_idx)
    config.set("model.num_classes", len(cell_states))
    logger.info(
        "训练标签映射: cell_states=%s, num_classes=%d", cell_states, len(cell_states)
    )

    # 运行前数据预检：drop_last=True 时训练样本数必须 >= batch_size。
    batch_size = config.get("training.batch_size", 32)
    drop_last = config.get("training.drop_last", True)
    if len(train_df) == 0:
        raise SystemExit("训练失败：预处理后训练集为空。")
    if drop_last and len(train_df) < batch_size:
        raise SystemExit(
            f"训练失败：启用 drop_last=True 时训练集样本数 ({len(train_df)}) "
            f"必须 >= batch_size ({batch_size})。请减小 batch_size 或设置 "
            f"training.drop_last=False。"
        )
    if len(test_df) == 0:
        logger.warning("测试集为空，训练后将跳过测试阶段。")

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
    data_module = PTMLightningDataModule(
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

    # 模型检查点目录——即使禁用 Lightning 自动 checkpoint，推理 artifact 也
    # 必须写入此目录（P0-2），因此 checkpoint_dir 始终解析。
    checkpoint_config = config.get("training.checkpoint", {})
    checkpoint_dir = config.get("paths.outputs_models", f"outputs/models/{args.model}")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if checkpoint_config.get("enabled", True):
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

    tb_logger = TensorBoardLogger(
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
    test_metrics: dict = {}
    if len(test_df) == 0:
        logger.warning("测试集为空，跳过测试阶段。")
    else:
        logger.info("=" * 60)
        logger.info("测试模型...")
        logger.info("=" * 60)
        try:
            test_results = trainer.test(datamodule=data_module)
            if test_results:
                # 打印测试结果
                logger.info("测试结果:")
                for key, value in test_results[0].items():
                    logger.info(f"  {key}: {value:.4f}")
                test_metrics = dict(test_results[0])
        except Exception as exc:
            logger.warning("测试阶段失败（通常因缺少 best checkpoint）: %s", exc)

    # P0-2: 统一推理 artifact 合约——导出与 train.py / train_lightning.py 对齐的
    # best_model.pt + best_model.config.yaml + artifact_manifest.json，使 CLI/API
    # 推理入口可直接消费预训练模型产物。Lightning .ckpt 仍保留用于恢复训练。
    # 从 Lightning ModelCheckpoint 中取最佳 .ckpt 提取裸权重；否则用当前模型权重。
    best_ckpt_path = None
    for cb in getattr(trainer, "checkpoint_callbacks", []) or []:
        try:
            cand = cb.best_model_path
            if cand and os.path.exists(cand):
                best_ckpt_path = cand
                break
        except Exception:
            best_ckpt_path = None

    if best_ckpt_path:
        logger.info("从最佳 Lightning checkpoint 提取裸权重: %s", best_ckpt_path)
        best_module = PTM2CellNetLightning.load_from_checkpoint(
            best_ckpt_path,
            model=PTM2CellNet.from_config(config.to_dict()),
            config=config.to_dict(),
        )
        export_source_model = best_module
    else:
        logger.warning("未找到最佳 Lightning checkpoint，使用当前模型权重导出推理 artifact。")
        export_source_model = lightning_model

    encoder_metadata = {
        "encoder_type": config.get("model.encoder_type"),
        "pretrained_backbone": args.model,
        "freeze_encoder": bool(config.get("model.freeze_encoder", False)),
    }
    export_inference_artifact(
        export_source_model,
        config,
        checkpoint_dir,
        cell_states=cell_states,
        is_demo_data=is_demo_data,
        data_source=data_source,
        extra_encoder_metadata=encoder_metadata,
    )

    # P1-1 / P1-3: artifact manifest 含 provenance + 评估指标 + 发布门禁字段。
    write_artifact_manifest(
        checkpoint_dir,
        model_class=type(model).__name__,
        cell_states=cell_states,
        config=config,
        training_entrypoint="scripts/train_pretrained.py",
        is_demo_data=is_demo_data,
        data_source=data_source,
        train_df=train_df,
        split_strategy=config.get("data.split_strategy")
        or f"random({config.get('data.split.train_ratio', 0.7)}/"
        f"{config.get('data.split.val_ratio', 0.15)}/"
        f"{config.get('data.split.test_ratio', 0.15)})",
        metrics=test_metrics or None,
        encoder_metadata=encoder_metadata,
        intended_use=(
            "Engineering pipeline validation only." if is_demo_data
            else "Cell-state prediction with a pretrained protein LM backbone."
        ),
    )

    logger.info("=" * 60)
    logger.info(f"{args.model} 训练完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
