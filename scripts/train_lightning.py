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
import os
import subprocess
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger

from src.utils.config import Config
from src.data.loaders import DataLoader
from src.data.preprocess import DataPreprocessor
from src.data.lightning_datamodule import PTMLightningDataModule
from src.models.architectures import PTM2CellNet
from src.training.logging_config import configure_default_logger
from src.training.lightning_module import PTM2CellNetLightning
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def _resolve_trainer_runtime(config: Config) -> tuple[str, int, str]:
    """Choose trainer runtime settings compatible with the current host."""
    accelerator = config.get("training.accelerator", "gpu")
    devices = config.get("training.devices", 1)
    precision = config.get("training.precision", "32")

    if accelerator == "gpu" and not torch.cuda.is_available():
        logger.warning("未检测到CUDA，回退到CPU训练配置")
        accelerator = "cpu"
        devices = 1

    if accelerator == "cpu" and precision != "32":
        logger.warning("CPU环境不支持混合精度，使用32位精度")
        precision = "32"

    return accelerator, devices, precision


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="PTM2CellNet Lightning训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/smoke/lightning_cnn_cpu.yaml",
        help="配置文件路径（首次使用推荐 configs/smoke/ 下的配置）",
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
    L.seed_everything(args.seed, workers=True)

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

    # P0-1: 从训练集推导标签映射并同步到 config
    from src.data.labels import derive_label_mapping

    cell_states, label_to_idx = derive_label_mapping(train_df)
    config.set("data.cell_states", cell_states)
    config.set("data.label_to_idx", label_to_idx)
    config.set("model.num_classes", len(cell_states))
    logger.info("训练标签映射: cell_states=%s, num_classes=%d", cell_states, len(cell_states))

    # P0-2: 运行前数据预检
    batch_size = config.get("training.batch_size", 32)
    drop_last = config.get("training.drop_last", True)
    if len(train_df) == 0:
        raise SystemExit("训练失败：预处理后训练集为空。")
    if drop_last and len(train_df) < batch_size:
        raise SystemExit(
            f"训练失败：启用 drop_last=True 时训练集样本数 ({len(train_df)}) "
            f"必须 >= batch_size ({batch_size})。"
            f"请减小 batch_size 或设置 training.drop_last=False。"
        )
    val_monitor = config.get("training.checkpoint.monitor", "val_loss")
    if val_monitor == "val_loss" and len(val_df) == 0:
        logger.warning("验证集为空，禁用 val_loss monitor。训练仍将继续但不会保存最佳 checkpoint。")
        config.set("training.checkpoint.monitor", "step")
        config.set("training.checkpoint.save_top_k", 0)
        config.set("training.early_stopping.enabled", False)
    if len(test_df) == 0:
        logger.warning("测试集为空，训练后将跳过测试阶段。")

    # 创建数据模块
    logger.info("创建数据模块...")
    data_module = PTMLightningDataModule(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=config.to_dict(),
    )

    # 创建模型
    logger.info("创建模型...")
    model = PTM2CellNet.from_config(config.to_dict())

    # 自动类别权重: 当 training.auto_class_weights=True 且未显式提供 class_weights 时，
    # 从训练标签计算逆频率权重并写入 config，使 FocalLoss 消费之。这激活了
    # PTM2CellNetLightning.compute_class_weights 静态方法（此前为死代码）。
    training_cfg = config.get("training", {}) or {}
    if (
        training_cfg.get("auto_class_weights", False)
        and training_cfg.get("class_weights") is None
    ):
        import torch as _torch
        from src.training.lightning_module import PTM2CellNetLightning as _LightningMod

        label_col = config.get("data.label_column", "cell_state")
        label_series = train_df[label_col].map(label_to_idx)
        label_series = label_series.dropna().astype(int)
        if len(label_series) > 0:
            labels_tensor = _torch.tensor(label_series.values, dtype=_torch.long)
            num_classes = len(cell_states)
            weights = _LightningMod.compute_class_weights(labels_tensor, num_classes)
            config.set("training.class_weights", weights.tolist())
            logger.info(
                "auto_class_weights 已计算类别权重: %s", weights.tolist()
            )

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
    checkpoint_dir = config.get("paths.outputs_models", "outputs/models")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_config = config.get("training.checkpoint", {})
    if checkpoint_config.get("enabled", True):
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
    log_dir = logger_config.get("save_dir", "outputs/logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    loggers = None
    if logger_config.get("type", "tensorboard") == "tensorboard":
        loggers = [
            configure_default_logger(
                save_dir=log_dir,
                name=logger_config.get("name", "lightning_logs"),
                kind="tensorboard",
            )
        ]
        logger.info(f"Lightning日志目录: {log_dir}")
    elif logger_config.get("type") == "wandb":
        wandb_logger = WandbLogger(
            project="ptm2cellnet",
            name=logger_config.get("name", "lightning_run"),
            save_dir=logger_config.get("save_dir", "outputs/logs"),
        )
        loggers = [wandb_logger]

    accelerator, devices, precision = _resolve_trainer_runtime(config)

    # 创建Trainer
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=config.get("training.strategy", "auto"),
        precision=precision,
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

    # 断点续训配置一致性校验: 若 --resume 指定 checkpoint，记录校验日志，
    # 提示 num_classes/label 映射/模型结构应与当前 config 一致；不一致会导致
    # Lightning 加载时静默错误。此处仅做存在性检查与日志记录，实际加载由
    # Lightning 的 ckpt_path 参数完成。
    if args.resume is not None:
        import os as _os
        if not _os.path.exists(args.resume):
            raise SystemExit(f"--resume 指定的 checkpoint 不存在: {args.resume}")
        logger.info("断点续训: 从 %s 恢复", args.resume)
        logger.info(
            "配置一致性校验: 请确保 checkpoint 的 num_classes=%s、label 映射及模型结构与当前配置一致；"
            "若不一致，恢复后的训练将产生静默错误。",
            config.get("model.num_classes", "未配置"),
        )

    trainer.fit(
        lightning_model,
        datamodule=data_module,
        ckpt_path=args.resume,
    )

    # 测试
    if len(test_df) == 0:
        logger.warning("测试集为空，跳过测试阶段。")
    else:
        logger.info("=" * 60)
        logger.info("测试模型...")
        logger.info("=" * 60)
        try:
            test_results = trainer.test(datamodule=data_module)
            logger.info("测试结果:")
            for key, value in test_results[0].items():
                logger.info(f"  {key}: {value:.4f}")
        except Exception as exc:
            logger.warning("测试阶段失败（通常因缺少 best checkpoint）: %s", exc)

    # P0-3: 统一 checkpoint 合约——除 Lightning 的 .ckpt 外，额外导出推理 artifact
    # （裸 best_model.pt + best_model.config.yaml + 标签映射），使 CLI/API 推理入口
    # 可直接消费 Lightning 训练产物。.ckpt 仍保留用于恢复训练。
    _export_inference_artifact(
        lightning_model, data_module, config, checkpoint_dir,
        is_demo_data=is_demo_data, data_source=data_source,
    )

    # P1-3: 导出 artifact manifest
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        git_commit = "unknown"
    manifest = {
        "checkpoint_path": os.path.join(checkpoint_dir, "best_model.pt"),
        "config_path": os.path.join(checkpoint_dir, "best_model.config.yaml"),
        "training_entrypoint": "scripts/train_lightning.py",
        "model_class": type(model).__name__,
        "num_classes": config.get("model.num_classes", 0),
        "cell_states": config.get("data.cell_states", []),
        "max_sequence_length": config.get("data.max_sequence_length", 1000),
        "ptm_types": config.get("data.ptm_types", None),
        "git_commit": git_commit,
        "created_at": __import__("datetime").datetime.now().isoformat(),
    }
    # P1-3: record data provenance + model_kind so downstream tooling can flag
    # demo models and prevent misuse for biological interpretation.
    model_kind = "demo" if is_demo_data else "real"
    manifest["model_kind"] = model_kind
    manifest["model_card"] = {
        "model_kind": model_kind,
        "training_data": "synthetic_random" if is_demo_data else data_source,
        "not_for_biological_use": is_demo_data,
        "description": (
            "Demo/smoke model trained on the synthetic sample fixture. "
            "Validates the engineering pipeline only; predictions have no "
            "biological meaning."
        ) if is_demo_data else (
            "Model trained on real data. Verify dataset provenance before use."
        ),
    }
    manifest["data_provenance"] = {
        "model_kind": model_kind,
        "training_data": "synthetic_random" if is_demo_data else data_source,
        "source": data_source,
        "not_for_biological_use": is_demo_data,
    }
    from src.utils.io import save_json

    save_json(manifest, os.path.join(checkpoint_dir, "artifact_manifest.json"))
    if is_demo_data:
        logger.warning(
            "⚠️ 该模型在合成/示例数据上训练，将标记为 model_kind=demo。"
            "产物仅用于工程链路验证，不可用于真实生物学预测。"
        )
    logger.info("artifact manifest 已导出")

    logger.info("=" * 60)
    logger.info("训练完成！")
    logger.info("=" * 60)


def _export_inference_artifact(
    lightning_model, data_module, config, checkpoint_dir, is_demo_data=False, data_source="unknown"
):
    """导出推理 artifact：裸 state_dict + config + 标签映射。

    优先使用 Lightning ModelCheckpoint 记录的最佳权重；若不可得则用当前权重。
    产物文件（与 scripts/train.py 对齐）：
        - ``best_model.pt``：裸 ``state_dict``，可被 ``predict.py`` / API 直接加载
        - ``best_model.config.yaml``：含 ``data.cell_states`` / ``data.label_to_idx``
    """
    import os

    from src.utils.io import save_model

    # 解析训练标签顺序（与 dataset_base 的 sorted(unique) 一致）
    train_ds = getattr(data_module, "train_dataset", None)
    cell_states = list(getattr(train_ds, "labels", []) or [])
    if not cell_states:
        logger.warning(
            "导出推理 artifact 时未解析到 cell_states，跳过导出。"
            "推理将无法可靠解码类别索引。"
        )
        return

    config.set("data.cell_states", cell_states)
    config.set(
        "data.label_to_idx",
        {label: idx for idx, label in enumerate(cell_states)},
    )
    config_num_classes = config.get("model.num_classes")
    actual_num_classes = len(cell_states)
    if config_num_classes is not None and config_num_classes != actual_num_classes:
        raise ValueError(
            f"model.num_classes ({config_num_classes}) 与 data.cell_states 数量 "
            f"({actual_num_classes}) 不一致。请确保模型创建前已同步。"
        )

    # 从 Lightning checkpoint 回调中取最佳 .ckpt，提取裸权重；否则用当前模型权重
    best_ckpt_path = None
    for cb in lightning_model.trainer.checkpoint_callbacks if hasattr(lightning_model, "trainer") else []:
        try:
            best_ckpt_path = cb.best_model_path
            if best_ckpt_path and os.path.exists(best_ckpt_path):
                break
        except Exception:
            best_ckpt_path = None

    if best_ckpt_path and os.path.exists(best_ckpt_path):
        logger.info("从最佳 Lightning checkpoint 提取裸权重: %s", best_ckpt_path)
        best_module = PTM2CellNetLightning.load_from_checkpoint(
            best_ckpt_path, model=PTM2CellNet.from_config(config.to_dict()), config=config.to_dict()
        )
        base_model = best_module.model
    else:
        logger.warning("未找到最佳 Lightning checkpoint，使用当前模型权重导出推理 artifact。")
        base_model = lightning_model.model

    artifact_path = os.path.join(checkpoint_dir, "best_model.pt")
    save_model(base_model, artifact_path)
    config_path = os.path.join(checkpoint_dir, "best_model.config.yaml")
    # P1-3: persist model_kind + data provenance into the config yaml so the
    # config is self-describing (the manifest is the other source of truth).
    config.set("model.model_kind", "demo" if is_demo_data else "real")
    config.set(
        "model_card",
        {
            "model_kind": "demo" if is_demo_data else "real",
            "training_data": "synthetic_random" if is_demo_data else data_source,
            "not_for_biological_use": is_demo_data,
        },
    )
    config.save(config_path)
    logger.info("已导出推理 artifact: %s + %s (cell_states=%s)", artifact_path, config_path, cell_states)


if __name__ == "__main__":
    main()
