"""
训练脚本
功能概述: 训练PTM2CellNet模型的主脚本
设计思路: 整合数据加载、模型创建、训练流程
"""

import argparse
import json
import os
import random
import subprocess
import sys

import numpy as np


def _ensure_project_root() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PTM2CellNet训练脚本")
    # P1-3: 默认指向 smoke 配置，使 ``python scripts/train.py`` 首次运行即
    # 轻量可跑（小型 CNN、CPU、1 epoch）。研究/生产训练请显式传 configs/research/*。
    parser.add_argument(
        "--config",
        type=str,
        default="configs/smoke/cnn_cpu.yaml",
        help="配置文件路径（默认 smoke 快速验证；研究训练请用 configs/research/*.yaml）",
    )
    parser.add_argument("--data", type=str, default=None, help="训练数据路径")
    parser.add_argument("--output", type=str, default="outputs", help="输出目录")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--pathway-analysis", action="store_true", default=False, help="启用信号通路分析（基于验证集）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（确保可复现性）")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="checkpoint 路径用于断点续训（恢复模型权重与 epoch 计数）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader 工作进程数（默认从 config data.num_workers 读取，缺省 0）",
    )
    return parser.parse_args()


def main():
    """主函数"""
    _ensure_project_root()
    # torch 在种子设置(下方 line ~81 torch.manual_seed)即被使用，须在此提前导入；
    # 原代码仅在 line ~166 的 DataLoader 段才局部 import torch，导致 main() 启动
    # 即 NameError: name 'torch' is not defined，--resume 等后续流程无从执行。
    import torch
    from src.utils.config import Config
    from src.utils.logging import setup_logger, get_timestamped_log_filename
    from src.data.loaders import DataLoader
    from src.data.preprocess import DataPreprocessor
    from src.data.datasets import PTMPlainDataModule
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # P1-3: 若用户使用了较重的研究/默认配置，提示首次运行可改用 smoke 配置。
    _heavy_configs = ("configs/default.yaml", "configs/research/", "configs/lightning.yaml")
    if args.config.endswith(tuple(_heavy_configs)):
        logger.info(
            "提示：当前配置 %s 偏研究级（transformer/多层/多 epoch）。"
            "首次 E2E 验证推荐 configs/smoke/cnn_cpu.yaml（小型 CNN、CPU、1 epoch）。",
            args.config,
        )

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
        # P1-3: 与 train_lightning.py 一致——把 sample/synthetic 文件名视为 demo，
        # 避免 sample_data.csv 这种合成 fixture 被当作真实数据误标记 model_kind=real。
        _data_basename = os.path.basename(args.data).lower()
        is_demo_data = "sample" in _data_basename or "synthetic" in _data_basename
        data_source = args.data
    else:
        df = loader.load_sample_data(num_samples=500)
        # P1-3: load_sample_data() generates random synthetic sequences/labels,
        # so any model trained this way is a demo/smoke model.
        is_demo_data = True
        data_source = "synthetic_random (DataLoader.load_sample_data)"

    logger.info("步骤 1.5: 数据质量预检")
    preprocessor_for_qc = DataPreprocessor(config.to_dict())
    qc_report = preprocessor_for_qc.validate_data_quality(df)
    for w in qc_report.get("warnings", []):
        logger.warning("数据质量警告: %s", w)
    if not qc_report["ok"]:
        # Hard failures make training meaningless; fail fast with actionable detail.
        raise SystemExit(
            "数据质量预检未通过（训练中止）：\n  - "
            + "\n  - ".join(qc_report["hard_failures"])
            + "\n请修正数据后重试，详见 docs/guides/data_integration.md。"
        )
    logger.info(
        "数据质量预检通过：样本数=%d，标签分布=%s，序列长度=%s",
        qc_report["row_count"],
        qc_report["label_distribution"],
        qc_report["sequence_length_stats"],
    )

    logger.info("步骤 2: 数据预处理")
    preprocessor = DataPreprocessor(config.to_dict())
    try:
        train_df, val_df, test_df = preprocessor.preprocess_pipeline(df)
    except Exception as exc:
        # P1-3: 将预处理期空数据集等错误转为明确的退出码与可操作信息
        raise SystemExit(f"数据预处理失败: {exc}") from exc

    logger.info("步骤 3: 创建数据模块")
    batch_size = config.get("training.batch_size", 32)
    num_workers = args.num_workers if args.num_workers is not None else config.get("data.num_workers", 0)
    datamodule = PTMPlainDataModule(
        train_df, val_df, test_df,
        config=config.to_dict(),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    from src.data.labels import derive_label_mapping
    cell_states, label_to_idx = derive_label_mapping(train_df)
    config.set("data.cell_states", cell_states)
    config.set("data.label_to_idx", label_to_idx)
    config.set("model.num_classes", len(cell_states))
    logger.info("持久化训练标签映射: cell_states=%s", cell_states)

    # Create WeightedRandomSampler for balanced batches
    # numpy/torch 已在模块顶部导入；函数内重复 import 会使整个函数体的 np/torch 成为
    # 局部变量，导致种子块(83-88行)在局部 import 执行前访问 np/torch 时 UnboundLocalError。
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

    # Build the train dataloader via the datamodule's dataset while injecting the
    # WeightedRandomSampler (datamodule.train_dataloader() uses shuffle=True,
    # which is incompatible with a sampler). num_workers is read from config/CLI
    # so multi-core hosts can parallelise data loading.
    from torch.utils.data import DataLoader
    pin_memory = config.get("data.pin_memory", False)
    persistent_workers = config.get("data.persistent_workers", False) and num_workers > 0
    train_loader = DataLoader(
        datamodule.train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    logger.info("步骤 4: 创建模型")
    model = PTM2CellNet.from_config(config.to_dict())
    logger.info("模型创建完成: %s", model.encoder_type)

    # 断点续训预检：仅校验 checkpoint 存在性与可访问性。
    # 实际权重加载与 num_classes 一致性校验统一由下方步骤 5 之后的
    # torch.load(args.resume) + load_state_dict(strict=False) 完成，
    # 避免在此处先 load_model 加载一次权重、随后又被权威加载覆盖的冗余调用。
    if args.resume:
        if not os.path.exists(args.resume):
            raise SystemExit(
                f"--resume 指定的 checkpoint 不存在: {args.resume}。"
                f"请确认路径正确且与当前 config 匹配。"
            )
        logger.info("检测到 --resume，将在训练器初始化后从 checkpoint 恢复: %s", args.resume)

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

    # 断点续训: 恢复模型权重与 epoch 计数，并校验 checkpoint 与当前 config 一致。
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise SystemExit(f"--resume 指定的 checkpoint 不存在: {args.resume}")
        logger.info("从 checkpoint 恢复: %s", args.resume)
        # 显式 weights_only=False：checkpoint 非纯 state_dict（含 epoch/global_step
        # 等 Python 对象），需要反序列化完整对象。仅加载受信任的自产 checkpoint。
        ckpt = torch.load(args.resume, map_location=trainer.device, weights_only=False)

        # 配置一致性校验: checkpoint 的 num_classes 必须与当前 config 一致，
        # 否则权重形状不匹配会静默失败或导致维度错误。
        ckpt_state = ckpt.get("model_state_dict", ckpt)
        # 探测分类头输出维度（兼容多种命名）
        ckpt_num_classes = None
        for key, tensor in ckpt_state.items():
            if key.endswith(("classifier.weight", "classifier.bias", "head.weight", "head.bias")):
                if tensor.dim() >= 2:
                    ckpt_num_classes = tensor.shape[0]
                else:
                    ckpt_num_classes = int(tensor.shape[0])
                break
        if ckpt_num_classes is not None and ckpt_num_classes != len(cell_states):
            raise SystemExit(
                f"checkpoint 与当前配置不一致: checkpoint num_classes={ckpt_num_classes}, "
                f"config num_classes={len(cell_states)}。请使用匹配的 checkpoint 或配置。"
            )

        missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
        if missing:
            logger.warning("恢复时缺失的键: %s", missing)
        if unexpected:
            logger.warning("恢复时多余的键: %s", unexpected)

        ckpt_epoch = ckpt.get("epoch", None)
        if ckpt_epoch is not None:
            trainer.epoch = int(ckpt_epoch)
            trainer.global_step = int(ckpt.get("global_step", 0))
            logger.info("恢复 epoch=%d, global_step=%d", trainer.epoch, trainer.global_step)
        # train.py 的 ModelCheckpoint 未持久化 optimizer/scheduler 状态，
        # 因此 optimizer/scheduler 从当前配置重新初始化（lr 重新预热）。
        logger.warning(
            "注意: optimizer/scheduler 状态未在 checkpoint 中保存，将使用当前配置重新初始化。"
        )

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

    logger.info("保存训练配置")
    trained_config = Config(config.to_dict())
    # P1-3: persist model_kind + data provenance into the config yaml so the
    # config is self-describing (the manifest is the other source of truth).
    trained_config.set("model.model_kind", "demo" if is_demo_data else "real")
    trained_config.set(
        "model_card",
        {
            "model_kind": "demo" if is_demo_data else "real",
            "training_data": "synthetic_random" if is_demo_data else data_source,
            "not_for_biological_use": is_demo_data,
        },
    )
    trained_config.save(os.path.join(args.output, "models", "best_model.config.yaml"))
    trained_config.save(os.path.join(args.output, "models", "best_model_last.config.yaml"))

    logger.info("步骤 8: 评估模型")
    evaluator = Evaluator(model, config=config.to_dict(), task_type="classification")
    test_metrics = evaluator.evaluate(datamodule.test_dataloader())

    metrics_path = os.path.join(args.output, "results", "test_metrics.json")
    # Sanitize NaN values for valid JSON output
    def _sanitize_nans(obj):
        if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize_nans(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_nans(v) for v in obj]
        return obj
    with open(metrics_path, "w") as f:
        json.dump(_sanitize_nans(test_metrics), f, indent=2)
    logger.info("测试指标已保存: %s", metrics_path)

    # P1-1 / P1-3: 导出 artifact manifest（provenance + 数据画像 + 发布门禁）。
    # 评估在 manifest 之前完成，使 deployable 由真实指标驱动而非手设。
    from src.data.data_contract import profile_dataset, validate_data_contract
    from src.training.artifacts import write_artifact_manifest

    contract_report = validate_data_contract(df)
    for w in contract_report.get("warnings", []):
        logger.warning("数据契约: %s", w)
    if not contract_report["ok"] and not is_demo_data:
        logger.warning(
            "数据契约硬失败（将标记 deployable=False）：%s",
            contract_report["hard_failures"],
        )
    dataset_profile = profile_dataset(df)

    release_thresholds = config.get("release_gate.thresholds", None)
    if is_demo_data:
        # demo 模型不参与发布门禁；metrics 仍记录用于回归。
        release_thresholds = None

    write_artifact_manifest(
        os.path.join(args.output, "models"),
        model_class=type(model).__name__,
        cell_states=cell_states,
        config=trained_config,
        training_entrypoint="scripts/train.py",
        is_demo_data=is_demo_data,
        data_source=data_source,
        train_df=df,
        split_strategy=config.get("data.split_strategy")
        or f"random({config.get('data.split.train_ratio', 0.7)}/"
        f"{config.get('data.split.val_ratio', 0.15)}/"
        f"{config.get('data.split.test_ratio', 0.15)})",
        metrics=_sanitize_nans(test_metrics) or None,
        dataset_profile=dataset_profile,
        release_thresholds=release_thresholds,
        intended_use=(
            "Engineering pipeline validation only." if is_demo_data
            else "Cell-state prediction. Validate on held-out biological data."
        ),
    )
    if is_demo_data:
        logger.warning(
            "⚠️ 该模型在合成/示例数据上训练，将标记为 model_kind=demo。"
            "产物仅用于工程链路验证，不可用于真实生物学预测。"
        )
    logger.info("artifact manifest 已导出")

    if args.pathway_analysis:
        logger.info("步骤 9: 信号通路分析")
        try:
            from src.models.signaling_network import SignalingNetworkMapper

            mapper = SignalingNetworkMapper()
            val_df = val_df.copy()
            ptm_cols = {"gene_symbol", "ptm_type", "effect", "delta_prob"}
            available = ptm_cols.intersection(val_df.columns)
            if available:
                ptm_data = val_df[list(available)].dropna(subset=["gene_symbol", "ptm_type"])
                if not ptm_data.empty:
                    if "effect" not in ptm_data.columns:
                        ptm_data["effect"] = "gain"
                    if "delta_prob" not in ptm_data.columns:
                        ptm_data["delta_prob"] = 1.0
                    ptm_data["delta_prob"] = ptm_data["delta_prob"].astype(float)
                    report = mapper.generate_network_report(ptm_data)
                    pathway_activities = report.get("pathway_activities", {})
                    logger.info("通路分析结果（%d 条PTM记录）:", len(ptm_data))
                    for pathway, activity in sorted(pathway_activities.items(), key=lambda x: abs(x[1]), reverse=True):
                        direction = "↑" if activity > 0 else "↓"
                        logger.info("  %s %s (%.3f)", direction, pathway, activity)
                    if not pathway_activities:
                        logger.info("  未检测到显著的通路活性变化")
                    pathway_report_path = os.path.join(args.output, "results", "pathway_analysis.json")
                    with open(pathway_report_path, "w") as f:
                        json.dump(report, f, indent=2, ensure_ascii=False)
                    logger.info("通路分析报告已保存: %s", pathway_report_path)
                else:
                    logger.info("验证集缺少 PTM 数据列，跳过通路分析")
            else:
                logger.info("验证集缺少 PTM 列（%s），跳过通路分析", ptm_cols - available)
        except ImportError:
            logger.warning("signaling_network模块未安装，跳过通路分析")
        except Exception as e:
            logger.warning("通路分析失败: %s", e)

    logger.info("=" * 60)
    logger.info("训练完成")
    if trainer.val_losses:
        logger.info("最佳验证损失: %.4f", min(trainer.val_losses))
    else:
        logger.info("未进行验证，无法计算最佳验证损失")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
