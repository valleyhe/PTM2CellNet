"""
训练脚本
功能概述: 训练PTM2CellNet模型的主脚本
设计思路: 整合数据加载、模型创建、训练流程
"""

import argparse
import json
import os
import subprocess
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
    parser.add_argument("--pathway-analysis", action="store_true", default=False, help="启用信号通路分析（基于验证集）")
    return parser.parse_args()


def main():
    """主函数"""
    _ensure_project_root()
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
        # P1-3: real data source provided → this is NOT a demo model.
        is_demo_data = False
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
    datamodule = PTMPlainDataModule(
        train_df, val_df, test_df,
        config=config.to_dict(),
        batch_size=batch_size,
    )
    from src.data.labels import derive_label_mapping
    cell_states, label_to_idx = derive_label_mapping(train_df)
    config.set("data.cell_states", cell_states)
    config.set("data.label_to_idx", label_to_idx)
    config.set("model.num_classes", len(cell_states))
    logger.info("持久化训练标签映射: cell_states=%s", cell_states)

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

    # P1-3: 导出 artifact manifest
    import subprocess
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        git_commit = "unknown"
    manifest = {
        "checkpoint_path": os.path.join(args.output, "models", "best_model.pt"),
        "config_path": os.path.join(args.output, "models", "best_model.config.yaml"),
        "training_entrypoint": "scripts/train.py",
        "model_class": type(model).__name__,
        "num_classes": len(cell_states),
        "cell_states": cell_states,
        "max_sequence_length": config.get("data.max_sequence_length", 1000),
        "ptm_types": config.get("data.ptm_types", None),
        "git_commit": git_commit,
        "created_at": __import__("datetime").datetime.now().isoformat(),
    }
    # P1-3: record data provenance + model_kind so downstream tooling (predict.py,
    # /model/info, container auto-init) can flag demo models and prevent misuse.
    model_kind = "demo" if is_demo_data else "real"
    manifest["model_kind"] = model_kind
    manifest["model_card"] = {
        "model_kind": model_kind,
        "training_data": "synthetic_random" if is_demo_data else data_source,
        "not_for_biological_use": is_demo_data,
        "description": (
            "Demo/smoke model trained on randomly generated sequences and labels. "
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
    save_json(manifest, os.path.join(args.output, "models", "artifact_manifest.json"))
    if is_demo_data:
        logger.warning(
            "⚠️ 该模型在合成/示例数据上训练，将标记为 model_kind=demo。"
            "产物仅用于工程链路验证，不可用于真实生物学预测。"
        )
    logger.info("artifact manifest 已导出")

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
