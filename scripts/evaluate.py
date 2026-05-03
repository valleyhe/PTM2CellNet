"""
评估脚本
功能概述: 评估训练好的模型性能
设计思路: 加载模型和数据，计算评估指标，生成可视化图表
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
    parser = argparse.ArgumentParser(description="PTM2CellNet评估脚本")
    parser.add_argument("--model", type=str, default="outputs/models/best_model.pt", help="模型权重路径")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--data", type=str, default=None, help="评估数据路径")
    parser.add_argument("--output", type=str, default="outputs/results", help="输出目录")
    parser.add_argument("--device", type=str, default=None, help="设备（cuda/cpu）")
    return parser.parse_args()


def main():
    """主函数"""
    _ensure_project_root()
    import torch

    from src.utils.config import Config
    from src.utils.logging import setup_logger, get_timestamped_log_filename
    from src.utils.io import load_model
    from src.data.loaders import DataLoader
    from src.data.preprocess import DataPreprocessor
    from src.data.datasets import PTMDataModule
    from src.models.architectures import PTM2CellNet
    from src.evaluation.evaluators import Evaluator
    from src.evaluation.visualization import (
        plot_roc_curve,
        plot_pr_curve,
        plot_confusion_matrix,
    )

    logger = setup_logger(__name__, get_timestamped_log_filename("evaluate"))
    logger.info("=" * 60)
    logger.info("PTM2CellNet 评估开始")
    logger.info("=" * 60)

    args = parse_args()
    config = Config.from_yaml(args.config)

    os.makedirs(args.output, exist_ok=True)

    logger.info("步骤 1: 加载数据")
    loader = DataLoader(config.to_dict())

    if args.data:
        df = loader.load_from_csv(args.data)
    else:
        df = loader.load_sample_data(num_samples=200)

    logger.info("步骤 2: 数据预处理")
    preprocessor = DataPreprocessor(config.to_dict())
    _, _, test_df = preprocessor.preprocess_pipeline(df)

    logger.info("步骤 3: 创建数据模块")
    batch_size = config.get("training.batch_size", 32)
    datamodule = PTMDataModule(
        test_df, None, test_df,
        config=config.to_dict(),
        batch_size=batch_size,
    )
    cell_states = datamodule.get_labels()

    logger.info("步骤 4: 加载模型")
    config.set("model.num_classes", len(cell_states))
    model = PTM2CellNet.from_config(config.to_dict())
    load_model(model, args.model)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("模型已加载到 %s", device)

    logger.info("步骤 5: 评估模型")
    evaluator = Evaluator(
        model,
        config=config.to_dict(),
        device=device,
        task_type="classification",
    )
    result = evaluator.evaluate(datamodule.test_dataloader(), return_predictions=True)

    metrics = result["metrics"]
    logger.info("评估指标:")
    logger.info("  Accuracy: %.4f", metrics.get("accuracy", 0))
    logger.info("  Precision (macro): %.4f", metrics.get("precision_macro", 0))
    logger.info("  Recall (macro): %.4f", metrics.get("recall_macro", 0))
    logger.info("  F1 (macro): %.4f", metrics.get("f1_macro", 0))
    if "auc_roc" in metrics:
        logger.info("  AUC-ROC: %.4f", metrics.get("auc_roc", 0))
    if "auc_pr" in metrics:
        logger.info("  AUC-PR: %.4f", metrics.get("auc_pr", 0))

    metrics_path = os.path.join(args.output, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("评估指标已保存: %s", metrics_path)

    logger.info("步骤 6: 生成可视化图表")

    if "predictions" in result and "targets" in result and "probabilities" in result:
        y_pred = result["predictions"]
        y_true = result["targets"]
        y_score = result["probabilities"]

        roc_path = os.path.join(args.output, "roc_curve.png")
        plot_roc_curve(y_true, y_score, roc_path, labels=cell_states)
        logger.info("ROC曲线已保存: %s", roc_path)

        pr_path = os.path.join(args.output, "pr_curve.png")
        plot_pr_curve(y_true, y_score, pr_path, labels=cell_states)
        logger.info("PR曲线已保存: %s", pr_path)

        cm_path = os.path.join(args.output, "confusion_matrix.png")
        plot_confusion_matrix(y_true, y_pred, cm_path, labels=cell_states)
        logger.info("混淆矩阵已保存: %s", cm_path)

    logger.info("=" * 60)
    logger.info("评估完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
