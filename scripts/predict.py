"""
预测脚本
功能概述: 使用训练好的模型进行预测
设计思路: 加载模型权重，处理输入数据，生成预测结果
"""

import argparse
import os
import sys


def _ensure_project_root() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PTM2CellNet预测脚本")
    parser.add_argument("--model", type=str, default="outputs/models/best_model.pt", help="模型权重路径")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--input", type=str, default=None, help="输入数据路径（CSV）")
    parser.add_argument("--sequence", type=str, default=None, help="单个蛋白质序列")
    parser.add_argument("--output", type=str, default="outputs/results/predictions.csv", help="输出路径")
    parser.add_argument("--device", type=str, default=None, help="设备（cuda/cpu）")
    return parser.parse_args()


def main():
    """主函数"""
    _ensure_project_root()
    import pandas as pd
    import torch

    from src.utils.config import Config
    from src.utils.logging import setup_logger, get_timestamped_log_filename
    from src.utils.io import load_model, save_dataframe
    from src.models.architectures import PTM2CellNet
    from src.api.schemas import PredictionRequest
    from src.api.routes import initialize_model, preprocess_request

    logger = setup_logger(__name__, get_timestamped_log_filename("predict"))
    logger.info("=" * 60)
    logger.info("PTM2CellNet 预测开始")
    logger.info("=" * 60)

    args = parse_args()
    config = Config.from_yaml(args.config)

    cell_states = config.get("data.cell_states")
    if not cell_states:
        cell_states = ["proliferation", "differentiation", "apoptosis", "quiescence"]

    logger.info("步骤 1: 加载模型")
    config.set("model.num_classes", len(cell_states))
    model = PTM2CellNet.from_config(config.to_dict())
    load_model(model, args.model)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    initialize_model(model, cell_states, device)
    logger.info("模型已加载到 %s", device)

    if args.sequence:
        logger.info("步骤 2: 单样本预测")
        request = PredictionRequest(
            sequence=args.sequence,
            ptm_sites=[],
        )

        batch = preprocess_request(request)

        with torch.no_grad():
            for key in batch:
                batch[key] = batch[key].to(device)

            outputs = model(batch)

            if isinstance(outputs, dict):
                probabilities = outputs.get("probabilities")
            else:
                logits = outputs
                probabilities = torch.softmax(logits, dim=-1)

            probs_np = probabilities[0].cpu().numpy()
            pred_idx = int(torch.argmax(probabilities[0]).item())
            pred_label = cell_states[pred_idx]
            confidence = float(probs_np[pred_idx])

            prob_dict = {
                cell_states[i]: float(probs_np[i])
                for i in range(len(probs_np))
            }

        logger.info("预测结果:")
        logger.info("  细胞状态: %s", pred_label)
        logger.info("  置信度: %.4f", confidence)
        logger.info("  概率分布: %s", prob_dict)

    elif args.input:
        logger.info("步骤 2: 批量预测")
        df = pd.read_csv(args.input)

        results = []
        for idx, row in df.iterrows():
            sequence = str(row.get("sequence", ""))
            request = PredictionRequest(
                sequence=sequence,
                ptm_sites=[],
            )

            batch = preprocess_request(request)

            with torch.no_grad():
                for key in batch:
                    batch[key] = batch[key].to(device)

                outputs = model(batch)

                if isinstance(outputs, dict):
                    probabilities = outputs.get("probabilities")
                else:
                    logits = outputs
                    probabilities = torch.softmax(logits, dim=-1)

                probs_np = probabilities[0].cpu().numpy()
                pred_idx = int(torch.argmax(probabilities[0]).item())
                pred_label = cell_states[pred_idx]
                confidence = float(probs_np[pred_idx])

            result = {
                "id": row.get("id", idx),
                "sequence": sequence,
                "predicted_cell_state": pred_label,
                "confidence": confidence,
            }
            results.append(result)

        result_df = pd.DataFrame(results)
        save_dataframe(result_df, args.output)
        logger.info("预测结果已保存: %s", args.output)
        logger.info("共处理 %d 个样本", len(results))

    else:
        logger.warning("未提供输入数据，请使用 --sequence 或 --input 参数")

    logger.info("=" * 60)
    logger.info("预测完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
