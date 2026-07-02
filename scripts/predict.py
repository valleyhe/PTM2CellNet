"""
预测脚本
功能概述: 使用训练好的模型进行预测
设计思路: 加载模型权重，处理输入数据，生成预测结果
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple


def _ensure_project_root() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def parse_args(args=None):
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PTM2CellNet预测脚本")
    parser.add_argument("--model", type=str, default="outputs/models/best_model.pt", help="模型权重路径")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--input", type=str, default=None, help="输入数据路径（CSV）")
    parser.add_argument("--sequence", type=str, default=None, help="单个蛋白质序列")
    parser.add_argument("--output", type=str, default="outputs/results/predictions.csv", help="输出路径")
    parser.add_argument("--device", type=str, default=None, help="设备（cuda/cpu）")
    parser.add_argument("--batch-size", type=int, default=32, help="批量推理的批大小")
    parser.add_argument("--pathway-analysis", action="store_true", default=False, help="启用信号通路分析")
    parser.add_argument(
        "--ptm-sites",
        type=str,
        default=None,
        help=(
            "单样本模式的 PTM 位点，JSON 字符串或 JSON 文件路径。"
            "例如 '[{\"position\":3,\"type\":\"phosphorylation\"}]'"
        ),
    )
    parser.add_argument(
        "--allow-demo-fallback",
        action="store_true",
        default=False,
        help=(
            "当配置缺少 data.cell_states 时，允许回退到内置演示标签顺序。"
            "仅用于 demo 模型，生产推理必须显式提供标签映射。"
        ),
    )
    return parser.parse_args(args)


def _run_prediction_on_batch(
    model,
    batch: Dict[str, "torch.Tensor"],
    device: str,
) -> Tuple:
    """对已经堆叠好的一个 batch 执行一次前向传播。

    参数:
        model: 已加载权重的 PTM2CellNet 模型。
        batch: 张量字典，每个张量形状为 [N, ...]。
        device: 计算设备。

    返回:
        probabilities: [N, num_classes] 的 CPU 概率张量。
    """
    import torch

    with torch.no_grad():
        for key in batch:
            batch[key] = batch[key].to(device)

        outputs = model(batch)

        if isinstance(outputs, dict):
            probabilities = outputs.get("probabilities")
        else:
            probabilities = torch.softmax(outputs, dim=-1)

    return probabilities.cpu()


def _is_demo_model(config, manifest: Optional[Dict] = None) -> bool:
    """Return True if the loaded model is a demo/smoke model (P1-3).

    A model is considered demo when any of these holds:
      - ``model.model_kind == 'demo'`` in the config
      - ``model_card.model_kind == 'demo'`` / ``data_provenance.model_kind == 'demo'``
      - sibling ``artifact_manifest.json`` declares ``model_kind: 'demo'``
      - training data is described as synthetic
    """
    def _kind_from(d):
        if not isinstance(d, dict):
            return None
        kind = d.get("model_kind")
        if isinstance(kind, str) and kind:
            return kind.lower()
        td = d.get("training_data")
        if isinstance(td, str) and "synthetic" in td.lower():
            return "demo"
        return None

    if manifest and isinstance(manifest, dict):
        for key in ("model_card", "data_provenance"):
            kind = _kind_from(manifest.get(key))
            if kind == "demo":
                return True
        if _kind_from(manifest) == "demo":
            return True

    if config is not None:
        cfg_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config or {})
        model_cfg = cfg_dict.get("model", {}) if isinstance(cfg_dict, dict) else {}
        if _kind_from(model_cfg) == "demo":
            return True
        for key in ("model_card", "data_provenance"):
            if _kind_from(cfg_dict.get(key) if isinstance(cfg_dict, dict) else None) == "demo":
                return True
    return False


def _warn_if_demo_model(checkpoint_path: str, config) -> None:
    """Emit a prominent warning when the loaded checkpoint is a demo model (P1-3).

    Looks at the config and a sibling ``artifact_manifest.json`` so demo models
    shipped with the repo (or trained on synthetic fixtures) are flagged without
    breaking real-model inference. Does not raise — smoke tests must still pass.
    """
    import json
    import os
    from src.utils.logging import setup_logger

    logger = setup_logger("predict")

    manifest = None
    ckpt_dir = os.path.dirname(checkpoint_path)
    manifest_path = os.path.join(ckpt_dir, "artifact_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError):
            manifest = None

    if _is_demo_model(config, manifest):
        logger.warning(
            "⚠️  DEMO 模型检测：%s 是合成数据训练的 demo/smoke 模型。\n"
            "    该模型仅用于验证工程链路（训练/推理/评估/API），\n"
            "    预测结果不具备真实生物学意义，请勿用于科研或临床解读。",
            checkpoint_path,
        )


def _predict_in_batches(
    model,
    preprocessed_rows: List[Dict],
    device: str,
    batch_size: int,
    cell_states: Optional[List[str]] = None,
) -> Tuple[List[int], List[float], List[Dict[str, float]]]:
    """将预处理后的样本按 batch_size 堆叠，逐批推理并返回预测结果。

    ``cell_states`` 控制逐类概率列 ``prob_<state>`` 的命名。历史调用方
    （和部分单元测试）只关心 ``(predictions, confidences)`` 二元组，因此
    此参数是可选的：缺省时按 ``class_<i>`` 生成占位列名，返回值仍为三元组
    以保持单一合约（P0-1）。需要消费概率列的调用方应显式传入真实标签。
    """
    import torch

    if cell_states is None:
        # Infer num_classes lazily from the first forward, then back-fill names.
        cell_states = []

    predictions: List[int] = []
    confidences: List[float] = []
    probabilities_per_row: List[Dict[str, float]] = []

    for start in range(0, len(preprocessed_rows), batch_size):
        chunk = preprocessed_rows[start : start + batch_size]
        batch = {
            key: torch.stack([row[key] for row in chunk], dim=0)
            for key in chunk[0].keys()
        }

        probabilities = _run_prediction_on_batch(model, batch, device)
        pred_indices = torch.argmax(probabilities, dim=-1)
        max_probs = probabilities.gather(
            dim=-1, index=pred_indices.unsqueeze(-1)
        ).squeeze(-1)

        # Back-fill default cell_states once we know num_classes.
        if not cell_states:
            num_classes = probabilities.shape[-1]
            cell_states = [f"class_{i}" for i in range(num_classes)]

        predictions.extend(pred_indices.tolist())
        confidences.extend(max_probs.tolist())
        for i in range(len(chunk)):
            probs_i = probabilities[i].tolist()
            probabilities_per_row.append({
                f"prob_{state}": probs_i[j]
                for j, state in enumerate(cell_states)
            })

    return predictions, confidences, probabilities_per_row


def _parse_ptm_sites_arg(ptm_sites_arg):
    """解析 ``--ptm-sites`` 参数（JSON 字符串或 JSON 文件路径）。

    返回 ``(ptm_sites_list, parse_errors)``：``ptm_sites_list`` 是可供
    ``PredictionRequest`` 使用的列表；``parse_errors`` 收集非法位点的行级错误信息
    （用于输出报告）。
    """
    import json
    import os

    parse_errors = []
    if ptm_sites_arg is None:
        return [], []

    # 若是已存在的文件路径，则读取文件内容作为 JSON
    raw = ptm_sites_arg
    if os.path.exists(ptm_sites_arg):
        with open(ptm_sites_arg, "r", encoding="utf-8") as f:
            raw = f.read()

    try:
        sites = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(
            f"--ptm-sites 无法解析为 JSON: {exc}。"
            "示例: '[{\"position\":3,\"type\":\"phosphorylation\"}]'"
        ) from exc

    if not isinstance(sites, list):
        raise ValueError("--ptm-sites JSON 顶层必须是列表。")

    cleaned = []
    for i, site in enumerate(sites):
        if not isinstance(site, dict) or "position" not in site:
            parse_errors.append(f"位点#{i}: 缺少 position 字段或非对象")
            continue
        pos = site.get("position")
        if not isinstance(pos, int) or pos < 1:
            parse_errors.append(f"位点#{i}: position={pos!r} 无效（需为 >=1 的整数）")
            continue
        cleaned.append({
            "position": pos,
            "type": str(site.get("type", "")),
            **({"amino_acid": site["amino_acid"]} if site.get("amino_acid") else {}),
        })

    return cleaned, parse_errors


def _parse_batch_ptm_sites(ptm_sites_value, sequence: str, row_index, parse_errors):
    """解析批量 CSV 中单个 ``ptm_sites`` 单元格。

    返回可供 ``PredictionRequest`` 使用的位点列表；非法位点记录到 ``parse_errors``。
    """
    import json

    if ptm_sites_value is None:
        return []
    if isinstance(ptm_sites_value, list):
        raw_sites = ptm_sites_value
    else:
        text = str(ptm_sites_value).strip()
        if not text or text.lower() in ("nan", "none", "null", "[]"):
            return []
        try:
            raw_sites = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            parse_errors.append(
                f"行{row_index}: ptm_sites JSON 解析失败: {exc}"
            )
            return []

    if not isinstance(raw_sites, list):
        parse_errors.append(f"行{row_index}: ptm_sites 顶层必须是列表")
        return []

    seq_len = len(sequence)
    cleaned = []
    for j, site in enumerate(raw_sites):
        if not isinstance(site, dict) or "position" not in site:
            parse_errors.append(f"行{row_index} 位点#{j}: 缺少 position 字段")
            continue
        pos = site.get("position")
        if not isinstance(pos, int) or pos < 1:
            parse_errors.append(
                f"行{row_index} 位点#{j}: position={pos!r} 无效"
            )
            continue
        if pos > seq_len:
            parse_errors.append(
                f"行{row_index} 位点#{j}: position {pos} 超出序列长度 {seq_len}"
            )
            continue
        cleaned.append({
            "position": pos,
            "type": str(site.get("type", "")),
            **({"amino_acid": site["amino_acid"]} if site.get("amino_acid") else {}),
        })
    return cleaned


def main():
    """主函数"""
    _ensure_project_root()
    import pandas as pd
    import torch

    from src.utils.config import Config
    from src.utils.logging import setup_logger, get_timestamped_log_filename
    from src.utils.io import load_model, save_dataframe
    from src.utils.checkpoint_utils import resolve_inference_config
    from src.models.architectures import PTM2CellNet
    from src.api.schemas import PredictionRequest
    from src.api.routes import initialize_model, preprocess_request

    logger = setup_logger(__name__, get_timestamped_log_filename("predict"))
    logger.info("=" * 60)
    logger.info("PTM2CellNet 预测开始")
    logger.info("=" * 60)

    args = parse_args()
    # 优先使用与 checkpoint 同目录的 .config.yaml，避免与默认 config 不匹配
    config, config_source = resolve_inference_config(args.model, args.config)
    logger.info("使用配置文件: %s", config_source)

    # P1-3: detect demo/smoke models and warn loudly so they are not mistaken
    # for biologically valid predictions. Resolution order: explicit config flag
    # -> sibling artifact_manifest.json -> data_provenance in config.
    _warn_if_demo_model(args.model, config)

    cell_states = config.get("data.cell_states")
    if not cell_states:
        # P0-1: 禁止静默 fallback 到硬编码标签顺序——这会把类别索引解码成
        # 错误的细胞状态。仅当显式 --allow-demo-fallback 时允许使用演示顺序。
        if not args.allow_demo_fallback:
            raise SystemExit(
                "配置缺少 data.cell_states，无法可靠解码预测类别索引。\n"
                "训练产物应通过 train.py 持久化 cell_states 到 best_model.config.yaml；\n"
                "若确为 demo 模型且接受硬编码标签顺序，可显式传入 --allow-demo-fallback。"
            )
        cell_states = ["proliferation", "differentiation", "apoptosis", "quiescence"]
        logger.warning(
            "使用 --allow-demo-fallback 回退到演示标签顺序 %s，"
            "生产推理请勿使用。", cell_states,
        )

    logger.info("步骤 1: 加载模型")
    config.set("model.num_classes", len(cell_states))
    model = PTM2CellNet.from_config(config.to_dict())
    try:
        load_model(model, args.model)
    except RuntimeError as exc:
        raise RuntimeError(
            f"加载 checkpoint 失败：checkpoint 与 config 不匹配。"
            f"请使用训练该 checkpoint 时的 config 文件，路径通常为 "
            f"outputs/models/<model>.config.yaml。原始错误: {exc}"
        ) from exc

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    initialize_model(model, cell_states, device, config=config.to_dict())
    logger.info("模型已加载到 %s", device)

    if args.sequence:
        logger.info("步骤 2: 单样本预测")
        # P1-2: 解析 --ptm-sites，使单样本推理真实消费 PTM 位点
        ptm_sites, ptm_parse_errors = _parse_ptm_sites_arg(args.ptm_sites)
        if ptm_parse_errors:
            for err in ptm_parse_errors:
                logger.warning("PTM 解析: %s", err)

        request = PredictionRequest(
            sequence=args.sequence,
            ptm_sites=ptm_sites,
        )

        batch = preprocess_request(request)
        batch = {key: val.unsqueeze(0) for key, val in batch.items()}

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
        logger.info("  PTM 位点: 已解析 %d 个", len(ptm_sites))

        # P1-2: 单样本模式也写出 --output，格式与批量结果兼容
        single_result = pd.DataFrame([{
            "id": 0,
            "sequence": args.sequence,
            "ptm_sites": json.dumps(ptm_sites) if ptm_sites else "",
            "ptm_count": len(ptm_sites),
            "predicted_cell_state": pred_label,
            "confidence": confidence,
            **{f"prob_{state}": prob_dict[state] for state in cell_states},
        }])
        save_dataframe(single_result, args.output)
        logger.info("单样本预测结果已保存: %s", args.output)

        if args.pathway_analysis:
            logger.info("步骤 3: 信号通路分析（单样本模式需要PTM位点数据）")
            logger.info("  --pathway-analysis 在单样本模式中需要额外的PTM位点信息，跳过")

    elif args.input:
        logger.info("步骤 2: 批量预测")
        df = pd.read_csv(args.input)

        preprocessed_rows = []
        sequences = []
        ids = []
        ptm_counts = []
        batch_parse_errors = []
        for idx, row in df.iterrows():
            sequence = str(row.get("sequence", ""))
            # P1-1: 解析 CSV 中的 ptm_sites 列，使批量推理真实消费 PTM 位点
            ptm_sites = _parse_batch_ptm_sites(
                row.get("ptm_sites"), sequence, idx, batch_parse_errors
            )
            request = PredictionRequest(
                sequence=sequence,
                ptm_sites=ptm_sites,
            )
            preprocessed_rows.append(preprocess_request(request))
            sequences.append(sequence)
            ids.append(row.get("id", idx))
            ptm_counts.append(len(ptm_sites))

        if batch_parse_errors:
            for err in batch_parse_errors:
                logger.warning("PTM 解析: %s", err)
            logger.warning(
                "批量 PTM 解析共出现 %d 处错误（非法位点已跳过）",
                len(batch_parse_errors),
            )

        predictions, confidences, prob_rows = _predict_in_batches(
            model, preprocessed_rows, device, args.batch_size, cell_states
        )

        results = []
        for sample_id, sequence, pred_idx, confidence, n_ptm, probs_row in zip(
            ids, sequences, predictions, confidences, ptm_counts, prob_rows
        ):
            results.append({
                "id": sample_id,
                "sequence": sequence,
                "ptm_count": n_ptm,
                "predicted_cell_state": cell_states[pred_idx],
                "confidence": confidence,
                **probs_row,
            })

        result_df = pd.DataFrame(results)
        save_dataframe(result_df, args.output)
        logger.info("预测结果已保存: %s", args.output)
        logger.info("共处理 %d 个样本（含 PTM 位点 %d 个）", len(results), sum(ptm_counts))

        if args.pathway_analysis:
            logger.info("步骤 3: 信号通路分析")
            try:
                from src.models.signaling_network import SignalingNetworkMapper

                mapper = SignalingNetworkMapper()
                ptm_columns = {"gene_symbol", "ptm_type", "effect", "delta_prob"}
                if ptm_columns.issubset(df.columns):
                    ptm_df = df[list(ptm_columns)].copy()
                    ptm_df["effect"] = ptm_df["effect"].fillna("gain")
                    ptm_df["delta_prob"] = ptm_df["delta_prob"].fillna(1.0).astype(float)
                    report = mapper.generate_network_report(ptm_df)
                    pathway_output = os.path.join(
                        os.path.dirname(args.output), "pathway_analysis.csv"
                    )
                    pathway_rows = []
                    for name, activity in report.get("pathway_activities", {}).items():
                        pathway_rows.append({
                            "pathway": name,
                            "activity": activity,
                            "key_genes": ",".join(mapper.pathways.get(name, {}).get("output_genes", [])),
                        })
                    if pathway_rows:
                        pd.DataFrame(pathway_rows).to_csv(pathway_output, index=False)
                        logger.info("通路分析已保存: %s", pathway_output)
                    else:
                        logger.info("未检测到显著的通路活性变化")
                else:
                    logger.info("输入数据缺少PTM列（%s），跳过通路分析", ptm_columns - set(df.columns))
            except ImportError:
                logger.warning("signaling_network模块未安装，跳过通路分析")
            except Exception as e:
                logger.warning("通路分析失败: %s", e)

    else:
        logger.warning("未提供输入数据，请使用 --sequence 或 --input 参数")

    logger.info("=" * 60)
    logger.info("预测完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
