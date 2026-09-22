"""
训练脚本
功能概述: 训练PTM2CellNet模型的主脚本
设计思路: 整合数据加载、模型创建、训练流程
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.data.datasets import PTMPlainDataModule
    from src.data.loaders import DataLoader
    from src.models.architectures import PTM2CellNet
    from src.training.trainers import Trainer
    from src.utils.config import Config


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
        help="完整训练状态 checkpoint 路径（checkpoint_best.pt / checkpoint_last.pt）"
        "用于精确断点续训（恢复模型权重、epoch、optimizer/scheduler/scaler/RNG）。"
        "也兼容旧版 best_model.pt（仅恢复权重与 epoch）。",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader 工作进程数（默认从 config data.num_workers 读取，缺省 0）",
    )
    return parser.parse_args()


def _set_global_seeds(seed: int) -> None:
    """Seed python/numpy/torch and force deterministic cuDNN behaviour."""

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_training_dataframe(loader: DataLoader, args: argparse.Namespace) -> tuple[Any, bool, str]:
    """Load the training table; sample/synthetic inputs are flagged as demo."""

    if args.data:
        df = loader.load_from_csv(args.data)
        # P1-3: 与 train_lightning.py 一致——把 sample/synthetic 文件名视为 demo，
        # 避免 sample_data.csv 这种合成 fixture 被当作真实数据误标记 model_kind=real。
        data_basename = os.path.basename(args.data).lower()
        is_demo_data = "sample" in data_basename or "synthetic" in data_basename
        return df, is_demo_data, args.data
    df = loader.load_sample_data(num_samples=500)
    # P1-3: load_sample_data() generates random synthetic sequences/labels,
    # so any model trained this way is a demo/smoke model.
    return df, True, "synthetic_random (DataLoader.load_sample_data)"


def _run_data_quality_gate(config: Config, df: Any, logger: logging.Logger) -> None:
    """Hard-fail on unusable data before any training work happens."""

    from src.data.preprocess import DataPreprocessor

    preprocessor = DataPreprocessor(config.to_dict())
    qc_report = preprocessor.validate_data_quality(df)
    for warning in qc_report.get("warnings", []):
        logger.warning("数据质量警告: %s", warning)
    if not qc_report["ok"]:
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


def _build_train_loader(
    datamodule: PTMPlainDataModule,
    train_df: Any,
    config: Config,
    cell_states: list[str],
    batch_size: int,
    num_workers: int,
    logger: logging.Logger,
):
    """Build a class-balanced train dataloader over the datamodule's dataset."""

    # numpy/torch 已在模块顶部导入；函数内重复 import 会使整个函数体的 np/torch 成为
    # 局部变量，导致种子块在局部 import 执行前访问 np/torch 时 UnboundLocalError。
    import torch
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.data import WeightedRandomSampler

    label_col = config.get("data.label_column", "cell_state")
    train_labels = train_df[label_col].map({cls: idx for idx, cls in enumerate(cell_states)}).values
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32), num_samples=len(train_labels), replacement=True
    )
    logger.info("使用WeightedRandomSampler平衡批次: 类别分布 %s", class_counts.tolist())

    # Build the train dataloader via the datamodule's dataset while injecting the
    # WeightedRandomSampler (datamodule.train_dataloader() uses shuffle=True,
    # which is incompatible with a sampler). num_workers is read from config/CLI
    # so multi-core hosts can parallelise data loading.
    pin_memory = config.get("data.pin_memory", False)
    persistent_workers = config.get("data.persistent_workers", False) and num_workers > 0
    return TorchDataLoader(
        datamodule.train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def _detect_checkpoint_num_classes(ckpt_state: dict) -> int | None:
    """Probe the classifier head width of a checkpoint state dict."""

    for key, tensor in ckpt_state.items():
        if key.endswith(("classifier.weight", "classifier.bias", "head.weight", "head.bias")):
            if tensor.dim() >= 2:
                return int(tensor.shape[0])
            return int(tensor.shape[0])
    return None


def _safe_restore_state(
    state: Any,
    target: Any,
    label: str,
    success_message: str,
    logger: logging.Logger,
) -> bool:
    """Load one optional state dict; incompatible states warn and re-init."""

    if state is None or target is None:
        return False
    try:
        target.load_state_dict(state)
        logger.info("%s", success_message)
        return True
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("%s 状态恢复失败（结构不兼容，将重新初始化）: %s", label, exc)
        return False


def _restore_callback_states(trainer: Trainer, callback_states: dict, logger: logging.Logger) -> bool:
    """Restore stateful callback (checkpoint/early-stopping) bookkeeping."""

    from src.training.callbacks import _callback_state_keys

    restored_callbacks = 0
    for index, key in _callback_state_keys(trainer.callbacks).items():
        load_fn = getattr(trainer.callbacks[index], "load_state_dict", None)
        state = callback_states.get(key)
        if not callable(load_fn) or state is None:
            continue
        try:
            load_fn(state)
            restored_callbacks += 1
            logger.info("已恢复 callback 状态: %s", key)
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("callback %s 状态恢复失败（将重新初始化）: %s", key, exc)
    return restored_callbacks > 0


def _log_state_dict_diff(missing: list, unexpected: list, logger: logging.Logger) -> None:
    """Warn about non-strictly matched keys after a resume load."""

    if missing:
        logger.warning("恢复时缺失的键: %s", missing)
    if unexpected:
        logger.warning("恢复时多余的键: %s", unexpected)


def _restore_from_checkpoint(
    args: argparse.Namespace,
    trainer: Trainer,
    model: PTM2CellNet,
    optimizer: Any,
    scheduler: Any,
    num_classes: int,
    logger: logging.Logger,
) -> None:
    """Exact resume: weights, epoch counters, optimizer/scheduler/scaler/RNG and callbacks."""

    if not os.path.exists(args.resume):
        raise SystemExit(f"--resume 指定的 checkpoint 不存在: {args.resume}")
    logger.info("从 checkpoint 恢复: %s", args.resume)
    # 显式 weights_only=False：checkpoint 非纯 state_dict（含 epoch/global_step
    # 等 Python 对象），需要反序列化完整对象。仅加载受信任的自产 checkpoint。
    # 通过 safe_torch_load 走，以便：(1) 复用统一的审计日志；(2) 让
    # PTM2CELLNET_SAFE_LOAD_ONLY=1 仍能全局禁止不安全加载；(3) 收敛 v15 安全
    # 复核 §6 中 "scripts/train.py 直接 torch.load" 的剩余风险点。
    from src.utils.io import safe_torch_load

    ckpt = safe_torch_load(
        args.resume,
        map_location=trainer.device,
        weights_only=False,
        enforce_safe_only=False,
    )

    # 配置一致性校验: checkpoint 的 num_classes 必须与当前 config 一致，
    # 否则权重形状不匹配会静默失败或导致维度错误。
    ckpt_state = ckpt.get("model_state_dict", ckpt)
    ckpt_num_classes = _detect_checkpoint_num_classes(ckpt_state)
    if ckpt_num_classes is not None and ckpt_num_classes != num_classes:
        raise SystemExit(
            f"checkpoint 与当前配置不一致: checkpoint num_classes={ckpt_num_classes}, "
            f"config num_classes={num_classes}。请使用匹配的 checkpoint 或配置。"
        )

    missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
    _log_state_dict_diff(missing, unexpected, logger)

    ckpt_epoch = ckpt.get("epoch", None)
    if ckpt_epoch is not None:
        # ModelCheckpoint records the epoch that has just completed.  The
        # Trainer loop treats ``trainer.epoch`` as the next epoch to run,
        # matching CrossScaleTrainer.load_checkpoint(), so resume must
        # advance by one instead of replaying the saved epoch.
        trainer.epoch = int(ckpt_epoch) + 1
        trainer.global_step = int(ckpt.get("global_step", 0))
        logger.info(
            "恢复完成 epoch=%d，下一轮起始 epoch=%d，global_step=%d",
            int(ckpt_epoch),
            trainer.epoch,
            trainer.global_step,
        )

    # P1-01: 精确续训——恢复 optimizer/scheduler/scaler/RNG 状态。
    # 旧 checkpoint（仅模型权重）仍受支持：缺少的状态字段保持重新初始化，
    # 并显式警告该续训不等价于连续训练。
    restored_any = False
    restored_any |= _safe_restore_state(
        ckpt.get("optimizer_state_dict"),
        optimizer,
        "optimizer",
        "已恢复 optimizer 状态（momentum/Adam 步数等）",
        logger,
    )
    restored_any |= _safe_restore_state(
        ckpt.get("scheduler_state_dict"), scheduler, "scheduler", "已恢复 scheduler 状态（当前 LR/步数）", logger
    )
    if trainer.scaler is not None:
        restored_any |= _safe_restore_state(
            ckpt.get("scaler_state_dict"), trainer.scaler, "AMP scaler", "已恢复 AMP scaler 状态", logger
        )
    rng_state = ckpt.get("rng_state")
    if isinstance(rng_state, dict):
        from src.training.callbacks import _restore_rng_state

        try:
            _restore_rng_state(rng_state)
            restored_any = True
            logger.info("已恢复 python/numpy/torch RNG 状态")
        except (TypeError, ValueError) as exc:
            logger.warning("RNG 状态恢复失败（将使用当前种子继续）: %s", exc)
    # TD-M01: 恢复 stateful callback（ModelCheckpoint/EarlyStopping）的
    # best/wait/top-k 状态，使 resume 后 early stopping 与 best 语义连续。
    # 旧 checkpoint 无 callback_states 字段时跳过，保持缺省状态。
    callback_states = ckpt.get("callback_states")
    if isinstance(callback_states, dict) and callback_states:
        restored_any |= _restore_callback_states(trainer, callback_states, logger)
    if not restored_any:
        logger.warning(
            "注意: checkpoint 未包含 optimizer/scheduler/scaler/RNG 状态"
            "（旧版 checkpoint 或纯 state_dict），将使用当前配置重新初始化——"
            "该续训不等价于连续训练，仅恢复模型权重与 epoch 计数。"
        )


def _export_inference_artifact(
    args: argparse.Namespace,
    trainer: Trainer,
    model: PTM2CellNet,
    logger: logging.Logger,
) -> None:
    """Export the bare state_dict best_model.pt for weights_only inference."""

    # P1-01: best_model.pt 可被 predict.py / API 以 weights_only 安全加载；
    # 完整训练状态保留在 checkpoint_best.pt。训练循环结束时 model 可能停留在
    # 最后一个 epoch，必须先恢复 checkpoint_best.pt，避免名为 best_model 的
    # 发布 artifact 实际包含最后一轮权重。
    from src.utils.io import safe_torch_load, save_model

    best_checkpoint_path = os.path.join(args.output, "models", "checkpoint_best.pt")
    if not os.path.isfile(best_checkpoint_path):
        # TD-M01（2026-08-09）: --resume 后 val_loss 未再改善时，
        # checkpoint_best.pt 保持 resume 前的旧状态或不存在于新输出目录。
        # 此时回退到本次训练最后保存的 checkpoint_best_last.pt，保证推理
        # artifact 仍可导出，同时保留告警以便审计。
        last_checkpoint_path = os.path.join(args.output, "models", "checkpoint_best_last.pt")
        if os.path.isfile(last_checkpoint_path):
            logger.warning(
                "checkpoint_best.pt 不存在（resume 后指标未改善），改用 checkpoint_best_last.pt 导出推理 artifact: %s",
                last_checkpoint_path,
            )
            best_checkpoint_path = last_checkpoint_path
        else:
            raise SystemExit(f"训练未生成最佳 checkpoint: {best_checkpoint_path}。无法安全导出 best_model.pt。")
    best_checkpoint = safe_torch_load(
        best_checkpoint_path,
        map_location=trainer.device,
        weights_only=False,
        enforce_safe_only=False,
    )
    best_state_dict = best_checkpoint.get("model_state_dict")
    if not isinstance(best_state_dict, dict):
        raise SystemExit(f"最佳 checkpoint 缺少有效 model_state_dict: {best_checkpoint_path}")
    model.load_state_dict(best_state_dict, strict=True)
    logger.info("已恢复最佳 checkpoint 权重用于推理 artifact: %s", best_checkpoint_path)

    save_model(model, os.path.join(args.output, "models", "best_model.pt"))
    logger.info("已导出推理 artifact: %s", os.path.join(args.output, "models", "best_model.pt"))


def _sanitize_nans(obj: Any) -> Any:
    """Replace NaN/inf floats with None for valid JSON output."""

    if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
        return None
    if isinstance(obj, dict):
        return {key: _sanitize_nans(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nans(value) for value in obj]
    return obj


def _run_pathway_analysis(args: argparse.Namespace, val_df: Any, logger: logging.Logger) -> None:
    """Optional step 9: pathway activity report from validation PTM records."""

    logger.info("步骤 9: 信号通路分析")
    try:
        from src.models.signaling_network import SignalingNetworkMapper

        mapper = SignalingNetworkMapper()
        val_df = val_df.copy()
        ptm_cols = {"gene_symbol", "ptm_type", "effect", "delta_prob"}
        available = ptm_cols.intersection(val_df.columns)
        if not available:
            logger.info("验证集缺少 PTM 列（%s），跳过通路分析", ptm_cols - available)
            return
        ptm_data = val_df[list(available)].dropna(subset=["gene_symbol", "ptm_type"])
        if ptm_data.empty:
            logger.info("验证集缺少 PTM 数据列，跳过通路分析")
            return
        if "effect" not in ptm_data.columns:
            ptm_data["effect"] = "gain"
        if "delta_prob" not in ptm_data.columns:
            ptm_data["delta_prob"] = 1.0
        ptm_data["delta_prob"] = ptm_data["delta_prob"].astype(float)
        report = mapper.generate_network_report(ptm_data)
        pathway_activities = report.get("pathway_activities", {})
        logger.info("通路分析结果（%d 条PTM记录）:", len(ptm_data))
        for pathway, activity in sorted(pathway_activities.items(), key=lambda item: abs(item[1]), reverse=True):
            direction = "↑" if activity > 0 else "↓"
            logger.info("  %s %s (%.3f)", direction, pathway, activity)
        if not pathway_activities:
            logger.info("  未检测到显著的通路活性变化")
        pathway_report_path = os.path.join(args.output, "results", "pathway_analysis.json")
        with open(pathway_report_path, "w") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        logger.info("通路分析报告已保存: %s", pathway_report_path)
    except ImportError:
        logger.warning("signaling_network模块未安装，跳过通路分析")
    except Exception as e:  # noqa: BLE001 -- 通路分析是可选增强，失败不阻断训练
        logger.warning("通路分析失败: %s", e)


def _write_artifact_manifest_gate(
    args: argparse.Namespace,
    config: Config,
    trained_config: Config,
    df: Any,
    cell_states: list[str],
    test_metrics: dict,
    is_demo_data: bool,
    data_source: str,
    model_class_name: str,
    logger: logging.Logger,
) -> bool:
    """Export the artifact manifest; returns whether the data contract was violated."""

    from src.data.data_contract import profile_dataset, validate_data_contract
    from src.training.artifacts import write_artifact_manifest

    # P2-1: 真实数据（非 demo）强制要求 provenance 列——没有它们无法做
    # dataset_hash/leakage 审计与发布门禁；demo/synthetic 数据保持 warning。
    # 契约硬失败不中止训练（工程上仍可迭代），但产物 manifest 标记不可发布。
    contract_report = validate_data_contract(df, require_recommended=not is_demo_data)
    for warning in contract_report.get("warnings", []):
        logger.warning("数据契约: %s", warning)
    contract_violated = not contract_report["ok"] and not is_demo_data
    if contract_violated:
        logger.warning(
            "数据契约硬失败（产物将标记 deployable=False）：%s",
            contract_report["hard_failures"],
        )
    dataset_profile = profile_dataset(df)

    release_thresholds = config.get("release_gate.thresholds", None)
    if is_demo_data:
        # demo 模型不参与发布门禁；metrics 仍记录用于回归。
        release_thresholds = None

    write_artifact_manifest(
        os.path.join(args.output, "models"),
        model_class=model_class_name,
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
        # P2-1: 契约硬失败（缺 provenance 列等）的真实数据模型不可发布，
        # 显式覆盖 release gate 判定。
        deployable=False if contract_violated else None,
        intended_use=(
            "Engineering pipeline validation only."
            if is_demo_data
            else "Cell-state prediction. Validate on held-out biological data."
        ),
    )
    if is_demo_data:
        logger.warning(
            "⚠️ 该模型在合成/示例数据上训练，将标记为 model_kind=demo。产物仅用于工程链路验证，不可用于真实生物学预测。"
        )
    return contract_violated


def _apply_cli_overrides(args: argparse.Namespace, config: Config) -> None:
    """Apply explicit CLI training overrides onto the loaded config."""

    if args.epochs is not None:
        config.set("training.max_epochs", args.epochs)
    if args.batch_size is not None:
        config.set("training.batch_size", args.batch_size)
    if args.lr is not None:
        config.set("training.learning_rate", args.lr)


def main():
    """主函数"""
    _ensure_project_root()
    # torch 在种子设置(下方 line ~81 torch.manual_seed)即被使用，须在此提前导入；
    # 原代码仅在 line ~166 的 DataLoader 段才局部 import torch，导致 main() 启动
    # 即 NameError: name 'torch' is not defined，--resume 等后续流程无从执行。
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

    _set_global_seeds(args.seed)

    # P1-3: 若用户使用了较重的研究/默认配置，提示首次运行可改用 smoke 配置。
    _heavy_configs = ("configs/default.yaml", "configs/research/", "configs/lightning.yaml")
    if args.config.endswith(tuple(_heavy_configs)):
        logger.info(
            "提示：当前配置 %s 偏研究级（transformer/多层/多 epoch）。"
            "首次 E2E 验证推荐 configs/smoke/cnn_cpu.yaml（小型 CNN、CPU、1 epoch）。",
            args.config,
        )

    config = Config.from_yaml(args.config)

    _apply_cli_overrides(args, config)

    logger.info("步骤 1: 加载数据")
    loader = DataLoader(config.to_dict())
    df, is_demo_data, data_source = _load_training_dataframe(loader, args)

    logger.info("步骤 1.5: 数据质量预检")
    _run_data_quality_gate(config, df, logger)

    logger.info("步骤 2: 数据预处理")
    preprocessor = DataPreprocessor(config.to_dict())
    try:
        train_df, val_df, test_df = preprocessor.preprocess_pipeline(df)
    except Exception as exc:
        # P1-3: 将预处理期空数据集等错误转为明确的退出码与可操作信息
        raise SystemExit(f"数据预处理失败: {exc}") from exc

    # P2-1: 划分后写回 split_group 推荐列，使数据契约的泄漏审计可用
    # （train/val/test 显式归属，与 homology 划分语义一致）。
    for _split_name, _split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        _split_df["split_group"] = _split_name

    logger.info("步骤 3: 创建数据模块")
    batch_size = config.get("training.batch_size", 32)
    num_workers = args.num_workers if args.num_workers is not None else config.get("data.num_workers", 0)
    datamodule = PTMPlainDataModule(
        train_df,
        val_df,
        test_df,
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

    # Create WeightedRandomSampler for balanced batches and build the train
    # loader over the datamodule's dataset (sampler is incompatible with
    # shuffle=True, hence the custom loader).
    train_loader = _build_train_loader(datamodule, train_df, config, cell_states, batch_size, num_workers, logger)

    logger.info("步骤 4: 创建模型")
    model = PTM2CellNet.from_config(config.to_dict())
    logger.info("模型创建完成: %s", model.encoder_type)

    # 断点续训预检：仅校验 checkpoint 存在性与可访问性。
    # 实际权重加载与 num_classes 一致性校验统一由下方步骤 5 之后的
    # torch.load(args.resume) + load_state_dict(strict=False) 完成，
    # 避免在此处先 load_model 加载一次权重、随后又被权威加载覆盖的冗余调用。
    if args.resume:
        if not os.path.exists(args.resume):
            raise SystemExit(f"--resume 指定的 checkpoint 不存在: {args.resume}。请确认路径正确且与当前 config 匹配。")
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
        filepath=os.path.join(args.output, "models", "checkpoint_best.pt"),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_last=True,
    )
    # P1-01: 挂载 optimizer/scheduler 到 checkpoint 回调，使保存的 checkpoint
    # 携带精确续训所需的优化器状态（momentum/Adam 步数）与调度器状态（当前 LR）。
    # checkpoint_best.pt / checkpoint_last.pt 为完整训练状态（供 --resume）；
    # best_model.pt 由训练结束后导出的裸 state_dict（供推理）——两套产物分离，
    # 避免完整状态中的 numpy RNG 等对象破坏 weights_only 推理加载契约。
    checkpoint_callback.optimizer = optimizer
    checkpoint_callback.scheduler = scheduler
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=config.get("training.early_stopping_patience", 10),
    )

    trainer.add_callback(checkpoint_callback)
    trainer.add_callback(early_stop_callback)

    # 断点续训: 精确恢复权重/epoch/优化器/调度器/RNG/callback 状态。
    if args.resume is not None:
        _restore_from_checkpoint(args, trainer, model, optimizer, scheduler, len(cell_states), logger)

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

    _export_inference_artifact(args, trainer, model, logger)

    logger.info("步骤 8: 评估模型")
    evaluator = Evaluator(model, config=config.to_dict(), task_type="classification")
    test_metrics = evaluator.evaluate(datamodule.test_dataloader())

    metrics_path = os.path.join(args.output, "results", "test_metrics.json")

    with open(metrics_path, "w") as f:
        json.dump(_sanitize_nans(test_metrics), f, indent=2)
    logger.info("测试指标已保存: %s", metrics_path)

    # P1-1 / P1-3: 导出 artifact manifest（provenance + 数据画像 + 发布门禁）。
    _write_artifact_manifest_gate(
        args,
        config,
        trained_config,
        df,
        cell_states,
        test_metrics,
        is_demo_data,
        data_source,
        type(model).__name__,
        logger,
    )
    logger.info("artifact manifest 已导出")

    if args.pathway_analysis:
        _run_pathway_analysis(args, val_df, logger)

    logger.info("=" * 60)
    logger.info("训练完成")
    if trainer.val_losses:
        logger.info("最佳验证损失: %.4f", min(trainer.val_losses))
    else:
        logger.info("未进行验证，无法计算最佳验证损失")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
