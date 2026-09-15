"""
两阶段筛选/解释脚本
"""

import argparse
import os
import sys


def _ensure_project_root() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def parse_args():
    parser = argparse.ArgumentParser(description="运行两阶段筛选/解释流程")
    parser.add_argument(
        "--config", type=str, default="configs/integration/two_stage_explanation.yaml", help="配置文件路径"
    )
    parser.add_argument("--input", type=str, required=True, help="输入 CSV 路径")
    parser.add_argument("--output", type=str, default="outputs/results/two_stage", help="输出目录")
    parser.add_argument("--model", type=str, default=None, help="模型权重路径，可选")
    parser.add_argument("--model-config", type=str, default="configs/default.yaml", help="模型配置文件路径")
    return parser.parse_args()


class PTMCountHeuristicModel:
    """A deterministic fallback model for pipeline smoke runs."""

    def __call__(self, batch):
        import torch

        ptm_count = batch["ptm_mask"].sum(dim=1).float()
        positive = torch.clamp(0.2 + 0.3 * ptm_count, max=0.95)
        probabilities = torch.stack([1.0 - positive, positive], dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        return {
            "probabilities": probabilities,
            "predictions": predictions,
        }


def _build_model(model_path: str | None, model_config_path: str):
    if not model_path:
        return PTMCountHeuristicModel()

    from src.models.architectures import PTM2CellNet
    from src.utils.config import Config
    from src.utils.io import load_model

    config = Config.from_yaml(model_config_path)
    model = PTM2CellNet.from_config(config.to_dict())
    load_model(model, model_path, device="cpu")
    model.eval()
    return model


def _assert_backend_ready(adapter) -> None:
    """在 pipeline 启动前校验 GenKI backend 依赖是否就绪。

    缺少 torch_geometric 等可选依赖时，给出明确的安装提示而非晦涩堆栈。
    依赖就绪与否由 adapter 的 ``validate_runtime_ready`` 决定——基于 fixture
    的后端会声明自己 runtime_ready，而真实 GenKI source 后端在缺 torch_geometric
    时会报告 missing_dependencies（P1-2）。
    """
    try:
        adapter.validate_runtime_ready()
    except RuntimeError as exc:
        raise SystemExit(
            f"{exc}\n"
            "GenKI source backend 依赖未就绪。请安装 genki 可选依赖：\n"
            "    pip install -e .[genki]\n"
            "注意 torch-geometric 需与当前 torch/CUDA 版本匹配，详见 README。"
        ) from exc


def main():
    _ensure_project_root()

    import pandas as pd

    from src.evaluation.explainers import LeaveOnePTMOutScorer, TwoStageExplanationPipeline
    from src.integration.genki_adapter import GenKIAdapter
    from src.integration.ptm_gene_mapper import ProteinGeneMapper
    from src.utils.config import Config
    from src.utils.io import save_json
    from src.utils.logging import get_timestamped_log_filename, setup_logger

    args = parse_args()
    logger = setup_logger(__name__, get_timestamped_log_filename("two_stage_explanation"))
    config = Config.from_yaml(args.config)
    integration_cfg = config.get("integration", {})

    logger.info("加载输入数据: %s", args.input)
    df = pd.read_csv(args.input)
    scorer = LeaveOnePTMOutScorer(
        target_class_index=int(integration_cfg.get("target_class_index", 1)),
        label_names=integration_cfg.get("label_names"),
        amino_acids=str(integration_cfg.get("amino_acids", "ACDEFGHIKLMNPQRSTVWY")),
        ptm_type_to_index=integration_cfg.get("ptm_type_to_index", {}),
    )
    adapter = GenKIAdapter(
        ref_root=str(integration_cfg["genki_ref_root"]),
        gene_list_file=integration_cfg.get("gene_list_file"),
        network_file=integration_cfg.get("network_file"),
        counts_file=integration_cfg.get("counts_file"),
        adata_file=integration_cfg.get("adata_file"),
        grn_file_dir=integration_cfg.get("grn_file_dir"),
        pcnet_name=str(integration_cfg.get("pcnet_name", "pcNet")),
        cutoff=int(integration_cfg.get("cutoff", 85)),
        target_cell=integration_cfg.get("target_cell"),
        obs_label=str(integration_cfg.get("obs_label", "ident")),
        scoring_method=str(integration_cfg.get("scoring_method", "shift")),
        trainer_epochs=int(integration_cfg.get("trainer_epochs", 10)),
        trainer_lr=float(integration_cfg.get("trainer_lr", 7e-4)),
        trainer_beta=float(integration_cfg.get("trainer_beta", 1e-4)),
        trainer_seed=integration_cfg.get("trainer_seed"),
        trainer_out_channels=int(integration_cfg.get("trainer_out_channels", 2)),
        null_permutations=int(integration_cfg.get("null_permutations", 32)),
        null_seed=integration_cfg.get("null_seed"),
        significance_alpha=float(integration_cfg.get("significance_alpha", 0.05)),
        bagging_threshold=float(integration_cfg.get("bagging_threshold", 0.05)),
        bagging_cutoff=float(integration_cfg.get("bagging_cutoff", 0.95)),
    )

    # 在进入耗时 pipeline 之前做 backend readiness 校验，避免因缺少可选依赖
    # （如 torch_geometric）而在 pipeline 内部抛出不友好的堆栈。
    _assert_backend_ready(adapter)

    pipeline = TwoStageExplanationPipeline(
        scorer=scorer,
        mapper=ProteinGeneMapper.from_csv(str(integration_cfg["mapping_file"])),
        genki_adapter=adapter,
        top_k_candidates=int(integration_cfg.get("top_k_candidates", 1)),
        perturbation_mode=str(integration_cfg.get("perturbation_mode", "hard_ko")),
    )
    model = _build_model(args.model, args.model_config)

    logger.info("开始执行两阶段解释流程")
    results = pipeline.run(model=model, df=df, output_dir=args.output)
    save_json(adapter.get_backend_info(), os.path.join(args.output, "backend_info.json"))
    logger.info("完成，两阶段解释结果数: %d", len(results))


if __name__ == "__main__":
    main()
