"""
PTM-aware virtual perturbation 对比脚本
"""

import argparse
import os
import sys
from dataclasses import asdict


def _ensure_project_root() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def parse_args():
    parser = argparse.ArgumentParser(description="运行 PTM-aware virtual perturbation")
    parser.add_argument(
        "--config", type=str, default="configs/integration/ptm_virtual_perturbation.yaml", help="配置文件路径"
    )
    parser.add_argument("--gene", type=str, required=True, help="目标基因符号")
    parser.add_argument("--magnitude", type=float, default=None, help="扰动强度")
    parser.add_argument("--output", type=str, default="outputs/results/ptm_virtual_perturbation", help="输出目录")
    return parser.parse_args()


def _overlap_at_k(left, right, k: int) -> int:
    return len(set(left[:k]) & set(right[:k]))


def _build_significance_payload(result):
    return {
        "gene_symbol": result.gene_symbol,
        "mode": result.mode,
        "distance_score": result.distance_score,
        "significant_genes": result.metadata.get("significant_genes", []),
        "null_distribution_summary": result.metadata.get("null_distribution_summary", {}),
        "empirical_pvalues": result.metadata.get("empirical_pvalues", {}),
        "adjusted_pvalues": result.metadata.get("adjusted_pvalues", {}),
    }


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

    from pathlib import Path

    from src.integration.contracts import GenePerturbationRequest
    from src.integration.genki_adapter import GenKIAdapter
    from src.integration.genki_reports import (
        build_comparison_summary_payload,
        build_generank_dataframe,
        build_significant_gene_dataframe,
        render_comparison_summary_markdown,
        save_gsea_ranked_tsv,
    )
    from src.utils.config import Config

    from src.utils.io import save_dataframe, save_json
    from src.utils.logging import get_timestamped_log_filename, setup_logger

    args = parse_args()
    logger = setup_logger(__name__, get_timestamped_log_filename("ptm_virtual_perturbation"))
    config = Config.from_yaml(args.config)
    integration_cfg = config.get("integration", {})
    magnitude = float(args.magnitude if args.magnitude is not None else integration_cfg.get("default_magnitude", 0.6))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    hard = adapter.run(GenePerturbationRequest(args.gene, "", "", -1, 1.0, "hard_ko"))
    soft = adapter.run(GenePerturbationRequest(args.gene, "", "", -1, magnitude, "soft_ptm"))

    save_json(asdict(hard), str(output_dir / "hard_ko_results.json"))
    save_json(asdict(soft), str(output_dir / "soft_ptm_results.json"))
    save_json(adapter.get_backend_info(), str(output_dir / "backend_info.json"))
    hard_generank = build_generank_dataframe(hard)
    soft_generank = build_generank_dataframe(soft)
    save_dataframe(hard_generank, str(output_dir / "hard_ko_gene_rank.csv"))
    save_dataframe(soft_generank, str(output_dir / "soft_ptm_gene_rank.csv"))
    save_dataframe(hard_generank, str(output_dir / "hard_ko_generank.csv"))
    save_dataframe(soft_generank, str(output_dir / "soft_ptm_generank.csv"))
    save_dataframe(build_significant_gene_dataframe(hard), str(output_dir / "hard_ko_significant_genes.csv"))
    save_dataframe(build_significant_gene_dataframe(soft), str(output_dir / "soft_ptm_significant_genes.csv"))
    save_gsea_ranked_tsv(hard, str(output_dir / "hard_ko_gsea_rank.tsv"))
    save_gsea_ranked_tsv(soft, str(output_dir / "soft_ptm_gsea_rank.tsv"))
    save_json(_build_significance_payload(hard), str(output_dir / "hard_ko_significance.json"))
    save_json(_build_significance_payload(soft), str(output_dir / "soft_ptm_significance.json"))

    comparison = {
        "gene_symbol": args.gene,
        "hard_distance": hard.distance_score,
        "soft_distance": soft.distance_score,
        "top_gene_overlap_at_3": _overlap_at_k(hard.ranked_genes, soft.ranked_genes, 3),
        "top_gene_overlap_at_10": _overlap_at_k(hard.ranked_genes, soft.ranked_genes, 10),
    }
    save_json(comparison, str(output_dir / "perturbation_comparison.json"))
    summary_payload = build_comparison_summary_payload(
        hard=hard,
        soft=soft,
        comparison=comparison,
        backend_info=adapter.get_backend_info(),
    )
    save_json(summary_payload, str(output_dir / "perturbation_summary.json"))
    (output_dir / "perturbation_summary.md").write_text(
        render_comparison_summary_markdown(summary_payload),
        encoding="utf-8",
    )
    (output_dir / "perturbation_comparison.md").write_text(
        render_comparison_summary_markdown(summary_payload),
        encoding="utf-8",
    )
    logger.info("输出已写入 %s", output_dir)


if __name__ == "__main__":
    main()
