#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据统计分析脚本
生成数据集的详细统计报告
"""

import sys
from pathlib import Path

# 确保项目根目录在sys.path中
def _ensure_project_root():
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

_ensure_project_root()

import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# 设置matplotlib后端
plt.switch_backend('Agg')


def analyze_dataframe(df, name):
    """分析DataFrame并返回统计信息"""
    stats = {
        "样本总数": len(df),
        "序列长度": {
            "平均值": df["sequence"].str.len().mean(),
            "中位数": df["sequence"].str.len().median(),
            "最小值": df["sequence"].str.len().min(),
            "最大值": df["sequence"].str.len().max(),
        },
        "细胞状态分布": df["cell_state"].value_counts().to_dict(),
    }

    # PTM统计
    ptm_counts = []
    for ptm_sites in df["ptm_sites"]:
        try:
            sites = json.loads(ptm_sites) if isinstance(ptm_sites, str) else ptm_sites
            ptm_counts.append(len(sites))
        except (json.JSONDecodeError, TypeError):
            ptm_counts.append(0)

    stats["PTM位点数量"] = {
        "平均值": sum(ptm_counts) / len(ptm_counts),
        "中位数": sorted(ptm_counts)[len(ptm_counts) // 2],
        "最小值": min(ptm_counts),
        "最大值": max(ptm_counts),
    }

    # PTM类型统计
    ptm_types = []
    for ptm_sites in df["ptm_sites"]:
        try:
            sites = json.loads(ptm_sites) if isinstance(ptm_sites, str) else ptm_sites
            for site in sites:
                ptm_types.append(site.get("type", "unknown"))
        except (json.JSONDecodeError, TypeError):
            pass

    if ptm_types:
        stats["PTM类型分布"] = dict(Counter(ptm_types))

    return stats


def generate_report():
    """生成数据统计报告"""
    print("=" * 60)
    print("PTM2CellNet 数据统计报告")
    print("=" * 60)

    # 加载数据
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    datasets = {
        "训练集": train_df,
        "验证集": val_df,
        "测试集": test_df,
    }

    # 分析每个数据集
    all_stats = {}
    for name, df in datasets.items():
        print(f"\n{'='*60}")
        print(f"{name}统计")
        print(f"{'='*60}")
        stats = analyze_dataframe(df, name)
        all_stats[name] = stats

        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.2f}")
                    else:
                        print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")

    # 生成可视化
    print(f"\n{'='*60}")
    print("生成可视化图表...")
    print(f"{'='*60}")

    output_dir = Path("outputs/statistics")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 序列长度分布
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (name, df) in enumerate(datasets.items()):
        lengths = df["sequence"].str.len()
        axes[i].hist(lengths, bins=30, edgecolor='black')
        axes[i].set_title(f"{name} - 序列长度分布")
        axes[i].set_xlabel("序列长度")
        axes[i].set_ylabel("样本数")
    plt.tight_layout()
    plt.savefig(output_dir / "sequence_length_distribution.png", dpi=150)
    print(f"  - 序列长度分布图: {output_dir / 'sequence_length_distribution.png'}")

    # 2. 细胞状态分布
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (name, df) in enumerate(datasets.items()):
        df["cell_state"].value_counts().plot(kind='bar', ax=axes[i])
        axes[i].set_title(f"{name} - 细胞状态分布")
        axes[i].set_xlabel("细胞状态")
        axes[i].set_ylabel("样本数")
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "cell_state_distribution.png", dpi=150)
    print(f"  - 细胞状态分布图: {output_dir / 'cell_state_distribution.png'}")

    print(f"\n报告生成完成！")
    print(f"=" * 60)

    return all_stats


if __name__ == "__main__":
    generate_report()
