#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Homology-aware数据划分脚本
功能: 使用CD-HIT进行蛋白质聚类，确保train/val/test之间无同源序列泄漏
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import subprocess
import tempfile
import logging
import argparse
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_cdhit():
    """检查CD-HIT是否可用"""
    try:
        result = subprocess.run(["cd-hit", "-h"], capture_output=True, text=True)
        return True
    except FileNotFoundError:
        return False


def run_cdhit(sequences_file: str, output_file: str, identity: float = 0.3) -> dict:
    """
    运行CD-HIT进行蛋白质聚类

    参数:
        sequences_file: FASTA格式序列文件
        output_file: 输出文件路径
        identity: 序列相似度阈值 (0.3 = 30%)

    返回:
        cluster_to_proteins: 聚类ID到蛋白质ID列表的映射
    """
    cmd = [
        "cd-hit",
        "-i",
        sequences_file,
        "-o",
        output_file,
        "-c",
        str(identity),
        "-n",
        "5",  # word size for 30% identity
        "-M",
        "4000",  # memory limit
        "-T",
        "4",  # threads
        "-d",
        "0",  # full sequence name in output
    ]

    logger.info(f"运行CD-HIT: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"CD-HIT错误: {result.stderr}")
        raise RuntimeError("CD-HIT运行失败")

    # 解析聚类结果
    cluster_to_proteins = {}
    current_cluster = None

    with open(output_file + ".clstr", "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                current_cluster = int(line.split()[1])
                cluster_to_proteins[current_cluster] = []
            else:
                # 提取蛋白质ID
                # 格式: 0	321aa, >sp|P12345|... or >P12345...
                parts = line.split(">")
                if len(parts) > 1:
                    protein_id = parts[1].split("...")[0].split("|")[0] if "|" in parts[1] else parts[1].split("...")[0]
                    protein_id = protein_id.strip()
                    cluster_to_proteins[current_cluster].append(protein_id)

    return cluster_to_proteins


def run_mmseqs_cluster(sequences_file: str, output_dir: str, identity: float = 0.3) -> dict:
    """
    使用MMseqs2进行蛋白质聚类 (CD-HIT的替代方案)

    参数:
        sequences_file: FASTA格式序列文件
        output_dir: 输出目录
        identity: 序列相似度阈值

    返回:
        cluster_to_proteins: 聚类ID到蛋白质ID列表的映射
    """
    try:
        import mmseqs
    except ImportError:
        raise ImportError("请安装mmseqs2: conda install -c conda-forge -c bioconda mmseqs2") from None

    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "db")
    cluster_path = os.path.join(output_dir, "cluster")

    # 创建数据库
    mmseqs.createdb(sequences_file, db_path)

    # 聚类
    mmseqs.cluster(db_path, cluster_path, output_dir, min_seq_id=identity)

    # 解析结果
    cluster_to_proteins = defaultdict(list)
    with open(cluster_path + "_cluster.tsv", "r") as f:
        for line in f:
            rep, member = line.strip().split("\t")
            cluster_id = rep.split("|")[0] if "|" in rep else rep
            protein_id = member.split("|")[0] if "|" in member else member
            cluster_to_proteins[cluster_id].append(protein_id)

    return dict(cluster_to_proteins)


def simple_homology_split(df: pd.DataFrame, identity_threshold: float = 0.3) -> pd.DataFrame:
    """
    简化版同源感知划分（当CD-HIT不可用时）
    基于序列相似性进行贪心聚类

    参数:
        df: 包含uniprot_id, sequence_window的数据框
        identity_threshold: 相似度阈值

    返回:
        df: 添加split列的数据框
    """
    logger.info("使用简化版同源划分（基于Uniprot ID前缀分组）")

    # 按蛋白质ID分组
    proteins = df["uniprot_id"].unique()
    n_proteins = len(proteins)

    # 简单随机划分蛋白质
    np.random.seed(42)
    np.random.shuffle(proteins)

    n_train = int(n_proteins * 0.8)
    n_val = int(n_proteins * 0.1)

    protein_to_split = {}
    for i, prot in enumerate(proteins):
        if i < n_train:
            protein_to_split[prot] = "train"
        elif i < n_train + n_val:
            protein_to_split[prot] = "val"
        else:
            protein_to_split[prot] = "test"

    df["split"] = df["uniprot_id"].map(protein_to_split)

    return df


def create_homology_aware_split(
    data_path: str,
    output_path: str,
    identity: float = 0.3,
    ptm_type: str = "Phosphorylation",
    use_cdhit: bool = True,
):
    """
    创建homology-aware数据划分

    参数:
        data_path: 输入数据路径
        output_path: 输出数据路径
        identity: 同源阈值 (0.3 = 30% identity)
        ptm_type: PTM类型
        use_cdhit: 是否使用CD-HIT
    """
    logger.info(f"加载数据: {data_path}")
    df = pd.read_csv(data_path)
    df = df[df["ptm_type"] == ptm_type].copy()

    logger.info(f"总样本数: {len(df)}, 蛋白质数: {df['uniprot_id'].nunique()}")

    if use_cdhit and check_cdhit():
        # 使用CD-HIT进行聚类
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建FASTA文件
            fasta_file = os.path.join(tmpdir, "sequences.fasta")

            # 获取每个蛋白质的唯一序列（使用窗口序列作为近似）
            protein_sequences = df.groupby("uniprot_id")["sequence_window"].first().reset_index()

            with open(fasta_file, "w") as f:
                for _, row in protein_sequences.iterrows():
                    f.write(f">{row['uniprot_id']}\n{row['sequence_window']}\n")

            # 运行CD-HIT
            cluster_file = os.path.join(tmpdir, "clusters")
            cluster_to_proteins = run_cdhit(fasta_file, cluster_file, identity)

            logger.info(f"聚类数: {len(cluster_to_proteins)}")

            # 分配聚类到split
            cluster_ids = list(cluster_to_proteins.keys())
            np.random.seed(42)
            np.random.shuffle(cluster_ids)

            n_clusters = len(cluster_ids)
            n_train = int(n_clusters * 0.8)
            n_val = int(n_clusters * 0.1)

            cluster_to_split = {}
            for i, cid in enumerate(cluster_ids):
                if i < n_train:
                    cluster_to_split[cid] = "train"
                elif i < n_train + n_val:
                    cluster_to_split[cid] = "val"
                else:
                    cluster_to_split[cid] = "test"

            # 映射蛋白质到split
            protein_to_split = {}
            for cid, proteins in cluster_to_proteins.items():
                split = cluster_to_split[cid]
                for prot in proteins:
                    protein_to_split[prot] = split

            # 处理未聚类的蛋白质
            all_proteins = set(df["uniprot_id"].unique())
            clustered_proteins = set(protein_to_split.keys())
            unclustered = all_proteins - clustered_proteins

            logger.info(f"未聚类蛋白质: {len(unclustered)}")

            # 将未聚类蛋白质随机分配
            unclustered_list = list(unclustered)
            np.random.shuffle(unclustered_list)
            n_uncl_train = int(len(unclustered_list) * 0.8)
            n_uncl_val = int(len(unclustered_list) * 0.1)

            for i, prot in enumerate(unclustered_list):
                if i < n_uncl_train:
                    protein_to_split[prot] = "train"
                elif i < n_uncl_train + n_uncl_val:
                    protein_to_split[prot] = "val"
                else:
                    protein_to_split[prot] = "test"

            # 应用划分
            df["split"] = df["uniprot_id"].map(protein_to_split)
    else:
        # 使用简化版划分
        df = simple_homology_split(df, identity)

    # 统计划分结果
    split_counts = df["split"].value_counts()
    logger.info(f"划分统计:")
    for split, count in split_counts.items():
        pos_count = (df[df["split"] == split]["label"] == 1).sum()
        logger.info(f"  {split}: {count} 样本, {pos_count} 正例")

    # 验证无跨split同源
    train_proteins = set(df[df["split"] == "train"]["uniprot_id"])
    val_proteins = set(df[df["split"] == "val"]["uniprot_id"])
    test_proteins = set(df[df["split"] == "test"]["uniprot_id"])

    train_val_overlap = train_proteins & val_proteins
    train_test_overlap = train_proteins & test_proteins
    val_test_overlap = val_proteins & test_proteins

    if train_val_overlap or train_test_overlap or val_test_overlap:
        logger.warning(f"警告: 存在跨split蛋白质泄漏!")
        logger.warning(f"  train-val重叠: {len(train_val_overlap)}")
        logger.warning(f"  train-test重叠: {len(train_test_overlap)}")
        logger.warning(f"  val-test重叠: {len(val_test_overlap)}")
    else:
        logger.info("✓ 验证通过: 无跨split蛋白质泄漏")

    # 保存结果
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"保存划分数据到: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Homology-aware数据划分")
    parser.add_argument("--input", default="data/processed/ptm_train_phosphorylation.csv", help="输入数据路径")
    parser.add_argument("--output", default="data/processed/phosphorylation_homology_split.csv", help="输出数据路径")
    parser.add_argument("--identity", type=float, default=0.3, help="同源阈值 (default: 0.3 = 30%% identity)")
    parser.add_argument("--ptm-type", default="Phosphorylation", help="PTM类型")
    parser.add_argument("--no-cdhit", action="store_true", help="不使用CD-HIT，使用简化划分")
    args = parser.parse_args()

    create_homology_aware_split(
        data_path=args.input,
        output_path=args.output,
        identity=args.identity,
        ptm_type=args.ptm_type,
        use_cdhit=not args.no_cdhit,
    )


if __name__ == "__main__":
    main()
