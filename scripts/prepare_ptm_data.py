#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PTM预训练数据准备脚本
功能: 从dbPTM和UniProt数据构建训练数据集
"""

import sys
from pathlib import Path
import pandas as pd
import gzip
import logging
from typing import Dict, List, Optional
from collections import defaultdict
import random

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PTMDataPreparer:
    """PTM训练数据准备器"""

    # PTM类型到修饰残基的映射
    PTM_RESIDUE_MAP = {
        'Phosphorylation': ['S', 'T', 'Y'],
        'Ubiquitination': ['K'],
        'Acetylation': ['K'],
        'Methylation': ['K', 'R'],
        'Sumoylation': ['K'],
        'Succinylation': ['K'],
        'Malonylation': ['K'],
        'N-linked Glycosylation': ['N'],
        'O-linked Glycosylation': ['S', 'T'],
        'S-nitrosylation': ['C'],
        'S-palmitoylation': ['C'],
        'Glutathionylation': ['C'],
    }

    def __init__(self, dbptm_dir: str, uniprot_dir: str, output_dir: str):
        self.dbptm_dir = Path(dbptm_dir)
        self.uniprot_dir = Path(uniprot_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sequences: Dict[str, str] = {}  # UniProt accession -> sequence
        self.ptm_data: Dict[str, pd.DataFrame] = {}  # PTM type -> DataFrame

    def load_uniprot_fasta(self, fasta_file: Optional[str] = None) -> int:
        """
        加载UniProt Swiss-Prot FASTA文件

        参数:
            fasta_file: FASTA文件路径（可以是.gz压缩文件）

        返回:
            加载的序列数量
        """
        if fasta_file is None:
            # 尝试查找文件
            for ext in ['.fasta', '.fasta.gz']:
                candidate = self.uniprot_dir / f"uniprot_sprot{ext}"
                if candidate.exists():
                    fasta_file = str(candidate)
                    break

        if fasta_file is None:
            logger.warning("UniProt FASTA文件未找到")
            return 0

        logger.info(f"加载UniProt序列: {fasta_file}")

        count = 0
        is_gz = fasta_file.endswith('.gz')

        open_func = gzip.open if is_gz else open
        mode = 'rt' if is_gz else 'r'

        with open_func(fasta_file, mode) as f:
            current_id = None
            current_seq = []

            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # 保存前一个序列
                    if current_id and current_seq:
                        self.sequences[current_id] = ''.join(current_seq)
                        count += 1

                    # 解析header
                    # 格式: >sp|P31946|1433B_HUMAN ... 或 >tr|...
                    parts = line.split('|')
                    if len(parts) >= 2:
                        current_id = parts[1]  # UniProt accession
                    else:
                        current_id = line[1:].split()[0]

                    current_seq = []
                else:
                    current_seq.append(line)

            # 保存最后一个序列
            if current_id and current_seq:
                self.sequences[current_id] = ''.join(current_seq)
                count += 1

        logger.info(f"加载完成: {count:,} 条序列")
        return count

    def load_dbptm_file(self, ptm_type: str) -> pd.DataFrame:
        """
        加载单个dbPTM数据文件

        参数:
            ptm_type: PTM类型名称

        返回:
            DataFrame
        """
        file_path = self.dbptm_dir / f"{ptm_type}.txt"

        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return pd.DataFrame()

        # 跳过可能的PaxHeader
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # 过滤掉PaxHeader行
        data_lines = [l for l in lines if not l.startswith('././@PaxHeader')]

        # 解析数据
        records = []
        for line in data_lines:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                records.append({
                    'protein_id': parts[0],
                    'uniprot_id': parts[1],
                    'position': int(parts[2]),
                    'ptm_type': parts[3],
                    'pubmed_id': parts[4],
                    'sequence_window': parts[5]
                })

        df = pd.DataFrame(records)
        logger.info(f"加载 {ptm_type}: {len(df):,} 条记录")
        return df

    def load_all_dbptm(self, ptm_types: Optional[List[str]] = None) -> int:
        """
        加载所有dbPTM数据

        参数:
            ptm_types: 要加载的PTM类型列表，None则加载所有

        返回:
            总记录数
        """
        if ptm_types is None:
            # 自动检测可用的PTM类型
            ptm_types = []
            for f in self.dbptm_dir.glob('*.txt'):
                ptm_types.append(f.stem)

        total = 0
        for ptm_type in ptm_types:
            df = self.load_dbptm_file(ptm_type)
            if not df.empty:
                self.ptm_data[ptm_type] = df
                total += len(df)

        logger.info(f"加载完成: {len(self.ptm_data)} 种PTM类型, {total:,} 条记录")
        return total

    def extract_sequence_window(self, sequence: str, position: int,
                                 window_size: int = 15) -> Optional[str]:
        """
        从序列中提取窗口

        参数:
            sequence: 蛋白质序列
            position: 位点位置（1-based）
            window_size: 每侧窗口大小

        返回:
            序列窗口字符串，中心为修饰位点
        """
        if position < 1 or position > len(sequence):
            return None

        # 0-based index
        idx = position - 1

        start = max(0, idx - window_size)
        end = min(len(sequence), idx + window_size + 1)

        window = sequence[start:end]

        # 用'-'填充
        left_pad = window_size - (idx - start)
        right_pad = window_size - (end - idx - 1)

        return '-' * left_pad + window + '-' * right_pad

    def prepare_training_data(self, ptm_type: str,
                               negative_ratio: float = 1.0,
                               min_samples: int = 5000) -> pd.DataFrame:
        """
        准备单个PTM类型的训练数据

        参数:
            ptm_type: PTM类型
            negative_ratio: 负样本比例
            min_samples: 最小样本数阈值

        返回:
            训练数据DataFrame
        """
        if ptm_type not in self.ptm_data:
            logger.warning(f"未加载PTM类型: {ptm_type}")
            return pd.DataFrame()

        df = self.ptm_data[ptm_type]
        if df.empty:
            return pd.DataFrame()

        # 获取该PTM类型的修饰残基
        target_residues = self.PTM_RESIDUE_MAP.get(ptm_type, [])
        if not target_residues:
            logger.warning(f"未知PTM类型的修饰残基: {ptm_type}")
            return pd.DataFrame()

        logger.info(f"准备 {ptm_type} 训练数据，修饰残基: {target_residues}")

        # 收集正样本
        positive_samples = []

        for _, row in df.iterrows():
            uniprot_id = row['uniprot_id']
            position = row['position']

            if uniprot_id not in self.sequences:
                continue

            sequence = self.sequences[uniprot_id]

            # 验证位点
            if position < 1 or position > len(sequence):
                continue

            aa = sequence[position - 1]
            if aa not in target_residues:
                continue  # 残基类型不匹配

            window = self.extract_sequence_window(sequence, position)
            if window:
                positive_samples.append({
                    'uniprot_id': uniprot_id,
                    'position': position,
                    'aa': aa,
                    'sequence_window': window,
                    'label': 1,
                    'ptm_type': ptm_type
                })

        logger.info(f"正样本: {len(positive_samples):,}")

        if len(positive_samples) < min_samples:
            logger.warning(f"样本数不足 {min_samples}，跳过")
            return pd.DataFrame()

        # 生成负样本
        negative_samples = []
        target_count = int(len(positive_samples) * negative_ratio)

        # 收集所有可能的负样本位点
        neg_candidates = defaultdict(list)

        for uniprot_id, sequence in self.sequences.items():
            for i, aa in enumerate(sequence):
                if aa in target_residues:
                    neg_candidates[uniprot_id].append(i + 1)  # 1-based

        # 排除正样本位点
        positive_sites = set()
        for sample in positive_samples:
            positive_sites.add((sample['uniprot_id'], sample['position']))

        # 随机采样负样本
        all_neg_sites = []
        for uniprot_id, positions in neg_candidates.items():
            for pos in positions:
                if (uniprot_id, pos) not in positive_sites:
                    all_neg_sites.append((uniprot_id, pos))

        random.shuffle(all_neg_sites)

        for uniprot_id, position in all_neg_sites[:target_count]:
            sequence = self.sequences[uniprot_id]
            aa = sequence[position - 1]
            window = self.extract_sequence_window(sequence, position)

            if window:
                negative_samples.append({
                    'uniprot_id': uniprot_id,
                    'position': position,
                    'aa': aa,
                    'sequence_window': window,
                    'label': 0,
                    'ptm_type': ptm_type
                })

        logger.info(f"负样本: {len(negative_samples):,}")

        # 合并
        all_samples = positive_samples + negative_samples
        result_df = pd.DataFrame(all_samples)

        # 打乱
        result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"总计: {len(result_df):,} 样本")
        return result_df

    def prepare_all_training_data(self, ptm_types: Optional[List[str]] = None,
                                   output_prefix: str = "ptm_train") -> Dict[str, int]:
        """
        准备所有PTM类型的训练数据

        参数:
            ptm_types: 要处理的PTM类型列表
            output_prefix: 输出文件前缀

        返回:
            各PTM类型的样本数量
        """
        if ptm_types is None:
            ptm_types = list(self.ptm_data.keys())

        results = {}

        for ptm_type in ptm_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"处理: {ptm_type}")

            train_df = self.prepare_training_data(ptm_type)

            if not train_df.empty:
                output_file = self.output_dir / f"{output_prefix}_{ptm_type.lower()}.csv"
                train_df.to_csv(output_file, index=False)
                logger.info(f"保存: {output_file}")
                results[ptm_type] = len(train_df)

        return results

    def generate_summary_report(self, output_file: str = "data_summary.md") -> str:
        """生成数据摘要报告"""
        report = []
        report.append("# PTM训练数据摘要\n")
        report.append(f"**生成日期**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        report.append("## 数据源\n\n")
        report.append(f"- UniProt序列: {len(self.sequences):,} 条\n")
        report.append(f"- PTM类型: {len(self.ptm_data)} 种\n\n")

        report.append("## 各PTM类型统计\n\n")
        report.append("| PTM类型 | 位点数 | 修饰残基 |\n")
        report.append("|---------|--------|----------|\n")

        for ptm_type, df in sorted(self.ptm_data.items(), key=lambda x: -len(x[1])):
            residues = ', '.join(self.PTM_RESIDUE_MAP.get(ptm_type, ['Unknown']))
            report.append(f"| {ptm_type} | {len(df):,} | {residues} |\n")

        report_content = ''.join(report)

        report_path = self.output_dir / output_file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"报告保存: {report_path}")
        return report_content


def main():
    import argparse
    parser = argparse.ArgumentParser(description='准备PTM训练数据')
    parser.add_argument('--dbptm-dir', '-d', default='data/dbptm',
                       help='dbPTM数据目录')
    parser.add_argument('--uniprot-dir', '-u', default='data/uniprot',
                       help='UniProt数据目录')
    parser.add_argument('--output-dir', '-o', default='data/processed',
                       help='输出目录')
    parser.add_argument('--ptm-types', '-t', nargs='+',
                       help='指定PTM类型')
    args = parser.parse_args()

    preparer = PTMDataPreparer(args.dbptm_dir, args.uniprot_dir, args.output_dir)

    # 加载UniProt序列
    preparer.load_uniprot_fasta()

    # 加载dbPTM数据
    preparer.load_all_dbptm(args.ptm_types)

    # 准备训练数据
    if preparer.sequences:
        results = preparer.prepare_all_training_data(args.ptm_types)
        print(f"\n准备完成: {results}")
    else:
        print("警告: UniProt序列未加载，无法准备训练数据")

    # 生成报告
    preparer.generate_summary_report()


if __name__ == '__main__':
    main()