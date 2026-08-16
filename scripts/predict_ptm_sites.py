#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PTM site prediction CLI.

Canonical entry point for batch PTM site prediction. This script predicts PTM
sites and PTM-variant effects from protein sequences; it does not predict cell
states.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.utils.io import safe_torch_load
from src.data.aa_constants import DEFAULT_PTM_WINDOW_SIZE, DEFAULT_PTM_HALF_WINDOW
import torch.nn as nn
import pandas as pd
from typing import Dict, List, Optional
import argparse
import logging
from tqdm import tqdm
from Bio import SeqIO
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiTaskPTMPredictor(nn.Module):
    """多任务PTM预测模型（用于加载预训练权重）"""

    AA_TO_IDX = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
        'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
        'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
        '-': 20,
    }

    def __init__(self, vocab_size=21, embed_dim=64, hidden_dim=128, ptm_types=None):
        super().__init__()
        self.ptm_types = ptm_types or ['Phosphorylation', 'Acetylation', 'Ubiquitination',
                                        'Methylation', 'Sumoylation', 'Succinylation']

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=20)
        self.encoder = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.classifiers = nn.ModuleDict({
            ptm: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 2),
            ) for ptm in self.ptm_types
        })

    def encode(self, x):
        embedded = self.embedding(x)
        encoded = self.encoder(embedded.transpose(1, 2))
        return self.pool(encoded).squeeze(-1)

    def forward(self, x, ptm_type):
        features = self.encode(x)
        logits = self.classifiers[ptm_type](features)
        probs = torch.softmax(logits, dim=-1)
        return {'logits': logits, 'probs': probs}


class BatchPTMPredictor:
    """批量PTM预测器"""

    # PTM类型对应的潜在修饰残基
    PTM_RESIDUES = {
        'Phosphorylation': ['S', 'T', 'Y'],
        'Acetylation': ['K'],
        'Ubiquitination': ['K'],
        'Methylation': ['K', 'R'],
        'Sumoylation': ['K'],
        'Succinylation': ['K'],
    }

    def __init__(
        self,
        model_path: str,
        model_type: str = 'cnn_multitask',
        device: str = 'cpu',
        batch_size: int = 256,
    ):
        """
        初始化批量预测器

        参数:
            model_path: 模型文件路径
            model_type: 模型类型 (cnn_multitask/esm2_t12/esm2_t6)
            device: 计算设备
            batch_size: 批处理大小
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.window_size = DEFAULT_PTM_WINDOW_SIZE
        self.half_window = DEFAULT_PTM_HALF_WINDOW

        # 加载模型
        self.model = self._load_model()
        self.model.eval()

        logger.info(f"模型已加载: {model_path}, 类型: {model_type}, 设备: {device}")

    def _load_model(self):
        """加载预训练模型"""
        if self.model_type == 'cnn_multitask':
            model = MultiTaskPTMPredictor()
            state_dict = safe_torch_load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            return model
        elif self.model_type.startswith('esm2'):
            # ESM-2模型加载
            return self._load_esm2_model()
        else:
            raise ValueError(f"未知模型类型: {self.model_type}")

    def _load_esm2_model(self):
        """加载ESM-2模型"""
        import esm

        # 确定ESM-2模型名称
        if self.model_type == 'esm2_t12':
            esm_model_name = 'esm2_t12_35M_UR50D'
        else:
            esm_model_name = 'esm2_t6_8M_UR50D'

        # 加载预训练ESM-2
        esm_model, alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
        batch_converter = alphabet.get_batch_converter()

        # 加载微调权重
        state_dict = safe_torch_load(self.model_path, map_location=self.device)

        # 构建模型包装器
        class ESM2Wrapper(nn.Module):
            def __init__(self, esm_model, state_dict, embed_dim):
                super().__init__()
                self.esm = esm_model
                self.classifier = nn.Sequential(
                    nn.Linear(embed_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 2),
                )
                # 加载分类器权重
                self.load_state_dict(state_dict, strict=False)

            def forward(self, sequences):
                # ESM-2编码
                batch_tokens = self._tokenize(sequences)
                with torch.no_grad():
                    results = self.esm(batch_tokens, repr_layers=[len(self.esm.layers)])
                embeddings = results['representations'][len(self.esm.layers)][:, 0]
                # 分类
                logits = self.classifier(embeddings)
                probs = torch.softmax(logits, dim=-1)
                return {'logits': logits, 'probs': probs}

        # 获取embed_dim
        embed_dim = esm_model.embed_dim
        model = ESM2Wrapper(esm_model, state_dict, embed_dim)
        model.to(self.device)
        model.batch_converter = batch_converter
        model.alphabet = alphabet

        return model

    def _tokenize(self, sequences):
        """将序列转换为ESM tokens"""
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.batch_converter(data)
        return batch_tokens.to(self.device)

    def _encode_sequence_window(self, sequence: str, position: int) -> torch.Tensor:
        """编码序列窗口"""
        # 提取窗口
        start = max(0, position - self.half_window)
        end = min(len(sequence), position + self.half_window + 1)

        window = sequence[start:end]

        # 填充到固定长度
        if len(window) < self.window_size:
            pad_left = max(0, self.half_window - position)
            pad_right = max(0, (position + self.half_window + 1) - len(sequence))
            window = '-' * pad_left + window + '-' * pad_right

        # 转换为索引
        indices = [self.model.AA_TO_IDX.get(aa, 20) for aa in window]
        return torch.tensor(indices, dtype=torch.long)

    def predict_protein(
        self,
        sequence: str,
        protein_id: str,
        ptm_types: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        预测单个蛋白质的所有潜在PTM位点

        参数:
            sequence: 蛋白质序列
            protein_id: 蛋白质ID
            ptm_types: 要预测的PTM类型列表
            threshold: 预测阈值

        返回:
            预测结果DataFrame
        """
        ptm_types = ptm_types or self.model.ptm_types
        results = []

        # 收集所有候选位点
        candidates = []
        for ptm_type in ptm_types:
            target_residues = self.PTM_RESIDUES.get(ptm_type, [])
            for i, aa in enumerate(sequence):
                if aa in target_residues:
                    candidates.append({
                        'position': i + 1,  # 1-based
                        'aa': aa,
                        'ptm_type': ptm_type,
                    })

        if not candidates:
            return pd.DataFrame()

        # 批量预测
        if self.model_type == 'cnn_multitask':
            # 按PTM类型分组预测
            for ptm_type in ptm_types:
                type_candidates = [c for c in candidates if c['ptm_type'] == ptm_type]
                if not type_candidates:
                    continue

                # 准备批量输入
                windows = []
                for c in type_candidates:
                    window_tensor = self._encode_sequence_window(sequence, c['position'] - 1)
                    windows.append(window_tensor)

                if not windows:
                    continue

                # 按 batch_size 分块进行真正批量推理
                all_probs = []
                for start in range(0, len(windows), self.batch_size):
                    chunk = windows[start:start + self.batch_size]
                    batch_tensor = torch.stack(chunk).to(self.device)

                    with torch.no_grad():
                        output = self.model(batch_tensor, ptm_type)
                        all_probs.append(output['probs'][:, 1].cpu())

                probs = torch.cat(all_probs, dim=0).numpy()

                # 收集结果
                for i, c in enumerate(type_candidates):
                    prob = probs[i]
                    if prob >= threshold:
                        results.append({
                            'protein_id': protein_id,
                            'position': c['position'],
                            'aa': c['aa'],
                            'ptm_type': ptm_type,
                            'probability': float(prob),
                            'sequence_window': sequence[max(0, c['position'] - (DEFAULT_PTM_HALF_WINDOW + 1)):c['position'] + DEFAULT_PTM_HALF_WINDOW],
                        })

        return pd.DataFrame(results)

    def predict_fasta(
        self,
        fasta_path: str,
        output_path: str,
        ptm_types: Optional[List[str]] = None,
        threshold: float = 0.5,
        max_proteins: Optional[int] = None,
    ) -> Dict:
        """
        批量预测FASTA文件中的所有蛋白质

        参数:
            fasta_path: FASTA文件路径
            output_path: 输出文件路径
            ptm_types: PTM类型列表
            threshold: 预测阈值
            max_proteins: 最大处理蛋白数量

        返回:
            统计信息字典
        """
        logger.info(f"开始批量预测: {fasta_path}")

        # 读取序列
        sequences = []
        protein_ids = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            protein_ids.append(record.id)
            sequences.append(str(record.seq))
            if max_proteins and len(sequences) >= max_proteins:
                break

        logger.info(f"读取 {len(sequences)} 个蛋白质序列")

        # 批量预测
        all_results = []
        stats = {
            'total_proteins': len(sequences),
            'total_predictions': 0,
            'ptm_counts': {},
        }

        for protein_id, sequence in tqdm(zip(protein_ids, sequences, strict=False), total=len(sequences)):
            df = self.predict_protein(sequence, protein_id, ptm_types, threshold)
            if len(df) > 0:
                all_results.append(df)
                stats['total_predictions'] += len(df)

                # 统计各PTM类型
                for ptm_type in df['ptm_type'].unique():
                    stats['ptm_counts'][ptm_type] = stats['ptm_counts'].get(ptm_type, 0) + len(df[df['ptm_type'] == ptm_type])

        # 合并结果
        if all_results:
            result_df = pd.concat(all_results, ignore_index=True)
            result_df = result_df.sort_values(['protein_id', 'position', 'ptm_type'])

            # 保存结果
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix == '.csv':
                result_df.to_csv(output_path, index=False)
            else:
                result_df.to_csv(output_path, sep='\t', index=False)

            logger.info(f"结果已保存至: {output_path}")
        else:
            logger.warning("未找到任何预测位点")
            result_df = pd.DataFrame()

        stats['output_file'] = str(output_path)
        return stats

    def predict_variants(
        self,
        variants_file: str,
        sequences_file: str,
        output_path: str,
        threshold: float = 0.5,
    ) -> Dict:
        """
        预测变异对PTM的影响

        参数:
            variants_file: 变异文件 (CSV: protein_id,position,ref_aa,alt_aa)
            sequences_file: 序列文件 (FASTA)
            output_path: 输出文件路径
            threshold: 预测阈值

        返回:
            统计信息
        """
        logger.info(f"开始变异PTM效应预测")

        # 读取序列
        sequences = {}
        for record in SeqIO.parse(sequences_file, "fasta"):
            sequences[record.id] = str(record.seq)

        # 读取变异
        variants_df = pd.read_csv(variants_file)

        results = []
        stats = {
            'total_variants': len(variants_df),
            'ptm_gain': 0,
            'ptm_loss': 0,
            'neutral': 0,
        }

        for _, row in tqdm(variants_df.iterrows(), total=len(variants_df)):
            protein_id = row['protein_id']
            position = int(row['position'])
            ref_aa = row['ref_aa']
            alt_aa = row['alt_aa']

            if protein_id not in sequences:
                continue

            sequence = sequences[protein_id]

            # 预测野生型PTM概率
            wt_results = self.predict_protein(
                sequence, protein_id,
                threshold=0.0,  # 获取所有概率
            )

            # 创建突变序列
            if position - 1 < len(sequence):
                mut_sequence = sequence[:position-1] + alt_aa + sequence[position:]
            else:
                continue

            # 预测突变型PTM概率
            mut_results = self.predict_protein(
                mut_sequence, protein_id,
                threshold=0.0,
            )

            # 比较PTM变化
            for ptm_type in self.model.ptm_types:
                wt_prob = wt_results[
                    (wt_results['position'] == position) &
                    (wt_results['ptm_type'] == ptm_type)
                ]['probability'].values if len(wt_results) > 0 else [0]

                mut_prob = mut_results[
                    (mut_results['position'] == position) &
                    (mut_results['ptm_type'] == ptm_type)
                ]['probability'].values if len(mut_results) > 0 else [0]

                wt_prob = wt_prob[0] if len(wt_prob) > 0 else 0
                mut_prob = mut_prob[0] if len(mut_prob) > 0 else 0

                delta_prob = mut_prob - wt_prob

                effect = 'neutral'
                if delta_prob > 0.2:
                    effect = 'gain'
                    stats['ptm_gain'] += 1
                elif delta_prob < -0.2:
                    effect = 'loss'
                    stats['ptm_loss'] += 1
                else:
                    stats['neutral'] += 1

                results.append({
                    'protein_id': protein_id,
                    'position': position,
                    'ref_aa': ref_aa,
                    'alt_aa': alt_aa,
                    'ptm_type': ptm_type,
                    'wt_probability': float(wt_prob),
                    'mut_probability': float(mut_prob),
                    'delta_probability': float(delta_prob),
                    'effect': effect,
                })

        # 保存结果
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_path, index=False)
        logger.info(f"变异预测结果已保存至: {output_path}")

        return stats


def _build_parser():
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description='批量PTM位点预测（预测蛋白质序列中的 PTM 位点）'
    )

    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='PTM位点预测模式')

    # FASTA批量预测
    fasta_parser = subparsers.add_parser('fasta', help='从FASTA文件批量预测PTM位点')
    fasta_parser.add_argument('--input', '-i', required=True, help='输入FASTA文件')
    fasta_parser.add_argument('--output', '-o', required=True, help='输出文件')
    fasta_parser.add_argument('--model', '-m', required=True, help='模型文件路径')
    fasta_parser.add_argument('--model-type', default='cnn_multitask',
                              choices=['cnn_multitask', 'esm2_t12', 'esm2_t6'],
                              help='模型类型')
    fasta_parser.add_argument('--ptm-types', nargs='+',
                              default=['Phosphorylation', 'Acetylation', 'Ubiquitination',
                                       'Methylation', 'Sumoylation', 'Succinylation'],
                              help='PTM类型')
    fasta_parser.add_argument('--threshold', type=float, default=0.5, help='预测阈值')
    fasta_parser.add_argument('--max-proteins', type=int, help='最大处理蛋白数')
    fasta_parser.add_argument('--device', default='cpu', help='计算设备')
    fasta_parser.add_argument('--batch-size', type=int, default=256, help='批大小')

    # 单序列预测
    single_parser = subparsers.add_parser('single', help='单条蛋白质序列PTM位点预测')
    single_parser.add_argument('--sequence', '-s', required=True, help='蛋白质序列')
    single_parser.add_argument('--protein-id', default='unknown', help='蛋白质ID')
    single_parser.add_argument('--model', '-m', required=True, help='模型文件路径')
    single_parser.add_argument('--model-type', default='cnn_multitask', help='模型类型')
    single_parser.add_argument('--threshold', type=float, default=0.5, help='预测阈值')
    single_parser.add_argument('--batch-size', type=int, default=256, help='批大小')
    single_parser.add_argument('--output', '-o', help='输出文件')

    # 变异预测
    variant_parser = subparsers.add_parser('variant', help='预测变异对PTM位点概率的影响')
    variant_parser.add_argument('--variants', '-v', required=True, help='变异文件')
    variant_parser.add_argument('--sequences', '-s', required=True, help='序列FASTA文件')
    variant_parser.add_argument('--output', '-o', required=True, help='输出文件')
    variant_parser.add_argument('--model', '-m', required=True, help='模型文件路径')
    variant_parser.add_argument('--model-type', default='cnn_multitask', help='模型类型')
    variant_parser.add_argument('--batch-size', type=int, default=256, help='批大小')

    return parser


def parse_args(args=None):
    """解析命令行参数。"""
    return _build_parser().parse_args(args)


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # 创建预测器
    predictor = BatchPTMPredictor(
        model_path=args.model,
        model_type=args.model_type,
        device=getattr(args, 'device', 'cpu'),
        batch_size=getattr(args, 'batch_size', 256),
    )

    # 执行预测
    if args.command == 'fasta':
        stats = predictor.predict_fasta(
            fasta_path=args.input,
            output_path=args.output,
            ptm_types=args.ptm_types,
            threshold=args.threshold,
            max_proteins=args.max_proteins,
        )
        logger.info(f"预测完成: {stats}")

    elif args.command == 'single':
        df = predictor.predict_protein(
            sequence=args.sequence,
            protein_id=args.protein_id,
            threshold=args.threshold,
        )
        print(df.to_string())
        if args.output:
            df.to_csv(args.output, index=False)

    elif args.command == 'variant':
        stats = predictor.predict_variants(
            variants_file=args.variants,
            sequences_file=args.sequences,
            output_path=args.output,
        )
        logger.info(f"变异预测完成: {stats}")


if __name__ == '__main__':
    main()
