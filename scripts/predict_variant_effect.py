#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
变异PTM效应预测脚本
功能: 预测氨基酸变异对PTM和信号网络的影响
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.variant_effect import VariantPTMEffectPredictor, predict_ptm_effects_for_variant
from src.models.signaling_network import PTMNetworkAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_sequences(fasta_path: str) -> dict:
    """加载FASTA序列"""
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id and current_seq:
                    sequences[current_id] = ''.join(current_seq)
                # 解析ID
                parts = line.split('|')
                if len(parts) >= 2:
                    current_id = parts[1]
                else:
                    current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        if current_id and current_seq:
            sequences[current_id] = ''.join(current_seq)

    return sequences


def main():
    parser = argparse.ArgumentParser(description='预测变异的PTM效应')
    parser.add_argument('--variant', '-v', type=str, required=True,
                        help='变异信息 (格式: UniProtID:Position:RefAA:AltAA, 如 P15056:600:V:E)')
    parser.add_argument('--model-dir', '-m', type=str,
                        default='outputs/ptm_pretrain',
                        help='模型目录')
    parser.add_argument('--fasta', '-f', type=str,
                        default='data/uniprot/uniprot_sprot.fasta',
                        help='UniProt FASTA文件')
    parser.add_argument('--ptm-types', '-p', type=str, nargs='+',
                        default=['Phosphorylation', 'Acetylation', 'Ubiquitination',
                                 'Methylation', 'Succinylation', 'Sumoylation'],
                        help='PTM类型')
    parser.add_argument('--output', '-o', type=str,
                        default='variant_prediction.json',
                        help='输出文件')
    args = parser.parse_args()

    # 解析变异
    parts = args.variant.split(':')
    if len(parts) != 4:
        logger.error("变异格式错误，应为: UniProtID:Position:RefAA:AltAA")
        sys.exit(1)

    uniprot_id, position, ref_aa, alt_aa = parts
    position = int(position)

    logger.info(f"分析变异: {uniprot_id}:{position}:{ref_aa}>{alt_aa}")

    # 加载序列
    if os.path.exists(args.fasta):
        logger.info(f"加载序列: {args.fasta}")
        sequences = load_sequences(args.fasta)
    else:
        logger.warning(f"FASTA文件不存在: {args.fasta}")
        sequences = {}

    if uniprot_id not in sequences:
        logger.error(f"未找到序列: {uniprot_id}")
        sys.exit(1)

    sequence = sequences[uniprot_id]
    logger.info(f"序列长度: {len(sequence)}")

    # 加载模型
    models = {}
    for ptm_type in args.ptm_types:
        model_path = Path(args.model_dir) / ptm_type.lower() / 'checkpoints'
        ckpt_files = list(model_path.glob('*.ckpt'))

        if not ckpt_files:
            logger.warning(f"未找到 {ptm_type} 模型")
            continue

        # 选择最佳模型（排除last.ckpt）
        ckpt_files = [f for f in ckpt_files if 'val_auroc' in f.stem]
        if not ckpt_files:
            logger.warning(f"未找到 {ptm_type} 验证模型")
            continue

        # 选择AUROC最高的模型
        best_model = sorted(ckpt_files, key=lambda x: x.stem, reverse=True)[0]
        logger.info(f"加载 {ptm_type} 模型: {best_model}")

        try:
            models[ptm_type] = VariantPTMEffectPredictor(
                model_path=str(best_model),
                ptm_type=ptm_type,
            )
        except Exception as e:
            logger.error(f"加载模型失败: {e}")

    if not models:
        logger.error("没有可用的模型")
        sys.exit(1)

    # 预测PTM效应
    logger.info("预测PTM效应...")
    ptm_effects = predict_ptm_effects_for_variant(
        uniprot_id, position, ref_aa, alt_aa, sequence, models
    )

    # 分析信号网络效应
    logger.info("分析信号网络效应...")
    analyzer = PTMNetworkAnalyzer()

    # 获取基因符号（简化处理）
    gene_symbol = uniprot_id  # 实际应从UniProt映射获取

    result = analyzer.analyze_variant(
        gene_symbol=gene_symbol,
        uniprot_id=uniprot_id,
        position=position,
        ref_aa=ref_aa,
        alt_aa=alt_aa,
        ptm_effects=ptm_effects,
    )

    # 输出结果
    print("\n" + "=" * 60)
    print(f"变异: {uniprot_id}:{position}:{ref_aa}>{alt_aa}")
    print("=" * 60)

    print("\n【PTM效应】")
    for ptm_type, effect in ptm_effects.items():
        print(f"  {ptm_type}:")
        print(f"    野生型概率: {effect['wildtype_prob']:.3f}")
        print(f"    突变型概率: {effect['mutant_prob']:.3f}")
        print(f"    变化: {effect['delta_prob']:+.3f} ({effect['effect']})")

    print("\n【信号网络效应】")
    network = result['network_effects']
    print(f"  影响的通路数: {network['summary']['affected_pathways']}")

    if network['key_pathways']:
        print("  关键通路:")
        for pathway, score in network['key_pathways']:
            direction = "激活" if score > 0 else "抑制"
            print(f"    - {pathway}: {direction} ({score:.2f})")

    print("\n【生物学解释】")
    print(network['interpretation'])

    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        # 转换为可序列化格式
        output = {
            'variant': result['variant'],
            'ptm_effects': ptm_effects,
            'network_effects': {
                'summary': network['summary'],
                'pathway_activities': network['pathway_activities'],
                'key_pathways': network['key_pathways'],
            },
        }
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"结果保存至: {args.output}")


if __name__ == '__main__':
    main()