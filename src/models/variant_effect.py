"""
变异效应预测模块
功能: 预测氨基酸变异对PTM位点的影响
"""

# mypy: disable-error-code="arg-type,assignment,dict-item,operator,return-value,name-defined"
import torch
import torch.nn as nn
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging

from src.utils.io import safe_torch_load

logger = logging.getLogger(__name__)


class VariantPTMEffectPredictor:
    """
    变异对PTM影响预测器

    用于预测氨基酸变异如何影响PTM位点的存在概率
    """

    # 氨基酸到索引的映射
    AA_TO_IDX = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
        'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
        'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
        '-': 20,  # padding
    }

    # PTM类型到修饰残基的映射
    PTM_RESIDUE_MAP = {
        'Phosphorylation': ['S', 'T', 'Y'],
        'Ubiquitination': ['K'],
        'Acetylation': ['K'],
        'Methylation': ['K', 'R'],
        'Sumoylation': ['K'],
        'Succinylation': ['K'],
    }

    def __init__(
        self,
        model_path: str,
        ptm_type: str = 'Phosphorylation',
        window_size: int = 15,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        初始化预测器

        参数:
            model_path: 预训练模型路径
            ptm_type: PTM类型
            window_size: 序列窗口大小（每侧）
            device: 计算设备
        """
        self.ptm_type = ptm_type
        self.window_size = window_size
        self.device = device

        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()

        logger.info(f"加载模型: {model_path}, PTM类型: {ptm_type}")

    def _load_model(self, model_path: str) -> nn.Module:
        """加载预训练模型"""
        # Imported at call time (not module load) so that test mocks patching
        # ``src.models.ptm_site_predictor.PTMSitePredictor`` take effect. The
        # package is already on sys.path, so no sys.path mutation is needed.
        from src.models.ptm_site_predictor import PTMSitePredictor

        checkpoint = safe_torch_load(
            model_path,
            map_location=self.device,
        )

        # Read hyperparameters from checkpoint, falling back to defaults that
        # match the original training config so checkpoints trained with other
        # encoder/dimension combinations load correctly.
        ckpt_params: Dict[str, Any] = {}
        if isinstance(checkpoint, dict):
            ckpt_params = checkpoint.get("hyperparams", {}) or {}

        vocab_size = int(ckpt_params.get("vocab_size", 21))
        embed_dim = int(ckpt_params.get("embed_dim", 64))
        hidden_dim = int(ckpt_params.get("hidden_dim", 128))
        encoder_type = str(ckpt_params.get("encoder_type", "cnn"))
        # window_size in checkpoint reflects training-time window; fall back to
        # the instance window_size (default 15) so encoding alignment matches.
        window_size = int(ckpt_params.get("window_size", self.window_size))
        # Additional architecture hyperparameters that determine the shape of
        # encoder/classifier weights. Reading them from the checkpoint is
        # required to avoid shape mismatches for checkpoints trained with
        # non-default num_layers / num_heads / num_classes.
        num_layers = int(ckpt_params.get("num_layers", 2))
        num_heads = int(ckpt_params.get("num_heads", 4))
        dropout = float(ckpt_params.get("dropout", 0.1))
        num_classes = int(ckpt_params.get("num_classes", 2))

        model = PTMSitePredictor(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            encoder_type=encoder_type,
            window_size=window_size,
            num_classes=num_classes,
        )

        # 加载状态字典（处理Lightning格式）
        if 'state_dict' in checkpoint:
            state_dict: Dict[str, Any] = {}
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('model.'):
                    state_dict[k[6:]] = v
                else:
                    state_dict[k] = v
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        return model

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """将氨基酸序列编码为索引张量"""
        indices = [self.AA_TO_IDX.get(aa, 20) for aa in sequence]
        return torch.tensor(indices, dtype=torch.long)

    def extract_window(
        self,
        sequence: str,
        position: int,
    ) -> Tuple[Optional[str], Optional[int]]:
        """
        提取序列窗口

        参数:
            sequence: 蛋白质序列
            position: 位点位置（1-based）

        返回:
            (窗口序列, 中心位置)
        """
        if position < 1 or position > len(sequence):
            return (None, None)

        # 0-based index
        idx = position - 1

        start = max(0, idx - self.window_size)
        end = min(len(sequence), idx + self.window_size + 1)

        window = sequence[start:end]

        # 填充
        left_pad = self.window_size - (idx - start)
        right_pad = self.window_size - (end - idx - 1)

        window = '-' * left_pad + window + '-' * right_pad

        return window, self.window_size  # 中心位置

    def predict_site(
        self,
        sequence: str,
        position: int,
    ) -> float:
        """
        预测单个位点的PTM概率

        参数:
            sequence: 蛋白质序列
            position: 位点位置（1-based）

        返回:
            PTM概率
        """
        window, center = self.extract_window(sequence, position)

        if window is None or center is None:
            return 0.0

        # 编码
        indices = self.encode_sequence(window).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            output = self.model(indices)
            prob = output['probs'][0, 1].item()  # 正样本概率

        return prob

    def predict_variant_effect(
        self,
        sequence: str,
        position: int,
        ref_aa: str,
        alt_aa: str,
    ) -> Dict[str, float]:
        """
        预测变异对PTM的影响

        参数:
            sequence: 蛋白质序列
            position: 变异位置（1-based）
            ref_aa: 参考氨基酸
            alt_aa: 变异氨基酸

        返回:
            {
                'wildtype_prob': 野生型PTM概率,
                'mutant_prob': 突变型PTM概率,
                'delta_prob': 概率变化,
                'effect': 'gain'/'loss'/'neutral'
            }
        """
        # 验证参考氨基酸
        if sequence[position - 1] != ref_aa:
            logger.warning(f"位置 {position} 的参考氨基酸不匹配: 期望 {ref_aa}, 实际 {sequence[position - 1]}")

        # 预测野生型
        wildtype_prob = self.predict_site(sequence, position)

        # 创建突变序列
        mutant_sequence = sequence[:position - 1] + alt_aa + sequence[position:]

        # 预测突变型
        mutant_prob = self.predict_site(mutant_sequence, position)

        # 计算效应
        delta_prob = mutant_prob - wildtype_prob

        if delta_prob > 0.1:
            effect = 'gain'
        elif delta_prob < -0.1:
            effect = 'loss'
        else:
            effect = 'neutral'

        return {
            'wildtype_prob': wildtype_prob,
            'mutant_prob': mutant_prob,
            'delta_prob': delta_prob,
            'effect': effect,
        }

    def predict_variants_batch(
        self,
        variants: List[Dict],
        sequences: Dict[str, str],
    ) -> pd.DataFrame:
        """
        批量预测变异效应

        参数:
            variants: 变异列表, 每个包含 {uniprot_id, position, ref_aa, alt_aa}
            sequences: UniProt ID到序列的映射

        返回:
            DataFrame包含预测结果
        """
        results = []

        for variant in variants:
            uniprot_id = variant['uniprot_id']
            position = variant['position']
            ref_aa = variant['ref_aa']
            alt_aa = variant['alt_aa']

            if uniprot_id not in sequences:
                logger.warning(f"未找到序列: {uniprot_id}")
                continue

            sequence = sequences[uniprot_id]

            try:
                effect = self.predict_variant_effect(
                    sequence, position, ref_aa, alt_aa
                )

                results.append({
                    'uniprot_id': uniprot_id,
                    'position': position,
                    'ref_aa': ref_aa,
                    'alt_aa': alt_aa,
                    'ptm_type': self.ptm_type,
                    **effect
                })
            except Exception as e:
                logger.error(f"预测失败: {uniprot_id}:{position}, 错误: {e}")

        return pd.DataFrame(results)


def predict_ptm_effects_for_variant(
    uniprot_id: str,
    position: int,
    ref_aa: str,
    alt_aa: str,
    sequence: str,
    models: Dict[str, VariantPTMEffectPredictor],
) -> Dict[str, Dict]:
    """
    使用多个PTM模型预测变异效应

    参数:
        uniprot_id: UniProt ID
        position: 变异位置
        ref_aa: 参考氨基酸
        alt_aa: 变异氨基酸
        sequence: 蛋白质序列
        models: PTM类型到预测器的映射

    返回:
        各PTM类型的预测结果
    """
    results = {}

    for ptm_type, predictor in models.items():
        try:
            effect = predictor.predict_variant_effect(
                sequence, position, ref_aa, alt_aa
            )
            results[ptm_type] = effect
        except Exception as e:
            logger.error(f"PTM {ptm_type} 预测失败: {e}")

    return results
