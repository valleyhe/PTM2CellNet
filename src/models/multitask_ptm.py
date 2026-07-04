# mypy: disable-error-code="annotation-unchecked"
"""
多任务PTM位点预测模型
功能: 联合训练多种PTM类型，提升K修饰类型区分能力
"""

import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import logging

from .encoders import PooledCNNEncoder, PooledTransformerEncoder, PooledLSTMEncoder

# Re-export pooled encoders under short names for backward compatibility.
# These accept pre-embedded input and produce (batch, hidden_dim).
CNNEncoder = PooledCNNEncoder
TransformerEncoder = PooledTransformerEncoder
LSTMEncoder = PooledLSTMEncoder

logger = logging.getLogger(__name__)


class MultiTaskPTMPredictor(nn.Module):
    """
    多任务PTM位点预测模型

    架构:
    1. 共享编码器 (CNN)
    2. 多个任务特定分类头
    3. 可选的对抗域适应
    """

    # PTM类型到修饰残基的映射
    PTM_RESIDUE_MAP = {
        'Phosphorylation': ['S', 'T', 'Y'],
        'Ubiquitination': ['K'],
        'Acetylation': ['K'],
        'Methylation': ['K', 'R'],
        'Sumoylation': ['K'],
        'Succinylation': ['K'],
    }

    # K修饰类型（难以区分的类型）
    K_MODIFICATIONS = ['Ubiquitination', 'Acetylation', 'Methylation', 'Sumoylation', 'Succinylation']

    def __init__(
        self,
        vocab_size: int = 21,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        encoder_type: str = "cnn",
        window_size: int = 31,
        ptm_types: Optional[List[str]] = None,
        share_encoder: bool = True,
        use_adversarial: bool = False,
    ):
        """
        初始化多任务模型

        参数:
            vocab_size: 氨基酸词表大小
            embed_dim: 嵌入维度
            hidden_dim: 隐藏层维度
            num_layers: 编码器层数
            num_heads: 注意力头数
            dropout: Dropout率
            encoder_type: 编码器类型 (cnn/transformer/lstm)
            window_size: 序列窗口大小
            ptm_types: PTM类型列表
            share_encoder: 是否共享编码器
            use_adversarial: 是否使用对抗域适应
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type
        self.window_size = window_size
        self.ptm_types: List[str] = ptm_types or [
            'Phosphorylation',
            'Acetylation',
            'Ubiquitination',
            'Methylation',
            'Succinylation',
            'Sumoylation',
        ]
        self.share_encoder = share_encoder
        self.use_adversarial = use_adversarial

        # Embedding层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=20)

        # 编码器
        if share_encoder:
            self.encoder = self._create_encoder(
                encoder_type,
                embed_dim,
                hidden_dim,
                num_layers,
                num_heads,
                dropout,
            )
        else:
            self.encoder = nn.ModuleDict({
                ptm: self._create_encoder(
                    encoder_type,
                    embed_dim,
                    hidden_dim,
                    num_layers,
                    num_heads,
                    dropout,
                )
                for ptm in self.ptm_types
            })

        # 任务特定分类头
        self.classifiers = nn.ModuleDict({
            ptm: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 2),
            ) for ptm in self.ptm_types
        })

        # 对抗域判别器（可选）
        if use_adversarial:
            self.domain_discriminator = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, len(self.ptm_types)),
            )

        # 初始化权重
        self.apply(self._init_weights)

    def _create_encoder(self, encoder_type, embed_dim, hidden_dim, num_layers, num_heads, dropout):
        """创建编码器"""
        if encoder_type == "cnn":
            return PooledCNNEncoder(embed_dim, hidden_dim, num_layers, dropout)
        elif encoder_type == "transformer":
            return PooledTransformerEncoder(embed_dim, hidden_dim, num_layers, num_heads, dropout)
        elif encoder_type == "lstm":
            return PooledLSTMEncoder(embed_dim, hidden_dim, num_layers, dropout)
        else:
            raise ValueError(f"未知编码器类型: {encoder_type}")

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def encode(self, x: torch.Tensor, ptm_type: Optional[str] = None) -> torch.Tensor:
        """
        编码序列

        参数:
            x: (batch, window_size) 序列索引
            ptm_type: PTM类型（不共享编码器时需要）

        返回:
            (batch, hidden_dim) 编码向量
        """
        # Embedding
        embedded = self.embedding(x)  # (batch, window_size, embed_dim)

        # 编码
        if self.share_encoder:
            encoded = self.encoder(embedded)
        else:
            if ptm_type is None:
                raise ValueError("不共享编码器时需要指定ptm_type")
            encoded = self.encoder[ptm_type](embedded)

        return typing.cast(torch.Tensor, encoded)

    def forward(
        self,
        sequence_indices: torch.Tensor,
        ptm_type: str,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            sequence_indices: (batch, window_size) 序列索引
            ptm_type: PTM类型
            return_features: 是否返回特征

        返回:
            dict: logits, probs, 可选features
        """
        # 编码
        features = self.encode(sequence_indices, ptm_type)

        # 分类
        logits = self.classifiers[ptm_type](features)
        probs = F.softmax(logits, dim=-1)

        result = {
            'logits': logits,
            'probs': probs,
        }

        if return_features:
            result['features'] = features

        return result

    def forward_all(
        self,
        sequence_indices: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        对所有PTM类型进行预测

        参数:
            sequence_indices: (batch, window_size) 序列索引

        返回:
            各PTM类型的预测结果
        """
        results = {}

        for ptm_type in self.ptm_types:
            results[ptm_type] = self.forward(sequence_indices, ptm_type)

        return results

    def get_adversarial_loss(self, features: torch.Tensor, ptm_indices: torch.Tensor) -> torch.Tensor:
        """
        计算对抗域损失

        参数:
            features: (batch, hidden_dim) 特征向量
            ptm_indices: (batch,) PTM类型索引

        返回:
            对抗损失
        """
        if not self.use_adversarial:
            return torch.tensor(0.0, device=features.device)

        # 梯度反转
        reversed_features = features  # 实际实现需要GradientReversalLayer

        # 域预测
        domain_logits = self.domain_discriminator(reversed_features)

        # 交叉熵损失
        loss = F.cross_entropy(domain_logits, ptm_indices)

        return loss


class MultiTaskLoss(nn.Module):
    """多任务损失函数"""

    def __init__(
        self,
        ptm_types: Optional[List[str]] = None,
        task_weights: Optional[Dict[str, float]] = None,
        use_uncertainty_weighting: bool = False,
    ):
        """
        初始化多任务损失

        参数:
            ptm_types: PTM类型列表
            task_weights: 任务权重
            use_uncertainty_weighting: 是否使用不确定性加权
        """
        super().__init__()

        self.ptm_types: List[str] = ptm_types if ptm_types is not None else []
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # 任务权重
        if task_weights is None:
            task_weights = {ptm: 1.0 for ptm in self.ptm_types}
        self.task_weights = task_weights

        # 不确定性参数（可学习）
        if use_uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.zeros(len(self.ptm_types)))

    def forward(
        self,
        outputs: Dict[str, Dict[str, torch.Tensor]],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失

        参数:
            outputs: 各PTM类型的预测输出
            targets: 各PTM类型的标签

        返回:
            总损失和各任务损失
        """
        losses = {}
        # 从第一个有效 target 推断设备，保证 total_loss 始终在正确设备上。
        device = torch.device('cpu')
        for ptm_type in self.ptm_types:
            if ptm_type in targets:
                device = targets[ptm_type].device
                break
        total_loss: torch.Tensor = torch.tensor(0.0, device=device)

        for i, ptm_type in enumerate(self.ptm_types):
            if ptm_type not in outputs or ptm_type not in targets:
                continue

            logits = outputs[ptm_type]['logits']
            target = targets[ptm_type]

            # 交叉熵损失
            loss = F.cross_entropy(logits, target)

            # 任务权重
            weight = self.task_weights.get(ptm_type, 1.0)

            # 不确定性加权
            if self.use_uncertainty_weighting:
                precision = torch.exp(-self.log_vars[i])
                loss = precision * loss + self.log_vars[i]

            weighted_loss = weight * loss
            losses[f'{ptm_type}_loss'] = loss
            total_loss = total_loss + weighted_loss

        losses['total_loss'] = total_loss

        return losses


def create_multitask_model(config: Dict) -> MultiTaskPTMPredictor:
    """根据配置创建多任务模型"""
    return MultiTaskPTMPredictor(
        vocab_size=config.get('vocab_size', 21),
        embed_dim=config.get('embed_dim', 64),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 2),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.1),
        encoder_type=config.get('encoder_type', 'cnn'),
        window_size=config.get('window_size', 31),
        ptm_types=config.get('ptm_types'),
        share_encoder=config.get('share_encoder', True),
        use_adversarial=config.get('use_adversarial', False),
    )
