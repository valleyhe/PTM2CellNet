"""
PTM位点预测模型
功能: 基于序列窗口预测PTM位点
"""

from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import PooledCNNEncoder, PooledTransformerEncoder, PooledLSTMEncoder


class PTMSitePredictor(nn.Module):
    """
    PTM位点预测模型

    架构:
    1. Embedding层 (可选)
    2. 编码器 (CNN / Transformer / LSTM)
    3. 分类头
    """

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
        num_classes: int = 2,
    ):
        """
        初始化模型

        参数:
            vocab_size: 氨基酸词表大小 (20 + padding)
            embed_dim: 嵌入维度
            hidden_dim: 隐藏层维度
            num_layers: 编码器层数
            num_heads: 注意力头数 (Transformer)
            dropout: Dropout率
            encoder_type: 编码器类型 (cnn / transformer / lstm)
            window_size: 序列窗口大小
            num_classes: 分类数 (默认2: 正/负样本)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type
        self.window_size = window_size

        # Embedding层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=20)

        # 编码器
        if encoder_type == "cnn":
            self.encoder = PooledCNNEncoder(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif encoder_type == "transformer":
            self.encoder = PooledTransformerEncoder(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
        elif encoder_type == "lstm":
            self.encoder = PooledLSTMEncoder(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"未知编码器类型: {encoder_type}")

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(
        self,
        sequence_indices: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            sequence_indices: (batch, window_size) 序列索引

        返回:
            dict:
                - logits: (batch, num_classes) 分类logits
                - probs: (batch, num_classes) 概率
        """
        # Embedding
        x = self.embedding(sequence_indices)  # (batch, window_size, embed_dim)

        # 编码
        encoded = self.encoder(x)  # (batch, hidden_dim)

        # 分类
        logits = self.classifier(encoded)  # (batch, num_classes)
        probs = F.softmax(logits, dim=-1)

        return {
            'logits': logits,
            'probs': probs,
        }


# Pooled encoder aliases for backward compatibility.
# The canonical definitions live in src.models.encoders as Pooled* variants.
from .encoders import (
    PooledCNNEncoder as CNNEncoder,
    PooledTransformerEncoder as TransformerEncoder,
    PooledLSTMEncoder as LSTMEncoder,
)

def create_model(config: Dict[str, Any]) -> PTMSitePredictor:
    """
    根据配置创建模型

    参数:
        config: 配置字典

    返回:
        PTMSitePredictor模型实例
    """
    return PTMSitePredictor(
        vocab_size=config.get('vocab_size', 21),
        embed_dim=config.get('embed_dim', 64),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 2),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.1),
        encoder_type=config.get('encoder_type', 'cnn'),
        window_size=config.get('window_size', 31),
        num_classes=config.get('num_classes', 2),
    )
