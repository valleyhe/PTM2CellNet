"""
PTM位点预测模型
功能: 基于序列窗口预测PTM位点
"""

# mypy: disable-error-code="arg-type,assignment,dict-item,operator,return-value,name-defined"
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


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
            self.encoder = CNNEncoder(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
        elif encoder_type == "lstm":
            self.encoder = LSTMEncoder(
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


class CNNEncoder(nn.Module):
    """CNN编码器"""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        kernel_size: int = 3,
    ):
        super().__init__()

        layers = []
        in_channels = embed_dim

        for i in range(num_layers):
            out_channels = hidden_dim if i == num_layers - 1 else embed_dim * 2
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        x = self.conv(x)  # (batch, hidden_dim, seq_len)
        x = self.pool(x).squeeze(-1)  # (batch, hidden_dim)
        return x


class TransformerEncoder(nn.Module):
    """Transformer编码器"""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出投影
        self.output_proj = nn.Linear(embed_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim)
        x = self.transformer(x)  # (batch, seq_len, embed_dim)
        x = x.mean(dim=1)  # 全局平均池化
        x = self.output_proj(x)  # (batch, hidden_dim)
        return x


class LSTMEncoder(nn.Module):
    """LSTM编码器"""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim)
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers * 2, batch, hidden_dim // 2)
        # 取最后一层的双向hidden state
        h_forward = h_n[-2]  # (batch, hidden_dim // 2)
        h_backward = h_n[-1]  # (batch, hidden_dim // 2)
        h = torch.cat([h_forward, h_backward], dim=-1)  # (batch, hidden_dim)
        return h


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
