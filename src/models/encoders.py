# mypy: ignore-errors
"""
序列编码器模块
功能概述: 将蛋白质序列编码为向量表示
"""

import math
from typing import cast

import torch
from torch import nn


class SequenceEncoder(nn.Module):
    """序列编码器基类"""

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class PositionalEncoding(nn.Module):
    """标准正弦位置编码（输入为 [seq_len, batch, dim]）"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return cast(torch.Tensor, self.dropout(x))


class CNNEncoder(SequenceEncoder):
    """基于CNN的序列编码器"""

    def __init__(self, vocab_size: int, embed_dim: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        x = self.embedding(sequences)  # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch, seq_len, embed_dim)
        return cast(torch.Tensor, self.dropout(x))


class TransformerEncoder(SequenceEncoder):
    """基于Transformer的序列编码器"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_len: int = 1000,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_emb = nn.Embedding(max_len, embed_dim)

        self.batch_first = True
        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dropout=dropout,
                batch_first=True,
            )
        except TypeError:
            self.batch_first = False
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dropout=dropout,
            )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = sequences.shape
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0).repeat(batch_size, 1)

        x = self.embedding(sequences) + self.position_emb(positions)
        x = self.dropout(x)

        if self.batch_first:
            x = self.encoder(x)
            return cast(torch.Tensor, x)

        x = x.transpose(0, 1)  # (seq_len, batch, embed_dim)
        x = self.encoder(x)
        return cast(torch.Tensor, x.transpose(0, 1))


class LSTMEncoder(SequenceEncoder):
    """基于LSTM的序列编码器"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_len: int = 1000,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.proj = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        x = self.embedding(sequences)
        x, _ = self.lstm(x)
        x = self.proj(x)
        return cast(torch.Tensor, self.dropout(x))
