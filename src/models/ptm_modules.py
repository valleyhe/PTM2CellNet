"""
PTM处理模块
功能概述: PTM类型与位置信息的嵌入与融合
"""

from typing import Optional, cast

import torch
from torch import nn


class PTMEmbedding(nn.Module):
    """PTM类型与位置信息嵌入"""

    def __init__(self, num_ptm_types: int, embed_dim: int, max_position: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.type_embedding = nn.Embedding(num_ptm_types + 1, embed_dim)
        self.position_embedding = nn.Embedding(max_position, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.max_position = max_position

    def forward(self, ptm_types: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        positions = positions.clamp(0, self.max_position - 1)
        return cast(
            torch.Tensor,
            self.dropout(self.type_embedding(ptm_types) + self.position_embedding(positions)),
        )


class PTMAttention(nn.Module):
    """PTM注意力融合"""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        sequence_emb: torch.Tensor,
        ptm_emb: torch.Tensor,
        ptm_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key_padding_mask = None
        if ptm_mask is not None:
            key_padding_mask = ptm_mask == 0
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked] = False

        attn_out, _ = self.attn(sequence_emb, ptm_emb, ptm_emb, key_padding_mask=key_padding_mask)
        if ptm_mask is not None:
            all_masked = ptm_mask.sum(dim=1) == 0
            if all_masked.any():
                attn_out = attn_out.clone()
                attn_out[all_masked] = 0.0
        return cast(torch.Tensor, self.norm(sequence_emb + attn_out))


class PTMModule(nn.Module):
    """PTM完整模块"""

    def __init__(
        self,
        num_ptm_types: int,
        embed_dim: int,
        max_position: int = 1000,
        num_attention_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        fusion_type: str = "attention",
    ):
        super().__init__()
        allowed = {"attention", "gated"}
        if fusion_type not in allowed:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}. Expected one of {sorted(allowed)}")
        self.fusion_type = fusion_type
        self.embedding = PTMEmbedding(num_ptm_types, embed_dim, max_position=max_position, dropout=dropout)
        if fusion_type == "attention":
            self.layers = nn.ModuleList(
                [PTMAttention(embed_dim, num_attention_heads, dropout=dropout) for _ in range(num_layers)]
            )
        else:
            self.layers = nn.ModuleList([GatedPTMFusion(embed_dim, dropout=dropout) for _ in range(num_layers)])

    def forward(
        self,
        sequence_emb: torch.Tensor,
        ptm_types: torch.Tensor,
        ptm_positions: torch.Tensor,
        ptm_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ptm_emb = self.embedding(ptm_types, ptm_positions)
        x = sequence_emb
        for layer in self.layers:
            x = layer(x, ptm_emb, ptm_mask)
        return x


class GatedPTMFusion(nn.Module):
    """门控PTM融合模块.

    通过可学习的门控机制动态控制序列特征和PTM特征的融合比例，
    实现更细粒度的特征交互。

    数学公式:
        gate = σ(W_g · [sequence_emb; ptm_emb] + b_g)
        fused = gate · sequence_emb + (1 - gate) · ptm_emb
        output = LayerNorm(Dropout(fused) + sequence_emb)  # 残差连接

    Args:
        embed_dim: 特征维度。
        dropout: Dropout概率，默认0.1。
        use_residual: 是否使用残差连接，默认True。
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1, use_residual: bool = True) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.use_residual = use_residual

    def forward(
        self,
        sequence_emb: torch.Tensor,
        ptm_emb: torch.Tensor,
        ptm_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        combined = torch.cat([sequence_emb, ptm_emb], dim=-1)
        gate = torch.sigmoid(self.gate_proj(combined))
        if ptm_mask is not None:
            gate = gate * ptm_mask.unsqueeze(-1).to(gate.dtype)
        fused = gate * sequence_emb + (1.0 - gate) * ptm_emb
        fused = self.dropout(fused)
        if self.use_residual:
            fused = fused + sequence_emb
        output = self.layer_norm(fused)
        return output
