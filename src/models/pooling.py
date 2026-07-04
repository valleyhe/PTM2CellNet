# mypy: disable-error-code="annotation-unchecked,assignment,no-any-return"
"""
注意力池化模块
功能概述: 提供基于attention的序列聚合方法，替代简单的mean pooling
"""

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    注意力池化层

    通过学习一个query向量，对序列特征进行加权聚合。
    相比mean pooling，可以更好地关注重要位置的特征。

    数学公式:
        attention = softmax(query · key^T / sqrt(dim))
        output = attention · value

    Args:
        hidden_dim: 特征维度
        num_heads: 注意力头数，默认1
        dropout: Dropout概率
    """

    def __init__(self, hidden_dim: int, num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == hidden_dim, "hidden_dim必须能被num_heads整除"

        # 可学习的query向量
        self.query = nn.Parameter(torch.randn(1, num_heads, 1, self.head_dim))

        # 投影层
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.query, std=0.02)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        参数:
            hidden_states: [batch_size, seq_len, hidden_dim] 输入特征
            attention_mask: [batch_size, seq_len] 注意力掩码（1表示有效，0表示padding）

        返回:
            pooled: [batch_size, hidden_dim] 池化后的特征
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 投影key和value
        key = self.key_proj(hidden_states)  # [B, L, H]
        value = self.value_proj(hidden_states)  # [B, L, H]

        # reshape为多头格式 [B, L, H] -> [B, num_heads, L, head_dim]
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        # query: [1, num_heads, 1, head_dim] -> [batch_size, num_heads, 1, head_dim]
        query = self.query.expand(batch_size, -1, -1, -1)

        # scores: [B, num_heads, 1, L]
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # 应用mask
        if attention_mask is not None:
            # attention_mask: [B, L] -> [B, 1, 1, L]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # softmax获取注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # [B, num_heads, 1, L]
        attn_weights = self.dropout(attn_weights)

        # 加权聚合
        # [B, num_heads, 1, L] @ [B, num_heads, L, head_dim] = [B, num_heads, 1, head_dim]
        output = torch.matmul(attn_weights, value)

        # reshape回原始维度
        output = output.transpose(1, 2).contiguous()  # [B, 1, num_heads, head_dim]
        output = output.view(batch_size, self.hidden_dim)  # [B, H]

        # 输出投影
        output = self.out_proj(output)

        return output


class MultiHeadAttentionPooling(nn.Module):
    """
    多头注意力池化

    使用多个query向量，每个头关注序列的不同方面，最后拼接。

    Args:
        hidden_dim: 特征维度
        num_heads: 注意力头数
        num_queries: query向量数量（每个头）
        dropout: Dropout概率
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_queries: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == hidden_dim, "hidden_dim必须能被num_heads整除"

        # 多个可学习的query向量
        self.query = nn.Parameter(
            torch.randn(1, num_heads, num_queries, self.head_dim)
        )

        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # 聚合多个query的输出
        if num_queries > 1:
            self.query_agg = nn.Linear(hidden_dim * num_queries, hidden_dim)
        else:
            self.query_agg = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        参数:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]

        返回:
            pooled: [batch_size, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 准备key padding mask
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        # 对每个query进行attention
        outputs = []
        for q_idx in range(self.num_queries):
            # 使用MultiheadAttention
            # 需要构造query: [B, num_queries, H]
            query = self.query[:, :, q_idx, :].expand(batch_size, -1, -1)
            query = query.reshape(batch_size, 1, self.hidden_dim)

            output, _ = self.attention(
                query, hidden_states, hidden_states,
                key_padding_mask=key_padding_mask
            )
            outputs.append(output.squeeze(1))  # [B, H]

        # 聚合多个query的输出
        if self.num_queries > 1:
            output = torch.cat(outputs, dim=-1)  # [B, H * num_queries]
            output = self.query_agg(output)  # [B, H]
        else:
            output = outputs[0]

        return output


class WeightedMeanPooling(nn.Module):
    """
    加权均值池化

    通过学习每个位置的权重，实现自适应的加权平均。

    Args:
        hidden_dim: 特征维度
        temperature: softmax温度参数，控制分布的尖锐程度
    """

    def __init__(self, hidden_dim: int, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.weight_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        参数:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]

        返回:
            pooled: [batch_size, hidden_dim]
        """
        # 计算位置权重
        weights = self.weight_proj(hidden_states).squeeze(-1)  # [B, L]
        weights = weights / self.temperature

        # 应用mask
        if attention_mask is not None:
            weights = weights.masked_fill(attention_mask == 0, float('-inf'))

        # softmax归一化
        weights = F.softmax(weights, dim=-1)  # [B, L]

        # 加权求和
        # [B, L, 1] * [B, L, H] -> sum -> [B, H]
        pooled = torch.sum(weights.unsqueeze(-1) * hidden_states, dim=1)

        return pooled


def create_pooling_layer(
    pool_type: str,
    hidden_dim: int,
    **kwargs
) -> nn.Module:
    """
    工厂函数：创建池化层

    参数:
        pool_type: 池化类型，"mean", "attention", "multihead_attention", "weighted_mean"
        hidden_dim: 特征维度
        **kwargs: 额外的参数

    返回:
        池化层实例
    """
    pool_type = pool_type.lower()

    if pool_type == "mean":
        # 简单的均值池化
        class MeanPooling(nn.Module):
            def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
                if mask is not None:
                    mask_expanded = mask.unsqueeze(-1).float()
                    x = x * mask_expanded
                    return x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
                return x.mean(dim=1)
        return MeanPooling()

    elif pool_type == "attention":
        return AttentionPooling(
            hidden_dim=hidden_dim,
            num_heads=kwargs.get("num_heads", 1),
            dropout=kwargs.get("dropout", 0.1),
        )

    elif pool_type == "multihead_attention":
        return MultiHeadAttentionPooling(
            hidden_dim=hidden_dim,
            num_heads=kwargs.get("num_heads", 4),
            num_queries=kwargs.get("num_queries", 1),
            dropout=kwargs.get("dropout", 0.1),
        )

    elif pool_type == "weighted_mean":
        return WeightedMeanPooling(
            hidden_dim=hidden_dim,
            temperature=kwargs.get("temperature", 1.0),
        )

    else:
        raise ValueError(f"未知的池化类型: {pool_type}")
