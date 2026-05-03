"""
Attention Modules for BiPerturb

Implements direction-aware multi-target attention for learning
interactions between genes with different perturbation types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiTargetAttention(nn.Module):
    """
    Multi-head self-attention for multi-target perturbation encoding.

    This is the baseline attention mechanism that combines multiple
    gene target embeddings into a unified perturbation embedding.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        # Multi-head attention
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head self-attention over targets.

        Args:
            x: [batch_size, num_targets, embed_dim]
            mask: optional attention mask

        Returns:
            output: [batch_size, num_targets, embed_dim]
            attention_weights: [batch_size, num_heads, num_targets, num_targets]
        """
        B, K, D = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: [B, num_heads, K, head_dim]

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # Shape: [B, num_heads, K, K]

        if mask is not None:
            if mask.shape == (B, K):
                key_mask = mask.to(dtype=torch.bool).unsqueeze(1).unsqueeze(1)
            elif mask.shape == (B, K, K):
                key_mask = mask.to(dtype=torch.bool).unsqueeze(1)
            elif mask.shape == (B, self.num_heads, K, K):
                key_mask = mask.to(dtype=torch.bool)
            else:
                raise ValueError(
                    f"mask must be [B, K], [B, K, K], or [B, H, K, K], got {tuple(mask.shape)}"
                )
            attn_scores = attn_scores.masked_fill(~key_mask, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        if mask is not None:
            attn_weights = attn_weights * key_mask.to(dtype=attn_weights.dtype)
            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-5)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        # Shape: [B, num_heads, K, head_dim]

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, K, D)
        output = self.out_proj(attn_output)

        return output, attn_weights.mean(dim=1)  # Average attention across heads


class DirectionAwareAttention(nn.Module):
    """
    Direction-aware multi-target attention.

    Key innovation: The attention mechanism is modulated by the perturbation
    direction (KO/KD/OE), allowing the model to learn different interaction
    patterns for activating vs inhibiting perturbations.

    For example:
    - KO + KO might have synergistic (positive) attention
    - KO + OE might have antagonistic (negative) attention
    """

    def __init__(
        self,
        embed_dim: int,
        direction_embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.direction_embed_dim = direction_embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Base attention
        self.base_attention = MultiTargetAttention(embed_dim, num_heads, dropout)

        # Direction modulation for query/key
        self.direction_q_mod = nn.Linear(direction_embed_dim, embed_dim)
        self.direction_k_mod = nn.Linear(direction_embed_dim, embed_dim)

        # Learnable per-head direction interaction matrix
        # [num_heads, 3, 3] for KO/KD/OE combinations
        self.direction_interaction = nn.Parameter(
            torch.stack([
                torch.eye(3) * 0.5 + torch.ones(3, 3) * 0.25
                for _ in range(num_heads)
            ])
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        direction_emb: torch.Tensor,
        directions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Direction-aware self-attention with per-head direction interaction.

        Args:
            x: [batch_size, num_targets, embed_dim] - gene embeddings
            direction_emb: [batch_size, num_targets, direction_embed_dim]
            directions: [batch_size, num_targets] - direction indices

        Returns:
            output: [batch_size, num_targets, embed_dim]
            attention_weights: [batch_size, num_targets, num_targets]
        """
        B, K, D = x.shape

        # Per-head direction interaction weights: [B, H, K, K]
        interaction_weights = self._compute_direction_interactions(directions)

        # Apply direction modulation to input
        q_mod = self.direction_q_mod(direction_emb)  # [B, K, D]

        x_modulated = x * (1 + torch.tanh(q_mod))  # [B, K, D]

        # Compute per-head Q, K, V directly
        base = self.base_attention
        H = base.num_heads
        head_dim = base.head_dim

        q = base.q_proj(x_modulated).view(B, K, H, head_dim).transpose(1, 2)  # [B, H, K, d]
        k = base.k_proj(x_modulated).view(B, K, H, head_dim).transpose(1, 2)
        v = base.v_proj(x_modulated).view(B, K, H, head_dim).transpose(1, 2)

        # Per-head attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / base.scale  # [B, H, K, K]
        if attention_mask is not None:
            key_mask = attention_mask.to(dtype=torch.bool).unsqueeze(1).unsqueeze(1)
            attn_scores = attn_scores.masked_fill(~key_mask, -1e9)

        base_weights = F.softmax(attn_scores, dim=-1)
        if attention_mask is not None:
            key_mask_float = attention_mask.to(dtype=base_weights.dtype).unsqueeze(1).unsqueeze(1)
            base_weights = base_weights * key_mask_float
            base_weights = base_weights / base_weights.sum(dim=-1, keepdim=True).clamp(min=1e-5)
        base_weights = base.dropout(base_weights)

        # Per-head direction-scaled weights
        scaled_weights = base_weights * interaction_weights  # [B, H, K, K]
        scaled_weights = scaled_weights / scaled_weights.sum(dim=-1, keepdim=True).clamp(min=1e-5)

        # Per-head value aggregation
        attn_output = torch.matmul(scaled_weights, v)  # [B, H, K, d]

        # Concat heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, K, D)
        output = base.out_proj(attn_output)

        return output, scaled_weights.mean(dim=1)  # [B, K, K] averaged for external use

    def _compute_direction_interactions(
        self,
        directions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-head direction interaction weights.

        Args:
            directions: [batch_size, num_targets] - direction indices

        Returns:
            interaction_weights: [batch_size, num_heads, num_targets, num_targets]
        """
        B, K = directions.shape
        H = self.num_heads

        dir_i = directions.unsqueeze(2).expand(-1, -1, K)  # [B, K, K]
        dir_j = directions.unsqueeze(1).expand(-1, K, -1)  # [B, K, K]

        # Expand to include head dimension
        dir_i_h = dir_i.unsqueeze(1).expand(-1, H, -1, -1)  # [B, H, K, K]
        dir_j_h = dir_j.unsqueeze(1).expand(-1, H, -1, -1)  # [B, H, K, K]

        # Head indices for batched lookup
        h_idx = torch.arange(H, device=directions.device).view(1, H, 1, 1).expand(B, -1, K, K)

        # self.direction_interaction: [H, 3, 3]
        interaction_weights = self.direction_interaction[h_idx, dir_i_h, dir_j_h]  # [B, H, K, K]

        return interaction_weights


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for multi-omics integration (future extension).

    Allows attention between different modalities (e.g., gene expression
    and protein abundance) for multi-omics perturbation prediction.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-modal attention.

        Args:
            query: [batch_size, num_queries, embed_dim]
            key_value: [batch_size, num_keys, embed_dim]

        Returns:
            output: [batch_size, num_queries, embed_dim]
            attention_weights: [batch_size, num_queries, num_keys]
        """
        attn_output, attn_weights = self.attention(
            query, key_value, key_value,
            need_weights=True,
            average_attn_weights=True
        )

        output = self.norm(query + attn_output)

        return output, attn_weights
