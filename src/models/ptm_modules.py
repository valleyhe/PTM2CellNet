"""
PTM处理模块
功能概述: PTM类型与位置信息的嵌入与融合
"""

from typing import Optional, Union, cast

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


class PTMTokenAdapter(nn.Module):
    """Scatter PTM tokens onto a residue-aligned sequence representation.

    The cross-scale model receives PTM positions in the project-wide 1-based
    convention.  This adapter keeps that convention explicit, validates active
    sites instead of clipping them into the last residue, and returns a dense
    ``[batch, sequence_length, embed_dim]`` token field that can be added to a
    pLM representation.  Type and position ``0`` are reserved for padding.

    ``sequence_length`` may be a single integer or one length per batch item;
    the latter is required when a padded batch contains variable-length
    proteins.
    """

    def __init__(
        self,
        num_ptm_types: int,
        embed_dim: int,
        max_position: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_ptm_types <= 0 or embed_dim <= 0 or max_position <= 0:
            raise ValueError("num_ptm_types, embed_dim and max_position must be positive")
        self.num_ptm_types = int(num_ptm_types)
        self.embed_dim = int(embed_dim)
        self.max_position = int(max_position)
        self.type_embedding = nn.Embedding(num_ptm_types + 1, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_position + 1, embed_dim, padding_idx=0)
        self.token_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.last_provenance = {
            "active_site_count": 0,
            "sequence_lengths": [],
            "position_convention": "one_based",
            "padding_id": 0,
        }

    @staticmethod
    def _sequence_lengths(
        sequence_length: Union[int, torch.Tensor], batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if isinstance(sequence_length, int):
            lengths = torch.full((batch_size,), sequence_length, device=device, dtype=torch.long)
        elif isinstance(sequence_length, torch.Tensor):
            if sequence_length.ndim == 0:
                lengths = torch.full(
                    (batch_size,), int(sequence_length.item()), device=device, dtype=torch.long
                )
            elif sequence_length.ndim == 1 and sequence_length.shape[0] == batch_size:
                if sequence_length.dtype not in (torch.int32, torch.int64):
                    raise ValueError("sequence_length tensor must use an integer dtype")
                lengths = sequence_length.to(device=device, dtype=torch.long)
            else:
                raise ValueError("sequence_length must be an integer or shape [batch_size]")
        else:
            raise ValueError("sequence_length must be an integer or torch.Tensor")
        if (lengths <= 0).any():
            raise ValueError("sequence_length values must be positive")
        return lengths

    def forward(
        self,
        ptm_types: torch.Tensor,
        ptm_positions: torch.Tensor,
        sequence_length: Union[int, torch.Tensor],
        ptm_mask: Optional[torch.Tensor] = None,
        valid_sequence_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not isinstance(ptm_types, torch.Tensor) or not isinstance(ptm_positions, torch.Tensor):
            raise ValueError("ptm_types and ptm_positions must be torch.Tensor values")
        if ptm_types.ndim != 2 or ptm_positions.shape != ptm_types.shape:
            raise ValueError("ptm_types and ptm_positions must have shape [B, P]")
        if ptm_types.dtype not in (torch.int32, torch.int64) or ptm_positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("ptm_types and ptm_positions must use integer dtypes")
        batch_size, num_sites = ptm_types.shape
        capacity_lengths = self._sequence_lengths(sequence_length, batch_size, ptm_types.device)
        if valid_sequence_lengths is None:
            lengths = capacity_lengths
        else:
            if not isinstance(valid_sequence_lengths, torch.Tensor):
                raise ValueError("valid_sequence_lengths must be a tensor")
            if valid_sequence_lengths.ndim != 1 or valid_sequence_lengths.shape[0] != batch_size:
                raise ValueError("valid_sequence_lengths must have shape [batch_size]")
            if valid_sequence_lengths.dtype not in (torch.int32, torch.int64):
                raise ValueError("valid_sequence_lengths must use an integer dtype")
            lengths = valid_sequence_lengths.to(device=ptm_types.device, dtype=torch.long)
            if (lengths <= 0).any() or (lengths > capacity_lengths).any():
                raise ValueError("valid_sequence_lengths must be within sequence_length capacity")
        max_sequence_length = int(capacity_lengths.max().item())

        if ptm_mask is None:
            mask = (ptm_types > 0).to(dtype=self.type_embedding.weight.dtype)
        else:
            if not isinstance(ptm_mask, torch.Tensor) or ptm_mask.shape != ptm_types.shape:
                raise ValueError("ptm_mask must match ptm_types shape [B, P]")
            mask = ptm_mask.to(device=ptm_types.device, dtype=self.type_embedding.weight.dtype)
            if not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
                raise ValueError("ptm_mask must contain finite values in [0, 1]")
        active = mask > 0
        invalid_types = active & ((ptm_types < 1) | (ptm_types > self.num_ptm_types))
        invalid_positions = active & (
            (ptm_positions < 1)
            | (ptm_positions > self.max_position)
            | (ptm_positions > lengths.unsqueeze(1))
        )
        if invalid_types.any():
            raise ValueError("active PTM types must be in [1, num_ptm_types]")
        if invalid_positions.any():
            raise ValueError(
                "active PTM positions must be one-based and within each sequence length"
            )

        safe_types = torch.where(active, ptm_types, torch.zeros_like(ptm_types))
        safe_positions = torch.where(active, ptm_positions, torch.zeros_like(ptm_positions))
        site_tokens = self.token_projection(
            self.type_embedding(safe_types) + self.position_embedding(safe_positions)
        )
        site_tokens = self.dropout(site_tokens) * mask.unsqueeze(-1)

        dense_tokens = torch.zeros(
            batch_size,
            max_sequence_length,
            self.embed_dim,
            device=site_tokens.device,
            dtype=site_tokens.dtype,
        )
        scatter_positions = (safe_positions - 1).clamp(min=0, max=max_sequence_length - 1)
        dense_tokens.scatter_add_(
            1,
            scatter_positions.unsqueeze(-1).expand(-1, -1, self.embed_dim),
            site_tokens,
        )
        self.last_provenance = {
            "active_site_count": int(active.sum().item()),
            "sequence_lengths": [int(value) for value in lengths.detach().cpu().tolist()],
            "position_convention": "one_based",
            "padding_id": 0,
        }
        return dense_tokens


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
        return cast(torch.Tensor, output)
