"""
Legacy DAVF 兼容层（checkpoint 代际兼容）。

背景（2026-08-04 实测）：仓库内全部 DAVF checkpoint
（``latent_davf_ibd_norman`` 等 7 个）均为旧架构，其 state_dict 结构为：

    gene_embed_table.weight   (5000, 1152)   # Geneformer 初始化嵌入表
    gene_embed_proj.weight    (192, 1152)    # 1152 -> 192 投影
    biperturb_encoder.*                      # 旧版 BiPerturb 编码器
    delta_mlp.*                              # 旧版速度网络（320 -> 64）

与当前 ``LatentDAVF``（``x_encoder`` / ``time_encoder`` / ``velocity_net`` /
``condition_projection``）不匹配；旧实现亦不存在于 git 历史。本模块按
state_dict 的参数字形重建旧架构，使 ``DAVFInferenceModule`` 能够加载这些
checkpoint 并复用其**条件编码器**（biperturb 部分）产出 DAVF 特征。

重建假设（已在论文附录 A.5 记录，供复核）：
1. 旧 ``BiPerturbEncoder`` 无交叉注意力（checkpoint 中不存在
   ``direction_cross_attention.*`` / ``direction_aware_attention.*``）；
   前向为：gene_emb + dir_emb 拼接 -> input_projection -> 方向调制 ->
   norm1 -> 残差 FFN -> norm2 -> output_projection -> 按 attention_mask
   的掩码均值池化。
2. ``DirectionEncoder`` 与当前实现完全同构（embedding + MLP 投影），直接复用。
3. ``delta_mlp`` 仅加载保存（旧速度网络，本模块不执行 ODE 积分）。

用途：离线环境（无外网基因解析）下，即使基因全部被掩码，兼容层也能让
checkpoint 真实加载（``model_source == "davf"``），并支持合成数据上的
方向敏感性验证。
"""

from typing import Any, Dict, Optional, cast

import torch
from torch import nn

from src.models.biperturb import DirectionEncoder


class LegacyBiPerturbEncoder(nn.Module):
    """旧版 BiPerturb 条件编码器（按 checkpoint 字形重建，无交叉注意力）。"""

    def __init__(
        self,
        gene_embed_dim: int = 192,
        direction_embed_dim: int = 64,
        hidden_dim: int = 256,
        num_directions: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.direction_encoder = DirectionEncoder(
            num_directions=num_directions,
            embed_dim=direction_embed_dim,
            dropout=dropout,
        )
        self.input_projection = nn.Linear(
            gene_embed_dim + direction_embed_dim, hidden_dim
        )
        self.direction_modulation = nn.Sequential(
            nn.Linear(direction_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        gene_embeddings: torch.Tensor,
        directions: torch.Tensor,
        magnitudes: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Any:
        """与 ``DAVFInferenceModule.forward`` 的调用签名兼容。

        Args:
            gene_embeddings: [B, K, gene_embed_dim]
            directions: [B, K] 方向码（0=KO, 1=KD, 2=OE）
            magnitudes: 已弃用，忽略
            return_attention: 兼容参数，恒返回 None
            attention_mask: [B, K] 有效性掩码（1=有效）

        Returns:
            (pooled_condition, None)：pooled_condition 形状 [B, hidden_dim]；
            全掩码时退化为纯偏置常数（无 NaN）。
        """
        dir_emb = self.direction_encoder(directions)  # [B, K, dir_dim]
        combined = torch.cat([gene_embeddings, dir_emb], dim=-1)  # [B, K, 256]
        h = self.input_projection(combined)  # [B, K, hidden]
        mod = self.direction_modulation(dir_emb)  # [B, K, hidden]
        h = h * mod
        h = self.norm1(h)
        h = h + self.ffn(self.norm2(h))
        h = self.output_projection(h)

        B, K, _ = h.shape
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float().to(h.dtype)
        else:
            mask = torch.ones(B, K, 1, device=h.device, dtype=h.dtype)
        counts = mask.sum(dim=1).clamp(min=1.0)  # 全掩码时取 1，避免除零
        pooled = (h * mask).sum(dim=1) / counts  # [B, hidden]
        return cast(tuple, (pooled, None))


class LegacyLatentDAVF(nn.Module):
    """旧版 LatentDAVF（条件编码部分），用于加载 2026 年前的 checkpoint。"""

    # 与 checkpoint 一致的固定字形容器（gene_embed_dim 1152 来自 Geneformer）
    GENE_TABLE_SIZE = 5000
    GENE_TABLE_DIM = 1152
    GENE_PROJ_DIM = 192
    HIDDEN_DIM = 256
    LATENT_DIM = 64

    def __init__(self) -> None:
        super().__init__()
        self.gene_embed_table = nn.Embedding(self.GENE_TABLE_SIZE, self.GENE_TABLE_DIM)
        self.gene_embed_proj = nn.Linear(self.GENE_TABLE_DIM, self.GENE_PROJ_DIM)
        self.biperturb_encoder = LegacyBiPerturbEncoder(
            gene_embed_dim=self.GENE_PROJ_DIM,
            hidden_dim=self.HIDDEN_DIM,
        )
        # 旧速度网络：输入 320 = 条件 256 + 时间 64，输出 latent 64。
        # 本兼容层不执行 ODE 积分，仅加载保存以备核对。
        self.delta_mlp = nn.Sequential(
            nn.Linear(320, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.LATENT_DIM),
        )

    def _get_gene_embeddings(self, gene_ids: torch.Tensor) -> torch.Tensor:
        """从 5000 基因的 Geneformer 初始化表中查询并投影到 192 维。

        Raises:
            ValueError: 基因索引越界（表大小为 5000）。
        """
        if (gene_ids < 0).any():
            raise ValueError("gene_ids must be non-negative")
        if gene_ids.numel() > 0 and gene_ids.max().item() >= self.GENE_TABLE_SIZE:
            raise ValueError(
                f"gene_ids must be in [0, {self.GENE_TABLE_SIZE}), "
                f"got max {gene_ids.max().item()}"
            )
        emb = self.gene_embed_table(gene_ids)  # [B, K, 1152]
        return cast(torch.Tensor, self.gene_embed_proj(emb))  # [B, K, 192]

    @staticmethod
    def is_legacy_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
        """按参数块前缀判断是否为旧架构 checkpoint。"""
        keys = set(state_dict.keys())
        return any(k.startswith("delta_mlp.") for k in keys) and any(
            k.startswith("biperturb_encoder.") for k in keys
        )
