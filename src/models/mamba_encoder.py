"""
Mamba编码器实现
基于Selective State Space Models (Mamba)的蛋白质序列编码器
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum


class SelectiveSSM(nn.Module):
    """
    选择性状态空间模型 (Selective State Space Model)

    实现Mamba的核心SSM机制，通过输入依赖的参数实现选择性记忆。

    Args:
        d_model: 模型维度
        d_state: 状态空间维度 (N)
        d_conv: 1D卷积核大小
        expand_factor: 内部维度扩展因子
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor

        # SSM参数（使用HiPPO初始化）
        self.A = nn.Parameter(self._init_A(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输入投影（投影到内部维度的2倍，用于门控）
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D因果卷积
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,  # Depthwise卷积
            bias=True,
        )

        # 选择性投影
        self.delta_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # 数值稳定性参数
        self.delta_min = 0.001
        self.delta_max = 0.1

    def _init_A(self, d_inner: int, d_state: int) -> torch.Tensor:
        """
        HiPPO-LegS初始化A矩阵

        A矩阵应该是负实数，确保状态空间稳定性。
        使用HiPPO（High-order Polynomial Projection Operators）初始化
        可以更好地保持长程依赖。

        简化版本：A[i,j] = -(2j+1) if i >= j else 0
        """
        A = torch.zeros(d_inner, d_state)
        for i in range(d_inner):
            for j in range(d_state):
                A[i, j] = -(2 * j + 1)
        return A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [B, L, D] 输入序列

        Returns:
            y: [B, L, D] 输出序列
        """
        B, L, D = x.shape

        # 1. 输入投影和分支
        x_proj = self.in_proj(x)  # [B, L, 2*d_inner]
        x_ssm, x_gate = x_proj.chunk(2, dim=-1)  # 各 [B, L, d_inner]

        # 2. 1D因果卷积
        x_conv = rearrange(x_ssm, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[:, :, :L]  # 截断到原始长度
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = F.silu(x_conv)  # SiLU激活

        # 3. 计算选择性参数（输入依赖）
        delta = self.delta_proj(x_conv)  # [B, L, d_inner]
        delta = F.softplus(delta)  # 确保 > 0
        delta = torch.clamp(delta, self.delta_min, self.delta_max)  # 数值稳定性

        B_param = self.B_proj(x_conv)  # [B, L, N]
        C_param = self.C_proj(x_conv)  # [B, L, N]

        # 4. SSM计��
        y = self._ssm_step(x_conv, delta, B_param, C_param)  # [B, L, d_inner]

        # 5. 门控和输出投影
        y = y * F.silu(x_gate)  # 门控
        y = self.out_proj(y)  # [B, L, D]

        return y

    def _ssm_step(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        SSM递推计算

        状态方程：
            h[t] = A̅·h[t-1] + B̅·x[t]
            y[t] = C·h[t] + D·x[t]

        其中 A̅ = exp(Δ·A), B̅ = Δ·B

        Args:
            x: [B, L, d_inner] 输入
            delta: [B, L, d_inner] 时间步长
            B: [B, L, N] 输入矩阵
            C: [B, L, N] 输出矩阵

        Returns:
            y: [B, L, d_inner] 输出
        """
        batch_size, seq_len, d_inner = x.shape

        # 离散化SSM参数
        # A̅ = exp(Δ·A)  [B, L, d_inner, N]
        A_bar = torch.exp(einsum(delta, self.A, 'b l d, d n -> b l d n'))

        # B̅ = Δ·B  [B, L, d_inner, N]
        B_bar = einsum(delta, B, 'b l d, b l n -> b l d n')

        # 递推计算状态
        h = torch.zeros(
            batch_size, d_inner, self.d_state,
            device=x.device, dtype=x.dtype
        )

        ys = []
        for t in range(seq_len):
            # h[t] = A̅[t]·h[t-1] + B̅[t]·x[t]
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t:t+1, :].transpose(1, 2)

            # y[t] = C[t]·h[t] + D·x[t]
            y_t = einsum(h, C[:, t], 'b d n, b n -> b d')
            y_t = y_t + self.D * x[:, t]

            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # [B, L, d_inner]
        return y


class MambaBlock(nn.Module):
    """
    Mamba块

    包含SelectiveSSM和残差连接。

    Args:
        d_model: 模型维度
        d_state: 状态空间维度
        d_conv: 1D卷积核大小
        expand_factor: 内部维度扩展因子
        dropout: Dropout概率
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand_factor)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [B, L, D] 输入

        Returns:
            output: [B, L, D] 输出
        """
        # Pre-norm + SSM + 残差
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        x = x + residual
        return x


class MambaEncoder(nn.Module):
    """
    Mamba编码器

    用于蛋白质序列编码的完整Mamba模型。

    Args:
        vocab_size: 词汇表大小
        hidden_dim: 隐藏层维度
        num_layers: Mamba块数量
        state_dim: 状态空间维度
        d_conv: 1D卷积核大小
        expand_factor: 内部维度扩展因子
        dropout: Dropout概率
        max_len: 最大序列长度
    """

    def __init__(
        self,
        vocab_size: int = 20,
        hidden_dim: int = 768,
        num_layers: int = 12,
        state_dim: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
        max_len: int = 1000,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Embedding层
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

        # Mamba块堆叠
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=hidden_dim,
                d_state=state_dim,
                d_conv=d_conv,
                expand_factor=expand_factor,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # 最终归一化
        self.norm = nn.LayerNorm(hidden_dim)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.embedding.weight, std=0.02)
        if self.embedding.padding_idx is not None:
            nn.init.constant_(self.embedding.weight[self.embedding.padding_idx], 0)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [B, L] 输入序列（token IDs）
            mask: [B, L] 可选的mask（1表示有效token，0表示padding）

        Returns:
            output: [B, L, hidden_dim] 编码后的序列
        """
        # Embedding
        x = self.embedding(x)  # [B, L, hidden_dim]

        # 逐层前向传播
        for layer in self.layers:
            x = layer(x)

        # 最终归一化
        x = self.norm(x)

        # 应用mask（处理padding）
        if mask is not None:
            # mask: [B, L]，1表���有效token，0表示padding
            # 将mask扩展到hidden_dim维度
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x = x * mask_expanded

        return x
