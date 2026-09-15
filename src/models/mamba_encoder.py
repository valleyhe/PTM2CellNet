"""
Mamba编码器实现
基于Selective State Space Models (Mamba)的蛋白质序列编码器
"""

from typing import Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    _HAS_MAMBA_SSM = True
except ImportError:
    _HAS_MAMBA_SSM = False


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
        x_conv = rearrange(x_ssm, "b l d -> b d l")
        x_conv = self.conv1d(x_conv)[:, :, :L]  # 截断到原始长度
        x_conv = rearrange(x_conv, "b d l -> b l d")
        x_conv = F.silu(x_conv)  # SiLU激活

        # 3. 计算选择性参数（输入依赖）
        delta = self.delta_proj(x_conv)  # [B, L, d_inner]
        delta = F.softplus(delta)  # 确保 > 0
        delta = torch.clamp(delta, self.delta_min, self.delta_max)  # 数值稳定性

        B_param = self.B_proj(x_conv)  # [B, L, N]
        C_param = self.C_proj(x_conv)  # [B, L, N]

        # 4. SSM 计算（修复历史乱码注释：原 "SSM计" 应为 "SSM 计算"）
        y = self._ssm_step(x_conv, delta, B_param, C_param)  # [B, L, d_inner]

        # 5. 门控和输出投影
        y = y * F.silu(x_gate)  # 门控
        y = self.out_proj(y)  # [B, L, D]

        return cast(torch.Tensor, y)

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

        Note:
            Two execution paths:
            1. If ``mamba_ssm`` is installed and input is on CUDA, use the fused
               ``selective_scan_fn`` for maximum performance.
            2. Otherwise, use the vectorized parallel scan
               (``_ssm_step_parallel``) via cumulative product + cumulative sum.
               This works on both CPU and GPU and is significantly faster than
               the sequential Python loop.
        """
        # Path 1: Fused CUDA kernel via mamba_ssm
        if _HAS_MAMBA_SSM and x.is_cuda:
            return self._ssm_step_fused(x, delta, B, C)

        # Fallback logging (one-time warning)
        if not getattr(self, "_fallback_logged", False):
            logger.info(
                "Mamba fused kernel not available (mamba_ssm not installed or no CUDA). "
                "Using vectorized parallel scan implementation."
            )
            self._fallback_logged = True

        # Path 2: Vectorized parallel scan (GPU or CPU)
        # The cumprod/cumsum approach is faster than a Python loop on both
        # GPU and CPU, though GPU benefits more from parallelism.
        return self._ssm_step_parallel(x, delta, B, C)

    def _ssm_step_sequential(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sequential SSM scan — Python loop, one timestep at a time.

        This is the original implementation, kept as a CPU fallback for
        debugging or when numerical stability of the parallel scan is a
        concern. For production use, prefer ``_ssm_step_parallel`` which
        is vectorized and significantly faster on both CPU and GPU.
        """
        batch_size, seq_len, d_inner = x.shape

        # 离散化SSM参数
        # A̅ = exp(Δ·A)  [B, L, d_inner, N]
        A_bar = torch.exp(einsum(delta, self.A, "b l d, d n -> b l d n"))

        # B̅ = Δ·B  [B, L, d_inner, N]
        B_bar = einsum(delta, B, "b l d, b l n -> b l d n")

        # Pre-compute the D-gated output term D*x once.
        Dx = self.D.unsqueeze(0).unsqueeze(0) * x
        Dx = Dx.permute(1, 0, 2).contiguous()  # [L, B, d_inner]

        # Pre-allocate the output buffer.
        y_out = torch.empty(
            seq_len,
            batch_size,
            d_inner,
            device=x.device,
            dtype=x.dtype,
        )

        h = torch.zeros(
            batch_size,
            d_inner,
            self.d_state,
            device=x.device,
            dtype=x.dtype,
        )

        x_perm = x.permute(1, 0, 2).contiguous()  # [L, B, d_inner]

        for t in range(seq_len):
            A_bar_t = A_bar[:, t]  # [B, d_inner, N]
            B_bar_t = B_bar[:, t]  # [B, d_inner, N]
            x_t = x_perm[t].unsqueeze(-1)  # [B, d_inner, 1]
            h = A_bar_t * h + B_bar_t * x_t  # [B, d_inner, N]

            C_t = C[:, t].unsqueeze(-1)  # [B, N, 1]
            y_t = torch.bmm(h, C_t).squeeze(-1)  # [B, d_inner]
            y_out[t] = y_t + Dx[t]

        return y_out.permute(1, 0, 2).contiguous()

    def _ssm_step_fused(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Use mamba_ssm's fused selective scan kernel (CUDA only).

        mamba_ssm's ``selective_scan_fn`` expects ``(B, D, L)``
        (batch, dim, length) layout for ``u``, ``delta``, ``B``, ``C``,
        while our code uses ``(B, L, D)``.  We permute at the boundary.
        """
        # Convert from our (B, L, D) to kernel's (B, D, L) layout
        u = x.permute(0, 2, 1).contiguous()  # [B, d_inner, L]
        delta_perm = delta.permute(0, 2, 1).contiguous()  # [B, d_inner, L]
        # B, C: (B, L, N) -> (B, N, L)  (gets rearranged to (B, 1, N, L) inside)
        B_perm = B.permute(0, 2, 1).contiguous()  # [B, N, L]
        C_perm = C.permute(0, 2, 1).contiguous()  # [B, N, L]

        A_param = self.A.contiguous()  # [d_inner, N]

        y = selective_scan_fn(
            u,  # [B, d_inner, L]
            delta_perm,  # [B, d_inner, L]
            A_param,  # [d_inner, N]
            B_perm,  # [B, N, L]
            C_perm,  # [B, N, L]
            z=None,
            D=self.D,
            delta_bias=None,
            delta_softplus=False,
            return_last_state=False,
        )  # -> [B, d_inner, L]

        # Convert back to our (B, L, D) layout
        return cast(torch.Tensor, y.permute(0, 2, 1).contiguous())

    def _ssm_step_parallel(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized parallel scan via cumulative product + cumulative sum.

        The SSM recurrence:
            h[t] = A_bar[t]*h[t-1] + B_bar[t]*x[t]
        can be expanded as:
            h[t] = sum_{s=0}^{t} (prod_{r=s+1}^{t} A_bar[r]) * B_bar[s]*x[s]

        Let cumprod_A[t] = prod_{s=0}^{t} A_bar[s]. Then:
            h[t] = cumprod_A[t] * sum_{s=0}^{t} B_bar[s]*x[s] / cumprod_A[s]

        This expresses the recurrence as cumprod * cumsum(Bx / cumprod),
        which runs in O(log L) depth on GPU.
        """
        batch_size, seq_len, d_inner = x.shape

        # Discretize: A_bar [B, L, d_inner, N], B_bar [B, L, d_inner, N]
        A_bar = torch.exp(einsum(delta, self.A, "b l d, d n -> b l d n"))
        B_bar = einsum(delta, B, "b l d, b l n -> b l d n")

        # Bx[t] = B_bar[t] * x[t]   [B, L, d_inner, N]
        Bx = B_bar * x.unsqueeze(-1)

        # Cumulative product of A_bar along the sequence dimension
        cumprod_A = torch.cumprod(A_bar, dim=1)  # [B, L, d_inner, N]

        # Normalized Bx: Bx[t] / cumprod_A[t]  (eps prevents division by zero)
        normalized_Bx = Bx * torch.reciprocal(cumprod_A + 1e-12)

        # Cumulative sum of normalized_Bx
        cumsum_Bx = torch.cumsum(normalized_Bx, dim=1)  # [B, L, d_inner, N]

        # h[t] = cumprod_A[t] * cumsum_Bx[t]
        h = cumprod_A * cumsum_Bx  # [B, L, d_inner, N]

        # y[t] = C[t] @ h[t]  (einsum: 'b l n, b l d n -> b l d')
        y = einsum(C, h, "b l n, b l d n -> b l d")

        # Add D*x
        Dx = self.D.unsqueeze(0).unsqueeze(0) * x  # [B, L, d_inner]
        y = y + Dx

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
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=hidden_dim,
                    d_state=state_dim,
                    d_conv=d_conv,
                    expand_factor=expand_factor,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

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
