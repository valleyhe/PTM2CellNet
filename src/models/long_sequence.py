"""
长序列处理模块
功能概述: 处理超过ESM-2最大长度(1022)的蛋白质序列
设计思路: 使用滑动窗口和层次化池化来聚合长序列表示

主要组件:
    - SlidingWindowHandler: 滑动窗口编码处理器（供ESM2/ESM3复用）
    - LongSequenceHandler: 位置映射工具类
    - SlidingWindowESM2: 滑动窗口ESM-2编码器
"""

import typing
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class LongSequenceHandler:
    """
    长序列处理工具类
    管理原始序列位置与窗口位置之间的映射
    """

    def __init__(
        self,
        sequence_length: int,
        window_size: int = 1022,
        overlap: int = 100,
    ):
        """
        初始化长序列处理器

        参数:
            sequence_length: 原始序列长度
            window_size: 每个窗口的大小（不包括特殊token）
            overlap: 窗口之间的重叠大小
        """
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.overlap = overlap
        self.stride = window_size - overlap

        # 计算窗口边界
        self.window_boundaries = self._compute_window_boundaries()

    def _compute_window_boundaries(self) -> List[Tuple[int, int]]:
        """
        计算所有窗口的边界

        返回:
            窗口边界列表，每个元素为(start, end)
        """
        boundaries = []
        start = 0

        while start < self.sequence_length:
            end = min(start + self.window_size, self.sequence_length)
            boundaries.append((start, end))

            # 如果已经到达序列末尾，退出
            if end >= self.sequence_length:
                break

            # 下一个窗口的起始位置
            start += self.stride

            # 确保最后一个窗口包含序列末尾
            if start + self.window_size > self.sequence_length:
                if self.sequence_length > self.window_size:
                    new_start = self.sequence_length - self.window_size
                    # 只有当新窗口与上一个窗口不同时才添加
                    if new_start > boundaries[-1][0]:
                        boundaries.append((new_start, self.sequence_length))
                break

        return boundaries

    def map_position_to_window(self, orig_pos: int) -> Tuple[int, int]:
        """
        将原始序列位置映射到窗口索引和窗口内位置

        参数:
            orig_pos: 原始序列中的位置（0-indexed）

        返回:
            (window_idx, window_pos): 窗口索引和窗口内位置

        异常:
            ValueError: 如果位置超出范围
        """
        if orig_pos < 0 or orig_pos >= self.sequence_length:
            raise ValueError(
                f"位置 {orig_pos} 超出范围 [0, {self.sequence_length})"
            )

        for window_idx, (start, end) in enumerate(self.window_boundaries):
            if start <= orig_pos < end:
                window_pos = orig_pos - start
                return window_idx, window_pos

        # 如果找不到，返回最后一个窗口（用于C端）
        last_idx = len(self.window_boundaries) - 1
        last_start, last_end = self.window_boundaries[last_idx]
        return last_idx, min(orig_pos - last_start, last_end - last_start - 1)

    def get_window_for_position(self, orig_pos: int) -> int:
        """
        获取包含指定位置的窗口索引

        参数:
            orig_pos: 原始序列中的位置

        返回:
            窗口索引
        """
        window_idx, _ = self.map_position_to_window(orig_pos)
        return window_idx

    def get_num_windows(self) -> int:
        """获取窗口数量"""
        return len(self.window_boundaries)


class SlidingWindowHandler:
    """
    滑动窗口编码处理器
    封装长序列的滑动窗口分块、编码、聚合逻辑，供ESM2Encoder和ESM3Encoder复用
    """

    def __init__(
        self,
        window_size: int = 1022,
        overlap: int = 100,
    ):
        """
        初始化滑动窗口处理器

        参数:
            window_size: 窗口大小（不包括特殊token）
            overlap: 窗口之间的重叠大小
        """
        self.window_size = window_size
        self.overlap = overlap

    def encode_sequences(
        self,
        encoder: nn.Module,
        sequences: List[str],
    ) -> torch.Tensor:
        """
        编码蛋白质序列列表，支持长序列(>window_size)的滑动窗口处理。

        参数:
            encoder: 具有tokenize/forward/hidden_dim属性的编码器实例
            sequences: 蛋白质序列字符串列表

        返回:
            embeddings: [batch_size, max_seq_len, hidden_dim] 每个残基的嵌入
        """
        try:
            device = next(encoder.parameters()).device
            dtype = next(encoder.parameters()).dtype
        except StopIteration:
            device = torch.device("cpu")
            dtype = torch.float32

        hidden_dim = encoder.hidden_dim
        batch_embeddings: List[torch.Tensor] = []
        max_seq_len = 0

        for seq in sequences:
            seq_len = len(seq)
            if seq_len > max_seq_len:
                max_seq_len = seq_len

            if seq_len <= self.window_size:
                # 短序列：直接tokenize并提取残基嵌入
                residue_emb = self._encode_short_sequence(
                    encoder, seq, device
                )
                batch_embeddings.append(residue_emb)
            else:
                # 长序列：使用滑动窗口
                residue_emb = self._encode_long_sequence(
                    encoder, seq, device, dtype, hidden_dim
                )
                batch_embeddings.append(residue_emb)

        # 填充到相同长度并堆叠
        return self._pad_and_stack(batch_embeddings, max_seq_len, hidden_dim, device, dtype)

    def _encode_short_sequence(
        self,
        encoder: nn.Module,
        seq: str,
        device: torch.device,
    ) -> torch.Tensor:
        """编码短序列（<= window_size）"""
        tokens = encoder.tokenize([seq])
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad() if not any(p.requires_grad for p in encoder.parameters()) else torch.enable_grad():
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)

        # 去掉 <cls> (位置0) 和 <eos> (位置 seq_len+1)
        token_len = outputs.size(1)
        residue_emb = outputs[:, 1:token_len - 1, :]
        # 截断到实际序列长度
        seq_len = len(seq)
        if residue_emb.size(1) > seq_len:
            residue_emb = residue_emb[:, :seq_len, :]
        return typing.cast(torch.Tensor, residue_emb.squeeze(0))

    def _encode_long_sequence(
        self,
        encoder: nn.Module,
        seq: str,
        device: torch.device,
        dtype: torch.dtype,
        hidden_dim: int,
    ) -> torch.Tensor:
        """编码长序列（> window_size），使用滑动窗口"""
        seq_len = len(seq)
        handler = LongSequenceHandler(
            sequence_length=seq_len,
            window_size=self.window_size,
            overlap=self.overlap,
        )
        accumulated = torch.zeros(seq_len, hidden_dim, device=device, dtype=dtype)
        counts = torch.zeros(seq_len, device=device, dtype=dtype)

        for start, end in handler.window_boundaries:
            window_seq = seq[start:end]
            tokens = encoder.tokenize([window_seq])
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            with torch.no_grad() if not any(p.requires_grad for p in encoder.parameters()) else torch.enable_grad():
                outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)

            token_len = outputs.size(1)
            window_emb = outputs[:, 1:token_len - 1, :]
            window_len = end - start
            if window_emb.size(1) > window_len:
                window_emb = window_emb[:, :window_len, :]

            window_emb = window_emb.squeeze(0)  # [window_len, hidden_dim]
            accumulated[start:end] += window_emb
            counts[start:end] += 1.0

        return accumulated / counts.unsqueeze(-1).clamp(min=1)

    @staticmethod
    def _pad_and_stack(
        embeddings: List[torch.Tensor],
        max_seq_len: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """将不等长的嵌入填充到相同长度并堆叠"""
        if len(embeddings) == 1:
            return embeddings[0].unsqueeze(0)

        padded = []
        for emb in embeddings:
            if emb.size(0) < max_seq_len:
                padding = torch.zeros(
                    max_seq_len - emb.size(0),
                    hidden_dim,
                    device=device,
                    dtype=emb.dtype,
                )
                emb = torch.cat([emb, padding], dim=0)
            padded.append(emb)

        return torch.stack(padded)


class SlidingWindowESM2(nn.Module):
    """
    滑动窗口ESM-2编码器
    处理超过ESM-2最大长度的序列，使用滑动窗口和层次化池化
    """

    def __init__(
        self,
        esm2_encoder,
        window_size: int = 1022,
        overlap: int = 100,
        pool_type: Literal["mean", "max", "attention"] = "attention",
        hidden_dim: Optional[int] = None,
    ):
        """
        初始化滑动窗口ESM-2编码器

        参数:
            esm2_encoder: ESM2Encoder实例
            window_size: 窗口大小（默认1022，ESM-2最大长度）
            overlap: 窗口重叠大小（默认100）
            pool_type: 池化类型，可选 "mean", "max", "attention"
            hidden_dim: 隐藏层维度（用于attention pooling）
        """
        super().__init__()
        self.esm2_encoder = esm2_encoder
        self.window_size = window_size
        self.overlap = overlap
        self.pool_type = pool_type
        self.stride = window_size - overlap

        # 获取ESM-2的隐藏维度
        if hidden_dim is None:
            hidden_dim = getattr(esm2_encoder, "hidden_dim", 128)
        self.hidden_dim: int = hidden_dim

        # 创建注意力池化（如果需要）
        if pool_type == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 4),
                nn.Tanh(),
                nn.Linear(self.hidden_dim // 4, 1),
            )

        logger.info(
            f"滑动窗口ESM-2初始化: window_size={window_size}, "
            f"overlap={overlap}, pool_type={pool_type}"
        )

    def _create_windows(self, sequence: str) -> List[str]:
        """
        将序列分割为重叠的窗口

        参数:
            sequence: 蛋白质序列字符串

        返回:
            窗口序列列表
        """
        seq_len = len(sequence)

        if seq_len <= self.window_size:
            return [sequence]

        windows = []
        start = 0

        while start < seq_len:
            end = min(start + self.window_size, seq_len)
            window_seq = sequence[start:end]
            windows.append(window_seq)

            # 如果已经到达序列末尾，退出
            if end >= seq_len:
                break

            # 下一个窗口的起始位置
            start += self.stride

            # 确保最后一个窗口包含序列末尾
            if start + self.window_size > seq_len:
                if seq_len > self.window_size:
                    new_start = seq_len - self.window_size
                    # 只有当新窗口与上一个窗口不同时才添加
                    if new_start > start - self.stride:  # 与上一个窗口起点比较
                        windows.append(sequence[new_start:seq_len])
                break

        return windows

    def _pool_windows(
        self,
        window_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        池化窗口嵌入

        参数:
            window_embeddings: [num_windows, seq_len, hidden_dim]
            attention_mask: [num_windows, seq_len] 注意力掩码

        返回:
            池化后的嵌入 [hidden_dim]
        """
        if self.pool_type == "mean":
            if attention_mask is not None:
                # 带掩码的均值池化
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_embeddings = (window_embeddings * mask_expanded).sum(dim=1)
                avg_embeddings = sum_embeddings / mask_expanded.sum(dim=1).clamp(min=1)
                # 对所有窗口取平均
                return avg_embeddings.mean(dim=0)
            else:
                return window_embeddings.mean(dim=(0, 1))

        elif self.pool_type == "max":
            # 最大池化
            if attention_mask is not None:
                # 将掩码位置设为很大的负数
                mask_expanded = attention_mask.unsqueeze(-1).float()
                window_embeddings = window_embeddings.masked_fill(
                    mask_expanded == 0, -1e9
                )
            pooled, _ = window_embeddings.max(dim=1)
            return pooled.mean(dim=0)

        elif self.pool_type == "attention":
            # 注意力加权池化
            # window_embeddings: [num_windows, seq_len, hidden_dim]
            num_windows, seq_len, hidden_dim = window_embeddings.shape

            # 计算注意力权重
            attn_scores = self.attention_pool(window_embeddings)  # [num_windows, seq_len, 1]

            if attention_mask is not None:
                # 应用掩码
                mask_expanded = attention_mask.unsqueeze(-1).float()
                attn_scores = attn_scores.masked_fill(mask_expanded == 0, -1e9)

            attn_weights = F.softmax(attn_scores, dim=1)  # [num_windows, seq_len, 1]

            # 加权求和
            weighted_embeddings = (window_embeddings * attn_weights).sum(dim=1)  # [num_windows, hidden_dim]

            # 对窗口取平均
            return weighted_embeddings.mean(dim=0)

        else:
            raise ValueError(f"未知的池化类型: {self.pool_type}")

    def forward(
        self,
        sequences: List[str],
        return_all_windows: bool = False,
    ) -> torch.Tensor:
        """
        前向传播

        参数:
            sequences: 蛋白质序列列表
            return_all_windows: 是否返回所有窗口的嵌入

        返回:
            如果return_all_windows=False: [batch_size, hidden_dim]
            如果return_all_windows=True: [batch_size, num_windows, hidden_dim]
        """
        batch_embeddings = []

        for sequence in sequences:
            if not sequence:
                raise ValueError("序列为空")

            # 创建窗口
            windows = self._create_windows(sequence)

            if len(windows) == 1:
                # 短序列，直接编码
                tokens = self.esm2_encoder.tokenize([windows[0]])
                input_ids = tokens["input_ids"]
                attention_mask = tokens.get("attention_mask")

                with torch.no_grad():
                    embeddings = self.esm2_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                # 取平均池化（去掉特殊token）
                if attention_mask is not None:
                    mask_expanded = attention_mask.unsqueeze(-1).float()
                    seq_embedding = (embeddings * mask_expanded).sum(dim=1)
                    seq_embedding = seq_embedding / mask_expanded.sum(dim=1).clamp(min=1)
                else:
                    seq_embedding = embeddings.mean(dim=1)

                batch_embeddings.append(seq_embedding.squeeze(0))
            else:
                # 长序列，使用滑动窗口
                window_embeddings_list = []

                for window in windows:
                    tokens = self.esm2_encoder.tokenize([window])
                    input_ids = tokens["input_ids"]
                    attention_mask = tokens.get("attention_mask")

                    with torch.no_grad():
                        embeddings = self.esm2_encoder(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )

                    # 取平均池化
                    if attention_mask is not None:
                        mask_expanded = attention_mask.unsqueeze(-1).float()
                        window_emb = (embeddings * mask_expanded).sum(dim=1)
                        window_emb = window_emb / mask_expanded.sum(dim=1).clamp(min=1)
                    else:
                        window_emb = embeddings.mean(dim=1)

                    window_embeddings_list.append(window_emb)

                # 堆叠窗口嵌入
                window_embeddings = torch.cat(window_embeddings_list, dim=0)  # [num_windows, hidden_dim]

                if return_all_windows:
                    batch_embeddings.append(window_embeddings)
                else:
                    # 池化所有窗口
                    # 扩展维度以适应池化函数 [1, num_windows, hidden_dim]
                    window_embeddings_expanded = window_embeddings.unsqueeze(0)
                    dummy_mask = torch.ones(1, window_embeddings.size(0), device=window_embeddings.device)
                    seq_embedding = self._pool_windows(window_embeddings_expanded, dummy_mask)
                    batch_embeddings.append(seq_embedding)

        if return_all_windows:
            # 填充到相同长度
            max_windows = max(emb.size(0) for emb in batch_embeddings)
            padded_embeddings = []
            for emb in batch_embeddings:
                if emb.size(0) < max_windows:
                    padding = torch.zeros(
                        max_windows - emb.size(0),
                        emb.size(1),
                        device=emb.device,
                        dtype=emb.dtype,
                    )
                    emb = torch.cat([emb, padding], dim=0)
                padded_embeddings.append(emb)
            return torch.stack(padded_embeddings)
        else:
            return torch.stack(batch_embeddings)

    def get_position_handler(self, sequence_length: int) -> LongSequenceHandler:
        """
        获取位置处理器

        参数:
            sequence_length: 序列长度

        返回:
            LongSequenceHandler实例
        """
        return LongSequenceHandler(
            sequence_length=sequence_length,
            window_size=self.window_size,
            overlap=self.overlap,
        )
