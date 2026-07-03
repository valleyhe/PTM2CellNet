# mypy: ignore-errors
"""
序列处理工具模块

提供共享的滑动窗口逻辑，消除 long_sequence.py 与 pretrained_encoders.py
之间的重复实现。

主要组件:
    - WindowedSequenceProcessor: 通用滑动窗口处理器
"""

from typing import List, Literal, Tuple

import torch

__all__ = ["WindowedSequenceProcessor"]


Reduction = Literal["mean", "max", "concat"]


class WindowedSequenceProcessor:
    """Shared sliding-window logic for long sequence processing.

    Splits a long sequence into overlapping windows, processes each independently,
    and reassembles results via configurable reduction (mean, max, concat).

    The windowing strategy matches the legacy implementations in
    ``src/models/long_sequence.py`` (``LongSequenceHandler``) and
    ``src/models/pretrained_encoders.py`` (``ESM2Encoder.encode_sequences``):

      * Windows are emitted with ``stride = window_size - overlap``.
      * The final window is shifted back so it always ends exactly at
        ``seq_len`` (covering the sequence tail), avoiding a truncated tail.
      * A duplicate final window is suppressed when the shift would land it
        on the same start as the previous window (i.e. when the sequence is
        only slightly longer than a single window).
    """

    def __init__(
        self,
        window_size: int = 1022,
        overlap: int = 100,
        reduction: Reduction = "mean",
    ) -> None:
        """
        初始化滑动窗口处理器

        参数:
            window_size: 每个窗口的大小（不包括特殊token），默认1022（ESM-2最大长度）
            overlap: 窗口之间的重叠大小，默认100
            reduction: 重叠区域的重叠归约方式，
                "mean" — 对每个原始位置取所有覆盖窗口的平均
                "max"  — 对每个原始位置取所有覆盖窗口的最大值
                "concat" — 按窗口顺序拼接（不进行重叠归约，仅截断到 original_len）

        异常:
            ValueError: 如果 window_size <= 0、overlap < 0 或
                overlap >= window_size（stride 必须为正）
        """
        if window_size <= 0:
            raise ValueError(f"window_size 必须为正整数，得到: {window_size}")
        if overlap < 0:
            raise ValueError(f"overlap 不能为负，得到: {overlap}")
        if overlap >= window_size:
            raise ValueError(
                f"overlap ({overlap}) 必须小于 window_size ({window_size})，"
                f"否则窗口无法前进"
            )
        if reduction not in ("mean", "max", "concat"):
            raise ValueError(
                f"未知的 reduction: {reduction}，可选: 'mean', 'max', 'concat'"
            )

        self.window_size: int = window_size
        self.overlap: int = overlap
        self.reduction: Reduction = reduction
        self.stride: int = window_size - overlap

    def split(self, seq_len: int) -> List[Tuple[int, int]]:
        """Return list of (start, end) tuples for each window.

        参数:
            seq_len: 原始序列长度

        返回:
            窗口边界列表，每个元素为 (start, end)，end 为开区间上界。
            当 ``seq_len <= window_size`` 时返回单个覆盖整段序列的窗口 ``[(0, seq_len)]``。

        异常:
            ValueError: 如果 seq_len < 0
        """
        if seq_len < 0:
            raise ValueError(f"seq_len 不能为负，得到: {seq_len}")

        # 空序列无窗口可生成，与 legacy LongSequenceHandler 行为一致
        if seq_len == 0:
            return []

        if seq_len <= self.window_size:
            return [(0, seq_len)]

        boundaries: List[Tuple[int, int]] = []
        start = 0

        while start < seq_len:
            end = min(start + self.window_size, seq_len)
            boundaries.append((start, end))

            # 已到达序列末尾
            if end >= seq_len:
                break

            start += self.stride

            # 确保最后一个窗口包含序列末尾：将窗口起点回退至
            # seq_len - window_size，使其恰好覆盖序列尾部。
            if start + self.window_size > seq_len:
                if seq_len > self.window_size:
                    new_start = seq_len - self.window_size
                    # 仅当回退后的窗口与上一个窗口不同时才添加
                    # （避免在序列仅略长于一个窗口时产生重复窗口）
                    if new_start > boundaries[-1][0]:
                        boundaries.append((new_start, seq_len))
                break

        return boundaries

    def reassemble(
        self,
        window_outputs: List[torch.Tensor],
        original_len: int,
    ) -> torch.Tensor:
        """Merge overlapping window outputs using configured reduction.

        参数:
            window_outputs: 每个窗口对应的输出张量列表。
                对于 "mean" / "max"：每个张量形状为 ``[window_len, *feature_dims]``，
                其中 ``window_len`` 必须与 ``split()`` 返回的对应窗口长度一致。
                对于 "concat"：每个张量形状为 ``[window_len, *feature_dims]``，
                按窗口顺序在第 0 维拼接。
            original_len: 原始序列长度，用于 "concat" 模式截断到正确长度，
                以及校验 "mean" / "max" 模式下窗口覆盖的完整性。

        返回:
            归约后的张量：
                - "mean" / "max": ``[original_len, *feature_dims]``
                - "concat": ``[min(original_len, total_window_len), *feature_dims]``

        异常:
            ValueError: 如果 window_outputs 为空，或窗口数量与 ``split(original_len)``
                不一致（mean/max 模式），或窗口长度与预期不符。
        """
        if not window_outputs:
            raise ValueError("window_outputs 不能为空")
        if original_len < 0:
            raise ValueError(f"original_len 不能为负，得到: {original_len}")

        boundaries = self.split(original_len)

        if self.reduction == "concat":
            # 拼接模式：按窗口顺序连接，截断到 original_len
            concatenated = torch.cat(window_outputs, dim=0)
            if concatenated.size(0) > original_len:
                concatenated = concatenated[:original_len]
            return concatenated

        # mean / max 模式：需要按位置归约重叠区域
        if len(window_outputs) != len(boundaries):
            raise ValueError(
                f"window_outputs 数量 ({len(window_outputs)}) 与 "
                f"窗口数量 ({len(boundaries)}) 不一致"
            )

        # 推断特征维度：取第一个窗口除第 0 维（window_len）外的形状
        feature_shape = tuple(window_outputs[0].shape[1:])
        # 统一设备与数据类型
        device = window_outputs[0].device
        dtype = window_outputs[0].dtype

        # 累加器：对 mean 收集 sum 与 count，对 max 收集逐元素最大值
        if self.reduction == "mean":
            accumulated = torch.zeros(
                (original_len, *feature_shape), device=device, dtype=dtype
            )
            counts = torch.zeros(original_len, device=device, dtype=dtype)

            for (start, end), window_out in zip(boundaries, window_outputs):
                expected_len = end - start
                if window_out.size(0) != expected_len:
                    raise ValueError(
                        f"窗口输出长度 ({window_out.size(0)}) 与窗口范围 "
                        f"({expected_len}) 不符，start={start}, end={end}"
                    )
                accumulated[start:end] += window_out
                counts[start:end] += 1.0

            return accumulated / counts.unsqueeze(-1).clamp(min=1)

        # reduction == "max"
        # 初始化为极小值，未被任何窗口覆盖的位置将保持为 -inf（理论上不应发生，
        # 因为 split() 保证窗口完整覆盖 [0, original_len)）
        result = torch.full(
            (original_len, *feature_shape),
            float("-inf"),
            device=device,
            dtype=dtype,
        )

        for (start, end), window_out in zip(boundaries, window_outputs):
            expected_len = end - start
            if window_out.size(0) != expected_len:
                raise ValueError(
                    f"窗口输出长度 ({window_out.size(0)}) 与窗口范围 "
                    f"({expected_len}) 不符，start={start}, end={end}"
                )
            result[start:end] = torch.maximum(result[start:end], window_out)

        return result
