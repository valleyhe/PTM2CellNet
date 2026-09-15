#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mamba 编码器独立测试脚本

功能: 在不依赖训练流程的情况下，验证 src.models.mamba_encoder 中的
      MambaEncoder 是否可以正常构建并完成一次前向传播。
"""

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径，与项目其他脚本保持一致
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models.mamba_encoder import MambaEncoder


def parse_args(args=None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="MambaEncoder 独立前向测试")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="测试输入的 batch size（默认: 2）",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="测试输入的序列长度（默认: 64）",
    )
    return parser.parse_args(args)


def main() -> None:
    """主函数：构建一个极小的 MambaEncoder 并完成一次随机输入前向传播。"""
    args = parse_args()

    batch_size = args.batch_size
    seq_len = args.seq_len
    vocab_size = 21
    hidden_dim = 64

    print(f"构建 tiny MambaEncoder: vocab_size={vocab_size}, hidden_dim={hidden_dim}")
    model = MambaEncoder(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=2,
        state_dim=8,
        d_conv=4,
        expand_factor=2,
        dropout=0.0,
        max_len=max(seq_len, 128),
    )
    model.eval()

    print(f"生成随机输入: shape=({batch_size}, {seq_len})")
    x = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

    try:
        with torch.no_grad():
            output = model(x)

        expected_shape = (batch_size, seq_len, hidden_dim)
        if output.shape != expected_shape:
            print(f"FAILURE: 输出形状不匹配。期望 {expected_shape}，实际 {tuple(output.shape)}")
            sys.exit(1)

        print(f"SUCCESS: 前向传播完成，输出形状 {tuple(output.shape)}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAILURE: 前向传播失败: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
