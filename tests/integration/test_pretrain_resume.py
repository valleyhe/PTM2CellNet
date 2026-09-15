"""Integration tests: N20 combined/masked pretraining checkpoint resume semantics.

Covers the full interrupt -> resume -> continue loop at the function level,
plus the safe-load contract (new full-state checkpoints must load through
``safe_torch_load`` with ``weights_only=True``).
"""

import os

import torch
from torch.utils.data import DataLoader

from src.training.self_supervised import (
    MaskedPTMPrediction,
    pretrain_combined,
    pretrain_masked_ptm,
)
from src.utils.safe_io import safe_torch_load


def _model_and_loader(seed: int = 0):
    torch.manual_seed(seed)
    model = MaskedPTMPrediction(num_ptm_types=5, embed_dim=12, max_position=16, mask_probability=0.5)
    dataset = [
        {
            "ptm_types": torch.tensor([1, 2, 3, 0], dtype=torch.long) + i % 2,
            "ptm_positions": torch.tensor([0, 1, 2, 3], dtype=torch.long),
            "ptm_mask": torch.tensor([1, 1, 1, 0], dtype=torch.float32),
        }
        for i in range(8)
    ]
    return model, DataLoader(dataset, batch_size=4, shuffle=False)


def test_masked_pretrain_full_loop_with_interrupt_and_resume(tmp_path):
    """中断（首段训练）→ 从 checkpoint 恢复 → 继续，history/状态语义完整。"""
    ckpt_dir = str(tmp_path)

    model, loader = _model_and_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # 首段：2 epochs，结束时必有 checkpoint（epoch 0 首次验证即 improved）
    losses1 = pretrain_masked_ptm(
        model,
        loader,
        optimizer,
        device="cpu",
        epochs=2,
        validation_split=0.5,
        validate_every=1,
        checkpoint_dir=ckpt_dir,
    )
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    assert os.path.exists(ckpt_path)
    state_before = torch.load(ckpt_path, weights_only=False)

    # 新格式必须能通过安全加载契约（weights_only=True）
    loaded = safe_torch_load(ckpt_path, map_location="cpu")
    assert isinstance(loaded, dict) and "optimizer" in loaded

    # 恢复训练：epochs=4 → 总 history 应等于 2（首段）+ 2（续段）
    model2, loader2 = _model_and_loader()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)
    losses2 = pretrain_masked_ptm(
        model2,
        loader2,
        optimizer2,
        device="cpu",
        epochs=4,
        validation_split=0.5,
        validate_every=1,
        checkpoint_dir=ckpt_dir,
        resume_from=ckpt_path,
    )
    assert len(losses2) == 4
    # 恢复的 history 前缀 = checkpoint 中保存的最佳时刻 history
    assert losses2[: len(state_before["history"])] == state_before["history"]

    state_after = torch.load(ckpt_path, weights_only=False)
    assert state_after["epoch"] >= state_before["epoch"]
    assert state_after["best_val_loss"] <= state_before["best_val_loss"] + 1e-9

    # 恢复后的模型可正常推理（forward 契约未破坏）
    with torch.no_grad():
        out = model2(
            torch.tensor([[1, 2, 3, 0]], dtype=torch.long),
            torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            torch.tensor([[1, 1, 1, 0]], dtype=torch.float32),
        )
    assert out["logits"].shape == (1, 4, 5)


def test_combined_pretrain_full_loop_with_interrupt_and_resume(tmp_path):
    """combined 训练中断→恢复：epoch/best_loss/history 语义 + 安全加载。"""
    ckpt_dir = str(tmp_path)

    model, loader = _model_and_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses1 = pretrain_combined(
        model,
        loader,
        optimizer,
        device="cpu",
        epochs=2,
        checkpoint_dir=ckpt_dir,
    )
    ckpt_path = os.path.join(ckpt_dir, "best_combined_model.pt")
    assert os.path.exists(ckpt_path)

    loaded = safe_torch_load(ckpt_path, map_location="cpu")
    assert isinstance(loaded, dict)
    for key in (
        "masked_model",
        "contrastive_module",
        "denoising_module",
        "optimizer",
        "epoch",
        "best_loss",
        "config",
        "history",
    ):
        assert key in loaded

    model2, loader2 = _model_and_loader()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)
    losses2 = pretrain_combined(
        model2,
        loader2,
        optimizer2,
        device="cpu",
        epochs=4,
        checkpoint_dir=ckpt_dir,
        resume_from=ckpt_path,
    )
    assert len(losses2) == 4
    assert losses2[: len(loaded["history"])] == loaded["history"]

    state_after = torch.load(ckpt_path, weights_only=False)
    assert state_after["epoch"] >= 1
    assert state_after["best_loss"] <= loaded["best_loss"] + 1e-9
