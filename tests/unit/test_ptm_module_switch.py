"""S0 消融开关测试：use_ptm_module=False 时模型结构与前向行为。

覆盖：
1. use_ptm_module=False：ptm_module 为 None，前向跳过融合，输出形状不变；
2. 默认 True：行为与历史一致（ptm_module 存在且前向可跑）；
3. from_config 路径：config model.use_ptm_module=false 正确传递；
4. 回归：False 模式下 DAVF 分支（零特征）仍可共存。
"""

import torch

from src.models.architectures import PTM2CellNet


def _make_batch(batch_size: int = 2, seq_len: int = 16, num_classes: int = 4):
    return {
        "sequence": torch.randint(1, 21, (batch_size, seq_len)),
        "ptm_types": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "ptm_positions": (torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)),
        "ptm_mask": torch.zeros(batch_size, seq_len),
        "label": torch.randint(0, num_classes, (batch_size,)),
    }


def test_use_ptm_module_false_skips_module():
    model = PTM2CellNet(
        encoder_type="cnn",
        embed_dim=32,
        num_classes=4,
        use_ptm_module=False,
    )
    assert model.ptm_module is None
    out = model(_make_batch())
    assert out["logits"].shape == (2, 4)
    assert torch.isfinite(out["logits"]).all()


def test_use_ptm_module_true_default():
    model = PTM2CellNet(encoder_type="cnn", embed_dim=32, num_classes=4)
    assert model.use_ptm_module is True
    assert model.ptm_module is not None
    out = model(_make_batch())
    assert out["logits"].shape == (2, 4)


def test_from_config_use_ptm_module():
    model = PTM2CellNet.from_config(
        {
            "model": {
                "encoder_type": "cnn",
                "hidden_dim": 32,
                "num_classes": 4,
                "use_ptm_module": False,
            },
            "data": {"max_sequence_length": 64},
        }
    )
    assert model.use_ptm_module is False
    assert model.ptm_module is None
    out = model(_make_batch())
    assert out["logits"].shape == (2, 4)


def test_use_ptm_module_false_with_davf_zero_features():
    model = PTM2CellNet(
        encoder_type="cnn",
        embed_dim=32,
        num_classes=4,
        use_ptm_module=False,
        use_davf=True,
        davf_config={
            "checkpoint_path": "checkpoints/does_not_exist.pt",
            "feature_dim": 16,
            "hidden_dim": 32,
            "freeze": True,
        },
    )
    # 无 davf_sites → DAVF 特征零回退，前向仍应可用（S2 消融路径）
    out = model(_make_batch())
    assert out["logits"].shape == (2, 4)
    assert torch.isfinite(out["logits"]).all()
