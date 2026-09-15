"""Unit tests for scripts/finetune_davf.py (V22-01 DAVF fine-tuning).

Tests the script's helpers, argument parsing, and the two-stage schedule
plumbing without running a full training loop (which would require a
real model checkpoint and data).
"""

import os
import sys

import pytest
import torch
from torch import nn

# Ensure project root is importable
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import scripts.finetune_davf as finetune_davf
from src.models.davf_inference import DAVFInferenceModule


class _DummyBackbone(nn.Module):
    """Minimal model with an embedded DAVFInferenceModule for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 3)
        # Embed a real DAVFInferenceModule so _find_davf_module works
        from src.models.davf_inference import DAVFInferenceConfig

        cfg = DAVFInferenceConfig(
            checkpoint_path="nonexistent.pt",  # graceful zero-feature fallback
            freeze=True,
        )
        self.davf = DAVFInferenceModule(cfg)


class TestArgparse:
    def test_parse_args_requires_data(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["finetune_davf.py", "--data", "x.csv"])
        args = finetune_davf.parse_args()
        assert args.data == "x.csv"
        assert args.stage1_epochs == 5
        assert args.stage2_epochs == 15
        assert args.stage1_lr == 1e-3
        assert args.stage2_lr == 1e-5

    def test_parse_args_custom_values(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "finetune_davf.py",
                "--data",
                "d.csv",
                "--stage1-epochs",
                "3",
                "--stage2-epochs",
                "7",
                "--stage2-lr",
                "2e-5",
            ],
        )
        args = finetune_davf.parse_args()
        assert args.stage1_epochs == 3
        assert args.stage2_epochs == 7
        assert args.stage2_lr == 2e-5


class TestHelpers:
    def test_count_trainable(self):
        m = nn.Linear(5, 3)
        n = finetune_davf._count_trainable(m)
        assert n == 5 * 3 + 3

    def test_count_trainable_with_frozen(self):
        m = nn.Linear(5, 3)
        for p in m.parameters():
            p.requires_grad = False
        assert finetune_davf._count_trainable(m) == 0

    def test_find_davf_module(self):
        backbone = _DummyBackbone()
        found = finetune_davf._find_davf_module(backbone)
        assert isinstance(found, DAVFInferenceModule)

    def test_find_davf_module_none(self):
        assert finetune_davf._find_davf_module(nn.Linear(3, 2)) is None

    def test_configure_optimizer_only_trainable(self):
        m = nn.Linear(5, 3)
        for p in m.parameters():
            p.requires_grad = False
        # All-frozen model raises a clear error (no empty optimizer)
        with pytest.raises(RuntimeError, match="No trainable parameters"):
            finetune_davf.configure_optimizer(m, lr=1e-3, weight_decay=0.0)

    def test_configure_optimizer_normal(self):
        m = nn.Linear(5, 3)
        opt = finetune_davf.configure_optimizer(m, lr=1e-3, weight_decay=0.0)
        assert opt is not None
        assert len(opt.param_groups) == 1

    def test_move_batch_to_device(self):
        device = torch.device("cpu")
        batch = {"x": torch.zeros(2, 3), "s": "keep", "n": 5}
        moved = finetune_davf._move_batch_to_device(batch, device)
        assert moved["s"] == "keep"
        assert moved["n"] == 5
        assert moved["x"].device == device


class TestStageScheduling:
    """Verify freeze/unfreeze is invoked during the two-stage schedule."""

    def test_stage1_freezes_then_stage2_unfreezes(self):
        """The two-stage flow calls freeze_davf then unfreeze_davf (D-15/D-16)."""
        backbone = _DummyBackbone()
        davf = backbone.davf

        # Stage 1: freeze
        davf.unfreeze_davf()  # start unfrozen
        assert all(p.requires_grad for p in davf.parameters())
        davf.freeze_davf()
        assert all(not p.requires_grad for p in davf.latent_davf.parameters())
        # DeltaProjection stays trainable (D-17)
        assert any(p.requires_grad for p in davf.delta_projection.parameters())

        # Stage 2: unfreeze
        davf.unfreeze_davf()
        assert all(p.requires_grad for p in davf.parameters())
