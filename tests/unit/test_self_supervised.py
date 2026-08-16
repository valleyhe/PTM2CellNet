"""Unit tests for masked PTM self-supervised pretraining."""

import os

import pytest
import torch
from torch.utils.data import DataLoader

from src.training.self_supervised import (
    MaskedPTMPrediction,
    pretrain_combined,
    pretrain_masked_ptm,
)


def _make_model(num_ptm_types: int = 5, **kwargs):
    torch.manual_seed(0)
    return MaskedPTMPrediction(
        num_ptm_types=num_ptm_types,
        embed_dim=12,
        max_position=16,
        mask_probability=0.5,
        **kwargs,
    )


def _make_dataloader(num_samples: int = 8, batch_size: int = 4):
    dataset = []
    for i in range(num_samples):
        t = torch.tensor([1, 2, 3, 0], dtype=torch.long)
        dataset.append(
            {
                "ptm_types": t + i % 2,
                "ptm_positions": torch.tensor([0, 1, 2, 3], dtype=torch.long),
                "ptm_mask": torch.tensor([1, 1, 1, 0], dtype=torch.float32),
            }
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class TestMaskedPTMPrediction:
    def test_forward_returns_logits_and_predictions(self):
        model = MaskedPTMPrediction(num_ptm_types=6, embed_dim=8, max_position=32)
        masked_ptm_types = torch.tensor([[1, 6, 2, 0], [3, 4, 6, 0]], dtype=torch.long)
        ptm_positions = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
        ptm_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.float32)

        output = model(masked_ptm_types, ptm_positions, ptm_mask)

        assert output["logits"].shape == (2, 4, 6)
        assert output["predictions"].shape == (2, 4)
        assert output["predictions"].dtype == torch.long

    def test_pretrain_masked_ptm_runs_and_returns_epoch_losses(self):
        torch.manual_seed(0)
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        dataloader = _make_dataloader()

        losses = pretrain_masked_ptm(model, dataloader, optimizer, device="cpu", epochs=2)

        assert len(losses) == 2
        assert all(loss >= 0.0 for loss in losses)


class TestMaskedPTMCheckpointResume:
    """N20: masked pretraining checkpoint must carry full training state."""

    def _train_and_checkpoint(self, tmp_path, epochs=2, resume_from=None):
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        dataloader = _make_dataloader()
        ckpt_dir = str(tmp_path)
        losses = pretrain_masked_ptm(
            model,
            dataloader,
            optimizer,
            device="cpu",
            epochs=epochs,
            validation_split=0.5,
            validate_every=1,
            checkpoint_dir=ckpt_dir,
            resume_from=resume_from,
        )
        return model, optimizer, losses

    def test_checkpoint_contains_full_state(self, tmp_path):
        self._train_and_checkpoint(tmp_path, epochs=2)
        ckpt_path = os.path.join(str(tmp_path), "best_model.pt")
        assert os.path.exists(ckpt_path)
        state = torch.load(ckpt_path, weights_only=False)
        assert "model" in state
        assert "optimizer" in state
        assert "epoch" in state
        assert "best_val_loss" in state
        assert "config" in state
        assert "history" in state
        assert state["config"]["epochs"] == 2
        assert len(state["history"]) >= 1
        # epoch 记录的是保存时刻已完成的 epoch（0-based）
        assert 0 <= state["epoch"] < 2

    def test_resume_continues_epoch_counter_and_history(self, tmp_path):
        # 先训练 2 epochs 生成 checkpoint，再 resume 继续 1 个 epoch
        self._train_and_checkpoint(tmp_path, epochs=2)
        ckpt_path = os.path.join(str(tmp_path), "best_model.pt")
        before = torch.load(ckpt_path, weights_only=False)

        model, _, losses = self._train_and_checkpoint(
            tmp_path, epochs=3, resume_from=ckpt_path
        )
        assert len(losses) == 3  # 恢复 history(2) + 新增(1)

        after = torch.load(ckpt_path, weights_only=False)
        # resume 语义：epoch 不倒退、best_val_loss 单调不增
        assert after["epoch"] >= before["epoch"]
        assert after["best_val_loss"] <= before["best_val_loss"] + 1e-9

    def test_resume_restores_optimizer_state(self, tmp_path):
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        dataloader = _make_dataloader()
        ckpt_dir = str(tmp_path)
        pretrain_masked_ptm(
            model, dataloader, optimizer, device="cpu", epochs=2,
            validation_split=0.5, validate_every=1, checkpoint_dir=ckpt_dir,
        )
        ckpt_path = os.path.join(ckpt_dir, "best_model.pt")

        # resume 后 optimizer 应恢复 step 计数（保存时刻 > 0）
        model2 = _make_model()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)
        dataloader2 = _make_dataloader()
        pretrain_masked_ptm(
            model2, dataloader2, optimizer2, device="cpu", epochs=3,
            validation_split=0.5, validate_every=1, checkpoint_dir=ckpt_dir,
            resume_from=ckpt_path,
        )
        assert any(
            pg["step"] > 0 for pg in optimizer2.state.values()
            if isinstance(pg, dict) and pg
        )

    def test_legacy_bare_state_dict_still_loads(self, tmp_path):
        # 旧格式：裸 state_dict（无 dict 包裹），必须兼容且从头继续
        model = _make_model()
        legacy_path = os.path.join(str(tmp_path), "legacy_model.pt")
        torch.save(model.state_dict(), legacy_path)

        model2 = _make_model()
        optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)
        losses = pretrain_masked_ptm(
            model2, _make_dataloader(), optimizer, device="cpu", epochs=2,
            resume_from=legacy_path,
        )
        assert len(losses) == 2  # 从头训练，不报错


class TestCombinedCheckpointResume:
    """N20: combined pretraining checkpoint must carry optimizer/epoch state."""

    def _run(self, tmp_path, epochs=2, resume_from=None, seed=1):
        torch.manual_seed(seed)
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        losses = pretrain_combined(
            model, _make_dataloader(), optimizer, device="cpu", epochs=epochs,
            checkpoint_dir=str(tmp_path), resume_from=resume_from,
        )
        return model, optimizer, losses

    def test_combined_checkpoint_contains_full_state(self, tmp_path):
        self._run(tmp_path, epochs=2)
        ckpt_path = os.path.join(str(tmp_path), "best_combined_model.pt")
        assert os.path.exists(ckpt_path)
        state = torch.load(ckpt_path, weights_only=False)
        for key in ("masked_model", "contrastive_module", "denoising_module",
                    "optimizer", "epoch", "best_loss", "config", "history"):
            assert key in state
        assert state["config"]["epochs"] == 2
        assert 0 <= state["epoch"] < 2

    def test_combined_resume_continues_history_and_epoch(self, tmp_path):
        self._run(tmp_path, epochs=2)
        ckpt_path = os.path.join(str(tmp_path), "best_combined_model.pt")
        before = torch.load(ckpt_path, weights_only=False)

        _, _, losses = self._run(tmp_path, epochs=4, resume_from=ckpt_path)
        assert len(losses) == 4  # 恢复 history + 新增 epoch

        after = torch.load(ckpt_path, weights_only=False)
        assert after["epoch"] >= before["epoch"]
        assert after["best_loss"] <= before["best_loss"] + 1e-9

    def test_combined_resume_restores_optimizer(self, tmp_path):
        self._run(tmp_path, epochs=2)
        ckpt_path = os.path.join(str(tmp_path), "best_combined_model.pt")

        _, optimizer, _ = self._run(tmp_path, epochs=3, resume_from=ckpt_path, seed=2)
        assert any(
            isinstance(pg, dict) and pg and pg["step"] > 0
            for pg in optimizer.state.values()
        )

    def test_combined_legacy_weights_only_checkpoint_warns(self, tmp_path, caplog):
        # 旧格式：仅三模块权重 dict，resume 应告警并继续
        from src.training.self_supervised import (
            PTMContrastiveLearning,
            PTMDenoisingAutoEncoder,
        )

        torch.manual_seed(0)
        model = _make_model()
        legacy_path = os.path.join(str(tmp_path), "legacy_combined.pt")
        # pretrain_combined 以 classifier.out_features 作为模块 embed_dim
        legacy_embed_dim = model.classifier.out_features
        torch.save(
            {
                "masked_model": model.state_dict(),
                "contrastive_module": PTMContrastiveLearning(
                    embed_dim=legacy_embed_dim
                ).state_dict(),
                "denoising_module": PTMDenoisingAutoEncoder(
                    num_ptm_types=5, embed_dim=legacy_embed_dim
                ).state_dict(),
            },
            legacy_path,
        )
        import logging

        with caplog.at_level(logging.WARNING):
            _, _, losses = self._run(tmp_path, epochs=2, resume_from=legacy_path)
        assert len(losses) == 2
        assert "only module weights" in caplog.text

    def test_combined_unrecognized_format_raises(self, tmp_path):
        bad_path = os.path.join(str(tmp_path), "bad.pt")
        torch.save({"unrelated": 1}, bad_path)
        with pytest.raises(ValueError, match="Unrecognized checkpoint format"):
            self._run(tmp_path, epochs=1, resume_from=bad_path)
