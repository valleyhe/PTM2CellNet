"""
数据增强模块单元测试
覆盖: SequenceAugmenter、PTMAugmenter、WeightedRandomSampler、辅助函数
"""

import random

import pytest
import torch

from src.data.augmentation import (
    SequenceAugmenter,
    PTMAugmenter,
    WeightedRandomSampler,
    compute_sample_weights,
    get_augmentation_config,
)


class TestSequenceAugmenter:
    def test_no_augment_when_prob_zero(self):
        augmenter = SequenceAugmenter(augment_prob=0.0)
        seq = torch.arange(1, 21, dtype=torch.long)
        out = augmenter(seq)
        assert torch.equal(out, seq)

    def test_random_truncate_reduces_length(self):
        random.seed(42)
        augmenter = SequenceAugmenter(augment_prob=1.0, max_truncate_ratio=0.2)
        seq = torch.arange(1, 21, dtype=torch.long)
        out = augmenter(seq)
        seq_len = (seq != 0).sum().item()
        out_len = (out != 0).sum().item()
        assert out_len <= seq_len
        assert out_len >= int(seq_len * (1 - 0.2))

    def test_random_mask_changes_tokens(self):
        random.seed(42)
        augmenter = SequenceAugmenter(augment_prob=1.0, mask_prob=0.5, mask_token_id=99)
        seq = torch.ones(20, dtype=torch.long)
        out = augmenter(seq)
        assert (out == 99).any()

    def test_random_swap_changes_order(self):
        random.seed(42)
        augmenter = SequenceAugmenter(augment_prob=1.0, random_swap_prob=1.0)
        seq = torch.arange(1, 21, dtype=torch.long)
        out = augmenter(seq)
        assert out.shape == seq.shape
        assert not torch.equal(out, seq)


class TestPTMAugmenter:
    def test_ptm_augmenter_adds_noise(self):
        torch.manual_seed(42)
        random.seed(42)
        augmenter = PTMAugmenter(add_noise_prob=1.0, noise_radius=1)
        mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
        types = torch.tensor([1, 0, 2, 0])
        out_mask, out_types = augmenter(mask, types)
        # add_noise_prob path adds new positive positions near existing ones
        assert out_mask.sum().item() >= mask.sum().item()

    def test_ptm_augmenter_dropout(self):
        torch.manual_seed(42)
        augmenter = PTMAugmenter(drop_prob=1.0)
        mask = torch.tensor([1.0, 1.0, 1.0, 0.0])
        types = torch.tensor([1, 2, 3, 0])
        out_mask, out_types = augmenter(mask, types)
        assert (out_mask == 0).any()


class TestWeightedRandomSampler:
    def test_sampler_length(self):
        sampler = WeightedRandomSampler(labels=[0, 0, 1, 1], num_samples=4)
        assert len(sampler) == 4

    def test_sampler_iterates(self):
        sampler = WeightedRandomSampler(labels=[0, 0, 1, 1], num_samples=4)
        indices = list(sampler)
        assert len(indices) == 4
        assert all(isinstance(i, int) for i in indices)


class TestHelpers:
    def test_compute_sample_weights_balanced(self):
        weights = compute_sample_weights([0, 0, 0, 1], mode="balanced")
        assert weights[3].item() == pytest.approx(weights[0].item() * 3, rel=1e-3)

    def test_get_augmentation_config_light(self):
        config = get_augmentation_config("light")
        assert "sequence_augment_prob" in config

    def test_get_augmentation_config_unknown(self):
        config = get_augmentation_config("unknown_type")
        assert isinstance(config, dict)
        assert "sequence_augment_prob" in config
