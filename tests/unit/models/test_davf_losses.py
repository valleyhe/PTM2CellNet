"""Numerical regression tests for the DAVF training losses (TD-13-14).

These anchor the exact numeric behaviour (magnitude scaling, sign handling,
padding masking, adaptive weight decay) that end-to-end tests only cover
indirectly.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.models.davf_losses import DAVFLoss, DirectionConsistencyLoss


def test_davf_loss_zero_when_prediction_matches_target():
    v_t = torch.tensor([[1.0, -2.0], [0.5, 0.5]])
    loss = DAVFLoss()(v_t, v_t.clone())

    assert float(loss["mse"]) == pytest.approx(0.0, abs=1e-7)
    assert float(loss["mag"]) == pytest.approx(0.0, abs=1e-7)
    assert float(loss["dir"]) == pytest.approx(0.0, abs=1e-6)
    assert float(loss["total"]) == pytest.approx(0.0, abs=1e-6)


def test_davf_loss_exact_values_for_known_inputs():
    v_t = torch.tensor([[2.0, 0.0]])
    u_t = torch.tensor([[1.0, 1.0]])
    loss = DAVFLoss(mse_weight=1.0, mag_weight=0.3)(v_t, u_t)

    # MSE = ((2-1)^2 + (0-1)^2) / 2 = 1.0
    assert float(loss["mse"]) == pytest.approx(1.0, rel=1e-6)
    # |mean(|v|) - mean(|u|)| / mean(|u|) = |1.0 - 1.0| / 1.0 = 0.0
    assert float(loss["mag"]) == pytest.approx(0.0, abs=1e-7)
    # cosine(v, u) = 2 / (2 * sqrt(2)) -> dir = 1 - 1/sqrt(2)
    cosine = 2.0 / (2.0 * math.sqrt(2.0))
    assert float(loss["dir"]) == pytest.approx(1.0 - cosine, rel=1e-6)
    # total = 1*mse + 0.3*mag + 0.1*dir
    assert float(loss["total"]) == pytest.approx(1.0 + 0.1 * (1.0 - cosine), rel=1e-5)


def test_davf_loss_magnitude_term_scales_with_relative_magnitude_error():
    u_t = torch.tensor([[4.0, 4.0]])
    over = DAVFLoss()(torch.tensor([[8.0, 8.0]]), u_t)
    under = DAVFLoss()(torch.tensor([[2.0, 2.0]]), u_t)

    # Both halves of the reference magnitude double / halve -> |1.0| relative error.
    assert float(over["mag"]) == pytest.approx(1.0, rel=1e-5)
    assert float(under["mag"]) == pytest.approx(0.5, rel=1e-5)


def test_davf_loss_adaptive_weights_decay_mse_and_grow_magnitude():
    v_t = torch.tensor([[1.0, -2.0]])
    u_t = torch.tensor([[0.5, 0.5]])
    adaptive = DAVFLoss(mse_weight=1.0, mag_weight=0.3, adaptive_weights=True)
    static = DAVFLoss(mse_weight=1.0, mag_weight=0.3)

    for _ in range(10000):
        adaptive(v_t, u_t)
    static(v_t, u_t)
    final = adaptive(v_t, u_t)

    # After >= 10000 steps decay saturates at 1.0: mse weight 1.0 -> 0.7,
    # mag weight 0.3 -> 0.45; the dir term is unaffected.
    mse, mag, dir_loss = static(v_t, u_t)["mse"], static(v_t, u_t)["mag"], static(v_t, u_t)["dir"]
    assert float(final["total"]) == pytest.approx(
        0.7 * float(mse) + 0.45 * float(mag) + 0.1 * float(dir_loss), rel=1e-5
    )


def test_direction_consistency_loss_rewards_correct_signs_and_penalizes_wrong():
    v_t = torch.tensor([[2.0, -2.0]])
    u_t = torch.zeros_like(v_t)
    gene_ids = torch.tensor([[0, 1]])
    directions = torch.tensor([[2, 0]])  # OE at gene 0 (expect +), KO at gene 1 (expect -)
    loss = DirectionConsistencyLoss()

    correct = loss(v_t, u_t, gene_ids, directions)
    flipped = loss(-v_t, u_t, gene_ids, directions)

    # BCE(logit=+2) < BCE(logit=-2): correct signs must cost strictly less.
    assert float(correct) < float(flipped)
    assert float(correct) == pytest.approx(_bce(2.0), rel=1e-6)
    assert float(flipped) == pytest.approx(_bce(-2.0), rel=1e-6)


def _bce(logit: float) -> float:
    return math.log1p(math.exp(-logit))


def test_direction_consistency_loss_masks_padding_targets():
    v_t = torch.tensor([[3.0, -3.0]])
    u_t = torch.zeros_like(v_t)
    gene_ids = torch.tensor([[0, -1]])  # -1 is padding
    directions = torch.tensor([[2, 2]])
    attention_mask = torch.tensor([[1.0, 0.0]])
    loss = DirectionConsistencyLoss()(v_t, u_t, gene_ids, directions, attention_mask)

    # Only the real target at gene 0 contributes; the padded -1 slot is masked out.
    assert float(loss) == pytest.approx(_bce(3.0), rel=1e-6)


def test_direction_consistency_loss_kd_uses_negative_sign_like_ko():
    v_t = torch.tensor([[1.5]])
    u_t = torch.zeros_like(v_t)
    loss = DirectionConsistencyLoss()(
        v_t,
        u_t,
        torch.tensor([[0]]),
        torch.tensor([[1]]),  # KD
    )
    # KD (direction=1) shares the KO expectation of a negative velocity,
    # so a positive prediction at the target is penalized like KO.
    assert float(loss) == pytest.approx(_bce(-1.5), rel=1e-6)
