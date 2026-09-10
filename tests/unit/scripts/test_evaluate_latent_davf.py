"""Tests for held-out current latent DAVF evaluation."""

from __future__ import annotations

import pytest
import torch

from scripts.evaluate_latent_davf import (
    _resolve_target_decoder_indices,
    compute_endpoint_metrics,
    compute_gene_direction_metrics,
    parse_args,
)


def test_compute_endpoint_metrics_reports_identity_baseline():
    z_0 = torch.zeros((2, 2))
    z_1 = torch.ones((2, 2))
    prediction = z_1.clone()

    metrics = compute_endpoint_metrics(z_0, z_1, prediction)

    assert metrics["endpoint_mse"] == 0.0
    assert metrics["identity_mse"] == 1.0
    assert metrics["relative_to_identity_mse"] == 0.0
    assert metrics["latent_delta_cosine_mean"] == pytest.approx(1.0)
    assert metrics["latent_delta_sign_accuracy"] == pytest.approx(1.0)
    assert metrics["predicted_delta_norm_ratio"] == pytest.approx(1.0)


def test_compute_endpoint_metrics_rejects_zero_identity_baseline():
    z_0 = torch.ones((1, 2))

    with pytest.raises(ValueError, match="zero identity baseline"):
        compute_endpoint_metrics(z_0, z_0, z_0)


def test_compute_gene_direction_metrics_uses_real_observed_delta_sign():
    metrics = compute_gene_direction_metrics(
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 3.0],
        [1.5, 1.5, 3.0],
    )

    assert metrics["gene_endpoint_mse"] == pytest.approx(0.5 / 3.0)
    assert metrics["gene_direction_sign_accuracy"] == pytest.approx(1.0)
    assert metrics["gene_direction_sign_total"] == 2.0
    assert metrics["gene_direction_inconclusive_fraction"] == pytest.approx(1.0 / 3.0)


def test_compute_gene_direction_metrics_rejects_all_inconclusive_observations():
    with pytest.raises(ValueError, match="all inconclusive"):
        compute_gene_direction_metrics([1.0, 2.0], [1.0, 2.0], [1.1, 1.9])


def test_target_decoder_indices_are_resolved_from_adapter_not_token_values():
    class _Adapter:
        def __init__(self):
            self.seen = None

        def resolve_target_gene_indices(self, gene_names):
            self.seen = list(gene_names)
            return [101, 202]

    adapter = _Adapter()
    indices = _resolve_target_decoder_indices(
        torch.tensor([[7, 0], [0, 3]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        token_to_gene={7: "ENSG7", 3: "ENSG3"},
        scvi_adapter=adapter,
    )

    assert indices.tolist() == [101, 202]
    assert adapter.seen == ["ENSG7", "ENSG3"]


def test_target_decoder_indices_reject_multiple_active_tokens():
    with pytest.raises(ValueError, match="exactly one active"):
        _resolve_target_decoder_indices(
            torch.tensor([[7, 3]]),
            torch.tensor([[1.0, 1.0]]),
            token_to_gene={7: "ENSG7", 3: "ENSG3"},
            scvi_adapter=object(),
        )


def test_parse_args_requires_direction_and_test_data():
    args = parse_args(
        [
            "--test-data", "test.npz",
            "--checkpoint", "best_model.pt",
            "--scvi-model", "scvi",
            "--embedding-asset", "asset",
            "--intervention-type", "KD",
            "--output", "report.json",
        ]
    )

    assert args.intervention_type == "KD"
    assert args.batch_size == 2048
    assert args.context_h5ad is None
    assert args.gene_direction_epsilon == 0.0
