"""Regression tests for the endpoint-aware latent DAVF trainer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from src.integration.perturbgen.donor_split import build_donor_split
from scripts.train_latent_davf import (
    _endpoint_losses,
    _endpoint_velocity_loss,
    _run_epoch,
    _validate_donor_split_metadata,
    parse_args,
)


class _RecordingFlowModel(torch.nn.Module):
    """Small test double that records the times used by the trainer."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.seen_times: list[torch.Tensor | None] = []

    def forward_flow_matching(self, *args, t=None, **kwargs):
        del kwargs
        self.seen_times.append(None if t is None else t.detach().clone())
        z_0, z_1 = args[:2]
        v_t = torch.zeros_like(z_1) + self.scale
        u_t = z_1 - z_0
        loss = (v_t - u_t).square().mean()
        return {"loss": loss, "v_t": v_t, "u_t": u_t}


def _batch() -> dict[str, torch.Tensor]:
    return {
        "z_0": torch.zeros(2, 4),
        "z_1": torch.ones(2, 4),
        "gene_ids": torch.zeros(2, 1, dtype=torch.long),
        "directions": torch.zeros(2, 1, dtype=torch.long),
        "attention_mask": torch.ones(2, 1),
    }


def test_endpoint_velocity_loss_evaluates_exact_t_zero() -> None:
    model = _RecordingFlowModel()

    loss = _endpoint_velocity_loss(model, _batch())

    assert loss.requires_grad
    assert len(model.seen_times) == 1
    assert model.seen_times[0] is not None
    assert torch.equal(model.seen_times[0], torch.zeros(2))


def test_endpoint_losses_include_direction_objective() -> None:
    model = _RecordingFlowModel()

    endpoint_loss, direction_loss = _endpoint_losses(model, _batch())

    assert endpoint_loss.requires_grad
    assert direction_loss.requires_grad
    assert len(model.seen_times) == 1
    assert model.seen_times[0] is not None
    assert torch.equal(model.seen_times[0], torch.zeros(2))


def test_run_epoch_reports_flow_endpoint_and_combined_losses() -> None:
    model = _RecordingFlowModel()
    loader = DataLoader([_batch()], batch_size=2)

    stats = _run_epoch(
        model,
        loader,
        torch.device("cpu"),
        endpoint_loss_weight=2.0,
    )

    assert set(stats) == {"loss", "flow_loss", "endpoint_loss", "direction_loss"}
    assert stats["loss"] == (stats["flow_loss"] + 2.0 * stats["endpoint_loss"] + stats["direction_loss"])
    assert len(model.seen_times) == 2
    assert model.seen_times[0] is not None
    assert torch.equal(model.seen_times[0], torch.full((1,), 0.5))
    assert model.seen_times[1] is not None
    assert torch.equal(model.seen_times[1], torch.zeros(1))


def test_parser_uses_endpoint_loss_by_default() -> None:
    args = parse_args(
        [
            "--train-data",
            "train.npz",
            "--val-data",
            "val.npz",
            "--scvi-model",
            "scvi",
            "--intervention-type",
            "KO",
            "--embedding-asset",
            "asset",
            "--output",
            "output",
        ]
    )

    assert args.endpoint_loss_weight == 1.0
    assert args.direction_loss_weight == 1.0
    assert args.train_donors is None
    assert args.held_out_donors is None
    assert args.require_donor_split is False


def test_parser_accepts_explicit_donor_split_flags() -> None:
    args = parse_args(
        [
            "--train-data",
            "train.npz",
            "--val-data",
            "val.npz",
            "--scvi-model",
            "scvi",
            "--intervention-type",
            "KO",
            "--embedding-asset",
            "asset",
            "--output",
            "output",
            "--train-donors",
            "D1,D2",
            "--held-out-donors",
            "D3,D4,D5",
            "--require-donor-split",
        ]
    )
    assert args.train_donors == "D1,D2"
    assert args.held_out_donors == "D3,D4,D5"
    assert args.require_donor_split is True


def test_donor_split_metadata_requires_train_only_donor_rows() -> None:
    donor_split = build_donor_split(["D1", "D2"], ["D3", "D4", "D5"])

    for dataset_metadata in ({}, {"donor_rows": ["D1", "D3"]}):
        dataset = SimpleNamespace(metadata={"donor_split": donor_split.to_payload(), "dataset": dataset_metadata})
        with pytest.raises(ValueError):
            _validate_donor_split_metadata(dataset, "train", donor_split)
