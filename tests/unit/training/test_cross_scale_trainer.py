"""Checkpoint and exact-resume tests for the cross-scale trainer."""

import copy
import random

import numpy as np
import torch

from src.data.cross_scale_dataset import CROSS_SCALE_DATA_SCHEMA_VERSION, CrossScaleDataModule
from src.models.cross_scale import CrossScaleConfig, CrossScalePTM2CellNet
from src.training.cross_scale_trainer import CrossScaleTrainer


def _write(path, samples):
    rng = np.random.default_rng(7)
    length, genes = 3, 4
    np.savez(
        path,
        schema_version=np.array(CROSS_SCALE_DATA_SCHEMA_VERSION),
        ankh39_embeddings=rng.normal(size=(samples, length, 2)).astype("float32"),
        esm2_embeddings=rng.normal(size=(samples, length, 3)).astype("float32"),
        prott5_embeddings=rng.normal(size=(samples, length, 4)).astype("float32"),
        signal_edge_index=np.array([[0, 1], [1, 2]], dtype="int64"),
        signal_gene_map=np.ones((length, genes), dtype="float32"),
        cell_edge_index=np.array([[0, 1, 2], [1, 2, 3]], dtype="int64"),
        delta_expression=rng.normal(size=(samples, genes)).astype("float32"),
        cell_state=np.arange(samples, dtype="int64") % 2,
    )


def _model():
    return CrossScalePTM2CellNet(
        config=CrossScaleConfig(
            protein_dim=5,
            signal_input_dim=5,
            signal_hidden_dim=6,
            signal_output_dim=5,
            cell_gene_feature_dim=2,
            cell_hidden_dim=6,
            num_cell_genes=4,
            num_cell_states=2,
            num_ptm_types=2,
            max_position=8,
            dropout=0.0,
        ),
        plm_backbone_dims={"ankh39": 2, "esm2": 3, "prott5": 4},
    )


def _trainer(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return CrossScaleTrainer(model, optimizer, scheduler=scheduler, artifact_context={"test": True})


def test_interrupted_resume_matches_continuous_training(tmp_path):
    for name, samples in (("train", 4), ("val", 2), ("test", 2)):
        _write(tmp_path / f"{name}.npz", samples)
    data = CrossScaleDataModule(
        tmp_path / "train.npz", tmp_path / "val.npz", tmp_path / "test.npz", batch_size=2, seed=11
    )
    torch.manual_seed(5)
    template = _model()
    initial = copy.deepcopy(template.state_dict())

    continuous_model = _model()
    continuous_model.load_state_dict(initial)
    continuous = _trainer(continuous_model)
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    continuous.fit(data.train_dataloader(), data.val_dataloader(), max_epochs=2, output_dir=tmp_path / "continuous")

    interrupted_model = _model()
    interrupted_model.load_state_dict(initial)
    interrupted = _trainer(interrupted_model)
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    interrupted.fit(data.train_dataloader(), data.val_dataloader(), max_epochs=1, output_dir=tmp_path / "interrupted")
    resumed_model = _model()
    resumed = _trainer(resumed_model)
    resumed.load_checkpoint(tmp_path / "interrupted" / "last.pt")
    assert resumed.start_epoch == 1
    resumed.fit(data.train_dataloader(), data.val_dataloader(), max_epochs=2, output_dir=tmp_path / "resumed")

    for key, expected in continuous.model.state_dict().items():
        assert torch.equal(expected, resumed.model.state_dict()[key]), key
