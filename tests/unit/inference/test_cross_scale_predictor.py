"""Artifact compatibility tests for offline cross-scale prediction."""

import json

import pytest
import torch

from src.inference.cross_scale_predictor import (
    CrossScaleArtifactError,
    load_cross_scale_artifact,
)
from src.models.cross_scale import CrossScalePTM2CellNet
from src.training.cross_scale_trainer import CrossScaleTrainer


def _artifact(tmp_path):
    config = {
        "cross_scale": {
            "protein_dim": 5,
            "signal_input_dim": 5,
            "signal_hidden_dim": 6,
            "signal_output_dim": 5,
            "cell_gene_feature_dim": 2,
            "cell_hidden_dim": 6,
            "num_cell_genes": 4,
            "num_cell_states": 2,
            "num_ptm_types": 2,
            "max_position": 8,
            "dropout": 0.0,
            "plm_backbone_dims": {"ankh39": 2, "esm2": 3, "prott5": 4},
        },
        "labels": {"cell_states": ["resting", "active"]},
    }
    model = CrossScalePTM2CellNet.from_config(config)
    trainer = CrossScaleTrainer(
        model,
        torch.optim.AdamW(model.parameters()),
        artifact_context={"config": config, "label_vocabulary": ["resting", "active"]},
    )
    trainer.best_val_loss = 1.0
    trainer.save_checkpoint(tmp_path / "best.pt", 0)
    manifest = {
        "artifact_schema_version": "ptm2cellnet.cross-scale.artifact.v1",
        "model_type": "cross_scale",
        "best_checkpoint": "best.pt",
        "config": config,
        "label_vocabulary": ["resting", "active"],
    }
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_load_cross_scale_artifact_roundtrip(tmp_path):
    _artifact(tmp_path)
    model, manifest, checkpoint = load_cross_scale_artifact(tmp_path)

    assert model.get_model_info()["model_class"] == "CrossScalePTM2CellNet"
    assert manifest["label_vocabulary"] == ["resting", "active"]
    assert checkpoint["epoch"] == 0


def test_load_cross_scale_artifact_rejects_manifest_checkpoint_drift(tmp_path):
    manifest = _artifact(tmp_path)
    manifest["config"]["cross_scale"]["propagation_alpha"] = 0.2
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CrossScaleArtifactError, match="config 不一致"):
        load_cross_scale_artifact(tmp_path)


def test_load_cross_scale_artifact_rejects_path_traversal(tmp_path):
    manifest = _artifact(tmp_path)
    manifest["best_checkpoint"] = "../best.pt"
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CrossScaleArtifactError, match="路径穿越"):
        load_cross_scale_artifact(tmp_path)
