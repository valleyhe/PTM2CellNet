"""Tests for the current latent-pair training data contract."""

from __future__ import annotations

import json
import numpy as np
import pytest
import torch

from src.data.latent_davf_dataset import LatentDAVFDataError, load_latent_davf_pairs


def _write_valid(path, **overrides):
    metadata = {
        "schema_version": "ptm2cellnet.latent-davf-pairs.v1",
        "scvi": {
            "model_path": "checkpoints/scvi/model",
            "latent_dim": 64,
            "num_genes": 4018,
            "gene_names": [f"gene_{index}" for index in range(4018)],
        },
        "embedding_asset": {
            "path": "outputs/perturbgen/embedding_asset",
            "vocab_size": 10,
            "embedding_dim": 768,
            "manifest": {"schema_version": 1},
        },
        "dataset": {
            "intervention_type": "KD",
            "direction_code": 1,
        },
    }
    values = {
        "metadata_json": np.asarray(json.dumps(metadata)),
        "z_0": np.zeros((3, 64), dtype=np.float32),
        "z_1": np.ones((3, 64), dtype=np.float32),
        "gene_ids": np.asarray([[0, 1], [1, 0], [0, 0]], dtype=np.int64),
        "directions": np.asarray([[2, 0], [0, 1], [2, 2]], dtype=np.int64),
        "attention_mask": np.asarray([[1, 1], [1, 0], [1, 1]], dtype=np.float32),
    }
    values.update(overrides)
    np.savez(path, **values)


def test_loads_valid_latent_pairs(tmp_path):
    path = tmp_path / "pairs.npz"
    _write_valid(path, magnitudes=np.ones((3, 2), dtype=np.float32))

    dataset = load_latent_davf_pairs(
        path,
        latent_dim=64,
        perturbgen_vocab_size=10,
    )

    assert len(dataset) == 3
    item = dataset[0]
    assert item["z_0"].shape == (64,)
    assert item["gene_ids"].dtype == torch.int64
    assert item["magnitudes"].shape == (2,)
    assert dataset.metadata["schema_version"] == "ptm2cellnet.latent-davf-pairs.v1"


def test_rejects_missing_provenance_metadata(tmp_path):
    path = tmp_path / "missing_metadata.npz"
    _write_valid(path)
    with np.load(path) as arrays:
        values = {key: arrays[key] for key in arrays.files if key != "metadata_json"}
    np.savez(path, **values)

    with pytest.raises(LatentDAVFDataError, match="metadata_json"):
        load_latent_davf_pairs(path, latent_dim=64, perturbgen_vocab_size=10)


def test_rejects_scvi_provenance_mismatch(tmp_path):
    path = tmp_path / "wrong_scvi.npz"
    _write_valid(path)

    with pytest.raises(LatentDAVFDataError, match="scVI model path"):
        load_latent_davf_pairs(
            path,
            latent_dim=64,
            perturbgen_vocab_size=10,
            expected_scvi_model_path="checkpoints/scvi/other-model",
        )


def test_rejects_embedding_manifest_mismatch(tmp_path):
    path = tmp_path / "wrong_asset.npz"
    _write_valid(path)

    with pytest.raises(LatentDAVFDataError, match="PerturbGen manifest"):
        load_latent_davf_pairs(
            path,
            latent_dim=64,
            perturbgen_vocab_size=10,
            expected_embedding_manifest={"schema_version": 999},
        )


def test_rejects_missing_required_array(tmp_path):
    path = tmp_path / "missing.npz"
    _write_valid(path)
    with np.load(path) as arrays:
        values = {key: arrays[key] for key in arrays.files if key != "z_1"}
    np.savez(path, **values)

    with pytest.raises(LatentDAVFDataError, match="z_1"):
        load_latent_davf_pairs(path, latent_dim=64, perturbgen_vocab_size=10)


def test_rejects_gene_id_outside_perturbgen_vocabulary(tmp_path):
    path = tmp_path / "bad_token.npz"
    _write_valid(path, gene_ids=np.asarray([[10, 0], [1, 0], [0, 0]], dtype=np.int64))

    with pytest.raises(LatentDAVFDataError, match="PerturbGen vocabulary"):
        load_latent_davf_pairs(path, latent_dim=64, perturbgen_vocab_size=10)


def test_rejects_gene_space_delta_shape_as_latent_input(tmp_path):
    path = tmp_path / "gene_delta.npz"
    _write_valid(path, z_0=np.zeros((3, 4018), dtype=np.float32))

    with pytest.raises(LatentDAVFDataError, match=r"\[N, 64\]"):
        load_latent_davf_pairs(path, latent_dim=64, perturbgen_vocab_size=10)


def test_rejects_non_binary_attention_mask(tmp_path):
    path = tmp_path / "bad_mask.npz"
    _write_valid(path, attention_mask=np.full((3, 2), 0.5, dtype=np.float32))

    with pytest.raises(LatentDAVFDataError, match="0/1"):
        load_latent_davf_pairs(path, latent_dim=64, perturbgen_vocab_size=10)


def test_rejects_samples_without_an_active_perturbation_target(tmp_path):
    path = tmp_path / "empty_target.npz"
    _write_valid(path, attention_mask=np.asarray([[0, 0], [1, 0], [1, 1]], dtype=np.float32))

    with pytest.raises(LatentDAVFDataError, match="active perturbation target"):
        load_latent_davf_pairs(path, latent_dim=64, perturbgen_vocab_size=10)


def test_rejects_pair_file_for_a_different_intervention_type(tmp_path):
    path = tmp_path / "wrong_direction.npz"
    _write_valid(path)
    with pytest.raises(LatentDAVFDataError, match="intervention type"):
        load_latent_davf_pairs(
            path,
            latent_dim=64,
            perturbgen_vocab_size=10,
            expected_intervention_type="KO",
        )


def test_accepts_the_declared_intervention_type(tmp_path):
    path = tmp_path / "kd.npz"
    _write_valid(path)
    dataset = load_latent_davf_pairs(
        path,
        latent_dim=64,
        perturbgen_vocab_size=10,
        expected_intervention_type="KD",
    )
    assert dataset.metadata["dataset"]["direction_code"] == 1
