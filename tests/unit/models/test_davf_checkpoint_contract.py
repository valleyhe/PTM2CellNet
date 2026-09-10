"""Tests for the strict current DAVF checkpoint contract."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from src.models.davf_checkpoint_contract import (
    DAVFCheckpointContractError,
    build_current_davf_checkpoint,
    load_current_davf_checkpoint,
)
from src.models.latent_davf import LatentDAVF, LatentDAVFConfig
from src.models.perturbgen_embedding import load_perturbgen_embedding_asset


def _asset(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    tensor_path = tmp_path / "gene_embeddings.safetensors"
    vocab_path = tmp_path / "vocabulary.json"
    matrix = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    save_file({"gene_embeddings": matrix}, str(tensor_path))
    vocab_path.write_text(json.dumps({"ENSG000001": 0, "ENSG000002": 1}), encoding="utf-8")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "embedding_shape": [2, 3],
                "files": {
                    tensor_path.name: {"sha256": sha(tensor_path)},
                    vocab_path.name: {"sha256": sha(vocab_path)},
                },
            }
        ),
        encoding="utf-8",
    )
    return load_perturbgen_embedding_asset(tmp_path)


class _ScVIAdapter:
    n_latent = 64
    n_genes = 4018
    gene_names = tuple(f"gene_{index}" for index in range(4018))

    def validate_compatibility(self, *, expected_latent_dim, expected_num_genes, expected_gene_names=None):
        assert expected_latent_dim == self.n_latent
        assert expected_num_genes == self.n_genes
        if expected_gene_names is not None:
            assert tuple(expected_gene_names) == self.gene_names


def _model(asset):
    config = LatentDAVFConfig(latent_dim=64, num_genes=4018)
    return LatentDAVF(config, pretrained_gene_embeddings=asset.embeddings)


def test_build_and_safe_reload_current_checkpoint(tmp_path):
    asset_dir = tmp_path / "asset"
    asset = _asset(asset_dir)
    model = _model(asset)
    adapter = _ScVIAdapter()
    payload = build_current_davf_checkpoint(
        model,
        asset=asset,
        embedding_asset_path=asset_dir,
        scvi_model_path=tmp_path / "scvi",
        scvi_adapter=adapter,
        training={"epoch": 1},
    )
    checkpoint = tmp_path / "best_model.pt"
    torch.save(payload, checkpoint)

    loaded, config = load_current_davf_checkpoint(
        checkpoint,
        asset=asset,
        scvi_adapter=adapter,
    )

    assert loaded["schema_version"] == 2
    assert loaded["model_type"] == "LatentDAVF"
    assert config.latent_dim == 64
    assert config.num_genes == 4018


def test_unversioned_legacy_state_dict_is_not_formal(tmp_path):
    asset = _asset(tmp_path / "asset")
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {"delta_mlp.0.weight": torch.zeros((512, 320))},
        checkpoint,
    )

    with pytest.raises(DAVFCheckpointContractError, match="schema_version=2"):
        load_current_davf_checkpoint(checkpoint, asset=asset, scvi_adapter=_ScVIAdapter())


def test_schema_v2_requires_an_exact_current_state_dict(tmp_path):
    asset_dir = tmp_path / "asset"
    asset = _asset(asset_dir)
    model = _model(asset)
    payload = build_current_davf_checkpoint(
        model,
        asset=asset,
        embedding_asset_path=asset_dir,
        scvi_model_path=tmp_path / "scvi",
        scvi_adapter=_ScVIAdapter(),
        training={"epoch": 1},
    )
    payload["model_state_dict"].pop("velocity_field.velocity_net.0.weight")

    with pytest.raises(DAVFCheckpointContractError, match="exact current LatentDAVF"):
        from src.models.davf_checkpoint_contract import validate_current_davf_checkpoint_payload

        validate_current_davf_checkpoint_payload(payload, asset=asset)


def test_schema_v2_rejects_non_finite_weights(tmp_path):
    asset_dir = tmp_path / "asset"
    asset = _asset(asset_dir)
    model = _model(asset)
    payload = build_current_davf_checkpoint(
        model,
        asset=asset,
        embedding_asset_path=asset_dir,
        scvi_model_path=tmp_path / "scvi",
        scvi_adapter=_ScVIAdapter(),
        training={"epoch": 1},
    )
    payload["model_state_dict"]["velocity_field.velocity_net.0.weight"][0, 0] = float("nan")

    with pytest.raises(DAVFCheckpointContractError, match="non-finite"):
        from src.models.davf_checkpoint_contract import validate_current_davf_checkpoint_payload

        validate_current_davf_checkpoint_payload(payload, asset=asset)


def test_schema_v2_rejects_updated_perturbgen_embedding_table(tmp_path):
    asset_dir = tmp_path / "asset"
    asset = _asset(asset_dir)
    model = _model(asset)
    payload = build_current_davf_checkpoint(
        model,
        asset=asset,
        embedding_asset_path=asset_dir,
        scvi_model_path=tmp_path / "scvi",
        scvi_adapter=_ScVIAdapter(),
        training={"epoch": 1},
    )
    payload["model_state_dict"]["gene_embed_table.weight"][0, 0] += 1.0

    with pytest.raises(DAVFCheckpointContractError, match="remain frozen"):
        from src.models.davf_checkpoint_contract import validate_current_davf_checkpoint_payload

        validate_current_davf_checkpoint_payload(payload, asset=asset)


def test_schema_v2_rejects_mismatched_training_direction_metadata(tmp_path):
    asset_dir = tmp_path / "asset"
    asset = _asset(asset_dir)
    model = _model(asset)
    payload = build_current_davf_checkpoint(
        model,
        asset=asset,
        embedding_asset_path=asset_dir,
        scvi_model_path=tmp_path / "scvi",
        scvi_adapter=_ScVIAdapter(),
        training={"intervention_type": "KO", "direction_code": 0},
    )
    payload["training"]["direction_code"] = 1

    with pytest.raises(DAVFCheckpointContractError, match="direction_code"):
        from src.models.davf_checkpoint_contract import validate_current_davf_checkpoint_payload

        validate_current_davf_checkpoint_payload(payload, asset=asset)


def test_schema_v2_rejects_checkpoint_when_expected_direction_differs(tmp_path):
    asset_dir = tmp_path / "asset"
    asset = _asset(asset_dir)
    model = _model(asset)
    payload = build_current_davf_checkpoint(
        model,
        asset=asset,
        embedding_asset_path=asset_dir,
        scvi_model_path=tmp_path / "scvi",
        scvi_adapter=_ScVIAdapter(),
        training={"intervention_type": "KO", "direction_code": 0},
    )

    with pytest.raises(DAVFCheckpointContractError, match="requested DAVF training direction"):
        from src.models.davf_checkpoint_contract import validate_current_davf_checkpoint_payload

        validate_current_davf_checkpoint_payload(
            payload,
            asset=asset,
            expected_intervention_type="KD",
        )


def test_inference_module_marks_only_schema_v2_as_formal(tmp_path):
    asset_dir = tmp_path / "asset"
    asset = _asset(asset_dir)
    model = _model(asset)
    payload = build_current_davf_checkpoint(
        model,
        asset=asset,
        embedding_asset_path=asset_dir,
        scvi_model_path=tmp_path / "scvi",
        scvi_adapter=_ScVIAdapter(),
        training={"epoch": 1},
    )
    checkpoint = tmp_path / "current.pt"
    torch.save(payload, checkpoint)

    from src.models.davf_inference import DAVFInferenceConfig, DAVFInferenceModule

    module = DAVFInferenceModule(
        DAVFInferenceConfig(
            checkpoint_path=str(checkpoint),
            embedding_asset_path=str(asset_dir),
            latent_dim=64,
            num_genes=4018,
            device="cpu",
        )
    )

    assert module._checkpoint_loaded is True
    assert module.current_checkpoint_contract_valid is True
