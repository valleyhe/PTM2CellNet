"""
Focused architecture integration tests for Phase 13 DAVF cascade fusion.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.api.schemas import PTMSite
from src.models.architectures import PTM2CellNet


def make_batch(batch_size: int = 2, seq_len: int = 32, device: torch.device | str = "cpu") -> dict:
    device = torch.device(device)
    return {
        "sequence": torch.randint(0, 20, (batch_size, seq_len), device=device),
        "ptm_types": torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
    }


def make_davf_inputs(batch_size: int = 2) -> tuple[list[list[PTMSite]], list[list[str]]]:
    ptm_site = PTMSite(position=7, type="phosphorylation", residue="S")
    return (
        [[ptm_site] for _ in range(batch_size)],
        [["TP53"] for _ in range(batch_size)],
    )


def make_config(use_davf: bool = True, davf_config: dict | None = None) -> dict:
    config = {
        "model": {
            "encoder_type": "transformer",
            "hidden_dim": 128,
            "num_classes": 4,
            "use_davf": use_davf,
        },
        "data": {},
    }
    if davf_config is not None:
        config["model"]["davf_config"] = davf_config
    return config


class TestDAVFArchitectureIntegration:
    def test_backward_compatibility_use_davf_false(self):
        model = PTM2CellNet(
            encoder_type="transformer",
            embed_dim=128,
            num_classes=4,
            use_davf=False,
        )

        assert model.davf_module is None
        assert model.ptm_mapper is None
        assert model.davf_feature_dim == 0

        output = model(make_batch())
        assert output["logits"].shape == (2, 4)

    def test_davf_enabled_creates_modules(self):
        model = PTM2CellNet(
            encoder_type="transformer",
            embed_dim=128,
            num_classes=4,
            use_davf=True,
        )

        assert model.ptm_mapper is not None
        assert model.davf_module is not None
        assert model.davf_feature_dim == 128

    def test_davf_expands_predictor_input_dim(self):
        model = PTM2CellNet(
            encoder_type="transformer",
            embed_dim=128,
            num_classes=4,
            use_davf=True,
        )

        first_linear = model.predictor.mlp[0]
        assert first_linear.in_features == 256

    def test_davf_config_from_dict(self):
        model = PTM2CellNet(
            encoder_type="transformer",
            embed_dim=128,
            num_classes=4,
            use_davf=True,
            davf_config={
                "feature_dim": 64,
                "scvi_model_path": "checkpoints/scvi/model.pt",
                "geneformer_path": "checkpoints/geneformer_embeddings.pt",
            },
        )

        assert model.davf_feature_dim == 64
        assert model.davf_module.config.feature_dim == 64
        assert model.davf_config["scvi_model_path"] == "checkpoints/scvi/model.pt"
        assert model.davf_config["geneformer_path"] == "checkpoints/geneformer_embeddings.pt"

    def test_forward_with_davf_sites(self):
        model = PTM2CellNet(
            encoder_type="transformer",
            embed_dim=128,
            num_classes=4,
            use_davf=True,
        )

        batch = make_batch()
        davf_sites, davf_gene_names = make_davf_inputs()
        batch["davf_sites"] = davf_sites
        batch["davf_gene_names"] = davf_gene_names

        output = model(batch)
        assert output["logits"].shape == (2, 4)

    def test_forward_without_davf_sites_uses_zero_features(self):
        model = PTM2CellNet(
            encoder_type="transformer",
            embed_dim=128,
            num_classes=4,
            use_davf=True,
        )

        output = model(make_batch())
        assert output["logits"].shape == (2, 4)

    def test_missing_checkpoint_uses_zero_features(self):
        model = PTM2CellNet(
            encoder_type="transformer",
            embed_dim=128,
            num_classes=4,
            use_davf=True,
            davf_config={"checkpoint_path": "missing/best_model.pt"},
        )

        assert model.davf_module is not None
        assert model.davf_module._checkpoint_loaded is False

        batch = make_batch()
        davf_sites, davf_gene_names = make_davf_inputs()
        batch["davf_sites"] = davf_sites
        batch["davf_gene_names"] = davf_gene_names

        output = model(batch)
        assert output["logits"].shape == (2, 4)

    def test_empty_ptm_sites_use_zero_features(self):
        model = PTM2CellNet(
            encoder_type="transformer",
            embed_dim=128,
            num_classes=4,
            use_davf=True,
        )

        batch = make_batch()
        batch["davf_sites"] = [[], []]
        batch["davf_gene_names"] = [[], []]
        output = model(batch)
        assert output["logits"].shape == (2, 4)

    def test_davf_with_multitask_raises(self):
        with pytest.raises(ValueError, match="DAVF integration not supported with multitask mode"):
            PTM2CellNet(
                encoder_type="transformer",
                embed_dim=128,
                use_davf=True,
                task_type="multitask",
                multitask_configs=[{"name": "task1", "type": "classification", "num_classes": 2}],
            )

    def test_from_config_with_davf(self):
        config = make_config(
            use_davf=True,
            davf_config={
                "feature_dim": 64,
                "scvi_model_path": "checkpoints/scvi/model.pt",
                "geneformer_path": "checkpoints/geneformer_embeddings.pt",
            },
        )
        model = PTM2CellNet.from_config(config)

        assert model.use_davf is True
        assert model.davf_module.config.feature_dim == 64
        assert model.davf_config["scvi_model_path"] == "checkpoints/scvi/model.pt"
        assert model.davf_config["geneformer_path"] == "checkpoints/geneformer_embeddings.pt"

    def test_from_config_without_davf(self):
        model = PTM2CellNet.from_config(make_config(use_davf=False))
        assert model.use_davf is False
        assert model.davf_module is None

    def test_from_config_with_yaml_defaults(self):
        model = PTM2CellNet.from_config(make_config(use_davf=True))

        assert model.davf_module.config.checkpoint_path == "checkpoints/latent_davf_ibd_norman/best_model.pt"
        assert model.davf_module.config.feature_dim == 128
        assert model.davf_module.config.num_steps == 50
        assert "scvi_model_path" in model.davf_config
        assert "geneformer_path" in model.davf_config

    def test_get_model_info_includes_davf(self):
        model = PTM2CellNet(
            encoder_type="transformer",
            embed_dim=128,
            num_classes=4,
            use_davf=True,
        )

        info = model.get_model_info()
        assert info["use_davf"] is True
        assert info["davf_feature_dim"] == 128
        assert "davf_config" in info

    def test_device_consistency(self, monkeypatch):
        model = PTM2CellNet(
            encoder_type="transformer",
            embed_dim=128,
            num_classes=4,
            use_davf=True,
        )

        cpu_batch = make_batch(device="cpu")
        cpu_sites, cpu_gene_names = make_davf_inputs()
        cpu_batch["davf_sites"] = cpu_sites
        cpu_batch["davf_gene_names"] = cpu_gene_names

        cpu_output = model(cpu_batch)
        assert cpu_output["logits"].device == cpu_batch["sequence"].device

        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable for DAVF device_consistency GPU assertion")

        model = model.to("cuda")
        captured: dict[str, torch.device] = {}
        original_forward = model.davf_module.forward

        def capture_forward(mapper_output, z_0=None):
            output = original_forward(mapper_output, z_0=z_0)
            captured["davf_device"] = output.davf_features.device
            return output

        monkeypatch.setattr(model.davf_module, "forward", capture_forward)

        gpu_batch = make_batch(device="cuda")
        davf_sites, davf_gene_names = make_davf_inputs()
        gpu_batch["davf_sites"] = davf_sites
        gpu_batch["davf_gene_names"] = davf_gene_names
        output = model(gpu_batch)

        assert captured["davf_device"] == output["logits"].device


def test_yaml_defaults_file_exists():
    path = Path("configs/davf_integration.yaml")
    assert path.exists()
