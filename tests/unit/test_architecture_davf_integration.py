"""
Focused architecture integration tests for Phase 13 DAVF cascade fusion.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.api.schemas import PTMSite
from src.models.architectures import PTM2CellNet


@pytest.fixture(autouse=True)
def _offline_gene_resolution(monkeypatch):
    """Neutralize UniProt network calls so tests are deterministic & offline.

    ``PTMDirectionMapper._resolve_gene_id`` falls back to a live UniProt
    ``GeneMapper`` lookup whenever a gene symbol (e.g. ``"TP53"``) is absent
    from the integer-indexed Geneformer vocabulary. In CI / offline runs that
    lookup hangs on network timeouts. These tests only assert output *shapes*,
    so masking the gene out (returning ``(0, 0)``) is equivalent and keeps the
    suite green without network access.
    """
    import src.analysis.gene_mapper as _gm

    monkeypatch.setattr(_gm.GeneMapper, "map_gene_to_uniprot", lambda self, gene, **kw: None)
    monkeypatch.setattr(_gm, "map_gene_to_uniprot", lambda gene, **kw: None)


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


def make_embedding_asset(tmp_path) -> str:
    """Build a minimal verified PerturbGen embedding asset directory.

    Schema v2 DAVF configs load the asset at construction time and fail fast
    when it is missing, so transparency tests need a real (tiny) asset.
    """
    import hashlib
    import json

    from safetensors.torch import save_file as save_safetensors

    asset_dir = tmp_path / "embedding_asset"
    asset_dir.mkdir()
    tensor_path = asset_dir / "gene_embeddings.safetensors"
    vocab_path = asset_dir / "vocabulary.json"
    matrix = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    save_safetensors({"gene_embeddings": matrix}, str(tensor_path))
    vocab_path.write_text(json.dumps({f"ENSG00000{i}": i for i in range(4)}), encoding="utf-8")

    def _sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (asset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "embedding_shape": [4, 2],
                "files": {
                    tensor_path.name: {"sha256": _sha(tensor_path)},
                    vocab_path.name: {"sha256": _sha(vocab_path)},
                },
            }
        ),
        encoding="utf-8",
    )
    return str(asset_dir)


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
        assert model.davf_path_class == "legacy_fusion_demo"

    def test_production_use_davf_without_asset_fails(self, monkeypatch):
        monkeypatch.setenv("PTM2CELLNET_ENV", "production")
        with pytest.raises(ValueError, match="embedding_asset_path"):
            PTM2CellNet(
                encoder_type="transformer",
                embed_dim=128,
                num_classes=4,
                use_davf=True,
            )

    def test_davf_expands_predictor_input_dim(self):
        model = PTM2CellNet(
            encoder_type="transformer",
            embed_dim=128,
            num_classes=4,
            use_davf=True,
        )

        first_linear = model.predictor.mlp[0]
        assert first_linear.in_features == 256

    def test_davf_config_from_dict(self, tmp_path):
        asset_path = make_embedding_asset(tmp_path)
        model = PTM2CellNet(
            encoder_type="transformer",
            embed_dim=128,
            num_classes=4,
            use_davf=True,
            davf_config={
                "feature_dim": 64,
                "scvi_model_path": "checkpoints/scvi/model.pt",
                "embedding_asset_path": asset_path,
            },
        )

        assert model.davf_feature_dim == 64
        assert model.davf_module.config.feature_dim == 64
        assert model.davf_config["scvi_model_path"] == "checkpoints/scvi/model.pt"
        assert model.davf_config["embedding_asset_path"] == asset_path
        assert model.davf_module.embedding_gene_to_token == {f"ENSG00000{i}": i for i in range(4)}
        assert model.ptm_mapper is not None
        assert model.ptm_mapper.gene_to_idx == model.davf_module.embedding_gene_to_token
        assert model.davf_path_class == "formal_schema_v2"

    def test_davf_config_rejects_removed_geneformer_path(self):
        """Schema v2: geneformer_path must fail loudly, not silently degrade."""
        with pytest.raises(ValueError, match="geneformer_path"):
            PTM2CellNet(
                encoder_type="transformer",
                embed_dim=128,
                num_classes=4,
                use_davf=True,
                davf_config={"geneformer_path": "checkpoints/geneformer_embeddings.pt"},
            )

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

    def test_davf_with_multitask_builds(self):
        model = PTM2CellNet(
            encoder_type="transformer",
            embed_dim=128,
            use_davf=True,
            task_type="multitask",
            multitask_configs=[
                {"name": "cell_state", "type": "classification", "num_classes": 4},
                {"name": "ptm_site", "type": "classification", "num_classes": 5},
            ],
        )

        assert model.use_davf is True
        assert model.davf_module is not None
        assert model.davf_feature_dim == 128
        assert model.predictor.get_task_names() == ["cell_state", "ptm_site"]
        # 任务头消费拼接后的 [序列+PTM ; DAVF] 联合表示
        assert model.predictor.input_dim == 128 + 128

    def test_davf_with_multitask_forward(self):
        model = PTM2CellNet(
            encoder_type="transformer",
            embed_dim=128,
            use_davf=True,
            task_type="multitask",
            multitask_configs=[
                {"name": "cell_state", "type": "classification", "num_classes": 4},
                {"name": "ptm_site", "type": "classification", "num_classes": 5},
            ],
        )

        batch = make_batch()
        davf_sites, davf_gene_names = make_davf_inputs()
        batch["davf_sites"] = davf_sites
        batch["davf_gene_names"] = davf_gene_names

        output = model(batch)
        assert output["cell_state"]["logits"].shape == (2, 4)
        assert output["cell_state"]["probabilities"].shape == (2, 4)
        assert output["ptm_site"]["logits"].shape == (2, 5)

    def test_davf_with_multitask_zero_features(self):
        model = PTM2CellNet(
            encoder_type="transformer",
            embed_dim=128,
            use_davf=True,
            task_type="multitask",
            multitask_configs=[
                {"name": "cell_state", "type": "classification", "num_classes": 4},
            ],
        )

        # 无 davf_sites / 无 checkpoint：DAVF 分支回退为零特征，前向仍应可用
        output = model(make_batch())
        assert output["cell_state"]["logits"].shape == (2, 4)

    def test_davf_multitask_gradient_flows_into_projection(self):
        """Joint multitask+DAVF mode: gradient reaches the trainable DAVF
        projection head (the rest of DAVF stays frozen by default).

        Mirrors the two-stage training contract (Section 3.5 of the methods):
        stage 1 keeps the DAVF backbone frozen and only the projection head +
        prediction heads train, so the projection must receive gradient while
        the DAVF encoder parameters must not.

        Uses ``state_space="gene"`` so the branch is active without requiring
        a pretrained checkpoint file (``_checkpoint_loaded=True``).
        """
        model = PTM2CellNet(
            encoder_type="transformer",
            embed_dim=128,
            use_davf=True,
            task_type="multitask",
            multitask_configs=[
                {"name": "cell_state", "type": "classification", "num_classes": 4},
                {"name": "ptm_site", "type": "classification", "num_classes": 5},
            ],
            davf_config={
                "state_space": "gene",
                "feature_dim": 64,
                "hidden_dim": 128,
                "freeze": True,
            },
        )
        assert model.davf_module is not None
        assert model.davf_module._checkpoint_loaded is True

        # DAVF backbone frozen by default; projection head always trainable.
        davf_encoder_frozen = all(not p.requires_grad for p in model.davf_module.gene_encoder.parameters())
        projection_trainable = any(p.requires_grad for p in model.davf_module.delta_projection.parameters())
        assert davf_encoder_frozen, "DAVF encoder must be frozen by default (stage 1)"
        assert projection_trainable, "DeltaProjection must remain trainable"

        batch = make_batch()
        davf_sites, davf_gene_names = make_davf_inputs()
        batch["davf_sites"] = davf_sites
        batch["davf_gene_names"] = davf_gene_names

        output = model(batch)
        loss = output["cell_state"]["logits"].sum() + output["ptm_site"]["logits"].sum()
        loss.backward()

        # The trainable projection head must receive gradient in joint mode.
        proj_grad = model.davf_module.delta_projection.net[0].weight.grad
        assert proj_grad is not None, "gradient must flow into DAVF projection"
        assert torch.isfinite(proj_grad).all()
        assert float(proj_grad.abs().sum()) > 0

    def test_from_config_with_multitask_and_davf(self):
        config = {
            "model": {
                "encoder_type": "transformer",
                "hidden_dim": 128,
                "task_type": "multitask",
                "use_davf": True,
                "multitask_configs": [
                    {"name": "cell_state", "type": "classification", "num_classes": 4},
                ],
            },
            "data": {},
        }
        model = PTM2CellNet.from_config(config)

        assert model.use_davf is True
        assert model.predictor.get_task_names() == ["cell_state"]
        output = model(make_batch())
        assert output["cell_state"]["logits"].shape == (2, 4)

    def test_from_config_with_davf(self, tmp_path):
        asset_path = make_embedding_asset(tmp_path)
        config = make_config(
            use_davf=True,
            davf_config={
                "feature_dim": 64,
                "scvi_model_path": "checkpoints/scvi/model.pt",
                "embedding_asset_path": asset_path,
            },
        )
        model = PTM2CellNet.from_config(config)

        assert model.use_davf is True
        assert model.davf_module.config.feature_dim == 64
        assert model.davf_config["scvi_model_path"] == "checkpoints/scvi/model.pt"
        assert model.davf_config["embedding_asset_path"] == asset_path

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
        assert "embedding_asset_path" in model.davf_config
        assert "geneformer_path" not in model.davf_config

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
