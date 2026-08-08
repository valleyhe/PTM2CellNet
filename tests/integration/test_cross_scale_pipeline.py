"""End-to-end fixture for protein/PTM → graph → cell-state composition."""

import torch

from src.models.cross_scale import CrossScaleConfig, CrossScalePTM2CellNet


def _chain(num_nodes: int) -> torch.Tensor:
    source = torch.arange(num_nodes - 1, dtype=torch.long)
    destination = source + 1
    return torch.stack([torch.cat([source, destination]), torch.cat([destination, source])])


def _batch() -> dict[str, torch.Tensor]:
    return {
        "ankh39_embeddings": torch.randn(2, 4, 5),
        "esm2_embeddings": torch.randn(2, 4, 6),
        "prott5_embeddings": torch.randn(2, 4, 7),
        "ptm_types": torch.tensor([[1, 2], [2, 1]]),
        "ptm_positions": torch.tensor([[1, 3], [2, 4]]),
        "ptm_mask": torch.ones(2, 2),
        "signal_edge_index": _chain(4),
        "signal_gene_map": torch.ones(4, 6),
        "cell_edge_index": _chain(6),
        "cell_gene_features": torch.randn(2, 6, 3),
        "gene_mask": torch.ones(2, 6),
        "signal_graph_version": "fixture-v1",
    }


def test_cross_scale_forward_loss_and_checkpoint_roundtrip():
    config = CrossScaleConfig(
        protein_dim=8,
        signal_input_dim=8,
        signal_hidden_dim=10,
        signal_output_dim=9,
        cell_gene_feature_dim=3,
        cell_hidden_dim=11,
        num_cell_genes=6,
        num_cell_states=3,
        num_ptm_types=3,
        max_position=16,
        dropout=0.0,
    )
    model = CrossScalePTM2CellNet(
        config=config,
        plm_backbone_dims={"ankh39": 5, "esm2": 6, "prott5": 7},
        data_manifest="data/manifests/datasets.yaml",
        signal_graph_version="fixture-v1",
    )
    batch = _batch()
    output = model(batch)
    assert output["delta_expression"].shape == (2, 6)
    assert output["cell_state_logits"].shape == (2, 3)
    assert output["sensitivity_matrix"].shape == (4, 4)
    assert output["ptm_token_embeddings"].shape == (2, 4, 8)
    assert output["provenance"]["ptm_adapter"]["active_site_count"] == 4
    assert output["provenance"]["data_manifest_digest"]
    assert output["provenance"]["graph_approximation_used"] is False
    loss = model.compute_loss(
        output,
        {"delta_expression": torch.randn(2, 6), "cell_state": torch.tensor([0, 1])},
    )
    loss.backward()
    assert torch.isfinite(loss)

    restored = CrossScalePTM2CellNet(
        config=config,
        plm_backbone_dims={"ankh39": 5, "esm2": 6, "prott5": 7},
        data_manifest="data/manifests/datasets.yaml",
        signal_graph_version="fixture-v1",
    )
    restored.load_state_dict(model.state_dict())
    restored.eval()
    model.eval()
    with torch.no_grad():
        first = model(batch)["cell_state_logits"]
        second = restored(batch)["cell_state_logits"]
    assert torch.allclose(first, second)
    info = restored.get_model_info()
    assert info["model_class"] == "CrossScalePTM2CellNet"
    assert info["data_manifest"] == "data/manifests/datasets.yaml"
