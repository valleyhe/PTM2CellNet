"""Unit contracts for the opt-in cross-scale scientific model."""

import pytest
import torch
from torch import nn
from types import SimpleNamespace

from src.models.cross_scale import (
    CIGNNSignalBridge,
    CellGraphCompassHead,
    CrossScaleContractError,
    CrossScaleConfig,
    CrossScalePTM2CellNet,
    MultiPLMEncoder,
    compute_sensitivity_matrix,
    normalized_adjacency,
)
from src.models.ptm_modules import PTMTokenAdapter


def _chain(num_nodes: int) -> torch.Tensor:
    source = torch.arange(num_nodes - 1, dtype=torch.long)
    destination = source + 1
    return torch.stack([torch.cat([source, destination]), torch.cat([destination, source])])


def test_sensitivity_matrix_has_formula_shape_and_stochastic_rows():
    edges = _chain(4)
    sensitivity = compute_sensitivity_matrix(edges, 4, alpha=0.8)
    assert sensitivity.shape == (4, 4)
    assert torch.allclose(sensitivity.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.isfinite(sensitivity).all()


def test_sensitivity_rejects_invalid_alpha_and_edges():
    with pytest.raises(CrossScaleContractError, match="alpha"):
        compute_sensitivity_matrix(_chain(3), 3, alpha=1.0)
    with pytest.raises(CrossScaleContractError, match="outside"):
        compute_sensitivity_matrix(torch.tensor([[0, 4], [1, 2]]), 3)


def test_sensitivity_matrix_matches_explicit_formula_and_edge_gradients():
    edges = _chain(3)
    weights = torch.tensor([0.2, 0.4, 0.3, 0.1], requires_grad=True)
    alpha = 0.7
    sensitivity = compute_sensitivity_matrix(edges, 3, edge_weight=weights, alpha=alpha)
    graph = normalized_adjacency(edges, 3, edge_weight=weights)
    identity = torch.eye(3)
    expected = (1.0 - alpha) * torch.linalg.solve(identity - alpha * graph, identity)
    assert torch.allclose(sensitivity, expected, atol=1e-6)
    sensitivity.square().sum().backward()
    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()


def test_cignn_bridge_propagates_gradients_and_requires_graph():
    bridge = CIGNNSignalBridge(5, hidden_dim=7, output_dim=6, num_layers=2)
    features = torch.randn(2, 4, 5, requires_grad=True)
    output = bridge(features, _chain(4))
    assert output.shape == (2, 4, 6)
    output.square().mean().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    with pytest.raises(CrossScaleContractError, match="edge_index"):
        bridge(features.detach(), None)


def test_typed_signal_edges_are_validated_and_provenanced():
    bridge = CIGNNSignalBridge(
        4,
        hidden_dim=5,
        num_edge_types=2,
        require_edge_type=True,
        dropout=0.0,
    )
    features = torch.randn(1, 3, 4)
    edges = _chain(3)
    with pytest.raises(CrossScaleContractError, match="edge_type"):
        bridge(features, edges)
    details = bridge.forward_with_details(features, edges, edge_type=torch.tensor([0, 1, 0, 1]))
    assert details["node_embeddings"].shape == (1, 3, 5)
    assert bridge.last_provenance["edge_types_provided"] is True
    with pytest.raises(CrossScaleContractError, match="integer dtype"):
        bridge(features, edges, edge_type=torch.tensor([0.0, 1.0, 0.0, 1.0]))


def test_multiplm_fuses_three_backbones_and_records_provenance():
    encoder = MultiPLMEncoder(
        backbone_dims={"ankh39": 4, "esm2": 5, "prott5": 6},
        output_dim=8,
        dropout=0.0,
    )
    source = {
        "ankh39_embeddings": torch.randn(2, 3, 4),
        "esm2_embeddings": torch.randn(2, 3, 5),
        "prott5_embeddings": torch.randn(2, 3, 6),
    }
    nodes = encoder.encode_nodes(source)
    pooled = encoder(source)
    assert nodes.shape == (2, 3, 8)
    assert pooled.shape == (2, 8)
    assert encoder.last_provenance["fallback_used"] is False
    assert set(encoder.last_provenance["provided_backbones"]) == {"ankh39", "esm2", "prott5"}


def test_multiplm_requires_explicit_alignment_for_different_token_lengths():
    encoder = MultiPLMEncoder(
        backbone_dims={"ankh39": 4, "esm2": 5},
        output_dim=6,
        required_backbones=("ankh39", "esm2"),
        dropout=0.0,
    )
    source = {
        "ankh39_embeddings": torch.randn(1, 3, 4),
        "esm2_embeddings": torch.randn(1, 5, 5),
    }
    with pytest.raises(CrossScaleContractError, match=r"share \[B, L\]"):
        encoder.encode_nodes(source)
    source["residue_alignment"] = {
        "ankh39": torch.ones(3, 4),
        "esm2": torch.ones(5, 4),
    }
    aligned = encoder.encode_nodes(source)
    assert aligned.shape == (1, 4, 6)
    assert encoder.last_provenance["residue_alignment_used"] is True


def test_multiplm_missing_required_backbone_is_not_silent():
    strict = MultiPLMEncoder(
        backbone_dims={"ankh39": 4, "esm2": 5},
        output_dim=8,
        required_backbones=("ankh39", "esm2"),
    )
    with pytest.raises(CrossScaleContractError, match="required PLM embedding missing"):
        strict.encode_nodes({"ankh39_embeddings": torch.randn(1, 2, 4)})

    fallback = MultiPLMEncoder(
        backbone_dims={"ankh39": 4, "esm2": 5},
        output_dim=8,
        required_backbones=("ankh39", "esm2"),
        allow_fallback=True,
    )
    nodes = fallback.encode_nodes({"ankh39_embeddings": torch.randn(1, 2, 4)})
    assert nodes.shape == (1, 2, 8)
    assert fallback.last_provenance["fallback_used"] is True


def test_cell_head_outputs_masked_expression_and_loss():
    head = CellGraphCompassHead(
        signal_dim=6,
        num_genes=5,
        gene_feature_dim=3,
        hidden_dim=8,
        num_cell_states=3,
        dropout=0.0,
    )
    signal = torch.randn(2, 4, 6)
    mapping = torch.ones(4, 5)
    outputs = head(
        signal,
        cell_edge_index=_chain(5),
        signal_gene_map=mapping,
        gene_features=torch.randn(2, 5, 3),
        gene_mask=torch.tensor([[1, 1, 1, 0, 1], [1, 1, 1, 1, 1]], dtype=torch.float32),
    )
    assert outputs["delta_expression"].shape == (2, 5)
    assert outputs["cell_state_logits"].shape == (2, 3)
    assert outputs["delta_expression"][0, 3].item() == 0.0
    loss = head.compute_loss(
        outputs,
        {"delta_expression": torch.randn(2, 5), "cell_state": torch.tensor([0, 2])},
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_cell_head_requires_explicit_mapping_and_graph():
    head = CellGraphCompassHead(signal_dim=4, num_genes=3, require_signal_gene_map=True)
    with pytest.raises(CrossScaleContractError, match="signal_gene_map"):
        head(torch.randn(1, 2, 4), cell_edge_index=_chain(3), signal_gene_map=None)
    with pytest.raises(CrossScaleContractError, match="cell_edge_index"):
        head(torch.randn(1, 2, 4), cell_edge_index=None, signal_gene_map=torch.ones(2, 3))
    with pytest.raises(CrossScaleContractError, match="gene_mask"):
        head(
            torch.randn(1, 2, 4),
            cell_edge_index=_chain(3),
            signal_gene_map=torch.ones(2, 3),
            gene_mask=torch.tensor([[1.0, 2.0, 1.0]]),
        )


def test_cross_scale_config_preserves_plm_controls_and_injected_policy():
    configured = CrossScalePTM2CellNet.from_config(
        {
            "cross_scale": {
                "protein_dim": 8,
                "signal_input_dim": 8,
                "signal_hidden_dim": 8,
                "signal_output_dim": 8,
                "num_cell_genes": 3,
                "plm_backbone_dims": {"ankh39": 4, "esm2": 5},
                "plm_model_names": {"ankh39": "local/ankh39", "esm2": "local/esm2"},
                "required_backbones": ["esm2"],
                "freeze_plm_backbones": False,
            }
        }
    )
    info = configured.get_model_info()
    assert info["configured_plm_model_names"] == {
        "ankh39": "local/ankh39",
        "esm2": "local/esm2",
    }
    assert info["required_backbones"] == ["esm2"]
    assert info["freeze_plm_backbones"] is False
    assert info["uninitialized_parameter_count"] > 0

    injected = MultiPLMEncoder(
        backbone_dims={"ankh39": 4},
        backbone_names=("ankh39",),
        required_backbones=("ankh39",),
        output_dim=8,
        allow_fallback=True,
    )
    injected_model = CrossScalePTM2CellNet(
        config=CrossScaleConfig(
            protein_dim=8,
            signal_input_dim=8,
            signal_hidden_dim=8,
            signal_output_dim=8,
            num_cell_genes=3,
        ),
        protein_encoder=injected,
    )
    assert injected_model.get_model_info()["allow_plm_fallback"] is True


def test_ptm_token_adapter_scatter_is_one_based_and_variable_length_safe():
    adapter = PTMTokenAdapter(num_ptm_types=3, embed_dim=5, max_position=8, dropout=0.0)
    tokens = adapter(
        torch.tensor([[1, 2, 0], [1, 0, 0]]),
        torch.tensor([[1, 3, 0], [2, 0, 0]]),
        sequence_length=4,
        valid_sequence_lengths=torch.tensor([4, 2]),
    )
    assert tokens.shape == (2, 4, 5)
    assert torch.allclose(tokens[1, 2:], torch.zeros(2, 5))
    assert adapter.last_provenance["active_site_count"] == 3
    with pytest.raises(ValueError, match="within each sequence length"):
        adapter(
            torch.tensor([[1]]),
            torch.tensor([[3]]),
            sequence_length=2,
        )
    with pytest.raises(ValueError, match="active PTM types"):
        adapter(
            torch.tensor([[4]]),
            torch.tensor([[1]]),
            sequence_length=2,
        )


class _FakeTokenizer:
    def __call__(self, sequences, **kwargs):
        rows = []
        masks = []
        specials = []
        for sequence in sequences:
            # One leading and trailing special token, plus right padding.
            rows.append([1] + list(range(2, 2 + len(sequence))) + [99])
            masks.append([1] * (len(sequence) + 2))
            specials.append([1] + [0] * len(sequence) + [1])
        width = max(len(row) for row in rows)
        for row, mask, special in zip(rows, masks, specials, strict=False):
            pad = width - len(row)
            row.extend([0] * pad)
            mask.extend([0] * pad)
            special.extend([1] * pad)
        return {
            "input_ids": torch.tensor(rows),
            "attention_mask": torch.tensor(masks),
            "special_tokens_mask": torch.tensor(specials),
        }

    def get_special_tokens_mask(self, row, already_has_special_tokens=True):
        return [int(token in {0, 1, 99}) for token in row]


class _FakeModel(nn.Module):
    def __init__(self, hidden_dim=4):
        super().__init__()
        self.embedding = nn.Embedding(128, hidden_dim)

    def forward(self, input_ids, attention_mask=None):
        return type("Output", (), {"last_hidden_state": self.embedding(input_ids)})()


def test_multiplm_raw_sequence_path_preserves_residue_nodes_and_mask():
    encoder = MultiPLMEncoder(
        backbone_dims={"ankh39": 4},
        output_dim=6,
        backbone_names=("ankh39",),
        required_backbones=("ankh39",),
        dropout=0.0,
    )
    encoder.backbone_models["ankh39"] = _FakeModel()
    encoder._tokenizers["ankh39"] = _FakeTokenizer()
    nodes, mask = encoder.encode_sequence_nodes(["ABCDE", "FG"], max_length=16)
    assert nodes.shape == (2, 5, 6)
    assert torch.equal(mask.sum(dim=1), torch.tensor([5.0, 2.0]))
    assert encoder.last_provenance["raw_sequence_pooling"] == "residue_mean_pool"


def test_multiplm_loader_uses_t5_encoder_for_ankh_and_prott5(monkeypatch):
    """T5 pLMs must expose encoder states and derive width from d_model."""

    import transformers

    calls = {}

    class _FakeT5Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(d_model=12)
            self.weight = nn.Parameter(torch.ones(1))

    class _FakeT5Tokenizer:
        pass

    def fake_t5_model_from_pretrained(model_name, **kwargs):
        calls["model"] = (model_name, kwargs)
        return _FakeT5Encoder()

    def fake_t5_tokenizer_from_pretrained(model_name, **kwargs):
        calls["tokenizer"] = (model_name, kwargs)
        return _FakeT5Tokenizer()

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        staticmethod(
            lambda model_name, **kwargs: SimpleNamespace(
                model_type="t5",
                architectures=["T5ForConditionalGeneration"],
                d_model=12,
            )
        ),
    )

    class _FakeT5EncoderFactory:
        from_pretrained = staticmethod(fake_t5_model_from_pretrained)

    class _FakeT5TokenizerFactory:
        from_pretrained = staticmethod(fake_t5_tokenizer_from_pretrained)

    # ``transformers`` exposes optional objects through module ``__getattr__``.
    # Looking up T5Tokenizer before patching raises when SentencePiece is not
    # installed, so inject the fakes into the module namespace directly.
    monkeypatch.setitem(transformers.__dict__, "T5EncoderModel", _FakeT5EncoderFactory)
    monkeypatch.setitem(transformers.__dict__, "T5Tokenizer", _FakeT5TokenizerFactory)

    encoder = MultiPLMEncoder(
        backbone_dims={"prott5": 5},
        output_dim=6,
        backbone_names=("prott5",),
        required_backbones=("prott5",),
        dropout=0.0,
    )
    model = encoder.load_pretrained_backbone(
        "prott5",
        model_name="local/prot-t5",
        local_files_only=True,
    )

    assert isinstance(model, _FakeT5Encoder)
    assert calls["model"][0] == "local/prot-t5"
    assert calls["tokenizer"][0] == "local/prot-t5"
    assert encoder.backbone_dims["prott5"] == 12
    assert encoder.projections["prott5"].in_features == 12


def test_multiplm_projection_accepts_fp16_backbone_states():
    """FP16 ProtT5 states are projected through the FP32 fusion stack."""

    encoder = MultiPLMEncoder(
        backbone_dims={"prott5": 4},
        output_dim=6,
        backbone_names=("prott5",),
        required_backbones=("prott5",),
        dropout=0.0,
    )
    nodes = encoder.encode_nodes({"prott5_embeddings": torch.randn(1, 3, 4, dtype=torch.float16)})

    assert nodes.dtype == torch.float32
    assert nodes.shape == (1, 3, 6)


def test_cross_scale_raw_sequences_keep_ptm_sites_addressable():
    encoder = MultiPLMEncoder(
        backbone_dims={"ankh39": 4},
        output_dim=8,
        backbone_names=("ankh39",),
        required_backbones=("ankh39",),
        dropout=0.0,
    )
    encoder.backbone_models["ankh39"] = _FakeModel()
    encoder._tokenizers["ankh39"] = _FakeTokenizer()
    model = CrossScalePTM2CellNet(
        config=CrossScaleConfig(
            protein_dim=8,
            signal_input_dim=8,
            signal_hidden_dim=8,
            signal_output_dim=8,
            cell_gene_feature_dim=3,
            cell_hidden_dim=8,
            num_cell_genes=3,
            num_cell_states=2,
            num_ptm_types=2,
            max_position=8,
            dropout=0.0,
        ),
        protein_encoder=encoder,
    )
    output = model(
        {
            "sequence": ["ABCDE", "FG"],
            "ptm_types": torch.tensor([[1, 2], [1, 0]]),
            "ptm_positions": torch.tensor([[1, 3], [2, 0]]),
            "ptm_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.float32),
            "signal_edge_index": _chain(5),
            "signal_gene_map": torch.ones(5, 3),
            "cell_edge_index": _chain(3),
        }
    )
    assert output["protein_node_embeddings"].shape == (2, 5, 8)
    assert output["ptm_token_embeddings"].shape == (2, 5, 8)
    assert output["provenance"]["ptm_adapter"]["active_site_count"] == 3
