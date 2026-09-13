"""Regression tests for the DAVF direction → PerturbGen mainline boundary."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from fastapi import HTTPException

from src.api.routes import predictions
from src.api.schemas import PTMSite, PredictionRequest
from src.integration.perturbgen.contracts import (
    DAVFDirectionEvidence,
    PTMSiteDirectionProposal,
)
from src.integration.perturbgen.mainline import evaluate_davf_perturbgen_candidate
from src.integration.perturbgen.reports import build_candidate_report_payload
from src.models.davf_inference import DAVFInferenceConfig, DAVFInferenceModule
from src.models.ptm_direction_mapper import PTMDirectionMapper, PTMDirectionMapperOutput


def _proposal(direction: str = "down") -> PTMSiteDirectionProposal:
    return PTMSiteDirectionProposal(
        gene_symbol="STAT3",
        ensembl_id="ENSG00000168610",
        position=12,
        ptm_type="phosphorylation",
        proposed_direction=direction,
        site_probability=0.95,
        provenance="ptm-site-model/test",
    )


def _evidence(direction: str = "down") -> DAVFDirectionEvidence:
    delta = -1.2 if direction == "down" else 1.2
    return DAVFDirectionEvidence(
        gene_symbol="STAT3",
        ensembl_id="ENSG00000168610",
        predicted_direction=direction,
        predicted_delta=delta,
        model_source="davf",
        checkpoint_provenance="checkpoints/davf/best_model.pt",
        embedding_provenance="outputs/embedding_asset",
        confidence=0.5,
    )


def _overexpress_path(path: str) -> list[dict[str, object]]:
    return [
        {
            "status": "evaluable",
            "path": path,
            "mode": "overexpress",
            "rescue_excl_target": 0.4 - seed * 0.05,
            "evaluable_donors": 3,
            "donor_consistency": 2 / 3,
            "seed": seed,
            "reason_code": None,
            "matched_null_count": 99,
            "empirical_pvalue": 0.01,
        }
        for seed in (1, 2, 3)
    ]


def test_mainline_requires_direction_agreement_before_perturbgen() -> None:
    decision = evaluate_davf_perturbgen_candidate(
        _proposal("down"),
        _evidence("up"),
        cell_type="T cell",
        ptm_context="test",
        observed_log2fc=-1.0,
        observed_fdr=0.01,
        observed_direction="down",
        intervention_type="KO",
        path_results=_overexpress_path("source_intervention") + _overexpress_path("within_state"),
        q_value=0.01,
        unperturbed_quality_status="pass",
    )

    assert decision.verdict == "fail"
    assert decision.candidate is None
    assert decision.dual_path is None
    assert "direction_evidence_disagreement" in decision.reasons


def test_mainline_passes_direction_gate_then_evaluates_both_paths() -> None:
    decision = evaluate_davf_perturbgen_candidate(
        _proposal("down"),
        _evidence("down"),
        cell_type="T cell",
        ptm_context="test",
        observed_log2fc=-1.0,
        observed_fdr=0.01,
        observed_direction="down",
        intervention_type="KO",
        path_results=_overexpress_path("source_intervention") + _overexpress_path("within_state"),
        q_value=0.01,
        unperturbed_quality_status="pass",
    )

    assert decision.verdict == "pass"
    assert decision.direction_gate.status == "pass"
    assert decision.candidate is not None
    assert decision.candidate.davf_action == "oe"
    assert decision.dual_path is not None
    assert {item.path for item in decision.dual_path.path_decisions} == {
        "source_intervention",
        "within_state",
    }


def test_mainline_report_keeps_direction_gate_and_candidate_evidence() -> None:
    decision = evaluate_davf_perturbgen_candidate(
        _proposal("down"),
        _evidence("down"),
        cell_type="T cell",
        ptm_context="test",
        observed_log2fc=-1.0,
        observed_fdr=0.01,
        observed_direction="down",
        intervention_type="KO",
        q_value=0.01,
        unperturbed_quality_status="pass",
    )

    payload = build_candidate_report_payload(
        decision,
        manifest={"run_id": "mainline-test"},
    )

    assert payload["direction_gate"]["status"] == "pass"
    assert payload["candidate"]["direction_gate_status"] == "pass"
    assert payload["candidate_gene"] == "STAT3"
    assert payload["verdict"] == "inconclusive"


def test_mapper_accepts_api_davf_mapping_and_rejects_truncated_batch() -> None:
    mapper = PTMDirectionMapper(
        geneformer_loader=SimpleNamespace(_gene_to_idx={"TP53": 7}),
        gene_mapper=SimpleNamespace(map_gene_to_uniprot=lambda _gene: None),
    )

    output = mapper.map_ptms(
        [{"position": 3, "ptm_type": "phosphorylation"}],
        ["TP53"],
    )
    assert output.gene_ids[0, 0].item() == 7
    assert output.directions[0, 0].item() == 2
    with pytest.raises(ValueError, match="same batch size"):
        mapper.map_batch([[{"position": 3, "ptm_type": "phosphorylation"}]], [])


def test_asset_backed_mapper_uses_only_verified_gene_token_mapping() -> None:
    mapper = PTMDirectionMapper(
        gene_to_idx={"ENSG00000168610": 4},
        gene_mapper=SimpleNamespace(map_gene_to_uniprot=lambda _gene: "must-not-run"),
    )

    known = mapper.map_ptms(
        [{"position": 3, "type": "phosphorylation"}],
        ["ENSG00000168610"],
    )
    unknown = mapper.map_ptms(
        [{"position": 3, "type": "phosphorylation"}],
        ["ENSG00000999999"],
    )

    assert known.gene_ids[0, 0].item() == 4
    assert known.attention_mask[0, 0].item() == 1
    assert unknown.attention_mask[0, 0].item() == 0


def test_asset_backed_mapper_normalizes_symbol_case_and_whitespace() -> None:
    mapper = PTMDirectionMapper(gene_to_idx={"TP53": 4})

    output = mapper.map_ptms(
        [{"position": 3, "type": "phosphorylation"}],
        ["  tp53  "],
    )

    assert output.gene_ids[0, 0].item() == 4
    assert output.attention_mask[0, 0].item() == 1


def test_davf_asset_mapper_resolves_api_symbol_through_verified_alias_table(tmp_path) -> None:
    alias_path = tmp_path / "genes.tsv"
    alias_path.write_text(
        "ENSG000000\tTP53\nENSG000001\tEGFR\nENSG000002\tTP53\n",
        encoding="utf-8",
    )
    asset_gene_to_token = {"ENSG000000": 0, "ENSG000001": 1, "ENSG000002": 2}
    aliases = DAVFInferenceModule._load_embedding_symbol_aliases(
        alias_path,
        asset_gene_to_token,
    )
    mapper = PTMDirectionMapper(
        gene_to_idx={**asset_gene_to_token, **aliases},
    )
    output = mapper.map_ptms(
        [{"type": "phosphorylation"}],
        [" egfr "],
    )

    assert output.gene_ids[0, 0].item() == 1
    assert output.attention_mask[0, 0].item() == 1
    assert aliases == {"EGFR": 1}


def test_latent_davf_rejects_an_all_masked_target_row_before_attention() -> None:
    from src.models.latent_davf import LatentDAVF, LatentDAVFConfig

    model = LatentDAVF(
        LatentDAVFConfig(
            latent_dim=2,
            num_genes=3,
            gene_embed_dim=4,
            hidden_dim=8,
            num_heads=2,
            direction_embed_dim=4,
            x_encoder_hidden=8,
            velocity_hidden=8,
            num_velocity_layers=2,
            dropout=0.0,
            attention_dropout=0.0,
        )
    )

    with pytest.raises(ValueError, match="at least one valid PerturbGen gene token"):
        model._resolve_condition(
            torch.zeros((1, 2), dtype=torch.long),
            torch.zeros((1, 2), dtype=torch.long),
            None,
            "internal_targets",
            None,
            torch.zeros((1, 2)),
        )


def test_api_davf_path_rejects_missing_gene_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(predictions.STATE, "model", SimpleNamespace(use_davf=True))
    monkeypatch.setattr(predictions.STATE, "feature_extractor", None)
    monkeypatch.setattr(predictions.STATE, "max_sequence_length", 100)
    monkeypatch.setattr(predictions.STATE, "ptm_type_to_idx", {})
    request = PredictionRequest(
        sequence="ACDE",
        ptm_sites=[PTMSite(position=2, type="phosphorylation")],
        use_davf=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        predictions.preprocess_request(request)
    assert exc_info.value.status_code == 400
    assert "gene_symbol" in str(exc_info.value.detail)


class _FakeLatentDAVF(torch.nn.Module):
    def predict(self, z_0, **_kwargs):
        return z_0 + torch.ones_like(z_0)


class _LegacyLikeLatentDAVF(torch.nn.Module):
    pass


class _FakeScVIAdapter:
    n_latent = 2
    n_genes = 2
    gene_names = ("BASE", "STAT3")
    config = SimpleNamespace(model_path="checkpoints/scvi/model")

    def validate_compatibility(self, *, expected_latent_dim, expected_num_genes, expected_gene_names=None):
        if expected_latent_dim != self.n_latent:
            raise ValueError("DAVF/scVI latent dimension mismatch")
        if expected_num_genes != self.n_genes:
            raise ValueError("DAVF/scVI gene dimension mismatch")
        if expected_gene_names is not None and tuple(expected_gene_names) != self.gene_names:
            raise ValueError("DAVF/scVI gene vocabulary order does not match")

    def decode(self, latent, **_kwargs):
        values = np.asarray(latent, dtype=float)
        return np.stack([values[:, 0], 2.0 * values[:, 0]], axis=1)


def _bare_direction_module(*, loaded: bool = True, with_assets: bool = True) -> DAVFInferenceModule:
    module = DAVFInferenceModule.__new__(DAVFInferenceModule)
    torch.nn.Module.__init__(module)
    module.config = DAVFInferenceConfig(
        state_space="scvi_latent",
        checkpoint_path="checkpoints/davf/best_model.pt",
        scvi_model_path="checkpoints/scvi/model",
        embedding_asset_path="outputs/embedding_asset" if with_assets else None,
        latent_dim=2,
        num_genes=2,
    )
    module._checkpoint_loaded = loaded
    module._current_checkpoint_contract_valid = loaded
    module._embedding_symbol_to_ensembl = {}
    module._checkpoint_contract = {
        "scvi": {
            "model_path": str(Path("checkpoints/scvi/model").resolve()),
            "latent_dim": 2,
            "num_genes": 2,
            "gene_names": ["BASE", "STAT3"],
        }
    }
    module.register_parameter("_test_anchor", torch.nn.Parameter(torch.zeros(1)))
    module.latent_davf = _FakeLatentDAVF()
    return module


def test_davf_decodes_gene_delta_into_direction_evidence() -> None:
    module = _bare_direction_module()
    mapper_output = PTMDirectionMapperOutput(
        gene_ids=torch.zeros((1, 2), dtype=torch.long),
        directions=torch.zeros((1, 2), dtype=torch.long),
        attention_mask=torch.tensor([[1.0, 0.0]]),
    )

    evidence = module.predict_expression_direction(
        mapper_output,
        torch.zeros((1, 2)),
        _FakeScVIAdapter(),
        target_gene_indices=[1],
        target_gene_symbols=["STAT3"],
        target_ensembl_ids=["ENSG00000168610"],
    )

    assert len(evidence) == 1
    assert evidence[0].predicted_direction == "up"
    assert evidence[0].predicted_delta == pytest.approx(2.0)
    assert evidence[0].confidence == pytest.approx(1.0)
    assert evidence[0].model_source == "davf"


def test_davf_direction_accepts_scvi_adapter_numpy_latent() -> None:
    module = _bare_direction_module()
    mapper_output = PTMDirectionMapperOutput(
        gene_ids=torch.zeros((1, 2), dtype=torch.long),
        directions=torch.zeros((1, 2), dtype=torch.long),
        attention_mask=torch.tensor([[1.0, 0.0]]),
    )

    evidence = module.predict_expression_direction(
        mapper_output,
        np.zeros((1, 2), dtype=np.float32),
        _FakeScVIAdapter(),
        target_gene_symbols=["STAT3"],
        target_ensembl_ids=["ENSG00000168610"],
    )

    assert evidence[0].predicted_delta == pytest.approx(2.0)


def test_davf_direction_rejects_all_masked_scvi_targets() -> None:
    module = _bare_direction_module()
    mapper_output = PTMDirectionMapperOutput(
        gene_ids=torch.zeros((1, 2), dtype=torch.long),
        directions=torch.zeros((1, 2), dtype=torch.long),
        attention_mask=torch.zeros((1, 2)),
    )

    with pytest.raises(ValueError, match="at least one valid PerturbGen gene token"):
        module.predict_expression_direction(
            mapper_output,
            torch.zeros((1, 2)),
            _FakeScVIAdapter(),
            target_gene_symbols=["STAT3"],
            target_ensembl_ids=["ENSG00000168610"],
        )


def test_davf_direction_resolves_scvi_target_index_when_not_supplied() -> None:
    module = _bare_direction_module()
    mapper_output = PTMDirectionMapperOutput(
        gene_ids=torch.zeros((1, 2), dtype=torch.long),
        directions=torch.zeros((1, 2), dtype=torch.long),
        attention_mask=torch.tensor([[1.0, 0.0]]),
    )

    evidence = module.predict_expression_direction(
        mapper_output,
        torch.zeros((1, 2)),
        _FakeScVIAdapter(),
        target_gene_symbols=["STAT3"],
        target_ensembl_ids=["ENSG00000168610"],
    )

    assert evidence[0].predicted_delta == pytest.approx(2.0)


def test_davf_direction_rejects_request_for_another_trained_modality() -> None:
    module = _bare_direction_module()
    module.config = DAVFInferenceConfig(
        state_space="scvi_latent",
        checkpoint_path="checkpoints/davf/ko.pt",
        scvi_model_path="checkpoints/scvi/model",
        embedding_asset_path="outputs/embedding_asset",
        latent_dim=2,
        num_genes=2,
        intervention_type="KO",
    )
    module._checkpoint_contract = {
        **module._checkpoint_contract,
        "training": {"intervention_type": "KO", "direction_code": 0},
    }
    mapper_output = PTMDirectionMapperOutput(
        gene_ids=torch.zeros((1, 2), dtype=torch.long),
        directions=torch.ones((1, 2), dtype=torch.long),
        attention_mask=torch.tensor([[1.0, 0.0]]),
    )

    with pytest.raises(RuntimeError, match="cannot serve active directions"):
        module.predict_expression_direction(
            mapper_output,
            torch.zeros((1, 2)),
            _FakeScVIAdapter(),
            target_gene_symbols=["STAT3"],
            target_ensembl_ids=["ENSG00000168610"],
        )


def test_davf_direction_rejects_indices_from_the_wrong_vocabulary() -> None:
    module = _bare_direction_module()
    mapper_output = PTMDirectionMapperOutput(
        gene_ids=torch.zeros((1, 2), dtype=torch.long),
        directions=torch.zeros((1, 2), dtype=torch.long),
        attention_mask=torch.tensor([[1.0, 0.0]]),
    )

    with pytest.raises(ValueError, match="does not match scVI gene order"):
        module.predict_expression_direction(
            mapper_output,
            torch.zeros((1, 2)),
            _FakeScVIAdapter(),
            target_gene_indices=[0],
            target_gene_symbols=["STAT3"],
            target_ensembl_ids=["ENSG00000168610"],
        )


def test_davf_binds_a_schema_compatible_scvi_adapter() -> None:
    module = _bare_direction_module()

    bound = module.bind_scvi_adapter(_FakeScVIAdapter())

    assert bound is module.scvi_adapter


def test_davf_rechecks_live_scvi_provenance_for_formal_checkpoint() -> None:
    module = _bare_direction_module()
    module._checkpoint_contract = {
        "scvi": {
            "model_path": str(Path("checkpoints/scvi/model").resolve()),
            "latent_dim": 2,
            "num_genes": 2,
            "gene_names": ["BASE", "STAT3"],
        }
    }
    adapter = _FakeScVIAdapter()
    adapter.config = SimpleNamespace(model_path="checkpoints/scvi/other-model")

    with pytest.raises(ValueError, match="model path does not match checkpoint provenance"):
        module.bind_scvi_adapter(adapter)


def test_davf_load_scvi_adapter_uses_configured_checkpoint() -> None:
    module = _bare_direction_module()
    module.device = torch.device("cpu")
    adapter = _FakeScVIAdapter()

    with patch(
        "src.models.scvi_adapter.ScVIAdapter.from_trained_model",
        return_value=adapter,
    ) as load:
        assert module.load_scvi_adapter(adata="context") is adapter

    load.assert_called_once()
    args, kwargs = load.call_args
    assert args == ("checkpoints/scvi/model",)
    assert kwargs["adata"] == "context"
    assert kwargs["config"].n_latent == 2
    assert kwargs["config"].device == "cpu"


def test_davf_direction_rejects_scvi_latent_dimension_mismatch() -> None:
    module = _bare_direction_module()
    adapter = _FakeScVIAdapter()
    adapter.n_latent = 3
    mapper_output = PTMDirectionMapperOutput(
        gene_ids=torch.zeros((1, 1), dtype=torch.long),
        directions=torch.zeros((1, 1), dtype=torch.long),
        attention_mask=torch.ones((1, 1)),
    )

    with pytest.raises(ValueError, match="latent dimension mismatch"):
        module.predict_expression_direction(
            mapper_output,
            torch.zeros((1, 2)),
            adapter,
            target_gene_indices=[1],
            target_gene_symbols=["STAT3"],
            target_ensembl_ids=["ENSG00000168610"],
        )


def test_davf_direction_rejects_scvi_gene_dimension_mismatch() -> None:
    module = _bare_direction_module()
    adapter = _FakeScVIAdapter()
    adapter.n_genes = 3
    mapper_output = PTMDirectionMapperOutput(
        gene_ids=torch.zeros((1, 1), dtype=torch.long),
        directions=torch.zeros((1, 1), dtype=torch.long),
        attention_mask=torch.ones((1, 1)),
    )

    with pytest.raises(ValueError, match="gene dimension mismatch"):
        module.predict_expression_direction(
            mapper_output,
            torch.zeros((1, 2)),
            adapter,
            target_gene_indices=[1],
            target_gene_symbols=["STAT3"],
            target_ensembl_ids=["ENSG00000168610"],
        )


def test_davf_direction_rejects_target_gene_order_mismatch() -> None:
    module = _bare_direction_module()
    mapper_output = PTMDirectionMapperOutput(
        gene_ids=torch.zeros((1, 1), dtype=torch.long),
        directions=torch.zeros((1, 1), dtype=torch.long),
        attention_mask=torch.ones((1, 1)),
    )

    with pytest.raises(ValueError, match="does not match scVI gene order"):
        module.predict_expression_direction(
            mapper_output,
            torch.zeros((1, 2)),
            _FakeScVIAdapter(),
            target_gene_indices=[1],
            target_gene_symbols=["WRONG"],
            target_ensembl_ids=["ENSG00000168610"],
        )


def test_davf_direction_evidence_rejects_fallback_or_missing_embedding() -> None:
    mapper_output = PTMDirectionMapperOutput(
        gene_ids=torch.zeros((1, 1), dtype=torch.long),
        directions=torch.zeros((1, 1), dtype=torch.long),
        attention_mask=torch.ones((1, 1)),
    )
    kwargs = {
        "target_gene_indices": [0],
        "target_gene_symbols": ["STAT3"],
        "target_ensembl_ids": ["ENSG00000168610"],
    }
    with pytest.raises(RuntimeError, match="checkpoint"):
        _bare_direction_module(loaded=False).predict_expression_direction(
            mapper_output, torch.zeros((1, 2)), _FakeScVIAdapter(), **kwargs
        )
    with pytest.raises(RuntimeError, match="embedding_asset_path"):
        _bare_direction_module(with_assets=False).predict_expression_direction(
            mapper_output, torch.zeros((1, 2)), _FakeScVIAdapter(), **kwargs
        )


def test_davf_direction_evidence_rejects_unversioned_loaded_checkpoint() -> None:
    module = _bare_direction_module()
    module._current_checkpoint_contract_valid = False
    mapper_output = PTMDirectionMapperOutput(
        gene_ids=torch.zeros((1, 1), dtype=torch.long),
        directions=torch.zeros((1, 1), dtype=torch.long),
        attention_mask=torch.ones((1, 1)),
    )

    with pytest.raises(RuntimeError, match="schema_version=2"):
        module.predict_expression_direction(
            mapper_output,
            torch.zeros((1, 2)),
            _FakeScVIAdapter(),
            target_gene_indices=[1],
            target_gene_symbols=["STAT3"],
            target_ensembl_ids=["ENSG00000168610"],
        )


def test_davf_direction_evidence_rejects_legacy_without_latent_predict() -> None:
    module = _bare_direction_module()
    module.latent_davf = _LegacyLikeLatentDAVF()
    mapper_output = PTMDirectionMapperOutput(
        gene_ids=torch.zeros((1, 1), dtype=torch.long),
        directions=torch.zeros((1, 1), dtype=torch.long),
        attention_mask=torch.ones((1, 1)),
    )

    with pytest.raises(RuntimeError, match="does not expose latent expression prediction"):
        module.predict_expression_direction(
            mapper_output,
            torch.zeros((1, 2)),
            _FakeScVIAdapter(),
            target_gene_indices=[0],
            target_gene_symbols=["STAT3"],
            target_ensembl_ids=["ENSG00000168610"],
        )


def test_variant_cell_state_prediction_handles_davf_gene_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(predictions.STATE, "model", SimpleNamespace(use_davf=True))
    monkeypatch.setattr(predictions.STATE, "feature_extractor", None)
    monkeypatch.setattr(predictions.STATE, "max_sequence_length", 100)
    monkeypatch.setattr(predictions.STATE, "ptm_type_to_idx", {})

    async def run() -> tuple[str | None, list[str]]:
        warnings: list[str] = []
        result = await predictions._variant_cell_state_prediction("ACDE", ["phosphorylation"], warnings)
        return result, warnings

    result, warnings = asyncio.run(run())
    assert result is None
    assert warnings and "gene_symbol" in warnings[0]
