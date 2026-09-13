from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from src.integration.perturbgen.contracts import (
    DAVFDirectionEvidence,
    PTMSiteDirectionProposal,
)
from src.integration.perturbgen.orchestrator import (
    DAVFPerturbGenOrchestrator,
    materialize_candidate_config,
    merge_route_preparations,
    merge_route_reports,
)
from src.models.ptm_direction_mapper import PTMDirectionMapper


GENE = "STAT3"
ENSEMBL = "ENSG00000168610"


def _semantic_context(route: str = "KO") -> dict[str, str]:
    return {
        "context": "disease",
        "intervention": route,
        "comparison_baseline": "normal",
        "reference_axis": "disease-minus-normal",
        "research_objective": "replication",
        "evidence_source": "donor_expression+davf_decode",
        "cohort": "formal",
    }


def _proposal() -> PTMSiteDirectionProposal:
    return PTMSiteDirectionProposal(
        gene_symbol=GENE,
        ensembl_id=ENSEMBL,
        position=12,
        ptm_type="phosphorylation",
        proposed_direction="down",
        site_probability=0.95,
        provenance="ptm-site/test",
    )


class _FakeDAVF:
    def __init__(self, intervention_type: str = "KO", predicted_direction: str = "down"):
        self.config = SimpleNamespace(intervention_type=intervention_type)
        self._embedding_symbol_to_ensembl = {GENE: ENSEMBL}
        self.predicted_direction = predicted_direction
        self.last_mapper_output = None

    @property
    def embedding_symbol_to_ensembl(self):
        return dict(self._embedding_symbol_to_ensembl)

    def build_perturbgen_direction_mapper(self):
        return PTMDirectionMapper(gene_to_idx={GENE: 17})

    def predict_expression_direction(
        self,
        mapper_output,
        z_0,
        scvi_adapter,
        *,
        target_gene_symbols,
        target_ensembl_ids,
        **_kwargs,
    ):
        self.last_mapper_output = mapper_output
        assert torch.as_tensor(z_0).shape[0] == len(target_gene_symbols)
        assert scvi_adapter is None
        delta = 1.0 if self.predicted_direction == "up" else -1.0
        return [
            DAVFDirectionEvidence(
                gene_symbol=symbol,
                ensembl_id=ensembl_id,
                predicted_direction=self.predicted_direction,
                predicted_delta=delta,
                model_source="davf",
                checkpoint_provenance="checkpoints/davf/formal.pt",
                embedding_provenance="outputs/perturbgen/asset",
                confidence=0.5,
            )
            for symbol, ensembl_id in zip(
                target_gene_symbols, target_ensembl_ids, strict=True
            )
        ]


def test_prepare_candidate_uses_explicit_ko_route_not_ptm_type_direction():
    davf = _FakeDAVF("KO")
    orchestrator = DAVFPerturbGenOrchestrator(davf_module=davf)

    preparation = orchestrator.prepare_candidate(
        _proposal(),
        torch.zeros((1, 64)),
        cell_type="K562",
        ptm_context="STAT3:S12",
        observed_log2fc=-1.0,
        observed_fdr=0.01,
        observed_direction="down",
        semantic_context=_semantic_context(),
    )

    assert preparation.status == "pass"
    assert preparation.invocation is not None
    assert preparation.invocation.intervention_type == "KO"
    assert preparation.invocation.target_token_id == 17
    assert preparation.invocation.perturbation_mode == "overexpress"
    assert davf.last_mapper_output.directions[0, 0].item() == 0


def test_invocation_rejects_davf_score_that_is_not_bound_to_confidence():
    davf = _FakeDAVF("KO")
    preparation = DAVFPerturbGenOrchestrator(davf_module=davf).prepare_candidate(
        _proposal(),
        torch.zeros((1, 64)),
        cell_type="K562",
        ptm_context="STAT3:S12",
        observed_log2fc=-1.0,
        observed_fdr=0.01,
        observed_direction="down",
        semantic_context=_semantic_context(),
    )
    assert preparation.invocation is not None
    candidate = replace(preparation.invocation.candidate, davf_score=0.4)
    with pytest.raises(ValueError, match="exactly match"):
        replace(preparation.invocation, candidate=candidate)


def test_prepare_candidate_blocks_perturbgen_when_direction_gate_fails():
    davf = _FakeDAVF("KD", predicted_direction="up")
    orchestrator = DAVFPerturbGenOrchestrator(davf_module=davf)

    preparation = orchestrator.prepare_candidate(
        _proposal(),
        torch.zeros((1, 64)),
        cell_type="Jurkat",
        ptm_context="STAT3:S12",
        observed_log2fc=-1.0,
        observed_fdr=0.01,
        observed_direction="down",
        semantic_context=_semantic_context("KD"),
    )

    assert preparation.status == "fail"
    assert preparation.invocation is None
    assert preparation.candidate is None
    assert "direction_evidence_disagreement" in preparation.direction_gate.reasons
    assert davf.last_mapper_output.directions[0, 0].item() == 1


def test_prepare_candidates_requires_symbol_ensembl_pair_from_verified_aliases():
    davf = _FakeDAVF("KO")
    orchestrator = DAVFPerturbGenOrchestrator(davf_module=davf)
    bad = PTMSiteDirectionProposal(
        gene_symbol=GENE,
        ensembl_id="ENSG00000141510",
        position=12,
        ptm_type="phosphorylation",
        proposed_direction="down",
        site_probability=0.95,
        provenance="ptm-site/test",
    )

    with pytest.raises(ValueError, match="symbol/Ensembl pair"):
        orchestrator.prepare_candidate(
            bad,
            torch.zeros((1, 64)),
            cell_type="K562",
            ptm_context="STAT3:S12",
            observed_log2fc=-1.0,
            observed_fdr=0.01,
            observed_direction="down",
            semantic_context=_semantic_context(),
        )


def test_materialize_candidate_config_rewrites_target_and_records_route(tmp_path):
    davf = _FakeDAVF("KO")
    preparation = DAVFPerturbGenOrchestrator(davf_module=davf).prepare_candidate(
        _proposal(),
        torch.zeros((1, 64)),
        cell_type="K562",
        ptm_context="STAT3:S12",
        observed_log2fc=-1.0,
        observed_fdr=0.01,
        observed_direction="down",
        semantic_context=_semantic_context(),
    )
    assert preparation.invocation is not None
    config = {
        "pipeline": {"output_root": str(tmp_path / "base")},
        "stages": {
            "perturb": {
                "perturb_config": {
                    "trainer": {
                        "genes_to_perturb": ["LCK"],
                        "perturbation_mode": "mask",
                    },
                    "datamodule": {},
                },
                "expected_outputs": [
                    {
                        "path": "results",
                        "discover_glob": "*_gLCK_s*_tmask.h5ad",
                    }
                ],
            }
        },
    }

    materialized = materialize_candidate_config(
        config,
        preparation.invocation,
        output_root=tmp_path / "ko" / "STAT3",
    )

    perturb = materialized["stages"]["perturb"]
    assert materialized["pipeline"]["intervention_type"] == "KO"
    assert materialized["pipeline"]["candidate_ensembl_id"] == ENSEMBL
    assert perturb["perturb_config"]["trainer"]["genes_to_perturb"] == [GENE]
    assert perturb["perturb_config"]["trainer"]["perturbation_mode"] == "overexpress"
    assert perturb["expected_outputs"][0]["discover_glob"] == "*_gSTAT3_s*_toverexpress.h5ad"


def test_merge_route_preparations_merges_only_by_ensembl_id():
    ko = DAVFPerturbGenOrchestrator(davf_module=_FakeDAVF("KO")).prepare_candidate(
        _proposal(),
        torch.zeros((1, 64)),
        cell_type="K562",
        ptm_context="STAT3:S12",
        observed_log2fc=-1.0,
        observed_fdr=0.01,
        observed_direction="down",
        semantic_context=_semantic_context(),
    )
    kd = DAVFPerturbGenOrchestrator(davf_module=_FakeDAVF("KD")).prepare_candidate(
        _proposal(),
        torch.zeros((1, 64)),
        cell_type="Jurkat",
        ptm_context="STAT3:S12",
        observed_log2fc=-1.0,
        observed_fdr=0.01,
        observed_direction="down",
        semantic_context=_semantic_context("KD"),
    )

    merged = merge_route_preparations([ko, kd])

    assert len(merged) == 1
    assert merged[0]["ensembl_id"] == ENSEMBL
    assert set(merged[0]["routes"]) == {"KO", "KD"}


def test_merge_route_reports_keeps_independent_run_artifacts_under_ensembl():
    ko = DAVFPerturbGenOrchestrator(davf_module=_FakeDAVF("KO")).prepare_candidate(
        _proposal(),
        torch.zeros((1, 64)),
        cell_type="K562",
        ptm_context="STAT3:S12",
        observed_log2fc=-1.0,
        observed_fdr=0.01,
        observed_direction="down",
        semantic_context=_semantic_context(),
    )
    kd = DAVFPerturbGenOrchestrator(davf_module=_FakeDAVF("KD")).prepare_candidate(
        _proposal(),
        torch.zeros((1, 64)),
        cell_type="Jurkat",
        ptm_context="STAT3:S12",
        observed_log2fc=-1.0,
        observed_fdr=0.01,
        observed_direction="down",
        semantic_context=_semantic_context("KD"),
    )
    reports = [
        {
            "intervention_type": "KO",
            "davf_config": "configs/davf_ko.yaml",
            "context_h5ad": "ko.h5ad",
            "candidates": [ko.to_dict()],
            "perturbgen_runs": [{"ensembl_id": ENSEMBL, "output_root": "ko-out"}],
        },
        {
            "intervention_type": "KD",
            "davf_config": "configs/davf_kd.yaml",
            "context_h5ad": "kd.h5ad",
            "candidates": [kd.to_dict()],
            "perturbgen_runs": [{"ensembl_id": ENSEMBL, "output_root": "kd-out"}],
        },
    ]

    merged = merge_route_reports(reports)

    assert len(merged) == 1
    assert set(merged[0]["routes"]) == {"KO", "KD"}
    assert merged[0]["routes"]["KO"]["perturbgen_runs"][0]["output_root"] == "ko-out"
    assert merged[0]["routes"]["KD"]["perturbgen_runs"][0]["output_root"] == "kd-out"


def test_merge_route_reports_rejects_two_reports_from_same_route():
    report = {
        "intervention_type": "KO",
        "candidates": [],
        "perturbgen_runs": [],
    }

    with pytest.raises(ValueError, match="requires exactly one KO and one KD"):
        merge_route_reports([report, report])
