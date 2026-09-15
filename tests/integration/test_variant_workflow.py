"""Integration tests for variant effect workflow."""

import pytest
from unittest.mock import Mock, patch
import requests

from src.analysis.variant_workflow import (
    VariantEffectWorkflow,
    VariantEffectResult,
)
from src.analysis.variant_parser import VariantComponents


class TestVariantEffectWorkflow:
    """Test complete variant effect prediction workflow."""

    @pytest.fixture
    def mock_predictor(self):
        """Create mock VariantPTMEffectPredictor."""
        with patch("src.analysis.variant_workflow.VariantPTMEffectPredictor") as mock:
            mock_instance = Mock()
            mock_instance.predict_variant_effect.return_value = {
                "wildtype_prob": 0.8,
                "mutant_prob": 0.3,
                "delta_prob": -0.5,
                "effect": "loss",
            }
            mock.return_value = mock_instance
            yield mock

    @pytest.fixture
    def workflow(self, mock_predictor):
        """Create workflow fixture with mocked predictor."""
        return VariantEffectWorkflow(
            model_path="/fake/model.pt",
            ptm_types=["Phosphorylation", "Ubiquitination"],
        )

    def test_predict_from_hgvs_with_sequence(self, workflow):
        """Test complete workflow with provided sequence."""
        hgvs = "NP_004324.2:p.Val600Glu"
        sequence = "A" * 599 + "V" + "A" * 100  # Position 599 is 'V'

        result = workflow.predict_from_hgvs(hgvs, sequence=sequence)

        assert isinstance(result, VariantEffectResult)
        assert result.variant["hgvs"] == hgvs
        assert result.variant["position"] == 600
        assert result.variant["ref_aa"] == "V"
        assert result.variant["alt_aa"] == "E"
        assert result.sequence_info["length"] == 700
        assert result.sequence_info["validated"] is True
        assert "Phosphorylation" in result.ptm_effects
        assert "Ubiquitination" in result.ptm_effects

    def test_predict_from_hgvs_reference_mismatch(self, workflow):
        """Test workflow when reference amino acid doesn't match sequence."""
        hgvs = "NP_004324.2:p.Val600Glu"
        # Sequence has 'L' at position 599, not 'V'
        sequence = "A" * 599 + "L" + "A" * 100

        result = workflow.predict_from_hgvs(hgvs, sequence=sequence)

        assert result.sequence_info["validated"] is False

    def test_predict_from_hgvs_requires_sequence(self, workflow):
        """Test that workflow raises error when sequence not provided and gene mapping fails."""
        hgvs = "BRAF:p.Val600Glu"
        parsed_variant = VariantComponents(
            gene_symbol="BRAF",
            accession=None,
            position=600,
            position_0based=599,
            ref_aa="V",
            alt_aa="E",
            hgvs_string=hgvs,
        )

        with patch.object(workflow.parser, "parse", return_value=parsed_variant):
            with patch.object(workflow.gene_mapper, "map_gene_to_uniprot", return_value=None):
                with pytest.raises(ValueError) as exc_info:
                    workflow.predict_from_hgvs(hgvs)

        assert "Could not map gene" in str(exc_info.value)

    def test_predict_from_hgvs_uses_gene_mapper(self, workflow):
        """Test that workflow uses GeneMapper when no accession is provided."""
        hgvs = "BRAF:p.Val600Glu"
        sequence = "A" * 599 + "V" + "A" * 100  # Position 599 is 'V'
        parsed_variant = VariantComponents(
            gene_symbol="BRAF",
            accession=None,
            position=600,
            position_0based=599,
            ref_aa="V",
            alt_aa="E",
            hgvs_string=hgvs,
        )

        with patch.object(workflow.parser, "parse", return_value=parsed_variant):
            with patch.object(workflow.gene_mapper, "map_gene_to_uniprot", return_value="P15056"):
                with patch.object(workflow, "fetch_sequence_from_uniprot", return_value=sequence) as mock_fetch:
                    result = workflow.predict_from_hgvs(hgvs)

        mock_fetch.assert_called_once_with("P15056")
        assert result.sequence_info["length"] == 700
        assert result.sequence_info["validated"] is True

    def test_predict_from_hgvs_refseq_accession_maps_to_uniprot_before_fetch(self, workflow):
        """Test that RefSeq HGVS accessions are resolved before UniProt sequence fetch."""
        hgvs = "NP_004324.2:p.Val600Glu"
        sequence = "A" * 599 + "V" + "A" * 100

        with patch.object(workflow, "resolve_accession_to_uniprot", return_value="P15056") as mock_resolve:
            with patch.object(workflow, "fetch_sequence_from_uniprot", return_value=sequence) as mock_fetch:
                result = workflow.predict_from_hgvs(hgvs)

        mock_resolve.assert_called_once_with("NP_004324.2")
        mock_fetch.assert_called_once_with("P15056")
        assert result.sequence_info["validated"] is True

    def test_fetch_sequence_from_uniprot_rejects_non_uniprot_accessions(self, workflow):
        """Test that raw RefSeq accessions are not sent directly to UniProtKB endpoint."""
        with pytest.raises(ValueError) as exc_info:
            workflow.fetch_sequence_from_uniprot("NP_004324.2")

        assert "not a UniProt accession" in str(exc_info.value)

    def test_predict_from_hgvs_populates_pathway_impacts(self, workflow):
        """Test that pathway_impacts is populated via PTMNetworkAnalyzer."""
        hgvs = "BRAF:p.Val600Glu"
        sequence = "A" * 599 + "V" + "A" * 100

        # Mock predictors to return effects that will trigger MAPK pathway
        for _ptm_type, predictor in workflow.predictors.items():
            predictor.predict_variant_effect = Mock(
                return_value={
                    "wildtype_prob": 0.8,
                    "mutant_prob": 0.2,
                    "delta_prob": -0.6,
                    "effect": "loss",
                }
            )

        with patch.object(workflow.gene_mapper, "map_gene_to_uniprot", return_value="P15056"):
            result = workflow.predict_from_hgvs(hgvs, sequence=sequence)

        assert result.pathway_impacts is not None
        assert isinstance(result.pathway_impacts, dict)
        # BRAF is in MAPK/ERK pathway
        assert "MAPK/ERK" in result.pathway_impacts
        impact = result.pathway_impacts["MAPK/ERK"]
        assert "activity" in impact
        assert "genes" in impact
        assert "confidence" in impact
        assert impact["confidence"] in ("high", "medium")

    def test_predict_from_hgvs_pathway_analysis_graceful_degradation(self, workflow):
        """Test that pathway analysis failure results in empty pathway_impacts."""
        hgvs = "BRAF:p.Val600Glu"
        sequence = "A" * 599 + "V" + "A" * 100

        # Mock predictors
        for _ptm_type, predictor in workflow.predictors.items():
            predictor.predict_variant_effect = Mock(
                return_value={
                    "wildtype_prob": 0.8,
                    "mutant_prob": 0.2,
                    "delta_prob": -0.6,
                    "effect": "loss",
                }
            )

        # Mock network analyzer to fail
        workflow.network_analyzer.analyze_variant = Mock(side_effect=RuntimeError("Network error"))

        with patch.object(workflow.gene_mapper, "map_gene_to_uniprot", return_value="P15056"):
            result = workflow.predict_from_hgvs(hgvs, sequence=sequence)

        assert result.pathway_impacts == {}
        assert result.ptm_effects is not None
        assert len(result.ptm_effects) > 0

    def test_resolve_accession_to_uniprot_raises_on_lookup_failure(self, workflow):
        """Test that unresolved non-UniProt accessions raise a clear error."""
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"results": []}

        with patch("src.analysis.variant_workflow.requests.get", return_value=response):
            with pytest.raises(ValueError) as exc_info:
                workflow.resolve_accession_to_uniprot("NP_004324.2")

        assert "Could not resolve accession" in str(exc_info.value)

    def test_resolve_accession_to_uniprot_wraps_request_errors(self, workflow):
        """Test that accession lookup network errors are surfaced as ValueError."""
        with patch(
            "src.analysis.variant_workflow.requests.get",
            side_effect=requests.RequestException("network error"),
        ):
            with pytest.raises(ValueError) as exc_info:
                workflow.resolve_accession_to_uniprot("NP_004324.2")

        assert "Failed to resolve accession" in str(exc_info.value)

    def test_predict_from_hgvs_parses_shorthand(self, workflow):
        """Test workflow with shorthand notation."""
        hgvs = "BRAF:p.Val600Glu"
        sequence = "A" * 599 + "V" + "A" * 100

        result = workflow.predict_from_hgvs(hgvs, sequence=sequence)

        assert result.variant["gene_symbol"] == "BRAF"
        assert result.variant["position"] == 600

    def test_predict_batch(self, workflow):
        """Test batch prediction."""
        variants = [
            {"hgvs": "NP_004324.2:p.Val600Glu", "sequence": "A" * 599 + "V" + "A" * 100},
            {"hgvs": "NP_000546.2:p.Arg175His", "sequence": "A" * 174 + "R" + "A" * 100},
        ]

        results = workflow.predict_batch(variants)

        assert len(results) == 2
        assert all(isinstance(r, VariantEffectResult) for r in results)
        assert results[0].variant["position"] == 600
        assert results[1].variant["position"] == 175

    def test_predict_batch_handles_errors(self, workflow):
        """Test batch prediction handles individual errors gracefully."""
        variants = [
            {"hgvs": "NP_004324.2:p.Val600Glu", "sequence": "A" * 599 + "V" + "A" * 100},
            {"hgvs": "invalid_hgvs"},  # No sequence, will fail
        ]

        results = workflow.predict_batch(variants)

        assert len(results) == 2
        # First should succeed
        assert "error" not in results[0].variant
        # Second should have error info
        assert "error" in results[1].variant

    def test_workflow_integration_components(self, workflow):
        """Test that workflow integrates all components."""
        hgvs = "NP_004324.2:p.Val600Glu"
        sequence = "A" * 599 + "V" + "A" * 100

        result = workflow.predict_from_hgvs(hgvs, sequence=sequence)

        # Verify parser worked
        assert result.variant["ref_aa"] == "V"
        assert result.variant["alt_aa"] == "E"

        # Verify predictor was called
        assert len(result.ptm_effects) > 0
        for _ptm_type, effect in result.ptm_effects.items():
            assert "wildtype_prob" in effect
            assert "mutant_prob" in effect
            assert "delta_prob" in effect
            assert "effect" in effect

    def test_variant_effect_result_dataclass(self):
        """Test VariantEffectResult dataclass structure."""
        result = VariantEffectResult(
            variant={"hgvs": "test", "position": 100},
            sequence_info={"length": 200},
            ptm_effects={"Phosphorylation": {"effect": "gain"}},
        )

        assert result.variant["hgvs"] == "test"
        assert result.sequence_info["length"] == 200
        assert result.ptm_effects["Phosphorylation"]["effect"] == "gain"
        assert result.pathway_impacts is None  # Default value

    def test_workflow_with_partial_predictor_failure(self, mock_predictor):
        """Test workflow when some predictors fail to load."""

        # Make one predictor fail
        def side_effect(*args, **kwargs):
            if kwargs.get("ptm_type") == "Phosphorylation":
                raise RuntimeError("Failed to load")
            mock_instance = Mock()
            mock_instance.predict_variant_effect.return_value = {
                "wildtype_prob": 0.5,
                "mutant_prob": 0.5,
                "delta_prob": 0.0,
                "effect": "neutral",
            }
            return mock_instance

        mock_predictor.side_effect = side_effect

        workflow = VariantEffectWorkflow(
            model_path="/fake/model.pt",
            ptm_types=["Phosphorylation", "Ubiquitination"],
        )

        # Should have only one predictor
        assert "Phosphorylation" not in workflow.predictors
