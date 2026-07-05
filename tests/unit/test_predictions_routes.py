"""Tests for prediction API schema validation and helper functions.

Routes themselves are tested via TestClient in ``test_api_routes.py``.
This file covers Pydantic model validation and standalone helper functions
from ``src.api.routes.predictions`` that do not require a running server.
"""

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    PTMSite,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    PTMEffect,
    PathwayImpact,
)

from src.api.routes.predictions import (
    _collect_davf_inputs,
    _model_supports_davf,
    _DEFAULT_EFFECT,
    _DEFAULT_DELTA_PROB,
)


# ---------------------------------------------------------------------------
# PTMSite schema
# ---------------------------------------------------------------------------

class TestPTMSite:
    """Tests for PTMSite Pydantic model."""

    def test_valid_ptm_site(self):
        """PTMSite should accept valid data."""
        site = PTMSite(position=100, type="phosphorylation", amino_acid="S")
        assert site.position == 100
        assert site.type == "phosphorylation"

    def test_minimal_ptm_site(self):
        """PTMSite should work with minimal data."""
        site = PTMSite(position=1, type="acetylation", amino_acid="K")
        assert site.position == 1
        assert site.type == "acetylation"

    def test_optional_gene_symbol(self):
        """PTMSite gene_symbol should be optional and settable."""
        site = PTMSite(
            position=50, type="phosphorylation", amino_acid="Y",
            gene_symbol="TP53",
        )
        assert site.gene_symbol == "TP53"

    def test_gene_symbol_defaults_to_none(self):
        """PTMSite without gene_symbol should default to None."""
        site = PTMSite(position=10, type="methylation", amino_acid="K")
        assert site.gene_symbol is None

    def test_optional_amino_acid(self):
        """amino_acid field is optional."""
        site = PTMSite(position=5, type="phosphorylation", amino_acid="S")
        assert site.amino_acid == "S"

    def test_position_must_be_positive(self):
        """PTMSite position must be >= 1 (ge=1)."""
        with pytest.raises(ValidationError):
            PTMSite(position=0, type="phosphorylation", amino_acid="S")


# ---------------------------------------------------------------------------
# PredictionRequest schema
# ---------------------------------------------------------------------------

class TestPredictionRequest:
    """Tests for PredictionRequest Pydantic model."""

    def test_valid_request(self):
        """PredictionRequest should accept valid data."""
        sites = [PTMSite(position=100, type="phosphorylation", amino_acid="S")]
        req = PredictionRequest(sequence="MALWMRLLPL", ptm_sites=sites)
        assert req.sequence == "MALWMRLLPL"
        assert len(req.ptm_sites) == 1

    def test_multiple_sites(self):
        """PredictionRequest should accept multiple PTM sites."""
        sites = [
            PTMSite(position=10, type="phosphorylation", amino_acid="S"),
            PTMSite(position=20, type="acetylation", amino_acid="K"),
        ]
        req = PredictionRequest(sequence="MALWMRLLPL", ptm_sites=sites)
        assert len(req.ptm_sites) == 2

    def test_ptm_sites_defaults_to_empty_list(self):
        """ptm_sites should default to an empty list."""
        req = PredictionRequest(sequence="ACDEFGHIKL")
        assert req.ptm_sites == []

    def test_use_davf_defaults_false(self):
        """use_davf should default to False."""
        req = PredictionRequest(sequence="ACDEFGHIKL")
        assert req.use_davf is False

    def test_use_davf_can_be_set(self):
        """use_davf should accept True."""
        req = PredictionRequest(sequence="MALWMRLLPL", ptm_sites=[], use_davf=True)
        assert req.use_davf is True


# ---------------------------------------------------------------------------
# BatchPredictionRequest schema
# ---------------------------------------------------------------------------

class TestBatchPredictionRequest:
    """Tests for BatchPredictionRequest Pydantic model."""

    def test_valid_batch(self):
        """Batch request should accept valid samples."""
        batch = BatchPredictionRequest(
            samples=[
                PredictionRequest(sequence="ACDEFGHIKL", ptm_sites=[]),
                PredictionRequest(sequence="LMNPQRSTVWY", ptm_sites=[]),
            ]
        )
        assert len(batch.samples) == 2

    def test_rejects_excessive_batch_size(self):
        """Batch of over 1000 samples should be rejected by validator."""
        many_samples = [
            PredictionRequest(sequence="A", ptm_sites=[]) for _ in range(1001)
        ]
        with pytest.raises(Exception, match="exceed 1000"):
            BatchPredictionRequest(samples=many_samples)


# ---------------------------------------------------------------------------
# PredictionResponse schema
# ---------------------------------------------------------------------------

class TestPredictionResponse:
    """Tests for PredictionResponse Pydantic model."""

    def test_minimal_response(self):
        """PredictionResponse should accept required fields."""
        resp = PredictionResponse(
            cell_state="proliferation",
            confidence=0.95,
            probabilities={"proliferation": 0.95, "apoptosis": 0.05},
        )
        assert resp.cell_state == "proliferation"
        assert resp.confidence == 0.95
        assert resp.probabilities["proliferation"] == 0.95

    def test_predicted_cell_state_fallback(self):
        """predicted_cell_state should default to cell_state."""
        resp = PredictionResponse(
            cell_state="apoptosis",
            confidence=0.8,
            probabilities={"apoptosis": 0.8},
        )
        assert resp.predicted_cell_state == "apoptosis"

    def test_with_pathway_impacts(self):
        """PredictionResponse should accept optional pathway_impacts."""
        impacts = [
            PathwayImpact(
                pathway_name="Apoptosis",
                activity_change=0.7,
                confidence="high",
                key_genes=["TP53", "BAX"],
            )
        ]
        resp = PredictionResponse(
            cell_state="apoptosis",
            confidence=0.9,
            probabilities={"apoptosis": 0.9},
            pathway_impacts=impacts,
            processing_time_ms=12.5,
        )
        assert resp.pathway_impacts is not None
        assert len(resp.pathway_impacts) == 1
        assert resp.pathway_impacts[0].pathway_name == "Apoptosis"
        assert resp.processing_time_ms == 12.5


# ---------------------------------------------------------------------------
# PTMEffect schema
# ---------------------------------------------------------------------------

class TestPTMEffect:
    """Tests for PTMEffect Pydantic model."""

    def test_valid_effect(self):
        """PTMEffect should accept valid numeric ranges."""
        effect = PTMEffect(
            ptm_type="Phosphorylation",
            wildtype_prob=0.8,
            mutant_prob=0.2,
            delta_prob=-0.6,
            effect="loss",
        )
        assert effect.ptm_type == "Phosphorylation"
        assert effect.delta_prob == -0.6
        assert effect.effect == "loss"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class MockPTMSite:
    """Minimal mock mimicking the PTMSite schema's attribute interface."""
    def __init__(self, position, type, amino_acid="S", gene_symbol=None):
        self.position = position
        self.type = type
        self.amino_acid = amino_acid
        self.gene_symbol = gene_symbol

    def model_dump(self):
        d = {"position": self.position, "type": self.type, "amino_acid": self.amino_acid}
        if self.gene_symbol is not None:
            d["gene_symbol"] = self.gene_symbol
        return d


class TestCollectDAVFInputs:
    """Tests for _collect_davf_inputs helper."""

    def test_skips_sites_without_gene_symbol(self):
        """Sites without gene_symbol should be skipped."""
        sites = [MockPTMSite(position=10, type="phosphorylation", gene_symbol=None)]
        result = _collect_davf_inputs(sites, max_len=100)
        assert result == {}

    def test_collects_sites_with_gene_symbol(self):
        """Sites with gene_symbol should be collected."""
        sites = [
            MockPTMSite(position=10, type="phosphorylation", gene_symbol="BRAF"),
        ]
        result = _collect_davf_inputs(sites, max_len=100)
        assert result["davf_positions"] == [10]
        assert result["davf_ptm_types"] == ["phosphorylation"]
        assert result["davf_gene_names"] == ["BRAF"]

    def test_skips_out_of_range_position(self):
        """Positions beyond max_len should be skipped."""
        sites = [
            MockPTMSite(position=200, type="phosphorylation", gene_symbol="BRAF"),
        ]
        result = _collect_davf_inputs(sites, max_len=100)
        assert result == {}

    def test_mixed_sites(self):
        """Mixed sites should collect only valid ones with gene_symbol."""
        sites = [
            MockPTMSite(position=10, type="phosphorylation", gene_symbol="BRAF"),
            MockPTMSite(position=5, type="acetylation", gene_symbol=None),  # skipped
            MockPTMSite(position=50, type="ubiquitination", gene_symbol="TP53"),
        ]
        result = _collect_davf_inputs(sites, max_len=100)
        assert result["davf_positions"] == [10, 50]
        assert result["davf_ptm_types"] == ["phosphorylation", "ubiquitination"]
        assert result["davf_gene_names"] == ["BRAF", "TP53"]

    def test_returns_empty_dict_for_empty_input(self):
        """Empty site list should return empty dict."""
        result = _collect_davf_inputs([], max_len=100)
        assert result == {}

    def test_returns_empty_dict_for_none(self):
        """None input should return empty dict."""
        result = _collect_davf_inputs(None, max_len=100)
        assert result == {}


class TestModelSupportsDAVF:
    """Tests for _model_supports_davf helper."""

    def test_request_flag_true_returns_true(self):
        """When request_flag is True, DAVF is supported."""
        assert _model_supports_davf(True) is True

    def test_request_flag_false_checks_model(self):
        """When request_flag is False, DAVF depends on model.use_davf."""
        # STATE.model is None on import, so getattr returns False
        assert _model_supports_davf(False) is False


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    """Sanity checks for module-level defaults."""

    def test_default_effect_is_loss(self):
        assert _DEFAULT_EFFECT == "loss"

    def test_default_delta_prob(self):
        assert _DEFAULT_DELTA_PROB == 0.5
