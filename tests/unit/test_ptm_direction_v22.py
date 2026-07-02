"""
Unit tests for V22-03 / V22-05 enhancements to the PTM direction mapper.

Covers:
    - Extended PTM type registry (V22-05): succinylation, glycosylation, etc.
    - Runtime registration API (register_ptm_direction).
    - Pathway-context-aware direction overrides (V22-03): methylation is
      repressive in chromatin/DNA-damage contexts.
"""

import pytest
import torch
from unittest.mock import Mock

from src.data.schemas import PTMSite
from src.models.ptm_direction_mapper import (
    PTMDirectionMapper,
    PTM_DIRECTION_MAP,
    DEFAULT_DIRECTION,
    register_ptm_direction,
    register_pathway_context_override,
    PTM_PATHWAY_CONTEXT_OVERRIDES,
    DIRECTION_KO,
    DIRECTION_KD,
    DIRECTION_OE,
)


@pytest.fixture
def mock_geneformer_loader():
    loader = Mock()
    loader._gene_to_idx = {"TP53": 100, "BRAF": 200, "EGFR": 300}
    loader.get_vocab_size = Mock(return_value=5000)
    loader.embedding_dim = 1152
    return loader


@pytest.fixture
def mock_gene_mapper():
    mapper = Mock()
    mapper.map_gene_to_uniprot = Mock(return_value=None)
    return mapper


@pytest.fixture
def ptm_mapper(mock_geneformer_loader, mock_gene_mapper):
    return PTMDirectionMapper(
        geneformer_loader=mock_geneformer_loader,
        gene_mapper=mock_gene_mapper,
        max_targets=32,
    )


class TestExtendedPTMRegistry:
    """V22-05: extended PTM types beyond the original 6."""

    def test_registry_contains_extended_types(self):
        # Originally only 6; V22-05 added many more (≥10).
        assert len(PTM_DIRECTION_MAP) >= 10
        for required in [
            "phosphorylation", "ubiquitination", "acetylation", "methylation",
            "sumoylation", "neddylation",
            # V22-05 additions
            "succinylation", "glycosylation", "palmitoylation", "lactylation",
        ]:
            assert required in PTM_DIRECTION_MAP, f"missing {required}"

    def test_direction_constants(self):
        assert DIRECTION_KO == 0
        assert DIRECTION_KD == 1
        assert DIRECTION_OE == 2
        assert DEFAULT_DIRECTION == DIRECTION_OE

    def test_extended_ptm_directions(self, ptm_mapper):
        """Extended PTM types resolve to their registered direction."""
        cases = {
            "succinylation": DIRECTION_OE,
            "glycosylation": DIRECTION_OE,
            "malonylation": DIRECTION_KD,
            "citrullination": DIRECTION_KD,
        }
        for ptm_type, expected in cases.items():
            ptm_sites = [PTMSite(position=1, type=ptm_type)]
            result = ptm_mapper.map_ptms(ptm_sites, ["TP53"])
            assert result.directions[0, 0].item() == expected, ptm_type

    def test_register_new_ptm_direction(self, ptm_mapper):
        """register_ptm_direction adds a new type at runtime (V22-05)."""
        register_ptm_direction("bogusptm", DIRECTION_KD)
        try:
            ptm_sites = [PTMSite(position=1, type="bogusptm")]
            result = ptm_mapper.map_ptms(ptm_sites, ["TP53"])
            assert result.directions[0, 0].item() == DIRECTION_KD
        finally:
            PTM_DIRECTION_MAP.pop("bogusptm", None)

    def test_register_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction must be"):
            register_ptm_direction("foo", 9)

    def test_register_duplicate_without_overwrite_raises(self):
        register_ptm_direction("dupptm", DIRECTION_KO)
        try:
            with pytest.raises(ValueError, match="already registered"):
                register_ptm_direction("dupptm", DIRECTION_OE)
            # overwrite=True succeeds
            register_ptm_direction("dupptm", DIRECTION_OE, overwrite=True)
            assert PTM_DIRECTION_MAP["dupptm"] == DIRECTION_OE
        finally:
            PTM_DIRECTION_MAP.pop("dupptm", None)

    def test_register_ptm_type_classmethod(self, ptm_mapper):
        PTMDirectionMapper.register_ptm_type("classmethodptm", DIRECTION_KO)
        try:
            ptm_sites = [PTMSite(position=1, type="classmethodptm")]
            result = ptm_mapper.map_ptms(ptm_sites, ["TP53"])
            assert result.directions[0, 0].item() == DIRECTION_KO
        finally:
            PTM_DIRECTION_MAP.pop("classmethodptm", None)


class TestPathwayContextOverrides:
    """V22-03: context-aware direction resolution."""

    def test_methylation_default_is_oe(self, ptm_mapper):
        """Without context, methylation defaults to OE (stabilizing)."""
        ptm_sites = [PTMSite(position=1, type="methylation")]
        result = ptm_mapper.map_ptms(ptm_sites, ["TP53"])
        assert result.directions[0, 0].item() == DIRECTION_OE

    def test_methylation_is_repressive_in_chromatin_context(self, ptm_mapper):
        """V22-03: methylation → KD in chromatin/DNA-damage contexts."""
        ptm_sites = [PTMSite(position=1, type="methylation")]
        for ctx in ["Chromatin remodeling", "Histone code", "DNA Damage response",
                    "gene regulation", "transcription"]:
            result = ptm_mapper.map_ptms(ptm_sites, ["TP53"], pathway_context=ctx)
            assert result.directions[0, 0].item() == DIRECTION_KD, f"ctx={ctx}"

    def test_methylation_remains_oe_in_signaling_context(self, ptm_mapper):
        """In a non-repressive context, methylation keeps its OE default."""
        ptm_sites = [PTMSite(position=1, type="methylation")]
        result = ptm_mapper.map_ptms(ptm_sites, ["TP53"], pathway_context="MAPK signaling")
        assert result.directions[0, 0].item() == DIRECTION_OE

    def test_context_override_registry_runtime(self, ptm_mapper):
        """register_pathway_context_override applies at runtime (V22-03)."""
        register_pathway_context_override("phosphorylation", "apoptosis", DIRECTION_KD)
        try:
            assert ("phosphorylation", "apoptosis") in PTM_PATHWAY_CONTEXT_OVERRIDES
            ptm_sites = [PTMSite(position=1, type="phosphorylation")]
            result = ptm_mapper.map_ptms(ptm_sites, ["TP53"], pathway_context="apoptosis")
            assert result.directions[0, 0].item() == DIRECTION_KD
        finally:
            PTM_PATHWAY_CONTEXT_OVERRIDES.pop(("phosphorylation", "apoptosis"), None)

    def test_context_override_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction must be"):
            register_pathway_context_override("foo", "bar", 9)

    def test_map_batch_accepts_per_sample_contexts(self, ptm_mapper):
        """V22-03: map_batch propagates per-sample pathway contexts."""
        batch_ptm_sites = [
            [PTMSite(position=1, type="methylation")],   # repressive ctx
            [PTMSite(position=1, type="methylation")],   # signaling ctx
        ]
        batch_gene_names = [["TP53"], ["TP53"]]
        contexts = ["Chromatin remodeling", "MAPK signaling"]

        result = ptm_mapper.map_batch(batch_ptm_sites, batch_gene_names,
                                      pathway_contexts=contexts)
        assert result.directions[0, 0].item() == DIRECTION_KD  # sample 0
        assert result.directions[1, 0].item() == DIRECTION_OE  # sample 1

    def test_map_batch_contexts_length_mismatch_raises(self, ptm_mapper):
        with pytest.raises(ValueError, match="pathway_contexts"):
            ptm_mapper.map_batch(
                [[PTMSite(position=1, type="methylation")]],
                [["TP53"]],
                pathway_contexts=["ctx1", "ctx2"],
            )

    def test_context_override_is_case_insensitive(self, ptm_mapper):
        """Context matching is case-insensitive on both type and context."""
        ptm_sites = [PTMSite(position=1, type="METHYLATION")]
        result = ptm_mapper.map_ptms(ptm_sites, ["TP53"], pathway_context="CHROMATIN")
        assert result.directions[0, 0].item() == DIRECTION_KD
