"""
Unit tests for PTMDirectionMapper module.

Tests PTM type → direction mapping and gene name → ID resolution
for DAVF inference pipeline.
"""

import pytest
import torch
from unittest.mock import Mock
from pydantic import BaseModel

from src.data.schemas import PTMSite


# Mock fixtures for isolated testing
@pytest.fixture
def mock_geneformer_loader():
    """Create mock GeneformerEmbeddingLoader with known vocabulary."""
    loader = Mock()
    loader._gene_to_idx = {"TP53": 100, "BRAF": 200, "EGFR": 300}
    loader.get_vocab_size = Mock(return_value=5000)
    loader.embedding_dim = 1152
    return loader


@pytest.fixture
def mock_gene_mapper():
    """Create mock GeneMapper that returns None for unknown genes."""
    mapper = Mock()
    mapper.map_gene_to_uniprot = Mock(return_value=None)
    return mapper


@pytest.fixture
def ptm_mapper(mock_geneformer_loader, mock_gene_mapper):
    """Create PTMDirectionMapper with mocked dependencies."""
    from src.models.ptm_direction_mapper import PTMDirectionMapper

    return PTMDirectionMapper(geneformer_loader=mock_geneformer_loader, gene_mapper=mock_gene_mapper, max_targets=32)


class TestPTMDirectionMapper:
    """Test suite for PTMDirectionMapper."""

    def test_mapper_uses_data_layer_ptm_site_container(self):
        """Mapper should depend on a data-layer PTMSite, not the API schema."""
        from src.api.schemas import PTMSite as ApiPTMSite
        from src.models import ptm_direction_mapper as mapper_module

        assert issubclass(ApiPTMSite, BaseModel)
        assert mapper_module.PTMSite is not ApiPTMSite
        assert mapper_module.PTMSite.__module__ == "src.data.schemas"

    def test_phosphorylation_direction(self, ptm_mapper):
        """Phosphorylation on known gene produces direction=2 (OE), mask=1."""
        ptm_sites = [PTMSite(position=100, type="phosphorylation")]
        gene_names = ["TP53"]

        result = ptm_mapper.map_ptms(ptm_sites, gene_names)

        assert result.directions.shape == (1, 32)
        assert result.directions[0, 0].item() == 2  # OE
        assert result.attention_mask[0, 0].item() == 1.0  # Valid

    def test_ubiquitination_direction(self, ptm_mapper):
        """Ubiquitination on known gene produces direction=0 (KO), mask=1."""
        ptm_sites = [PTMSite(position=50, type="ubiquitination")]
        gene_names = ["BRAF"]

        result = ptm_mapper.map_ptms(ptm_sites, gene_names)

        assert result.directions[0, 0].item() == 0  # KO
        assert result.attention_mask[0, 0].item() == 1.0

    def test_unknown_gene_masked(self, ptm_mapper):
        """Unknown gene names produce attention_mask=0."""
        ptm_sites = [PTMSite(position=10, type="phosphorylation")]
        gene_names = ["UNKNOWNGENE123"]

        result = ptm_mapper.map_ptms(ptm_sites, gene_names)

        assert result.attention_mask[0, 0].item() == 0.0  # Masked
        assert result.gene_ids[0, 0].item() == 0  # Placeholder ID

    def test_batch_shapes(self, ptm_mapper):
        """Batch of mixed PTM types produces correct tensor shapes [B, K]."""
        batch_ptm_sites = [
            [PTMSite(position=1, type="phosphorylation")],
            [PTMSite(position=2, type="ubiquitination")],
            [PTMSite(position=3, type="acetylation")],
        ]
        batch_gene_names = [["TP53"], ["BRAF"], ["EGFR"]]

        result = ptm_mapper.map_batch(batch_ptm_sites, batch_gene_names)

        assert result.gene_ids.shape == (3, 32)
        assert result.directions.shape == (3, 32)
        assert result.attention_mask.shape == (3, 32)
        assert result.gene_ids.dtype == torch.long
        assert result.directions.dtype == torch.long
        assert result.attention_mask.dtype == torch.float

    def test_empty_input(self, ptm_mapper):
        """Empty PTM list returns zero tensors with shape [1, 32]."""
        result = ptm_mapper.map_ptms([], [])

        assert result.gene_ids.shape == (1, 32)
        assert result.directions.shape == (1, 32)
        assert result.attention_mask.shape == (1, 32)
        assert torch.all(result.attention_mask == 0.0)

    def test_ptm_type_normalization(self, ptm_mapper):
        """PTM type normalization handles case, hyphens, underscores."""
        # All variants should map to direction=2 (phosphorylation)
        variants = [
            "Phosphorylation",
            "PHOSPHORYLATION",
            "phospho-rylation",
            "phospho_rylation",
            "phospho rylation",
        ]

        for variant in variants:
            ptm_sites = [PTMSite(position=1, type=variant)]
            gene_names = ["TP53"]

            result = ptm_mapper.map_ptms(ptm_sites, gene_names)
            assert result.directions[0, 0].item() == 2, f"Failed for: {variant}"

    def test_all_ptm_types(self, ptm_mapper):
        """All 6 defined PTM types map to correct directions."""
        ptm_directions = {
            "phosphorylation": 2,
            "ubiquitination": 0,
            "acetylation": 2,
            "methylation": 2,
            "sumoylation": 1,
            "neddylation": 2,
        }

        for ptm_type, expected_dir in ptm_directions.items():
            ptm_sites = [PTMSite(position=1, type=ptm_type)]
            gene_names = ["TP53"]

            result = ptm_mapper.map_ptms(ptm_sites, gene_names)
            assert result.directions[0, 0].item() == expected_dir, f"Failed for {ptm_type}: expected {expected_dir}"

    def test_default_unknown_ptm(self, ptm_mapper):
        """Unknown PTM type defaults to direction=2 (OE)."""
        ptm_sites = [PTMSite(position=1, type="unknown_modification")]
        gene_names = ["TP53"]

        result = ptm_mapper.map_ptms(ptm_sites, gene_names)

        assert result.directions[0, 0].item() == 2  # Default OE

    def test_multiple_ptms_same_gene(self, ptm_mapper):
        """Multiple PTMs on same gene produce separate entries."""
        ptm_sites = [
            PTMSite(position=100, type="phosphorylation"),
            PTMSite(position=200, type="ubiquitination"),
        ]
        gene_names = ["TP53", "TP53"]

        result = ptm_mapper.map_ptms(ptm_sites, gene_names)

        assert result.directions[0, 0].item() == 2  # Phosphorylation
        assert result.directions[0, 1].item() == 0  # Ubiquitination
        assert result.attention_mask[0, 0].item() == 1.0
        assert result.attention_mask[0, 1].item() == 1.0

    def test_max_targets_truncation(self, ptm_mapper):
        """More than 32 PTMs truncated to max_targets."""
        # Create 40 PTM sites (position starts at 1 per PTMSite validation)
        ptm_sites = [PTMSite(position=i + 1, type="phosphorylation") for i in range(40)]
        gene_names = ["TP53"] * 40

        result = ptm_mapper.map_ptms(ptm_sites, gene_names)

        assert result.directions.shape == (1, 32)
        # First 32 should be valid
        assert result.attention_mask[0, :32].sum().item() == 32.0

    def test_mismatched_lengths_raises(self, ptm_mapper):
        """Mismatched ptm_sites and gene_names lengths raises ValueError."""
        ptm_sites = [PTMSite(position=1, type="phosphorylation")]
        gene_names = ["TP53", "BRAF"]  # Wrong length

        with pytest.raises(ValueError, match="must have same length"):
            ptm_mapper.map_ptms(ptm_sites, gene_names)

    @pytest.mark.parametrize(
        ("intervention_type", "expected_direction"),
        [("KO", 0), ("KD", 1), ("OE", 2)],
    )
    def test_formal_intervention_mapping_uses_explicit_route(self, intervention_type, expected_direction):
        from src.models.ptm_direction_mapper import PTMDirectionMapper

        mapper = PTMDirectionMapper(gene_to_idx={"TP53": 100})
        result = mapper.map_intervention_targets(["TP53"], intervention_type)

        assert result.gene_ids[0, 0].item() == 100
        assert result.directions[0, 0].item() == expected_direction
        assert result.attention_mask[0, 0].item() == 1.0

    def test_formal_intervention_mapping_does_not_mask_unknown_gene(self):
        from src.models.ptm_direction_mapper import PTMDirectionMapper

        mapper = PTMDirectionMapper(gene_to_idx={"TP53": 100})
        with pytest.raises(KeyError, match="absent or ambiguous"):
            mapper.map_intervention_targets(["UNKNOWNGENE"], "KO")

    def test_formal_intervention_mapping_does_not_silently_truncate_targets(self):
        from src.models.ptm_direction_mapper import PTMDirectionMapper

        mapper = PTMDirectionMapper(gene_to_idx={"TP53": 100}, max_targets=1)
        with pytest.raises(ValueError, match="exceeds max_targets"):
            mapper.map_intervention_targets(["TP53", "TP53"], "KO")

    def test_formal_intervention_mapping_requires_verified_asset(self, ptm_mapper):
        with pytest.raises(RuntimeError, match="verified PerturbGen gene-token mapping"):
            ptm_mapper.map_intervention_targets(["TP53"], "KO")


class TestPTMDirectionMapperOutput:
    """Tests for PTMDirectionMapperOutput dataclass."""

    def test_output_dtypes(self, ptm_mapper):
        """Output tensors have correct dtypes."""
        ptm_sites = [PTMSite(position=1, type="phosphorylation")]
        gene_names = ["TP53"]

        result = ptm_mapper.map_ptms(ptm_sites, gene_names)

        assert result.gene_ids.dtype == torch.long
        assert result.directions.dtype == torch.long
        assert result.attention_mask.dtype == torch.float

    def test_output_structure(self, ptm_mapper):
        """Output has all required fields."""
        ptm_sites = [PTMSite(position=1, type="phosphorylation")]
        gene_names = ["TP53"]

        result = ptm_mapper.map_ptms(ptm_sites, gene_names)

        assert hasattr(result, "gene_ids")
        assert hasattr(result, "directions")
        assert hasattr(result, "attention_mask")
