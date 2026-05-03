"""Tests for HGVS variant parser."""
import pytest
from src.analysis.variant_parser import (
    HGVSVariantParser,
    VariantComponents,
    parse_variant,
)


class TestHGVSVariantParser:
    """Test HGVS variant parsing functionality."""

    @pytest.fixture
    def parser(self):
        """Create parser fixture."""
        return HGVSVariantParser()

    def test_parse_full_hgvs_notation(self, parser):
        """Test parsing full HGVS notation like 'NP_004324.2:p.Val600Glu'."""
        variant = parser.parse("NP_004324.2:p.Val600Glu")

        assert variant.accession == "NP_004324.2"
        assert variant.position == 600
        assert variant.position_0based == 599  # 0-based conversion
        assert variant.ref_aa == "V"
        assert variant.alt_aa == "E"
        assert variant.gene_symbol is None  # No gene symbol in full notation

    def test_parse_shorthand_notation(self, parser):
        """Test parsing shorthand notation like 'BRAF:p.V600E'."""
        # Note: hgvs library may not support shorthand directly
        # This tests the expected behavior if supported
        variant = parser.parse("BRAF:p.Val600Glu")

        assert variant.gene_symbol == "BRAF"
        assert variant.position == 600
        assert variant.ref_aa == "V"
        assert variant.alt_aa == "E"

    def test_invalid_hgvs_raises_valueerror(self, parser):
        """Test that invalid HGVS raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parser.parse("invalid_hgvs_string")

        assert "Invalid HGVS notation" in str(exc_info.value)

    def test_position_conversion_1based_to_0based(self, parser):
        """Test that HGVS 1-based position converts to 0-based Python index."""
        # HGVS position 1 -> Python index 0
        variant1 = parser.parse("NP_004324.2:p.Val1Glu")
        assert variant1.position == 1
        assert variant1.position_0based == 0

        # HGVS position 600 -> Python index 599
        variant600 = parser.parse("NP_004324.2:p.Val600Glu")
        assert variant600.position == 600
        assert variant600.position_0based == 599

    def test_reference_validation_matches(self, parser):
        """Test reference validation when sequence matches."""
        variant = parser.parse("NP_004324.2:p.Val600Glu")

        # Create a sequence where position 599 (0-based) is 'V'
        sequence = "A" * 599 + "V" + "A" * 100

        assert parser.validate_reference(variant, sequence) is True

    def test_reference_validation_mismatch(self, parser):
        """Test reference validation when sequence doesn't match."""
        variant = parser.parse("NP_004324.2:p.Val600Glu")

        # Create a sequence where position 599 is NOT 'V'
        sequence = "A" * 599 + "L" + "A" * 100

        assert parser.validate_reference(variant, sequence) is False

    def test_reference_validation_position_exceeds_length(self, parser):
        """Test reference validation when position exceeds sequence length."""
        variant = parser.parse("NP_004324.2:p.Val600Glu")

        # Short sequence that doesn't reach position 600
        sequence = "A" * 100

        assert parser.validate_reference(variant, sequence) is False

    def test_aa_3to1_mapping(self, parser):
        """Test 3-letter to 1-letter amino acid code conversion."""
        test_cases = [
            ("NP_004324.2:p.Ala100Gly", "A", "G"),
            ("NP_004324.2:p.Arg50Lys", "R", "K"),
            ("NP_004324.2:p.Cys200Ter", "C", "*"),  # Stop codon
        ]

        for hgvs, expected_ref, expected_alt in test_cases:
            variant = parser.parse(hgvs)
            assert variant.ref_aa == expected_ref
            assert variant.alt_aa == expected_alt

    def test_convenience_function_parse_variant(self):
        """Test the convenience function parse_variant."""
        variant = parse_variant("NP_004324.2:p.Val600Glu")

        assert variant.position == 600
        assert variant.ref_aa == "V"
        assert variant.alt_aa == "E"

    def test_variant_components_namedtuple(self):
        """Test that VariantComponents is a proper NamedTuple."""
        variant = VariantComponents(
            gene_symbol="BRAF",
            accession="NP_004324.2",
            position=600,
            position_0based=599,
            ref_aa="V",
            alt_aa="E",
            hgvs_string="NP_004324.2:p.Val600Glu",
        )

        # Access by name
        assert variant.gene_symbol == "BRAF"
        assert variant.position == 600

        # Access by index
        assert variant[0] == "BRAF"
        assert variant[2] == 600

    def test_extract_gene_symbol_from_shorthand(self, parser):
        """Test gene symbol extraction from shorthand notation."""
        # Shorthand notation
        assert parser._extract_gene_symbol("BRAF:p.V600E") == "BRAF"
        assert parser._extract_gene_symbol("TP53:p.R175H") == "TP53"

        # Full accession notation - no gene symbol
        assert parser._extract_gene_symbol("NP_004324.2:p.Val600Glu") is None

        # No colon - no gene symbol
        assert parser._extract_gene_symbol("invalid") is None
