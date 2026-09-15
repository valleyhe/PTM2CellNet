"""HGVS variant parsing module (FEAT-01)."""

import logging
from typing import NamedTuple, Optional
import hgvs.parser
from hgvs.exceptions import HGVSError

logger = logging.getLogger(__name__)


class VariantComponents(NamedTuple):
    """Parsed variant components."""

    gene_symbol: Optional[str]
    accession: Optional[str]
    position: int  # 1-based position from HGVS
    position_0based: int  # 0-based for Python indexing
    ref_aa: str
    alt_aa: str
    hgvs_string: str


class HGVSVariantParser:
    """Parser for HGVS variant notation."""

    # 3-letter to 1-letter amino acid code mapping
    AA_3TO1 = {
        "Ala": "A",
        "Arg": "R",
        "Asn": "N",
        "Asp": "D",
        "Cys": "C",
        "Gln": "Q",
        "Glu": "E",
        "Gly": "G",
        "His": "H",
        "Ile": "I",
        "Leu": "L",
        "Lys": "K",
        "Met": "M",
        "Phe": "F",
        "Pro": "P",
        "Ser": "S",
        "Thr": "T",
        "Trp": "W",
        "Tyr": "Y",
        "Val": "V",
        "Ter": "*",
        "Sec": "U",
        "Pyl": "O",
    }

    def __init__(self):
        self._parser = hgvs.parser.Parser()

    def parse(self, hgvs_string: str) -> VariantComponents:
        """
        Parse HGVS protein variant notation.

        Args:
            hgvs_string: HGVS notation like "NP_004324.2:p.Val600Glu" or "BRAF:p.V600E"

        Returns:
            VariantComponents with parsed data

        Raises:
            ValueError: If HGVS string is invalid
        """
        try:
            variant = self._parser.parse_hgvs_variant(hgvs_string)
        except HGVSError as e:
            raise ValueError(f"Invalid HGVS notation: {hgvs_string}") from e

        # Extract components
        accession = variant.ac if variant.ac else None
        gene_symbol = self._extract_gene_symbol(hgvs_string)

        # Position is 1-based in HGVS
        position = variant.posedit.pos.start.base
        position_0based = position - 1  # Convert to 0-based

        # Amino acids (3-letter in HGVS)
        ref_aa_3 = variant.posedit.pos.start.aa
        alt_aa_3 = variant.posedit.edit.alt

        ref_aa = self.AA_3TO1.get(ref_aa_3, ref_aa_3)
        alt_aa = self.AA_3TO1.get(alt_aa_3, alt_aa_3)

        return VariantComponents(
            gene_symbol=gene_symbol,
            accession=accession,
            position=position,
            position_0based=position_0based,
            ref_aa=ref_aa,
            alt_aa=alt_aa,
            hgvs_string=hgvs_string,
        )

    def _extract_gene_symbol(self, hgvs_string: str) -> Optional[str]:
        """Extract gene symbol from shorthand notation like 'BRAF:p.V600E'."""
        if ":" in hgvs_string and not hgvs_string.startswith("NP_"):
            parts = hgvs_string.split(":")
            if len(parts) >= 2:
                return parts[0]
        return None

    def validate_reference(
        self,
        variant: VariantComponents,
        sequence: str,
    ) -> bool:
        """
        Validate that reference amino acid matches sequence at position.

        Args:
            variant: Parsed variant
            sequence: Protein sequence

        Returns:
            True if reference matches, False otherwise
        """
        if variant.position_0based >= len(sequence):
            logger.warning("Position %s exceeds sequence length %s", variant.position, len(sequence))
            return False

        actual_aa = sequence[variant.position_0based]
        if actual_aa != variant.ref_aa:
            logger.warning(
                f"Reference mismatch at position {variant.position}: expected {variant.ref_aa}, found {actual_aa}"
            )
            return False

        return True


def parse_variant(hgvs_string: str) -> VariantComponents:
    """Convenience function to parse HGVS variant."""
    parser = HGVSVariantParser()
    return parser.parse(hgvs_string)
