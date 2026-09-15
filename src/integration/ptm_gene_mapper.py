"""Protein to gene mapping helpers for explainability workflows."""

import logging
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd

from .contracts import CandidateRecord

logger = logging.getLogger(__name__)


class ProteinGeneMapper:
    """Maps protein identifiers to gene symbols and vice versa.

    Supports:
      - Forward mapping: protein_id → gene_symbol
      - Reverse mapping: gene_symbol → set of protein_ids
      - UniProt accession resolution
      - Case-insensitive lookup with normalized keys
      - Fuzzy matching for common ID format variations

    Constructed from a dict, a CSV file, or an iterable of
    (protein_id, gene_symbol) pairs.
    """

    def __init__(self, mapping: Dict[str, str]) -> None:
        self.mapping: Dict[str, str] = {k.strip().upper(): v.strip() for k, v in mapping.items()}
        # Keep original keys for reverse lookup
        self._original_keys: Dict[str, str] = {k.strip().upper(): k.strip() for k in mapping}
        # Reverse mapping: gene_symbol → set of protein_ids
        self._reverse: Dict[str, Set[str]] = {}
        for protein_id, gene_symbol in mapping.items():
            normalized_gene = gene_symbol.strip().upper()
            self._reverse.setdefault(normalized_gene, set()).add(protein_id.strip())

    @classmethod
    def from_csv(cls, file_path: str) -> "ProteinGeneMapper":
        """Load mapping from a CSV file with protein_id and gene_symbol columns."""
        df = pd.read_csv(file_path)
        required = {"protein_id", "gene_symbol"}
        if not required.issubset(df.columns):
            raise ValueError(f"Mapping file must contain columns: {sorted(required)}")
        mapping = {str(row["protein_id"]): str(row["gene_symbol"]) for _, row in df.iterrows()}
        return cls(mapping)

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, str]]) -> "ProteinGeneMapper":
        """Create mapper from an iterable of (protein_id, gene_symbol) tuples."""
        return cls(dict(pairs))

    def map_protein(self, protein_id: str) -> Optional[str]:
        """Map a protein identifier to a gene symbol.

        Performs case-insensitive lookup with normalization.
        Returns None if the protein_id is not found.
        """
        normalized = protein_id.strip().upper()
        return self.mapping.get(normalized)

    def map_many(self, protein_ids: Iterable[str]) -> List[str]:
        """Map multiple protein IDs to gene symbols, deduplicating results."""
        genes: List[str] = []
        seen: Set[str] = set()
        for protein_id in protein_ids:
            gene_symbol = self.map_protein(protein_id)
            if gene_symbol and gene_symbol not in seen:
                genes.append(gene_symbol)
                seen.add(gene_symbol)
        return genes

    def map_candidate(self, candidate: CandidateRecord) -> str:
        """Map a CandidateRecord's protein_id to a gene symbol.

        Raises KeyError if the protein_id is not in the mapping.
        """
        gene_symbol = self.map_protein(candidate.protein_id)
        if gene_symbol is None:
            # Try fuzzy match as fallback
            fuzzy = self._fuzzy_match(candidate.protein_id)
            if fuzzy is not None:
                logger.debug(
                    "Fuzzy match: %s -> %s (exact not found)",
                    candidate.protein_id,
                    fuzzy,
                )
                return fuzzy
            raise KeyError(f"Protein id not found in mapper: {candidate.protein_id}")
        return gene_symbol

    def reverse_map(self, gene_symbol: str) -> Set[str]:
        """Map a gene symbol back to the set of protein IDs it maps from.

        Performs case-insensitive lookup.
        Returns an empty set if the gene_symbol is not found.
        """
        normalized = gene_symbol.strip().upper()
        return self._reverse.get(normalized, set())

    def resolve_uniprot(self, uniprot_id: str) -> Optional[str]:
        """Resolve a UniProt accession to a gene symbol.

        Handles common UniProt ID formats:
          - P04637 (bare accession)
          - UniProt:P04637 (prefixed)
          - sp|P04637|P53_HUMAN (pipe-delimited)

        Returns None if the ID is not in the mapping.
        """
        # Strip common prefixes
        cleaned = uniprot_id.strip()
        if cleaned.upper().startswith("UNIPROT:"):
            cleaned = cleaned[8:]
        elif "|" in cleaned:
            parts = cleaned.split("|")
            # Take the accession part (usually second field in sp|P04637|...)
            for part in parts:
                if len(part) >= 6 and part[0].isalpha():
                    cleaned = part
                    break

        return self.map_protein(cleaned)

    def _fuzzy_match(self, protein_id: str) -> Optional[str]:
        """Attempt fuzzy matching for common ID format variations.

        Tries:
          - With/without organism prefix (e.g., "9606.P04637" → "P04637")
          - With/without isoform suffix (e.g., "P04637-2" → "P04637")
          - With/without version suffix (e.g., "P04637.1" → "P04637")
        """
        cleaned = protein_id.strip()

        # Try stripping organism prefix (e.g., "9606.P04637")
        if "." in cleaned:
            parts = cleaned.split(".", 1)
            if parts[1]:
                result = self.map_protein(parts[1])
                if result:
                    return result

        # Try stripping isoform suffix (e.g., "P04637-2")
        if "-" in cleaned:
            base = cleaned.rsplit("-", 1)[0]
            result = self.map_protein(base)
            if result:
                return result

        # Try stripping version suffix (e.g., "P04637.1")
        if "." in cleaned:
            base = cleaned.rsplit(".", 1)[0]
            result = self.map_protein(base)
            if result:
                return result

        return None

    @property
    def size(self) -> int:
        """Number of mappings in the mapper."""
        return len(self.mapping)

    def __contains__(self, protein_id: str) -> bool:
        normalized = protein_id.strip().upper()
        return normalized in self.mapping

    def __len__(self) -> int:
        return len(self.mapping)
