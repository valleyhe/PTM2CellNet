"""Protein to gene mapping helpers for explainability workflows."""

from typing import Dict, Iterable, List

import pandas as pd

from .contracts import CandidateRecord


class ProteinGeneMapper:
    """Maps protein identifiers to gene symbols."""

    def __init__(self, mapping: Dict[str, str]) -> None:
        self.mapping = mapping

    @classmethod
    def from_csv(cls, file_path: str) -> "ProteinGeneMapper":
        df = pd.read_csv(file_path)
        required = {"protein_id", "gene_symbol"}
        if not required.issubset(df.columns):
            raise ValueError(f"Mapping file must contain columns: {sorted(required)}")
        mapping = {str(row["protein_id"]): str(row["gene_symbol"]) for _, row in df.iterrows()}
        return cls(mapping)

    def map_many(self, protein_ids: Iterable[str]) -> List[str]:
        genes: List[str] = []
        for protein_id in protein_ids:
            gene_symbol = self.mapping.get(protein_id)
            if gene_symbol and gene_symbol not in genes:
                genes.append(gene_symbol)
        return genes

    def map_candidate(self, candidate: CandidateRecord) -> str:
        gene_symbol = self.mapping.get(candidate.protein_id)
        if gene_symbol is None:
            raise KeyError(f"Protein id not found in mapper: {candidate.protein_id}")
        return gene_symbol
