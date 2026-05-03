"""Typed contracts shared by the PTM2CellNet integration pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class CandidateRecord:
    sample_id: str
    protein_id: str
    ptm_type: str
    ptm_position: int
    baseline_label: str
    baseline_probability: float
    perturbed_probability: float
    delta_probability: float


@dataclass(frozen=True)
class GenePerturbationRequest:
    gene_symbol: str
    source_protein_id: str
    source_ptm_type: str
    source_ptm_position: int
    magnitude: float
    mode: str


@dataclass(frozen=True)
class PerturbationResult:
    gene_symbol: str
    mode: str
    distance_score: float
    ranked_genes: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
