"""Typed contracts shared by the PTM2CellNet integration pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


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

    def validate(self) -> List[str]:
        """Validate this record and return a list of issues (empty if valid)."""
        issues = []
        if not self.sample_id:
            issues.append("sample_id must not be empty")
        if not self.protein_id:
            issues.append("protein_id must not be empty")
        if self.ptm_position < 1:
            issues.append("ptm_position must be >= 1")
        if not 0.0 <= self.baseline_probability <= 1.0:
            issues.append("baseline_probability must be in [0, 1]")
        if not 0.0 <= self.perturbed_probability <= 1.0:
            issues.append("perturbed_probability must be in [0, 1]")
        return issues


@dataclass(frozen=True)
class GenePerturbationRequest:
    gene_symbol: str
    source_protein_id: str
    source_ptm_type: str
    source_ptm_position: int
    magnitude: float
    mode: str

    def validate(self) -> List[str]:
        """Validate this request and return a list of issues (empty if valid)."""
        issues = []
        if not self.gene_symbol:
            issues.append("gene_symbol must not be empty")
        if self.mode not in ("hard_ko", "soft_ptm"):
            issues.append(f"mode must be 'hard_ko' or 'soft_ptm', got '{self.mode}'")
        if self.magnitude <= 0:
            issues.append("magnitude must be positive")
        if self.source_ptm_position < 1:
            issues.append("source_ptm_position must be >= 1")
        return issues


@dataclass(frozen=True)
class PerturbationResult:
    gene_symbol: str
    mode: str
    distance_score: float
    ranked_genes: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Validate this result and return a list of issues (empty if valid)."""
        issues = []
        if not self.gene_symbol:
            issues.append("gene_symbol must not be empty")
        if not self.ranked_genes:
            issues.append("ranked_genes must not be empty")
        return issues


@dataclass(frozen=True)
class BatchPerturbationRequest:
    """Batch of gene perturbation requests for parallel execution."""

    requests: Sequence[GenePerturbationRequest]
    parallel: bool = True
    fail_fast: bool = False

    def validate(self) -> List[str]:
        """Validate the batch and return a list of issues."""
        issues = []
        if not self.requests:
            issues.append("requests must not be empty")
        for i, req in enumerate(self.requests):
            req_issues = req.validate()
            for issue in req_issues:
                issues.append(f"requests[{i}]: {issue}")
            if self.fail_fast and issues:
                break
        return issues


@dataclass(frozen=True)
class BatchPerturbationResult:
    """Results from a batch perturbation execution."""

    results: Sequence[PerturbationResult]
    failed_requests: Sequence[str] = field(default_factory=tuple)
    total_time_ms: float = 0.0

    def validate(self) -> List[str]:
        """Validate the batch result."""
        issues = []
        for i, result in enumerate(self.results):
            result_issues = result.validate()
            for issue in result_issues:
                issues.append(f"results[{i}]: {issue}")
        return issues
