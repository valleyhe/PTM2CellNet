"""Integration utilities for PTM2CellNet external explainability pipelines."""

from .contracts import CandidateRecord, GenePerturbationRequest, PerturbationResult
from .genki_reports import (
    build_comparison_summary_payload,
    build_generank_dataframe,
    build_gsea_ranked_dataframe,
    build_result_summary_payload,
    build_significant_gene_dataframe,
    build_two_stage_summary_payload,
    render_comparison_summary_markdown,
    render_two_stage_summary_markdown,
    save_gsea_ranked_tsv,
)
from .ptm_gene_mapper import ProteinGeneMapper
from .ptm_virtual_perturbation import PTMPerturbationProfile, apply_soft_perturbation

try:
    from .genki import GraphUtilities, PerturbationExecutor, ReferenceDataLoader, SignificanceAnalyzer
except ImportError:
    ReferenceDataLoader = None  # type: ignore[misc,assignment]
    PerturbationExecutor = None  # type: ignore[misc,assignment]
    SignificanceAnalyzer = None  # type: ignore[misc,assignment]
    GraphUtilities = None  # type: ignore[misc,assignment]

try:
    from .genki_adapter import GenKIAdapter
except ImportError:
    GenKIAdapter = None  # type: ignore[misc,assignment]

__all__ = [
    "CandidateRecord",
    "GenePerturbationRequest",
    "PerturbationResult",
    "GenKIAdapter",
    "ReferenceDataLoader",
    "PerturbationExecutor",
    "SignificanceAnalyzer",
    "GraphUtilities",
    "build_comparison_summary_payload",
    "build_generank_dataframe",
    "build_gsea_ranked_dataframe",
    "build_result_summary_payload",
    "build_significant_gene_dataframe",
    "build_two_stage_summary_payload",
    "render_comparison_summary_markdown",
    "render_two_stage_summary_markdown",
    "save_gsea_ranked_tsv",
    "ProteinGeneMapper",
    "PTMPerturbationProfile",
    "apply_soft_perturbation",
]
